from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
import torch
from event_flow_bench.config import DEFAULT, validate, write_json, upstream_config
from event_flow_bench.data import EventFile, encode, sample, batch, load_manifest, split_manifest
from event_flow_bench.engine import evaluate, load_checkpoint, train
from event_flow_bench.experiments import collect, generate_grid
from event_flow_bench.fixtures import make_demo, write_fixture
from event_flow_bench.metrics import flow_metrics, mask_metrics, WarpMetrics
from event_flow_bench.models import MODEL_NAMES, build_model
from event_flow_bench.reproduce import load_weights, reproduce
from event_flow_bench._vendor.event_flow.loss.flow import EventWarping

torch.set_num_threads(1)

class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.c = make_demo(self.root)
    def tearDown(self):
        self.tmp.cleanup()

    def test_temporal_voxel_and_count_polarity(self):
        e = (np.array([3,3,3]), np.array([2,2,2]), np.array([0.,.5,1.]), np.array([1.,-1.,1.]))
        x = encode(e, [16,16], 5)
        self.assertEqual(x["event_cnt"][:,2,3].tolist(), [2,1])
        self.assertEqual(x["event_voxel"][:,2,3].tolist(), [1,0,-1,0,1])

    def test_zero_duration_stays_finite(self):
        e = tuple(np.array(v) for v in ([1,2],[1,2],[10.,10.],[1.,-1.]))
        x = encode(e,[16,16],5)
        self.assertTrue(torch.isfinite(x["event_voxel"]).all())
        self.assertEqual(float(x["dt_input"]),0)

    def test_causal_history_and_current_mask(self):
        path = self.root/'sequence-0.h5'
        with EventFile(path,[16,16]) as stream:
            x = sample(stream,32,48,self.c)
            self.assertEqual(float(x['event_cnt'].sum()),32)
            self.assertEqual(len(x['event_list']),16)
            now = encode(stream.read(32,48),[16,16],2)
            torch.testing.assert_close(x['event_mask'], now['event_mask'])
        with h5py.File(path,'r+') as f:
            f['events/xs'][48:] = 0
        with EventFile(path,[16,16]) as stream:
            y = sample(stream,32,48,self.c)
        for k in x:
            torch.testing.assert_close(x[k],y[k])

    def test_float64_timestamp_origin(self):
        path = write_fixture(self.root/'epoch.h5',epoch_timestamp=True)
        with EventFile(path,[16,16]) as stream:
            e = stream.read(0,16)
            self.assertGreater(e[2][-1]-e[2][0],.001)
            self.assertTrue(torch.isfinite(encode(e,[16,16],5)['event_voxel']).all())

    def test_left_search_repeated_timestamp(self):
        path=self.root/'sequence-0.h5'
        with h5py.File(path,'r+') as f:
            f['events/ts'][5:8]=.0005
        with EventFile(path,[16,16]) as stream:
            self.assertEqual(stream.index(.0005),5)

    def test_split_rejects_recording_leakage(self):
        path=self.root/'manifest.json'
        m=json.loads(path.read_text()); m['val'][0]['group']=m['train'][0]['group']; write_json(path,m)
        with self.assertRaisesRegex(ValueError,'leakage'):
            load_manifest(path)

    def test_grouped_split_is_reproducible(self):
        path=self.root/'manifest.json'
        m=json.loads(path.read_text()); m['train']+=m.pop('val'); write_json(path,m)
        a=split_manifest(path,self.root/'a.json',.33,42)
        b=split_manifest(path,self.root/'b.json',.33,42)
        self.assertEqual(a,b)
        self.assertFalse({e['group'] for e in a['train']} & {e['group'] for e in a['val']})

    def test_config_rejects_confounded_window(self):
        c=deepcopy(self.c); c['window']['loss_events']=65
        with self.assertRaises(ValueError): validate(c)

    def test_history_may_exceed_loss_without_changing_stride(self):
        c=deepcopy(self.c); c['window']['history_events']=160
        validate(c)
        self.assertEqual(c['window']['stride_events'],16)

    def test_out_of_bounds_events_fail(self):
        path=self.root/'sequence-0.h5'
        with h5py.File(path,'r+') as f: f['events/xs'][0]=99
        with EventFile(path,[16,16]) as stream:
            with self.assertRaises(ValueError): stream.read(0,16)

class MetricTests(unittest.TestCase):
    def test_valid_static_gt_and_no_event_mask(self):
        pred=torch.zeros(1,2,2,2); gt=torch.zeros_like(pred)
        pred[0,0,0,0]=2
        valid=torch.ones(1,2,2,dtype=torch.bool); event=valid.clone(); event[0,0,1]=False
        result=flow_metrics(pred,gt,valid,event)
        self.assertEqual(result['valid_pixels'],3)
        self.assertAlmostEqual(result['aee'],2/3,6)

    def test_gt_and_prediction_threshold_conventions(self):
        gt=torch.tensor([[[[100.]],[[0.]]]])
        pred=torch.tensor([[[[95.]],[[0.]]]])
        mask=torch.ones(1,1,1,dtype=torch.bool)
        self.assertEqual(flow_metrics(pred,gt,mask,mask,'gt')['outlier_pct'],0)
        self.assertEqual(flow_metrics(pred,gt,mask,mask,'prediction')['outlier_pct'],100)

    def test_empty_valid_mask_is_not_zero_error(self):
        z=torch.zeros(1,2,2,2); mask=torch.zeros(1,2,2,dtype=torch.bool)
        self.assertIsNone(flow_metrics(z,z,mask,mask)['aee'])

    def test_nonfinite_prediction_fails(self):
        z=torch.zeros(1,2,2,2); p=z.clone(); p[0,0,0,0]=float('nan')
        mask=torch.ones(1,2,2,dtype=torch.bool)
        with self.assertRaises(FloatingPointError): flow_metrics(p,z,mask,mask)

    def test_nan_gt_is_excluded(self):
        gt=torch.zeros(1,2,2,2); gt[0,0,0,0]=float('nan'); mask=torch.ones(1,2,2,dtype=torch.bool)
        self.assertEqual(flow_metrics(torch.zeros_like(gt),gt,mask,mask)['valid_pixels'],3)

    def test_mask_metrics_empty_and_exact(self):
        mask=np.zeros((3,3),bool); self.assertEqual(mask_metrics(mask,mask)['iou'],1)
        mask[1,1]=True; self.assertEqual(mask_metrics(mask,mask)['detected'],1)
        self.assertEqual(mask_metrics(~mask,mask)['detected'],0)

    def test_zero_flow_warp_ratios_and_contrast_gradient(self):
        c=deepcopy(DEFAULT); c['data']['resolution']=[16,16]; c['eval']['flow_scale']=16
        e=tuple(np.array(v) for v in ([3,4,5,6],[4,4,4,4],[0.,.3,.6,1.],[1.,1.,-1.,-1.]))
        inputs=batch([encode(e,[16,16],2)],'cpu')
        metric=WarpMetrics(c); metric.add([torch.zeros(1,2,16,16)],inputs)
        result=metric.compute()
        self.assertAlmostEqual(result['fwl'],1,6); self.assertAlmostEqual(result['rsat'],1,6)
        self.assertAlmostEqual(result['warped_event_fraction'],1,6)
        loss=EventWarping(upstream_config(c),'cpu',flow_scaling=16)
        flow=(torch.randn(1,2,16,16)*.001).requires_grad_()
        loss.event_flow_association([flow],inputs['event_list'].clone(),inputs['event_list_pol_mask'],inputs['event_mask'])
        value=loss(); value.backward()
        self.assertTrue(torch.isfinite(value)); self.assertTrue(torch.isfinite(flow.grad).all())
        self.assertGreater(float(flow.grad.abs().sum()),0)

class ModelTests(unittest.TestCase):
    def test_all_upstream_models_shapes_gradients_and_reset(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                c=deepcopy(DEFAULT); c['data']['resolution']=[16,16]; c['model'].update(name=name,base_num_channels=2)
                if any(x in name for x in ('LIF','Spiking')):
                    c['model']['activations']=['arctanspike','arctanspike']
                model=build_model(c)
                inputs={'event_cnt':torch.rand(1,2,16,16),'event_voxel':torch.rand(1,2,16,16)}
                first=model(inputs)['flow'][-1]
                self.assertEqual(tuple(first.shape),(1,2,16,16))
                first.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                model.detach(); model(inputs); model.reset()
                reset=model(inputs)['flow'][-1]
                torch.testing.assert_close(first.detach(),reset.detach())

    def test_voxel_input_for_both_feedforward_families(self):
        for name in ('EVFlowNet','FireFlowNet'):
            c=deepcopy(DEFAULT); c['data']['resolution']=[16,16]
            c['model'].update(name=name,base_num_channels=2,encoding='voxel',num_bins=5)
            out=build_model(c)({'event_cnt':torch.zeros(1,2,16,16),'event_voxel':torch.rand(1,5,16,16)})
            self.assertEqual(out['flow'][-1].shape,torch.Size([1,2,16,16]))

    @unittest.skipUnless(os.environ.get('EFB_LAYERSEG_SOURCE'),'optional checkout not configured')
    def test_layerseg_adapter_forward_and_backward(self):
        c=deepcopy(DEFAULT); c['data'].update(dataset='affineobjects',resolution=[16,16])
        c['model'].update(name='EVLayerSegNet',encoding='voxel',num_bins=5,source=os.environ['EFB_LAYERSEG_SOURCE'])
        model=build_model(c)
        out=model({'event_cnt':torch.rand(1,2,16,16),'event_voxel':torch.rand(1,5,16,16)})
        self.assertEqual(out['flow'][0].shape,torch.Size([1,2,16,16]))
        self.assertEqual(out['alpha_masks'].shape,torch.Size([1,2,16,16]))
        out['flow'][0].square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name); self.c=make_demo(self.root/'fixture')
    def tearDown(self): self.tmp.cleanup()

    def test_train_reload_eval_and_resume_match_uninterrupted(self):
        best=train(self.c,self.root/'one')
        model,state=load_checkpoint(best,self.c)
        a=evaluate(self.c,model,'test',self.root/'test.json')
        b=evaluate(self.c,model,'test')
        self.assertEqual(a['summary'],b['summary'])
        self.assertGreater(a['summary']['rsat_n'],0)
        c=deepcopy(self.c); c['train']['epochs']=2
        train(c,self.root/'one',self.root/'one/latest.pt')
        train(c,self.root/'two')
        x=torch.load(self.root/'one/latest.pt',weights_only=True)
        y=torch.load(self.root/'two/latest.pt',weights_only=True)
        for k in x['state_dict']: torch.testing.assert_close(x['state_dict'][k],y['state_dict'][k],rtol=0,atol=0)
        rows=collect([self.root/'test.json'],self.root/'table.csv')
        self.assertTrue(rows)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            collect([self.root/'test.json',self.root/'test.json'],self.root/'dup.csv')

    def test_checkpoint_history_mismatch_fails(self):
        best=train(self.c,self.root/'one')
        c=deepcopy(self.c); c['window']['history_events']=64
        with self.assertRaisesRegex(ValueError,'history'): load_checkpoint(best,c)

    def test_gt_interval_flow_units_dt1_dt4(self):
        class ConstantVelocity(torch.nn.Module):
            def reset(self): pass
            def forward(self,inputs):
                flow=torch.zeros(1,2,16,16)
                flow[:,0]=inputs['dt_input'].float() * (1/.0064) / 16
                return {'flow':[flow]}
        c=deepcopy(self.c)
        for mode in ('gtflow_dt1','gtflow_dt4'):
            c['eval']['mode']=mode
            c['window']['warmup_events']=0
            result=evaluate(c,ConstantVelocity(),'test')
            self.assertLess(result['summary']['aee'],1e-5)
            self.assertEqual(result['summary']['outlier_pct'],0)

    def test_upstream_reproduction_loads_state_dict_and_scores_both_dt(self):
        c=deepcopy(self.c); model=build_model(c)
        checkpoint=self.root/'weights.pt'; torch.save(model.net.state_dict(),checkpoint)
        training={'model':c['model'],'loss':{'flow_regul_weight':.001},'spiking_neuron':None}
        single=self.root/'mvsec'; single.mkdir(); write_fixture(single/'outdoor_day1.h5')
        for dt in (1,4):
            result=reproduce(single,checkpoint,training,dt,resolution=[16,16])
            self.assertEqual(result['protocol'],'hagenaars2021-upstream')
            self.assertEqual(result['summary']['aee_invalid'],0)
        with self.assertRaises(FileNotFoundError): load_weights(self.root/'missing.pt')

    def test_grid_expected_cells(self):
        jobs=generate_grid(self.root/'grid','manifest.json',histories=[1000,5000,15000,50000])
        self.assertEqual(len(jobs),60)
        self.assertEqual(len({j['name'] for j in jobs}),60)

if __name__=='__main__': unittest.main()
