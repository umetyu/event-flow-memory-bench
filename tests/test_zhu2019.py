from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import h5py
import torch
import yaml

from event_flow_bench.data import encode
from event_flow_bench.fixtures import make_demo, write_fixture
from event_flow_bench.zhu2019 import (PROTOCOL, ZhuFlowNet, displacement, read_config,
    resize_flow, smoothness_loss, temporal_loss, timestamp_image, train, evaluate)

torch.set_num_threads(1)


class ZhuMathTests(unittest.TestCase):
    def test_nine_bin_interpolation_and_sign(self):
        e = tuple(np.array(v) for v in ([3, 3, 3], [2, 2, 2], [0., .3125, 1.], [1., -1., 1.]))
        v = encode(e, [16, 16], 9)["event_voxel"][:, 2, 3]
        torch.testing.assert_close(v, torch.tensor([1., 0, -.5, -.5, 0, 0, 0, 0, 1]))

    def test_splat_polarities_and_partial_border_mass(self):
        e = torch.tensor([[[2., 1., 1., 1.], [4., 1., 1., -1.]]], dtype=torch.double)
        image, mass = timestamp_image(e, torch.tensor([[[1., 1.], [1., 1.]]], dtype=torch.double), [4, 4], 1e-12)
        torch.testing.assert_close(image[0, :, 1, 1], torch.tensor([2., 4.], dtype=torch.double))
        self.assertAlmostEqual(float(mass.sum()), 2.)
        _, partial = timestamp_image(e[:, :1], torch.tensor([[[-.25, 1.]]], dtype=torch.double), [4, 4])
        self.assertAlmostEqual(float(partial.sum()), .75)

    def test_bidirectional_uses_original_timestamp_without_pixel_normalization(self):
        # Both polarities remain separate: 2 * (2**2 + 4**2), NOT reversed time.
        e = torch.tensor([[[.25, 3., 3., 1.], [.5, 3., 3., -1.]]], dtype=torch.double)
        flow = torch.zeros(1, 2, 8, 8, dtype=torch.double)
        loss = temporal_loss(flow, e, [8, 8], eps=1e-12)
        self.assertAlmostEqual(float(loss), 40., places=8)
        e2 = e.clone(); e2[..., 2] += 1
        self.assertAlmostEqual(float(temporal_loss(flow, torch.cat((e, e2), 1), [8, 8], eps=1e-12)), 80., places=8)

    def test_temporal_gradient_matches_finite_difference(self):
        e = torch.tensor([[[.2, 2.1, 2.3, 1.], [.7, 2.9, 2.7, 1.]]], dtype=torch.double)
        scale = torch.tensor(.037, dtype=torch.double, requires_grad=True)
        template = torch.ones(1, 2, 6, 6, dtype=torch.double)
        self.assertTrue(torch.autograd.gradcheck(lambda s: temporal_loss(s * template, e, [6, 6]),
                                               (scale,), eps=1e-6, atol=1e-4))
        gradient = torch.autograd.grad(temporal_loss(scale * template, e, [6, 6]), scale)[0]
        self.assertGreater(abs(float(gradient)), 1e-4)

    def test_smoothness_does_not_cancel_opposite_components(self):
        f = torch.tensor([[[[0., 1.], [0., 1.]], [[0., -1.], [0., -1.]]]])
        expected = 8 * np.sqrt(1 + .001 ** 2) + 8 * .001
        self.assertAlmostEqual(float(smoothness_loss(f)), expected, places=5)

    def test_spatial_and_temporal_unit_conversion(self):
        f = torch.ones(1, 2, 2, 4)
        up = resize_flow(f, (4, 12))
        torch.testing.assert_close(up[:, 0], torch.full((1, 4, 12), 3.))
        torch.testing.assert_close(up[:, 1], torch.full((1, 4, 12), 2.))
        torch.testing.assert_close(displacement(f, .04, .01), f * 2)
        with self.assertRaises(ValueError): displacement(f, 0, .01)


class ZhuTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fixture = make_demo(self.root / "data")
        self.c = dict(protocol=PROTOCOL, seed=0, device="cpu", data=fixture["data"],
                      model=dict(num_bins=9, base_num_channels=2, kernel_size=3),
                      train=dict(epochs=1, batch_size=2, events_per_input=128, lr=.0002,
                                 clip_grad=100., smooth_weight=1., charbonnier_eps=.001,
                                 splat_eps=1e-6, augment=True))

    def tearDown(self):
        self.tmp.cleanup()

    def test_model_scales_and_backward(self):
        net = ZhuFlowNet(self.c)
        output = net(torch.randn(1, 9, 32, 32))
        self.assertEqual([list(f.shape[-2:]) for f in output], [[4, 4], [8, 8], [16, 16], [32, 32]])
        sum(f.square().mean() for f in output).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_config_refuses_wrong_bins_and_test_data(self):
        path = self.root / "c.yml"
        path.write_text(yaml.safe_dump(self.c))
        self.assertEqual(read_config(path), self.c)
        bad = deepcopy(self.c); bad['model']['num_bins'] = 5
        path.write_text(yaml.safe_dump(bad))
        with self.assertRaises(ValueError): read_config(path)
        bad = deepcopy(self.c); bad['data']['dataset'] = 'mvsec'
        path.write_text(yaml.safe_dump(bad))
        with self.assertRaises(ValueError): read_config(path)

    def test_train_resume_matches_continuous_training(self):
        train(self.c, self.root / "resumed")
        c2 = deepcopy(self.c); c2['train']['epochs'] = 2
        train(c2, self.root / "resumed", self.root / "resumed/latest.pt")
        train(c2, self.root / "continuous")
        a = torch.load(self.root / "resumed/latest.pt", weights_only=True)
        b = torch.load(self.root / "continuous/latest.pt", weights_only=True)
        self.assertEqual(a['epoch'], 1)
        for k in a['state_dict']:
            torch.testing.assert_close(a['state_dict'][k], b['state_dict'][k], rtol=0, atol=0)
        self.assertTrue((self.root / 'resumed/completed.json').exists())

    def test_smoke_is_not_a_full_training_checkpoint(self):
        result = train(self.c, self.root / "smoke", max_updates=1)
        self.assertEqual(result['status'], 'smoke_complete_not_full_training')
        self.assertTrue((self.root / 'smoke/smoke.json').exists())
        self.assertFalse((self.root / 'smoke/best.pt').exists())
        self.assertFalse((self.root / 'smoke/completed.json').exists())

    def test_mvsec_evaluation_is_causal_and_separates_score_definitions(self):
        path = write_fixture(self.root / 'indoor_flying1.h5', events=15520, resolution=(256, 256))
        manifest = self.root / 'mvsec.json'
        manifest.write_text(json.dumps({'version': 1, 'dataset': 'mvsec', 'test': [
            {'path': str(path), 'group': 'indoor_flying1', 'id': 'indoor_flying1'}]}))
        checkpoint = self.root / 'model.pt'
        torch.save({'config': self.c, 'state_dict': ZhuFlowNet(self.c).state_dict(),
                    'manifest_sha256': 'synthetic_training_fixture'}, checkpoint)
        first = evaluate(checkpoint, manifest, self.root / 'dt1.json', 'cpu', 1)
        self.assertTrue(first['rows'])
        row = first['rows'][0]
        self.assertEqual(row['end_event'] - row['input_start_event'], 15000)
        self.assertIn('outlier_pct_3px', row)
        with h5py.File(path, 'r+') as f:
            f['events/xs'][row['end_event']:] = 0
        again = evaluate(checkpoint, manifest, self.root / 'again.json', 'cpu', 1)
        self.assertEqual(row, again['rows'][0])
        dt4 = evaluate(checkpoint, manifest, self.root / 'dt4.json', 'cpu', 4)
        self.assertEqual(dt4['dt'], 4)
        self.assertFalse(dt4['evaluation']['matches_hagenaars_upstream_protocol'])


if __name__ == '__main__':
    unittest.main()
