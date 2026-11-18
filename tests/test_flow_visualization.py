import unittest
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import h5py
import numpy as np
import torch
from event_flow_bench.config import DEFAULT
from event_flow_bench.fixtures import write_fixture
from event_flow_bench.flow_visualization import aligned_chunks, flow_rgb, to_gt_displacement


class FlowVisualizationTests(unittest.TestCase):
    def test_final_input_is_full_without_duplicate_or_missing_events(self):
        for end in (1000, 500000, 500001, 500999):
            chunks = list(aligned_chunks(end, 1000))
            self.assertEqual(chunks[0][0], 0)
            self.assertEqual(chunks[-1], (end-1000, end))
            self.assertEqual(sum(b-a for a,b in chunks), end)
            self.assertTrue(all(chunks[i][1] == chunks[i+1][0] for i in range(len(chunks)-1)))
            self.assertTrue(all(b-a == 1000 for a,b in chunks[1:]))
        with self.assertRaises(ValueError): list(aligned_chunks(999, 1000))

    def test_fixed_scale_direction_zero_and_invalid_are_distinct(self):
        f=np.array([[[0.,5.,-5.,0.]],[[0.,0.,0.,5.]]])
        rgb=flow_rgb(f,5.)
        np.testing.assert_allclose(rgb[0,0],[1.,1.,1.])
        np.testing.assert_allclose(rgb[0,1],[1.,0.,0.])
        np.testing.assert_allclose(rgb[0,2],[0.,1.,1.])
        self.assertGreater(rgb[0,3,1],rgb[0,3,2])
        masked=flow_rgb(f,5.,np.array([[False,True,True,True]]))
        np.testing.assert_equal(masked[0,0],[0.,0.,0.])

    def test_same_vector_has_same_color_across_panels(self):
        a=np.array([[[1.,2.]],[[2.,1.]]])
        b=a.copy(); b[:,0,1]=1000
        np.testing.assert_equal(flow_rgb(a,5.)[0,0],flow_rgb(b,5.)[0,0])

    def test_gt_interval_units_use_final_stride(self):
        f=np.ones((2,3,3))*.01
        np.testing.assert_allclose(to_gt_displacement(f,128,.002,.02),12.8)
        with self.assertRaises(ValueError): to_gt_displacement(f,128,0,.02)

    def test_capture_uses_full_input_duration_and_never_future_events(self):
        script = Path(__file__).resolve().parents[1]/"scripts/plot_flow_comparison.py"
        spec = importlib.util.spec_from_file_location("flow_figure_script",script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        class ConstantModel:
            def reset(self): pass
            def __call__(self,inputs):
                out = torch.zeros(1,2,16,16); out[:,0] = 1/128
                return {"flow":[out]}
        c = deepcopy(DEFAULT)
        c["device"] = "cpu"; c["data"]["resolution"] = [16,16]
        c["window"].update(history_events=1000,stride_events=1000,warmup_events=0)
        with tempfile.TemporaryDirectory() as directory:
            path = write_fixture(Path(directory)/"sequence.h5",events=1300)
            entry = {"path":str(path),"id":"fixture"}
            arrays, record = module.capture([ConstantModel(),ConstantModel()],[c,c],entry,1,1000)
            self.assertEqual(record["input_events"],1000)
            self.assertEqual(record["start_event"],1024)
            self.assertEqual(record["end_event"],1088)
            self.assertAlmostEqual(record["input_seconds"],.0999)
            np.testing.assert_allclose(arrays["convgru_flow"][0],.0064/.0999,rtol=1e-6)
            self.assertAlmostEqual(record["convgru_metrics"]["aee"],1-.0064/.0999,places=6)
            np.testing.assert_array_equal(arrays["display_mask"],arrays["gt_valid"] & (arrays["event_counts"].sum(0)>0))
            # Alter everything beyond the selected GT endpoint: the result must
            # remain identical, including the recurrent warmup input sequence.
            with h5py.File(path,"r+") as f: f["events/xs"][1088:] = 0
            again, _ = module.capture([ConstantModel(),ConstantModel()],[c,c],entry,1,1000)
            for key in arrays: np.testing.assert_array_equal(arrays[key],again[key])


if __name__=="__main__": unittest.main()
