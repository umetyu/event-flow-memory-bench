import importlib.util
import json
import os
from pathlib import Path
import unittest

import numpy as np
import torch

from event_flow_bench.cmax import PatchCMax2018, SecretsCMax2022, secrets_to_xy, splat

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_cmax", ROOT/"scripts/run_cmax.py")
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


def moving_texture(resolution=64, velocity=(30., -20.), duration=.1):
    rng = np.random.RandomState(9)
    points = rng.uniform(10, 21, (80, 2))
    all_events = []
    for time in np.linspace(0, duration, 20):
        for py in range(resolution//32):
            for px in range(resolution//32):
                xy = points + [32*px, 32*py] + np.array(velocity)*time
                all_events.extend((x, y, time, 1.) for x, y in xy)
    return tuple(np.asarray(all_events, dtype=np.float64).T)


class CMaxTests(unittest.TestCase):
    def test_splat_gradient_mass_and_patch_separation(self):
        x = torch.tensor([3.2, 2.4], dtype=torch.double, requires_grad=True)
        y = torch.tensor([4.6, 1.3], dtype=torch.double, requires_grad=True)
        ids = torch.tensor([0, 1])
        self.assertTrue(torch.autograd.gradcheck(lambda a,b: splat(a,b,ids,2,8), (x,y)))
        torch.testing.assert_close(splat(x,y,ids,2,8).sum((1,2,3)), torch.ones(2,dtype=torch.double))

    def test_local_translation_sign_units_and_zero_sparse_patch(self):
        c = json.loads((ROOT/"configs/cmax.json").read_text())["patch2018"]
        c.update(iterations=60, initial_displacements=[-4, 0, 4])
        torch.set_num_threads(1)
        events = moving_texture()
        flow, diag = PatchCMax2018((64,64), c).estimate(events)
        # Bilinear rasterization approximates the continuous IWE: require subpixel
        # recovery (<0.3 px over this 0.1 s interval), not exact continuous velocity.
        self.assertLess(np.linalg.norm((flow[:,16,16]-[30.,-20.])*.1), .3)
        self.assertGreater(diag["contrast_ratio"], 1.)
        sparse = tuple(e[:1] for e in events)
        empty, diag = PatchCMax2018((64,64), c).estimate(sparse)
        self.assertEqual(float(np.abs(empty).sum()), 0.)
        self.assertEqual(diag["estimated_pixel_fraction"], 0.)

    def test_causal_window_and_gt_units_mask(self):
        self.assertEqual(runner.input_bounds(1300,1000), (300,1300))
        self.assertEqual(runner.input_bounds(300,1000), (0,300))
        velocity = np.zeros((2,4,5), np.float32); velocity[0] = 100
        gt = torch.zeros(1,2,4,5); gt[0,0,1,2] = 2
        valid = torch.ones(1,4,5,dtype=torch.bool)
        e = tuple(np.array(v) for v in ([2],[1],[.01],[1]))
        result = runner.score_velocity(velocity,gt,valid,e,.02)
        self.assertEqual(result["valid_pixels"],1)
        self.assertEqual(result["aee"],0.)
        invalid = runner.score_velocity(velocity,gt,valid & False,e,.02)
        self.assertIsNone(invalid["aee"])

    def test_official_axis_adapter(self):
        flow = np.stack((np.full((4,5),-20.), np.full((4,5),30.)))
        np.testing.assert_equal(secrets_to_xy(flow)[:,2,3], [30.,-20.])

    @unittest.skipUnless(os.environ.get("EFBENCH_SECRETS_CHECKOUT"), "official solver integration requires its optional environment")
    def test_official_warp_sign_and_seconds(self):
        c = json.loads((ROOT/"configs/cmax.json").read_text())["secrets2022"]
        solver = SecretsCMax2022((64,64),c,os.environ["EFBENCH_SECRETS_CHECKOUT"])
        # Upstream patch parameters are negative row/column velocity.
        s = solver.solver
        motion = np.zeros((2,*s.patch_image_size)); motion[0] = 20; motion[1] = -30
        dense = s.interpolate_dense_flow_from_patch_numpy(motion)
        np.testing.assert_allclose(secrets_to_xy(dense)[:,20,20],[30,-20],atol=1e-10)
        event = np.array([[20.,20.,0.,1.],[18.,23.,.1,1.]])
        warped,_ = s.warper.warp_event(event,dense*.1,"dense-flow",direction="first")
        np.testing.assert_allclose(warped[:,:2],[[20,20],[20,20]],atol=1e-10)

    @unittest.skipUnless(os.environ.get("EFBENCH_SECRETS_CHECKOUT"), "official solver integration requires its optional environment")
    def test_official_solver_recovers_known_translation(self):
        c = json.loads((ROOT/"configs/cmax.json").read_text())["secrets2022"]
        c["solver"]["patch"].update(scale=3, initialize="zero")
        c["optimizer"]["n_iter"] = 10
        solver = SecretsCMax2022((64,64),c,os.environ["EFBENCH_SECRETS_CHECKOUT"])
        flow, diagnostics = solver.estimate(moving_texture(), seed=0)
        error = np.linalg.norm((flow[:,16,16]-[30.,-20.])*.1)
        self.assertLess(error, .5)  # half-pixel displacement accuracy on held-out synthetic motion
        self.assertEqual(len(diagnostics["scales"]),2)


if __name__ == "__main__": unittest.main()
