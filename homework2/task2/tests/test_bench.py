"""Check physical forward propagation, fresh noise, regions, and feedback search."""

from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import BenchConfig, CameraBench, camera_regions, enhancement, search_focus
from schemas import FocusSettings


def settings(**changes):
    return FocusSettings(**{**dict(macropixel_size=4, focus_x=16, focus_y=16, focus_size=1,
        background_inner_radius=2, background_outer_radius=7, background_regions=4,
        iterations=100, flip_count=1, frames_per_measurement=2, validation_shots=5, seed=3), **changes})


class BenchTests(unittest.TestCase):
    def test_macro_and_incremental_fields_equal_full_physical_transmission(self):
        bench = CameraBench()
        rng = np.random.default_rng(8)
        pattern = rng.integers(0, 2, size=(8, 8))
        for _ in range(8):
            physical = bench.expand_pattern(pattern, 4)
            expected = np.abs(bench.transmission @ physical.ravel())**2
            actual = bench.noiseless_image(pattern, 4)
            np.testing.assert_allclose(actual.ravel(), expected, rtol=1e-6, atol=1e-6)
            pattern.ravel()[rng.integers(pattern.size)] ^= 1

    def test_unchanged_pattern_has_fresh_nonnegative_camera_noise(self):
        bench = CameraBench()
        pattern = np.ones((8, 8), dtype=np.uint8)
        first, second = bench.measure(pattern, 4), bench.measure(pattern, 4)
        self.assertFalse(np.array_equal(first, second))
        self.assertTrue((first >= 0).all() and (second >= 0).all())
        self.assertEqual(bench.shots, 2)

    def test_phase_or_nonbinary_patterns_are_rejected(self):
        bench = CameraBench()
        for value in (.5, 1j):
            with self.assertRaises(ValueError):
                bench.measure(np.full((8, 8), value), 4)

    def test_sectors_are_disjoint_and_uniform_image_has_unit_enhancement(self):
        focus, sectors = camera_regions(settings())
        self.assertFalse(np.any(focus & np.logical_or.reduce(sectors)))
        self.assertEqual(sum(sector.sum() for sector in sectors), np.logical_or.reduce(sectors).sum())
        result = enhancement(np.full((32, 32), 7.0), focus, sectors)
        self.assertAlmostEqual(result["enhancement"], 1)

    def test_search_returns_independent_validation_and_improves_this_seed(self):
        chosen = settings(iterations=150)
        result = search_focus(chosen)
        self.assertGreater(result["final"]["enhancement"], result["initial"]["enhancement"])
        self.assertEqual(len(result["history"]), chosen.iterations)
        self.assertEqual(len(result["final"]["shot_enhancements"]), chosen.validation_shots)
        self.assertTrue(np.isin(result["final_pattern"], (0, 1)).all())
        self.assertGreater(result["camera_shots"], chosen.iterations)


if __name__ == "__main__":
    unittest.main()
