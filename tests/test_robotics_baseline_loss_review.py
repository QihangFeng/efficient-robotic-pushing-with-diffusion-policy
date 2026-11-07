"""CPU-only tests for saved-record diagnostics; no policy or simulator imports."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_baseline_loss_review as review


def state(coverage=0., x=0.):
    return {"coverage": coverage, "block_pos": [x, 0.], "block_angle": 0.,
            "pusher_pos": [x, 0.], "image_sha256": str(x)}


def trace(values, starts=(1,), positions=None):
    positions = list(range(1, len(values))) if positions is None else positions
    return {"initial": state(values[0]), "steps": [
        {"step": i, "new_prediction": i in starts, "action": [float(i), 0.],
         "after": state(value, positions[i - 1])}
        for i, value in enumerate(values[1:], 1)]}


class WindowTests(unittest.TestCase):
    def test_drawdown_differs_from_net_loss_and_can_end_in_success(self):
        t = trace([.9, .7, .96])
        result = review.window(t["initial"], t["steps"])
        self.assertAlmostEqual(result["net_change"], .06)
        self.assertAlmostEqual(result["maximum_drawdown"], .2)
        self.assertEqual(result["movements"], [1, 2])

    def test_empty_window_is_not_a_four_action_extension(self):
        result = review.window(state(.8), [])
        self.assertEqual(result["net_change"], 0)
        self.assertEqual(result["movements"], [])

    def test_unit_stops_before_next_prediction(self):
        t = trace([0., .2, .3, .5, .6], starts=(1, 3))
        self.assertEqual([(i, len(rows)) for i, rows in review.units(t)], [(0, 2), (2, 2)])

    def test_extension_suffix_stops_at_early_interruption(self):
        t = trace([0.] * 13, starts=(1, 11))
        t["steps"][7]["plan_age"] = 8
        result = review.extension_windows({"events": [{"kind": "extend", "step": 8, "features": {}}]}, t)
        self.assertEqual(result[0]["movements"], [9, 10])
        self.assertFalse(result[0]["complete_four_movements"])


class PrefixTests(unittest.TestCase):
    def fixture(self):
        a = trace([0.] * 13, starts=(1, 9))
        b = trace([0.] * 13, starts=(1,))
        return a, b, [{"kind": "extend", "step": 8}]

    def test_identifies_exact_movements_before_extension(self):
        a, b, e = self.fixture()
        result = review.first_schedule_difference(a, b, e)
        self.assertEqual(result["exact_shared_movements"], 8)
        self.assertEqual(result["first_different_movement"], 9)
        self.assertEqual(result["adaptive_next_four"]["movements"], [9, 10, 11, 12])

    def test_rejects_changed_action_before_timing_diverges(self):
        a, b, e = self.fixture()
        b["steps"][3]["action"] = [-1., -1.]
        with self.assertRaises(AssertionError):
            review.first_schedule_difference(a, b, e)

    def test_rejects_changed_rgb_before_timing_diverges(self):
        a, b, e = self.fixture()
        b["steps"][3]["after"]["image_sha256"] = "changed"
        with self.assertRaises(AssertionError):
            review.first_schedule_difference(a, b, e)

    def test_only_coverage_has_roundoff_tolerance(self):
        a, b, e = self.fixture()
        b["steps"][3]["after"]["coverage"] += 1e-13
        review.first_schedule_difference(a, b, e)
        b["steps"][3]["after"]["block_pos"][0] += 1e-13
        with self.assertRaises(AssertionError):
            review.first_schedule_difference(a, b, e)

    def test_no_timing_difference_is_explicit(self):
        a, _, _ = self.fixture()
        self.assertIsNone(review.first_schedule_difference(a, a, [])["first_different_movement"])


class PreservationTests(unittest.TestCase):
    def test_existing_review_is_never_overwritten(self):
        parent = Path(__file__).resolve().parents[1] / "tmp"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            path = Path(folder) / "summary.json"
            review.save_once(path, {"a": 1})
            prior = path.read_bytes()
            review.save_once(path, {"a": 1})
            with self.assertRaises(AssertionError):
                review.save_once(path, {"a": 2})
            self.assertEqual(path.read_bytes(), prior)

    def test_review_cannot_write_into_previous_results(self):
        with self.assertRaises(AssertionError):
            review.run(review.SOURCE)

    def test_analysis_does_not_import_policy_or_simulator_libraries(self):
        # Inspect this module's imports instead of the process-wide import cache.
        import ast
        tree = ast.parse(Path(review.__file__).read_text())
        modules = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        modules += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any(m and m.split('.')[0] in {'torch', 'gymnasium', 'gym_pusht', 'lerobot'} for m in modules))


class SavedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result, cls.traces = review.build_review()

    def test_all_evidence_and_all_nine_losses_are_included(self):
        self.assertEqual(self.result['attempts_checked'], 320)
        self.assertEqual(self.result['movements_checked'], 70797)
        self.assertEqual(self.result['lost_attempts'], 9)
        self.assertEqual(self.result['distinct_lost_scenes'], 7)
        self.assertEqual(self.result['gained_attempts'], 23)

    def test_controls_are_retained_for_failure_marker_interpretation(self):
        groups = self.result['combined_method_comparison_groups']
        self.assertEqual({k:v['attempts'] for k,v in groups.items()},
                         {'retained':37,'gained':23,'lost':9,'shared_failure':11})
        self.assertEqual(sum(v['attempts'] for v in groups.values()), 80)
        self.assertEqual(sum(self.result['trained_first_schedule_difference_counts'].values()),80)

    def test_component_successes_are_not_mislabelled_as_causes(self):
        self.assertEqual(sum(l['methods']['recovery8']['success'] for l in self.result['losses']), 7)
        self.assertEqual(sum(l['methods']['milestone12']['success'] for l in self.result['losses']), 3)
        self.assertTrue(all(l['trained_schedule_divergence']['exact_shared_movements']==8 for l in self.result['losses']))
        self.assertTrue(any('caus' in s for s in self.result['limitations']))


if __name__ == '__main__':
    unittest.main()
