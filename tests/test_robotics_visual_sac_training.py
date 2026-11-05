"""Check the approved SAC budget and scoring boundaries without training."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_visual_sac_training as training


class Guard:
    reason = None

    def elapsed(self):
        return 0.0

    def safe(self):
        return True


class Learner:
    def __init__(self, partial=None):
        self.num_timesteps = 0
        self.calls = []
        self.partial = partial

    def learn(self, *, total_timesteps, reset_num_timesteps, callback, log_interval):
        if reset_num_timesteps:
            raise AssertionError("Must keep accumulated progress and replay")
        self.calls.append(total_timesteps)
        self.num_timesteps += self.partial or total_timesteps
        if self.partial:
            callback.reason = "time_limit"


class ScheduleTests(unittest.TestCase):
    def run_schedule(self, scores, learner=None, guard=None):
        learner = learner or Learner()
        scores = iter(scores)

        def record(model):
            successes, coverage = next(scores)
            return {"movements": model.num_timesteps,
                    "development": {"successes": successes, "mean_final_coverage": coverage}}

        result = training.train_segments(learner, guard or Guard(), record, {"mean_final_coverage": 0.2})
        return learner.calls, result[0], [row["movements"] for row in result[1]]

    def test_no_improvement_stops_after_both_intermediate_checkpoints(self):
        self.assertEqual(self.run_schedule([(0, 0.15), (0, 0.2)]),
                         ([25_000, 25_000], "no_progress_at_50000", [25_000, 50_000]))

    def test_either_success_or_improved_alignment_permits_continuation(self):
        for first, second in [((1, 0.1), (0, 0.15)), ((0, 0.15), (0, 0.4)), ((0, 0.4), (0, 0.15))]:
            with self.subTest(first=first, second=second):
                self.assertEqual(self.run_schedule([first, second, (2, 0.5)]),
                                 ([25_000, 25_000, 50_000], "movement_limit", [25_000, 50_000, 100_000]))

    def test_time_stop_records_last_safe_partial_checkpoint_only(self):
        self.assertEqual(self.run_schedule([(0, 0.15)], learner=Learner(partial=2400)),
                         ([25_000], "time_limit", [2400]))

    def test_resource_stop_prevents_training_and_evaluation(self):
        class NoResources(Guard):
            def safe(self):
                return False

        self.assertEqual(self.run_schedule([], guard=NoResources()), ([], "resource_guard", []))


if __name__ == "__main__":
    unittest.main()
