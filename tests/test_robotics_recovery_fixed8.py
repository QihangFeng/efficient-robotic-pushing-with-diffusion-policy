"""CPU-only guards for the missing fine-tuned/fixed-eight ablation cell."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_recovery_fixed8 as screen
from test_robotics_recovery_confirmation import ToyPolicy


def temporary():
    parent = Path(__file__).resolve().parents[1] / "tmp"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="recovery-fixed8-tests-", dir=parent)


def context(folder):
    output = Path(folder) / "output"
    screen.base.save_json(output / "protocol.json", {"scope": "test"})
    policy = ToyPolicy()
    original = screen.training.tail_state(policy)
    trained = {key: value + 1. for key, value in original.items()}
    protocol = {"original_head_digest": screen.training.tail_digest(original),
                "final_head": {"tensor_digest": screen.training.tail_digest(trained)}}
    return output, policy, {"original": original, "trained": trained}, protocol


def records():
    result = []
    for scene in screen.SCENES:
        for draw in screen.DRAWS:
            for arm in screen.ARMS:
                trained = arm in ("recovery8", "recovery12")
                queries = 10 if trained else 12
                result.append({"arm": arm, "trial": {
                    "scene": scene, "draw": draw, "method": screen.ARMS[arm],
                    "success": trained or scene % 2 == 0,
                    "final_coverage": .96 if trained else .90,
                    "model_queries": queries, "wall_seconds": float(queries * 5),
                    "monitor_seconds": 1. if screen.ARMS[arm] == "adaptive12" else 0.,
                    "prediction_seconds": [5.] * queries}})
    return result


class ScheduleTest(unittest.TestCase):
    def test_exactly_missing_eighty_attempts_match_previous_scenes_and_draws(self):
        self.assertEqual(screen.SCENES, list(range(170, 210)))
        self.assertEqual(screen.DRAWS, [60000, 70000])
        expected = [{"scene": scene, "draw": draw, "arm": "recovery8"}
                    for scene in range(170, 210) for draw in [60000, 70000]]
        self.assertEqual(screen.schedule(), expected)
        self.assertEqual(len(expected), 80)

    def test_existing_evaluator_arms_are_not_modified(self):
        self.assertEqual(screen.prior.ARMS, {"fixed8": "fixed8", "milestone12": "adaptive12", "recovery12": "adaptive12"})
        self.assertEqual(screen.ARMS["recovery8"], "fixed8")

    def test_cannot_write_previous_evidence(self):
        for path in [screen.REFERENCE, screen.prior.old.OUTPUT, Path("results"), Path("../outside")]:
            with self.subTest(path=path), self.assertRaises(AssertionError):
                screen.check_output(path)
        screen.check_output(screen.OUTPUT)
        screen.check_output(screen.OUTPUT.with_name(screen.OUTPUT.name + "-repeat"))


class BindingAndCollectionTest(unittest.TestCase):
    def test_trained_weights_are_used_despite_fixed8_inner_method(self):
        with temporary() as folder:
            output, policy, heads, protocol = context(folder)
            backbone = screen.training.backbone_digest(policy)
            item = screen.schedule()[0]

            def evaluate(actual, scene, draw, method, directory, max_steps):
                self.assertIs(actual, policy)
                self.assertEqual(method, "fixed8")
                self.assertEqual(max_steps, 300)
                self.assertEqual(screen.training.tail_digest(screen.training.tail_state(actual)), protocol["final_head"]["tensor_digest"])
                self.assertTrue(all(not value.requires_grad for value in actual.parameters()))
                return {"scene": scene, "draw": draw, "method": method}

            with patch.object(screen.gen, "evaluate_episode", side_effect=evaluate) as evaluator, \
                    patch.object(screen, "verify_record"):
                row = screen.collect(policy, item, output, protocol, heads)
            self.assertEqual(evaluator.call_count, 1)
            self.assertEqual(row["arm"], "recovery8")
            self.assertEqual(row["head_digest"], protocol["final_head"]["tensor_digest"])
            self.assertEqual(screen.training.backbone_digest(policy), backbone)
            binding = screen.base.read_json(output / "evaluation/recovery8/model-binding.json")
            self.assertEqual(binding["inner_method"], "fixed8")
            self.assertNotEqual(binding["head_digest"], protocol["original_head_digest"])

    def test_existing_trial_is_reused_without_new_prediction(self):
        with temporary() as folder:
            output, policy, heads, protocol = context(folder)
            item = screen.schedule()[0]
            directory = output / "evaluation/recovery8"
            screen.condition_binding(output, directory, "recovery8", protocol, create=True)
            trial = {"scene": item["scene"], "draw": item["draw"], "method": "fixed8"}
            screen.base.save_json(directory / (screen.gen.stem(item["scene"], item["draw"], "fixed8") + ".json"), trial)
            with patch.object(screen.gen, "evaluate_episode") as evaluator, patch.object(screen, "verify_record"):
                screen.collect(policy, item, output, protocol, heads)
            evaluator.assert_not_called()

    def test_original_weights_cannot_be_relabelled_as_trained(self):
        with temporary() as folder:
            output, _, _, protocol = context(folder)
            directory = output / "evaluation/recovery8"
            screen.condition_binding(output, directory, "recovery8", protocol, create=True)
            path = directory / "model-binding.json"
            changed = screen.base.read_json(path)
            changed["head_digest"] = protocol["original_head_digest"]
            screen.base.save_json(path, changed)
            with self.assertRaises(AssertionError):
                screen.condition_binding(output, directory, "recovery8", protocol)

    def test_missing_model_binding_cannot_be_created_during_verification(self):
        with temporary() as folder:
            output, _, _, protocol = context(folder)
            with self.assertRaises(AssertionError):
                screen.condition_binding(output, output / "evaluation/recovery8", "recovery8", protocol)


class FreezeAndResumeTest(unittest.TestCase):
    def test_cpu_prepare_freezes_before_collection_and_rejects_drift(self):
        with temporary() as folder:
            output = Path(folder) / "output"
            expected = {"checkpoint": "final600", "source_sha": "before"}
            with patch.object(screen, "check_output"), patch.object(screen, "reference_records"), \
                    patch.object(screen, "protocol_document", return_value=expected), \
                    patch.object(screen.adaptive, "load_policy") as loader:
                screen.prepare(output)
                before = (output / "protocol.json").read_bytes()
                screen.prepare(output)
                self.assertEqual((output / "protocol.json").read_bytes(), before)
            loader.assert_not_called()
            with patch.object(screen, "check_output"), \
                    patch.object(screen, "protocol_document", return_value={**expected, "source_sha": "after"}), \
                    self.assertRaises(AssertionError):
                screen.verify_protocol(output)

    def test_cannot_retroactively_freeze_existing_results(self):
        with temporary() as folder:
            output = Path(folder)
            screen.base.save_json(output / "summary.json", {"preserve": True})
            with patch.object(screen, "check_output"), self.assertRaises(AssertionError):
                screen.prepare(output)

    def test_empty_scan_is_cpu_only_and_rejects_orphan_trace(self):
        with temporary() as folder:
            output, _, _, protocol = context(folder)
            self.assertEqual(screen.scan_records(output, protocol), [])
            directory = output / "evaluation/recovery8"
            screen.condition_binding(output, directory, "recovery8", protocol, create=True)
            (directory / "orphan.trace.json.gz").write_bytes(b"preserve")
            with self.assertRaises(AssertionError):
                screen.scan_records(output, protocol)

    def test_partial_progress_index_is_rejected(self):
        with temporary() as folder:
            output, _, _, protocol = context(folder)
            (output / "evaluation.json.tmp").write_text("partial", encoding="utf-8")
            with self.assertRaises(AssertionError):
                screen.scan_records(output, protocol)

    def test_missing_earlier_attempt_cannot_be_silently_skipped(self):
        with temporary() as folder:
            output, _, _, protocol = context(folder)
            directory = output / "evaluation/recovery8"
            screen.condition_binding(output, directory, "recovery8", protocol, create=True)
            item = screen.schedule()[1]
            trial = {"scene": item["scene"], "draw": item["draw"], "method": "fixed8", "trace": "kept.trace.json.gz"}
            screen.base.save_json(directory / (screen.gen.stem(item["scene"], item["draw"], "fixed8") + ".json"), trial)
            (directory / trial["trace"]).write_bytes(b"preserve")
            with patch.object(screen, "verify_record"), self.assertRaises(AssertionError):
                screen.scan_records(output, protocol)


class PrefixTest(unittest.TestCase):
    def test_trained_prefix_is_matched_but_later_replanning_may_differ(self):
        commands = [[float(i), 0.] for i in range(12)]
        steps = [{"action": command, "after": {"step": i}} for i, command in enumerate(commands)]
        current = {"initial": {}, "steps": copy.deepcopy(steps[:8])}
        previous = {"initial": {}, "steps": copy.deepcopy(steps)}
        current["steps"][0]["predicted_commands"] = commands[:8]
        previous["steps"][0]["predicted_commands"] = commands
        row = {"trial": {}, "directory": "new"}
        old = {"trial": {"unit_lengths": [12]}, "directory": "old"}
        with patch.object(screen.gen, "read_trace", side_effect=[current, previous]), \
                patch.object(screen.adaptive, "states_match", side_effect=lambda a, b: a == b):
            screen.verify_prefix(row, old)

    def test_wrong_first_prediction_fails_the_matched_prefix_check(self):
        current = {"initial": {}, "steps": [{"predicted_commands": [[1., 0.]] * 8}]}
        previous = {"initial": {}, "steps": [{"predicted_commands": [[2., 0.]] * 12}]}
        with patch.object(screen.gen, "read_trace", side_effect=[current, previous]), \
                patch.object(screen.adaptive, "states_match", return_value=True), self.assertRaises(AssertionError):
            screen.verify_prefix({"trial": {}, "directory": "new"},
                                 {"trial": {"unit_lengths": [12]}, "directory": "old"})


class StatisticsTest(unittest.TestCase):
    def test_all_four_cells_and_both_draws_are_required(self):
        rows = records()
        table = screen.metric_table(rows)
        self.assertEqual(set(table), set(screen.ARMS))
        self.assertTrue(all(row["attempts"] == 80 for row in table.values()))
        self.assertEqual(table["recovery8"]["successes"], 80)
        for arm in screen.ARMS:
            self.assertEqual(screen.metric_table(rows, screen.DRAWS[0])[arm]["attempts"], 40)
        for changed in [rows[:-1], rows + [copy.deepcopy(rows[0])]]:
            with self.assertRaises(AssertionError):
                screen.metric_table(changed)

    def test_wrong_inner_method_or_unknown_draw_is_rejected(self):
        for field, value in [("method", "adaptive12"), ("draw", -1)]:
            rows = records()
            rows[0]["trial"][field] = value
            with self.assertRaises(AssertionError):
                screen.metric_table(rows)

    def test_paired_statistics_do_not_depend_on_row_order(self):
        rows = records()
        expected = screen.paired(rows, "recovery8", "fixed8")
        actual = screen.paired(list(reversed(rows)), "recovery8", "fixed8")
        self.assertEqual(expected, actual)
        self.assertEqual(expected["differences_method_minus_reference"]["model_queries"]["mean_difference"], -2.)
        self.assertEqual(len(expected["differences_method_minus_reference"]["success"]["scene_mean_differences"]), 40)

    def test_opposite_draw_effects_cancel_before_scene_bootstrap(self):
        rows = records()
        for row in rows:
            row["trial"]["model_queries"] = 10
            if row["arm"] == "recovery8":
                row["trial"]["model_queries"] += 2 if row["trial"]["draw"] == screen.DRAWS[0] else -2
        result = screen.paired(rows, "recovery8", "fixed8")["differences_method_minus_reference"]["model_queries"]
        self.assertEqual(result["scene_mean_differences"], [0.] * 40)
        self.assertEqual(result["paired_scene_bootstrap_95_interval"], [0., 0.])


if __name__ == "__main__":
    unittest.main()
