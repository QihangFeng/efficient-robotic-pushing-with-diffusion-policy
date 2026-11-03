"""Audit the completed four-cell ablation directly from saved JSON and traces.

This checker does not import the collector, load a policy, or run physics. It
independently recalculates metrics and scene-cluster intervals from raw records.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


OUTPUT = Path("results/robotics-recovery-fixed8")
ARMS = {"fixed8": "fixed8", "milestone12": "adaptive12",
        "recovery8": "fixed8", "recovery12": "adaptive12"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same_state(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if key == "coverage":
            assert abs(left[key] - right[key]) <= 1e-12
        else:
            assert left[key] == right[key], f"Physical state or RGB hash differs: {key}"


def trace(row):
    trial, directory = row["trial"], Path(row["directory"])
    stem = f"scene-{trial['scene']}-draw-{trial['draw']}-{trial['method']}"
    assert read(directory / (stem + ".json")) == trial
    assert trial["trace"] == stem + ".trace.json.gz"
    assert sha(directory / trial["trace"]) == trial["trace_sha256"]
    with gzip.open(directory / trial["trace"], "rt", encoding="utf-8") as stream:
        saved = json.load(stream)
    assert (saved["scene"], saved["draw"], saved["method"]) == (trial["scene"], trial["draw"], trial["method"])
    assert saved["initial"] == trial["initial"]
    steps = saved["steps"]
    assert len(steps) == trial["steps"] and 1 <= len(steps) <= 300
    assert steps[-1]["after"]["coverage"] == trial["final_coverage"]
    assert trial["success"] == (trial["final_coverage"] > .95)
    assert all(step["after"]["coverage"] <= .95 for step in steps[:-1])
    assert trial["success"] or (trial["truncated"] and len(steps) == 300)
    return saved


def audit(output=OUTPUT):
    assert output.resolve() in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + "-repeat").resolve()}
    protocol, summary = read(output / "protocol.json"), read(output / "summary.json")
    assert summary["complete"] and summary["protocol_sha256"] == sha(output / "protocol.json")
    reference = Path(protocol["reference"])
    for name, expected in protocol["reference_artifacts"].items():
        assert sha(reference / name) == expected, f"Prior evidence changed: {name}"
    for name, expected in protocol["fingerprints"].items():
        assert sha(name) == expected, f"Frozen source or input changed: {name}"
    assert sha(protocol["final_head"]["path"]) == protocol["final_head"]["sha256"]
    assert protocol["final_head"]["optimizer_step"] == 600
    old, new = read(reference / "evaluation.json"), read(output / "evaluation.json")
    assert old["complete"] and old["backbone_unchanged"] and new["complete"] and new["backbone_unchanged"]
    assert len(old["records"]) == 240 and len(new["records"]) == 80
    rows = old["records"] + new["records"]
    scenes, draws = protocol["scenes"], protocol["draw_offsets"]
    assert scenes == list(range(170, 210)) and draws == [60000, 70000]
    assert [(r["trial"]["scene"], r["trial"]["draw"]) for r in new["records"]] == [(s, d) for s in scenes for d in draws]
    lookup = {(r["trial"]["scene"], r["trial"]["draw"], r["arm"]): r for r in rows}
    assert len(rows) == len(lookup) == 320
    assert set(lookup) == {(s, d, arm) for s in scenes for d in draws for arm in ARMS}
    movements = 0
    for row in rows:
        trial, arm, directory = row["trial"], row["arm"], Path(row["directory"])
        trained = arm in ("recovery8", "recovery12")
        expected_head = protocol["final_head"]["tensor_digest"] if trained else protocol["original_head_digest"]
        binding = read(directory / "model-binding.json")
        assert row["head_digest"] == binding["head_digest"] == expected_head
        assert binding["arm"] == arm and binding["inner_method"] == trial["method"] == ARMS[arm]
        if arm == "recovery8":
            assert row["protocol_sha256"] == binding["protocol_sha256"] == sha(output / "protocol.json")
            assert directory == output / "evaluation/recovery8"
        else:
            assert row["eval_bindings_sha256"] == binding["eval_bindings_sha256"] == sha(reference / "eval-bindings.json")
            assert directory == reference / "evaluation" / arm
        actual = trace(row)
        movements += len(actual["steps"])
        same_state(actual["initial"], lookup[trial["scene"], trial["draw"], "fixed8"]["trial"]["initial"])
        assert trial["prediction_seed"] == trial["scene"] + trial["draw"]
        assert trial["horizon"] == 16
        assert all(math.isfinite(trial[key]) for key in ["final_coverage", "wall_seconds", "monitor_seconds"])
        if arm == "recovery8":
            assert trial["execution"] == 8 and trial["events"] == [] and trial["monitor_seconds"] == 0
            assert trial["model_queries"] == (trial["steps"] + 7) // 8
            assert trial["generated_shapes"] == [[1, 16, 2]] * trial["model_queries"]
            commands = None
            for index, step in enumerate(actual["steps"]):
                age = index % 8
                assert step["new_prediction"] == (age == 0)
                if age == 0:
                    commands = step["predicted_commands"]
                    assert len(commands) == 8
                assert step["action"] == commands[age] and step["plan_age"] == age + 1
                assert step["pending_before"] == step["pending_after"] == 7 - age
                assert "features" not in step
            previous = lookup[trial["scene"], trial["draw"], "recovery12"]
            previous_trace = trace(previous)
            assert actual["steps"][0]["predicted_commands"] == previous_trace["steps"][0]["predicted_commands"][:8]
            length = min(8, previous["trial"]["unit_lengths"][0], len(actual["steps"]))
            for left, right in zip(actual["steps"][:length], previous_trace["steps"][:length]):
                assert left["action"] == right["action"]
                same_state(left["after"], right["after"])
    for draw in [None, *draws]:
        table = summary["methods"] if draw is None else summary["per_draw"][str(draw)]
        for arm in ARMS:
            trials = [r["trial"] for r in rows if r["arm"] == arm and (draw is None or r["trial"]["draw"] == draw)]
            expected = {"attempts": len(trials), "successes": sum(t["success"] for t in trials),
                        **{f"mean_{key}": sum(t[key] for t in trials) / len(trials)
                           for key in ["final_coverage", "model_queries", "wall_seconds", "monitor_seconds"]},
                        "mean_prediction_seconds": sum(sum(t["prediction_seconds"]) / len(t["prediction_seconds"]) for t in trials) / len(trials)}
            for key, value in expected.items():
                assert abs(table[arm][key] - value) <= 1e-10, (arm, key, value, table[arm][key])
    indices = np.random.default_rng(protocol["statistics"]["bootstrap_seed"]).integers(
        0, 40, size=(protocol["statistics"]["bootstrap_samples"], 40))
    for comparison in summary["comparisons"]:
        method, reference_arm = comparison["method"], comparison["reference"]
        for key, saved in comparison["differences_method_minus_reference"].items():
            differences = np.array([sum(float(lookup[s, d, method]["trial"][key]) - float(lookup[s, d, reference_arm]["trial"][key]) for d in draws) / 2 for s in scenes])
            np.testing.assert_allclose(saved["scene_mean_differences"], differences, atol=1e-12, rtol=0)
            np.testing.assert_allclose(saved["mean_difference"], differences.mean(), atol=1e-12, rtol=0)
            np.testing.assert_allclose(saved["paired_scene_bootstrap_95_interval"], np.quantile(differences[indices].mean(axis=1), [.025, .975]), atol=1e-12, rtol=0)
        for name, first, second in [("gained_successes", method, reference_arm), ("lost_successes", reference_arm, method)]:
            expected = [{"scene": s, "draw": d} for s in scenes for d in draws if lookup[s, d, first]["trial"]["success"] and not lookup[s, d, second]["trial"]["success"]]
            assert comparison[name] == expected
    result = {"passed": True, "scored_attempts": 320, "new_attempts": 80, "reused_attempts": 240,
              "independent_scenes": 40, "draws_per_scene": 2, "movements_checked": movements,
              "new_trained_prefixes_checked": 80, "model_identities_checked": 320,
              "all_metrics_and_five_paired_comparisons_match": True,
              "prior_evidence_and_frozen_sources_unchanged": True,
              "auditor_sha256": sha(__file__), "summary_sha256": sha(output / "summary.json"),
              "scope": "Separate CPU raw-artifact audit; no neural prediction or physics replay."}
    path = output / "artifact-audit.json"
    if path.exists():
        assert read(path) == result
    else:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    print(audit(parser.parse_args().output_dir))
