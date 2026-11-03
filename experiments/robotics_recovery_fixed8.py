"""Complete the frozen four-configuration ablation with 80 trained/fixed8 trials.

The previous 240 confirmation trials and their sources remain read-only. Only
the missing cell is collected, using the same final600 checkpoint and evaluator.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

import robotics_recovery_confirmation as prior


base, gen, training, adaptive = prior.base, prior.gen, prior.training, prior.adaptive
OUTPUT = Path("results/robotics-recovery-fixed8")
REFERENCE = prior.OUTPUT
SCENES, DRAWS = list(prior.SCENES), list(prior.DRAWS)
ARMS = {"fixed8": "fixed8", "milestone12": "adaptive12",
        "recovery8": "fixed8", "recovery12": "adaptive12"}
SMOKE = [("before", "fixed8"), ("trained", "recovery8"), ("restored", "fixed8")]
STATISTICS = {"bootstrap_samples": 10000, "bootstrap_seed": 20261103,
              "unit": "paired initial scene, averaging both prediction draws before resampling",
              "independent_scenes": 40, "draws_per_scene": 2}
REFERENCE_FILES = ["protocol.json", "eval-bindings.json", "evaluation.json", "summary.json", "smoke.json"]


def schedule():
    return [{"scene": scene, "draw": draw, "arm": "recovery8"}
            for scene in SCENES for draw in DRAWS]


def check_output(output):
    assert Path(output).resolve() in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + "-repeat").resolve()}, \
        "Use the separate fixed-eight ablation output; never modify previous evidence."


def protocol_document():
    bindings = prior.verify_bindings(REFERENCE)
    reference = base.read_json(REFERENCE / "protocol.json")
    progress = base.read_json(REFERENCE / "evaluation.json")
    assert progress["complete"] and progress["backbone_unchanged"]
    assert base.read_json(REFERENCE / "summary.json")["complete"]
    assert reference["scenes"] == SCENES and reference["draw_offsets"] == DRAWS
    assert bindings["inputs"]["final_head"]["optimizer_step"] == 600
    sources = [Path(__file__).resolve().relative_to(Path.cwd()),
               Path("tests/test_robotics_recovery_fixed8.py")]
    return {"date": "2026-11-03", "scope": "Complete one missing ablation cell; no retraining or tuning",
            "reference": REFERENCE.as_posix(),
            "reference_artifacts": {name: base.sha(REFERENCE / name) for name in REFERENCE_FILES},
            "scenes": SCENES, "draw_offsets": DRAWS, "schedule": schedule(),
            "new_attempts": 80, "reused_attempts": 240, "max_steps": 300,
            "generation": reference["generation"], "execution": 8, "training": False,
            "success": reference["success"], "prediction_seed": reference["prediction_seed"],
            "final_head": bindings["inputs"]["final_head"],
            "original_head_digest": bindings["original_head_digest"],
            "statistics": STATISTICS,
            "fingerprints": {**reference["fingerprints"],
                             **{path.as_posix(): base.sha(path) for path in sources}},
            "smoke": {"scene": 0, "draw": 1234, "max_steps": 16, "order": [list(x) for x in SMOKE]},
            "limitations": ["These forty scenes are inspected confirmation scenes, not a fresh cohort.",
                            "One fixed trained checkpoint; no recovery-data-specific causal attribution.",
                            "The three reference arms were collected in earlier sessions. Runtime is descriptive.",
                            "Eighty attempts per arm represent forty scene clusters, not eighty independent scenes.",
                            "Clean paused simulation; not real-time or physical-robot performance."]}


def verify_protocol(output):
    check_output(output)
    saved = base.read_json(output / "protocol.json")
    assert saved == protocol_document(), "Frozen ablation source, checkpoint or prior evidence changed."
    return saved


def reference_records():
    bindings = prior.verify_bindings(REFERENCE)
    rows = prior.scan_records(REFERENCE, bindings)
    assert len(rows) == 240 and [prior.identity(row) for row in rows] == prior.schedule()
    return rows


def prepare(output=OUTPUT):
    check_output(output)
    if not (output / "protocol.json").exists():
        assert not output.exists() or not any(output.iterdir()), "Do not retroactively freeze existing results."
        reference_records()
        prior.old.pilot.write_once(output / "protocol.json", protocol_document())
    return verify_protocol(output)


def head_identity(protocol, arm):
    assert arm in ("fixed8", "recovery8")
    return protocol["final_head"]["tensor_digest"] if arm == "recovery8" else protocol["original_head_digest"]


def condition_binding(output, directory, arm, protocol, create=False):
    expected = {"arm": arm, "inner_method": "fixed8", "head_digest": head_identity(protocol, arm),
                "protocol_sha256": base.sha(output / "protocol.json")}
    path = directory / "model-binding.json"
    if path.exists():
        assert base.read_json(path) == expected, "Recorded model identity differs."
    else:
        assert create, "Missing recorded model identity."
        assert not directory.exists() or not any(directory.iterdir()), "Do not relabel existing trials."
        prior.old.pilot.write_once(path, expected)
    return expected


def verify_record(row, output, protocol, smoke=False):
    trial, arm = row["trial"], row["arm"]
    assert arm in ("fixed8", "recovery8") and (smoke or arm == "recovery8")
    directory = output / ("smoke" if smoke else "evaluation") / row.get("label", arm)
    assert Path(row["directory"]) == directory and trial["method"] == "fixed8"
    expected = condition_binding(output, directory, arm, protocol)
    assert row["head_digest"] == expected["head_digest"]
    assert row["protocol_sha256"] == expected["protocol_sha256"]
    path = directory / (gen.stem(trial["scene"], trial["draw"], "fixed8") + ".json")
    assert base.read_json(path) == trial
    return gen.verify_trial(trial, directory, [0] if smoke else SCENES,
                            [1234] if smoke else DRAWS, 16 if smoke else 300)


def record(trial, arm, directory, output, protocol, label=None):
    result = {"trial": trial, "arm": arm, "directory": directory.as_posix(),
              "head_digest": head_identity(protocol, arm),
              "protocol_sha256": base.sha(output / "protocol.json")}
    if label is not None:
        result["label"] = label
    return result


def scan_records(output, protocol, smoke=False):
    expected = ([{"scene": 0, "draw": 1234, "arm": arm, "label": label} for label, arm in SMOKE]
                if smoke else schedule())
    folder = output / ("smoke" if smoke else "evaluation")
    allowed = {item.get("label", item["arm"]) for item in expected}
    rows, found = [], set()
    if folder.exists():
        assert all(path.is_dir() and path.name in allowed for path in folder.iterdir()), "Unexpected condition directory."
    for item in expected:
        directory = folder / item.get("label", item["arm"])
        path = directory / (gen.stem(item["scene"], item["draw"], "fixed8") + ".json")
        if path.exists():
            trial = base.read_json(path)
            assert (trial["scene"], trial["draw"], trial["method"]) == (item["scene"], item["draw"], "fixed8")
            row = record(trial, item["arm"], directory, output, protocol, item.get("label"))
            verify_record(row, output, protocol, smoke)
            rows.append(row)
            found.update([path, directory / trial["trace"], directory / "model-binding.json"])
        elif (directory / "model-binding.json").exists():
            condition_binding(output, directory, item["arm"], protocol)
            found.add(directory / "model-binding.json")
    if folder.exists():
        assert all(path in found for directory in folder.iterdir() for path in directory.iterdir()), \
            "Orphan trace, partial sidecar or unexpected result; inspect without overwriting."
    keys = ["scene", "draw", "arm", "label"] if smoke else ["scene", "draw", "arm"]
    actual = [{**{key: row["trial"][key] for key in ["scene", "draw"]}, "arm": row["arm"],
               **({"label": row["label"]} if smoke else {})} for row in rows]
    assert actual == [{key: item[key] for key in keys} for item in expected[:len(rows)]], \
        "Completed attempts must be a predetermined schedule prefix."
    index = output / ("smoke.json" if smoke else "evaluation.json")
    assert not index.with_suffix(".json.tmp").exists(), "Inspect partial progress index."
    if index.exists():
        saved = base.read_json(index)
        assert saved["records"] == rows[:len(saved["records"])]
        if saved.get("complete") or saved.get("passed"):
            assert len(saved["records"]) == len(rows) == len(expected)
    return rows


def collect(policy, item, output, protocol, heads, smoke=False):
    arm = item["arm"]
    directory = output / ("smoke" if smoke else "evaluation") / item.get("label", arm)
    condition_binding(output, directory, arm, protocol, create=True)
    path = directory / (gen.stem(item["scene"], item["draw"], "fixed8") + ".json")
    if path.exists():
        trial = base.read_json(path)
    else:
        assert not path.with_suffix(".trace.json.gz").exists() and not path.with_suffix(".json.tmp").exists()
        prior.activate(policy, heads["trained" if arm == "recovery8" else "original"], head_identity(protocol, arm))
        trial = gen.evaluate_episode(policy, item["scene"], item["draw"], "fixed8", directory, 16 if smoke else 300)
        assert training.tail_digest(training.tail_state(policy)) == head_identity(protocol, arm)
    row = record(trial, arm, directory, output, protocol, item.get("label"))
    verify_record(row, output, protocol, smoke)
    return row


def verify_prefix(row, previous):
    current = gen.read_trace(row["trial"], Path(row["directory"]))
    reference = gen.read_trace(previous["trial"], Path(previous["directory"]))
    assert adaptive.states_match(current["initial"], reference["initial"]), "Initial physical scene changed."
    length = min(8, previous["trial"]["unit_lengths"][0], len(current["steps"]))
    assert current["steps"][0]["predicted_commands"] == reference["steps"][0]["predicted_commands"][:8], \
        "Same trained checkpoint and seed must produce the same first eight commands."
    for left, right in zip(current["steps"][:length], reference["steps"][:length]):
        assert left["action"] == right["action"] and adaptive.states_match(left["after"], right["after"]), \
            "Matched trained-prefix execution changed."


def verify_smoke(output, protocol):
    saved = base.read_json(output / "smoke.json")
    assert saved["passed"] and saved["backbone_unchanged"]
    rows = scan_records(output, protocol, True)
    old = {row["label"]: row for row in base.read_json(REFERENCE / "smoke.json")["records"]}
    for row in rows:
        verify_prefix(row, old[row["label"]])
        if row["arm"] == "fixed8":
            current = gen.read_trace(row["trial"], Path(row["directory"]))["steps"]
            reference = gen.read_trace(old[row["label"]]["trial"], Path(old[row["label"]]["directory"]))["steps"]
            assert len(current) == len(reference) == 16
            for left, right in zip(current, reference):
                assert left["action"] == right["action"] and adaptive.states_match(left["after"], right["after"])
    return saved


def smoke(output, protocol, policy, heads):
    if (output / "smoke.json").exists() and base.read_json(output / "smoke.json").get("passed"):
        return verify_smoke(output, protocol)
    rows = scan_records(output, protocol, True)
    backbone = training.backbone_digest(policy)
    try:
        for label, arm in SMOKE[len(rows):]:
            rows.append(collect(policy, {"scene": 0, "draw": 1234, "arm": arm, "label": label},
                                output, protocol, heads, True))
            base.save_json(output / "smoke.json", {"passed": False, "records": rows})
        assert training.backbone_digest(policy) == backbone
        base.save_json(output / "smoke.json", {"passed": True, "backbone_unchanged": True, "records": rows})
    finally:
        prior.activate(policy, heads["original"], protocol["original_head_digest"])
    return verify_smoke(output, protocol)


def metric_table(rows, draw=None):
    draws = DRAWS if draw is None else [draw]
    assert all(value in DRAWS for value in draws)
    assert all(row["arm"] in ARMS and row["trial"]["draw"] in DRAWS
               and row["trial"]["method"] == ARMS[row["arm"]] for row in rows)
    selected = [row for row in rows if row["trial"]["draw"] in draws]
    expected = {(scene, sample, arm) for scene in SCENES for sample in draws for arm in ARMS}
    assert len(selected) == len(expected)
    assert {(row["trial"]["scene"], row["trial"]["draw"], row["arm"]) for row in selected} == expected
    return {arm: {"attempts": len(trials), "successes": sum(trial["success"] for trial in trials),
                  **{f"mean_{key}": float(np.mean([trial[key] for trial in trials]))
                     for key in ["final_coverage", "model_queries", "wall_seconds", "monitor_seconds"]},
                  "mean_prediction_seconds": float(np.mean([np.mean(trial["prediction_seconds"]) for trial in trials]))}
            for arm in ARMS for trials in [[row["trial"] for row in selected if row["arm"] == arm]]}


def paired(rows, method, reference):
    assert method in ARMS and reference in ARMS and method != reference
    metric_table(rows)
    lookup = {(row["trial"]["scene"], row["trial"]["draw"], row["arm"]): row["trial"] for row in rows}
    indices = np.random.default_rng(STATISTICS["bootstrap_seed"]).integers(
        0, len(SCENES), size=(STATISTICS["bootstrap_samples"], len(SCENES)))
    differences = {}
    for key in ["success", "final_coverage", "model_queries", "wall_seconds"]:
        values = np.asarray([np.mean([float(lookup[scene, draw, method][key]) -
                                     float(lookup[scene, draw, reference][key]) for draw in DRAWS]) for scene in SCENES])
        differences[key] = {"mean_difference": float(values.mean()), "scene_mean_differences": values.tolist(),
                            "paired_scene_bootstrap_95_interval": np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()}
    discordant = lambda a, b: [{"scene": scene, "draw": draw} for scene in SCENES for draw in DRAWS
                               if lookup[scene, draw, a]["success"] and not lookup[scene, draw, b]["success"]]
    return {"method": method, "reference": reference, "gained_successes": discordant(method, reference),
            "lost_successes": discordant(reference, method), "differences_method_minus_reference": differences}


def verify(output=OUTPUT):
    protocol = verify_protocol(output)
    if (output / "smoke.json").exists() and base.read_json(output / "smoke.json").get("passed"):
        verify_smoke(output, protocol)
    else:
        scan_records(output, protocol, True)
    rows = scan_records(output, protocol)
    return {"complete": len(rows) == 80, "completed_attempts": len(rows), "expected_attempts": 80}


def summarize(output=OUTPUT):
    protocol = verify_protocol(output)
    verify_smoke(output, protocol)
    index = base.read_json(output / "evaluation.json")
    assert index["complete"] and index["backbone_unchanged"]
    rows = scan_records(output, protocol)
    assert len(rows) == 80
    reused = reference_records()
    lookup = {(row["trial"]["scene"], row["trial"]["draw"]): row for row in reused if row["arm"] == "recovery12"}
    for row in rows:
        verify_prefix(row, lookup[row["trial"]["scene"], row["trial"]["draw"]])
    all_rows = reused + rows
    contrasts = [("milestone12", "fixed8"), ("recovery8", "fixed8"),
                 ("recovery12", "milestone12"), ("recovery12", "recovery8"), ("recovery12", "fixed8")]
    result = {"complete": True, "new_attempts": 80, "reused_attempts": 240,
              "methods": metric_table(all_rows), "per_draw": {str(draw): metric_table(all_rows, draw) for draw in DRAWS},
              "comparisons": [paired(all_rows, method, reference) for method, reference in contrasts],
              "statistics": STATISTICS, "protocol_sha256": base.sha(output / "protocol.json"),
              "scope": "Frozen retrospective ablation completion on forty inspected scenes, two draws; no pilot pooling.",
              "timing": "All four runtime averages are descriptive across collection sessions, not a controlled speed benchmark.",
              "limitations": protocol["limitations"]}
    path = output / "summary.json"
    if path.exists():
        assert base.read_json(path) == result, "Completed summary changed."
    else:
        prior.old.pilot.write_once(path, result)
    return result


def _run(output):
    if (output / "summary.json").exists():
        return summarize(output)
    protocol = verify_protocol(output)
    verify(output)
    reused = reference_records()
    lookup = {(row["trial"]["scene"], row["trial"]["draw"]): row for row in reused if row["arm"] == "recovery12"}
    training.gpu_guard()
    policy, _ = adaptive.load_policy(base.MODEL)
    heads = {"original": training.tail_state(policy),
             "trained": training.read_tail(Path(protocol["final_head"]["path"]), protocol["final_head"]["sha256"])}
    assert training.tail_digest(heads["original"]) == protocol["original_head_digest"]
    assert training.tail_digest(heads["trained"]) == head_identity(protocol, "recovery8")
    backbone = training.backbone_digest(policy)
    smoke(output, protocol, policy, heads)
    rows = scan_records(output, protocol)
    try:
        for item in schedule()[len(rows):]:
            row = collect(policy, item, output, protocol, heads)
            verify_prefix(row, lookup[item["scene"], item["draw"]])
            rows.append(row)
            base.save_json(output / "evaluation.json", {"complete": False, "records": rows})
            print(f"Trained fixed-eight {len(rows)}/80: scene {item['scene']}, draw {item['draw']}, "
                  f"success={row['trial']['success']}, calls={row['trial']['model_queries']}", flush=True)
        assert training.backbone_digest(policy) == backbone
        base.save_json(output / "evaluation.json", {"complete": True, "backbone_unchanged": True, "records": rows})
    finally:
        prior.activate(policy, heads["original"], protocol["original_head_digest"])
    assert base.sha(base.MODEL / "model.safetensors") == base.WEIGHTS_SHA256
    return summarize(output)


def run(output=OUTPUT):
    check_output(output)
    if (output / "summary.json").exists():
        return summarize(output)
    protocol = verify_protocol(output)
    before = len(scan_records(output, protocol))
    resource = output / f"evaluation-resources-from-{before:03}.json"
    retry = 0
    while resource.exists():
        retry += 1
        resource = output / f"evaluation-resources-from-{before:03}-retry-{retry}.json"
    started, memory = time.perf_counter(), adaptive.MemorySampler()
    try:
        with memory:
            return _run(output)
    finally:
        index = output / "evaluation.json"
        after = len(base.read_json(index)["records"]) if index.exists() else before
        base.save_json(resource, {"wall_seconds": time.perf_counter() - started,
                                  "peak_process_rss_bytes": memory.peak_rss,
                                  "minimum_system_available_bytes": memory.minimum_available,
                                  "completed_attempts_before": before, "completed_attempts_after": after,
                                  "complete": (output / "summary.json").exists(),
                                  "scope": "This invocation including model load and verification; separate from attempt timing."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "verify", "run", "summarize"], required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = globals()[args.stage](args.output_dir)
    print(result if args.stage == "verify" else f"Fixed-eight ablation stage {args.stage} completed.", flush=True)


if __name__ == "__main__":
    main()
