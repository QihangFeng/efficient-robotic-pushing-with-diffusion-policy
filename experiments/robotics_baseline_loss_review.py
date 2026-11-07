"""Review saved negative flips without inference, training, or physics replay.

All simulator geometry here is retrospective diagnostic evidence. It is never
an input to a proposed controller. Source recordings and implementations stay
unchanged; a repeat must reproduce the same immutable review.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

ARMS = ("fixed8", "milestone12", "recovery8", "recovery12")
METHODS = dict(zip(ARMS, ("fixed8", "adaptive12", "fixed8", "adaptive12")))
SOURCE = Path("results/robotics-recovery-fixed8")
REFERENCE = Path("results/robotics-recovery-confirmation")
OUTPUT = Path("results/robotics-baseline-loss-review")
DROP = .03  # Existing descriptive screen, not an optimized runtime gate.
STATIONARY = 1e-9  # Same pose-change tolerance as the earlier loss review.


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(path, bindings):
    path = Path(path)
    digest = sha(path)
    assert bindings.setdefault(path.as_posix(), digest) == digest
    return path


def same_state(a, b):
    assert a.keys() == b.keys()
    return all(abs(a[k] - b[k]) <= 1e-12 if k == "coverage" else a[k] == b[k] for k in a)


def coverage_values(trace):
    return [trace["initial"]["coverage"]] + [r["after"]["coverage"] for r in trace["steps"]]


def before(trace, index):
    return trace["initial"] if index == 0 else trace["steps"][index - 1]["after"]


def window(start, steps):
    values = [start["coverage"]] + [r["after"]["coverage"] for r in steps]
    peak, drawdown = values[0], 0.
    for value in values[1:]:
        peak = max(peak, value)
        drawdown = max(drawdown, peak - value)
    return {"movements": [r["step"] for r in steps], "start_coverage": values[0],
            "end_coverage": values[-1], "net_change": values[-1] - values[0],
            "maximum_drawdown": drawdown}


def units(trace):
    rows = trace["steps"]
    starts = [i for i, row in enumerate(rows) if row["new_prediction"]]
    assert starts and starts[0] == 0
    return [(i, rows[i:j]) for i, j in zip(starts, starts[1:] + [len(rows)])]


def extension_windows(trial, trace):
    rows, result = trace["steps"], []
    for event in trial["events"]:
        if event["kind"] != "extend":
            continue
        step = event["step"]
        assert rows[step - 1]["plan_age"] == 8
        suffix = []
        for row in rows[step:step + 4]:
            if row["new_prediction"]:
                break
            suffix.append(row)
        result.append({"after_movement": step, "camera": event["features"],
                       "complete_four_movements": len(suffix) == 4,
                       **window(rows[step - 1]["after"], suffix)})
    return result


def first_schedule_difference(fixed, adaptive, events):
    """Describe an exactly shared prefix, not a causal label for a later loss."""
    assert same_state(fixed["initial"], adaptive["initial"])
    common = 0
    for a, b in zip(fixed["steps"], adaptive["steps"]):
        if a["new_prediction"] != b["new_prediction"]:
            event = next((e for e in events if e["step"] == common), None)
            assert event is not None and event["kind"] in ("extend", "interrupt")
            return {"first_different_movement": common + 1, "exact_shared_movements": common,
                    "decision_after_movement": common, "decision": event,
                    "fixed_requests_prediction": a["new_prediction"],
                    "adaptive_requests_prediction": b["new_prediction"],
                    "fixed_next_four": window(before(fixed, common), fixed["steps"][common:common + 4]),
                    "adaptive_next_four": window(before(adaptive, common), adaptive["steps"][common:common + 4])}
        assert a["action"] == b["action"] and same_state(a["after"], b["after"]), \
            "Same weights and prediction schedule unexpectedly differ before the first timing change."
        common += 1
    return {"first_different_movement": None, "exact_shared_movements": common}


def episode_review(trial, trace):
    rows, values = trace["steps"], coverage_values(trace)
    peak_step = max(range(len(values)), key=values.__getitem__)
    peak = values[peak_step]
    stationary_start = len(rows)
    for i in range(len(rows) - 1, -1, -1):
        a, b = before(trace, i), rows[i]["after"]
        translation = math.dist(a["block_pos"], b["block_pos"])
        rotation = abs((a["block_angle"] - b["block_angle"] + math.pi) % (2 * math.pi) - math.pi)
        if translation > STATIONARY or rotation > STATIONARY:
            break
        stationary_start = i
    stationary_rows = rows[stationary_start:]
    prefixes = []
    for index, chunk in units(trace):
        prefixes.append({"prediction_before_movement": index + 1,
                         "preceding_event": next((e["kind"] for e in trial["events"] if e["step"] == index), None),
                         **window(before(trace, index), chunk[:8])})
    largest = max(prefixes, key=lambda p: p["maximum_drawdown"])
    cameras = [{"movement": row["step"], "camera_coverage": row["features"]["coverage"],
                "true_coverage": row["after"]["coverage"]} for row in rows
               if row.get("features", {}).get("quality")]
    return {"success": trial["success"], "steps": len(rows), "predictions": trial["model_queries"],
            "final_coverage": values[-1], "peak_coverage": peak, "first_peak_after_movement": peak_step,
            "drop_from_peak_to_final": peak - values[-1],
            "predictions_after_peak": sum(r["new_prediction"] for r in rows[peak_step:]),
            "extensions": extension_windows(trial, trace),
            "interruptions": [e for e in trial["events"] if e["kind"] == "interrupt"],
            "fresh_prefixes": prefixes, "largest_fresh_prefix_drawdown": largest,
            "terminal_block_stationarity": {"after_movement": stationary_start,
                "movements": len(stationary_rows), "further_predictions": sum(r["new_prediction"] for r in stationary_rows),
                "coverage_at_start": values[stationary_start],
                "pusher_travel": sum(math.dist(before(trace, i)["pusher_pos"], rows[i]["after"]["pusher_pos"])
                                     for i in range(stationary_start, len(rows)))},
            "camera_peak": max(cameras, key=lambda c: c["camera_coverage"]) if cameras else None}


def load_sources():
    bindings = {}
    for name in [__file__, "tests/test_robotics_baseline_loss_review.py",
                 "experiments/robotics_adaptive_length.py", "experiments/robotics_generation_length.py",
                 "experiments/robotics_visual_monitor.py"]:
        bind(Path(name).resolve().relative_to(Path.cwd()), bindings)
    for directory in [SOURCE, REFERENCE]:
        for name in ["protocol.json", "evaluation.json", "summary.json"]:
            bind(directory / name, bindings)
    bind(SOURCE / "artifact-audit.json", bindings)
    bind(REFERENCE / "independent-audit.json", bindings)
    bind(REFERENCE / "eval-bindings.json", bindings)
    assert read(SOURCE / "artifact-audit.json")["passed"]
    assert read(REFERENCE / "independent-audit.json")["passed"]
    protocol = read(SOURCE / "protocol.json")
    for name, expected in protocol["fingerprints"].items():
        assert sha(name) == expected, f"Frozen implementation changed: {name}"
        bindings[name] = expected
    for name, expected in protocol["reference_artifacts"].items():
        assert sha(REFERENCE / name) == expected
    indices = [read(d / "evaluation.json") for d in [REFERENCE, SOURCE]]
    assert all(d["complete"] and d["backbone_unchanged"] for d in indices)
    rows = [r for index in indices for r in index["records"]]
    lookup, traces = {}, {}
    for row in rows:
        trial, arm, directory = row["trial"], row["arm"], Path(row["directory"])
        key = trial["scene"], trial["draw"], arm
        assert key not in lookup and arm in ARMS and trial["method"] == METHODS[arm]
        binding = read(bind(directory / "model-binding.json", bindings))
        head = protocol["final_head"]["tensor_digest"] if arm.startswith("recovery") else protocol["original_head_digest"]
        assert binding["head_digest"] == row["head_digest"] == head
        assert binding["arm"] == arm and binding["inner_method"] == trial["method"]
        if arm == "recovery8":
            assert directory == SOURCE / "evaluation" / arm
            assert binding["protocol_sha256"] == row["protocol_sha256"] == sha(SOURCE / "protocol.json")
        else:
            assert directory == REFERENCE / "evaluation" / arm
            assert binding["eval_bindings_sha256"] == row["eval_bindings_sha256"] == sha(REFERENCE / "eval-bindings.json")
        stem = f"scene-{trial['scene']}-draw-{trial['draw']}-{trial['method']}"
        assert read(bind(directory / (stem + ".json"), bindings)) == trial
        assert trial["trace"] == stem + ".trace.json.gz"
        path = bind(directory / trial["trace"], bindings)
        assert sha(path) == trial["trace_sha256"]
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            trace = json.load(stream)
        steps = trace["steps"]
        assert trace["initial"] == trial["initial"]
        assert [r["step"] for r in steps] == list(range(1, trial["steps"] + 1))
        assert len(steps) <= 300 and trial["horizon"] == 16
        assert trial["generated_shapes"] == [[1, 16, 2]] * trial["model_queries"]
        assert sum(r["new_prediction"] for r in steps) == trial["model_queries"]
        assert trial["prediction_seed"] == trial["scene"] + trial["draw"]
        assert trial["final_coverage"] == steps[-1]["after"]["coverage"]
        assert trial["success"] == (trial["final_coverage"] > .95)
        assert not any(r["after"]["coverage"] > .95 for r in steps[:-1])
        assert trial["success"] or (trial["truncated"] and len(steps) == 300)
        lookup[key], traces[key] = row, trace
    assert len(rows) == len(lookup) == 320
    assert set(lookup) == {(s, d, a) for s in range(170, 210) for d in [60000, 70000] for a in ARMS}
    for (scene, draw, arm), row in lookup.items():
        assert same_state(row["trial"]["initial"], lookup[scene, 60000, "fixed8"]["trial"]["initial"])
    return lookup, traces, bindings


def build_review():
    lookup, traces, bindings = load_sources()
    reviews = {k: episode_review(row["trial"], traces[k]) for k, row in lookup.items()}
    losses, gains, outcomes = [], [], {}
    patterns, timing_counts = {}, {}
    for scene in range(170, 210):
        for draw in [60000, 70000]:
            timing = first_schedule_difference(traces[scene, draw, "recovery8"], traces[scene, draw, "recovery12"],
                                               lookup[scene, draw, "recovery12"]["trial"]["events"])
            key = str(timing["first_different_movement"])
            timing_counts[key] = timing_counts.get(key, 0) + 1
            baseline = lookup[scene, draw, "fixed8"]["trial"]["success"]
            combined = lookup[scene, draw, "recovery12"]["trial"]["success"]
            category = ("retained" if combined else "lost") if baseline else ("gained" if combined else "shared_failure")
            outcomes[scene, draw] = category
            if category == "gained":
                gains.append({"scene": scene, "draw": draw})
            if category != "lost":
                continue
            a = reviews[scene, draw, "milestone12"]["success"]
            f = reviews[scene, draw, "recovery8"]["success"]
            label = f"original_adaptive_{int(a)}_trained_fixed8_{int(f)}"
            patterns[label] = patterns.get(label, 0) + 1
            losses.append({"scene": scene, "draw": draw, "pattern": label,
                           "methods": {arm: reviews[scene, draw, arm] for arm in ARMS},
                           "trained_schedule_divergence": first_schedule_difference(
                               traces[scene, draw, "recovery8"], traces[scene, draw, "recovery12"],
                               lookup[scene, draw, "recovery12"]["trial"]["events"])})
    summary = read(SOURCE / "summary.json")
    comparison = next(c for c in summary["comparisons"] if c["method"] == "recovery12" and c["reference"] == "fixed8")
    assert [{k: loss[k] for k in ("scene", "draw")} for loss in losses] == comparison["lost_successes"]
    assert gains == comparison["gained_successes"]
    groups = {}
    for category in ["retained", "gained", "lost", "shared_failure"]:
        selected = [reviews[s, d, "recovery12"] for (s, d), c in outcomes.items() if c == category]
        extensions = [e for r in selected for e in r["extensions"]]
        prefixes = [p for r in selected for p in r["fresh_prefixes"]]
        groups[category] = {"attempts": len(selected), "extensions": len(extensions),
            "extension_suffix_drawdowns_over_three_pp": sum(e["maximum_drawdown"] > DROP for e in extensions),
            "attempts_with_extension_suffix_drop_over_three_pp": sum(any(e["maximum_drawdown"] > DROP for e in r["extensions"]) for r in selected),
            "fresh_prefixes": len(prefixes), "fresh_prefix_drawdowns_over_three_pp": sum(p["maximum_drawdown"] > DROP for p in prefixes),
            "attempts_with_fresh_prefix_drop_over_three_pp": sum(any(p["maximum_drawdown"] > DROP for p in r["fresh_prefixes"]) for r in selected),
            "attempts_without_interruptions": sum(not r["interruptions"] for r in selected)}
    return {"scope": "Retrospective saved-record analysis; no prediction, training, or physics replay.",
            "scenes": list(range(170, 210)), "draws": [60000, 70000], "attempts_checked": 320,
            "movements_checked": sum(len(t["steps"]) for t in traces.values()),
            "lost_attempts": len(losses), "distinct_lost_scenes": len({r["scene"] for r in losses}),
            "gained_attempts": len(gains), "patterns": patterns,
            "trained_first_schedule_difference_counts": timing_counts,
            "settings": {"descriptive_drop": DROP, "stationary_pose_tolerance": STATIONARY},
            "combined_method_comparison_groups": groups, "losses": losses,
            "limitations": ["Selected failures and inspected scenes cannot provide new generalization evidence.",
                "Whole trajectories after divergence are different states, not matched-state counterfactuals.",
                "A first scheduling difference does not establish the cause of a later failure.",
                "Simulation truth is diagnostic-only; camera estimates are identified separately.",
                "Regression does not establish catastrophic forgetting; controller and sampling effects remain possible.",
                "No mitigation or changed controller is validated by this review."],
            "source_bindings": dict(sorted(bindings.items()))}, traces


def save_once(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, "Review or source content changed; preserve existing evidence."
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def plot_losses(summary, traces, output):
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = dict(zip(ARMS, ("#777777", "#3079ae", "#d78116", "#267746")))
    labels = dict(zip(ARMS, ("Original / fixed 8", "Original / adaptive", "Fine-tuned / fixed 8", "Fine-tuned / adaptive")))
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
    for ax, loss in zip(axes.flat, summary["losses"]):
        scene, draw = loss["scene"], loss["draw"]
        for arm in ARMS:
            values = coverage_values(traces[scene, draw, arm])
            ax.plot(range(len(values)), [v * 100 for v in values], color=colors[arm], lw=1.6)
        ax.axhline(95, color="#a13333", ls="--", lw=.8)
        ax.set_title(f"Scene {scene}, draw {draw}")
        ax.set_xlim(0, 300); ax.set_ylim(0, 101); ax.grid(alpha=.15)
    for ax in axes[-1]: ax.set_xlabel("Executed movement")
    for ax in axes[:, 0]: ax.set_ylabel("True target coverage (%)")
    fig.suptitle("Nine lost baseline successes: saved trajectories", fontsize=16)
    fig.legend([Line2D([0], [0], color=colors[a], lw=2) for a in ARMS], [labels[a] for a in ARMS],
               loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, .04, 1, .96])
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    path = output / "lost-success-coverage.png"
    if path.exists():
        assert path.read_bytes() == buffer.getvalue()
    else:
        path.write_bytes(buffer.getvalue())


def run(output=OUTPUT, plot=False):
    assert output.resolve() in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + "-repeat").resolve()}, \
        "Use the dedicated review directory; preserve old results."
    summary, traces = build_review()
    save_once(output / "summary.json", summary)
    if plot:
        plot_losses(summary, traces, output)
    assert all(sha(name) == value for name, value in summary["source_bindings"].items())
    return {k: summary[k] for k in ["attempts_checked", "movements_checked", "lost_attempts", "distinct_lost_scenes", "patterns"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--plot", action="store_true", help="Optional figure; requires matplotlib, not needed for diagnostics.")
    args = parser.parse_args()
    print(run(args.output_dir, args.plot))
