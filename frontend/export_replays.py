"""Export selected, matched PushT evaluation attempts for browser replay."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


FRONTEND = Path(__file__).resolve().parent
EVALUATION = FRONTEND.parent / "results" / "robotics-recovery-confirmation" / "evaluation"
OUTPUT = FRONTEND / "src" / "data" / "replays.json"
EXAMPLES = ((173, 60000, (False, True)), (177, 70000, (True, False)))

# PushT uses a square workspace and a fixed T-shaped goal at 45 degrees.
# These are world coordinates from gym-pusht 0.1.5, before image resizing.
TEE = ((-60, 0), (60, 0), (60, 30), (15, 30), (15, 120), (-15, 120), (-15, 30), (-60, 30))
TRACE_VERTEX_ORDER = (0, 1, 2, 7, 6, 3, 4, 5)
ARENA = {"size": 512, "pusherRadius": 15, "blockOutline": TEE, "target": (256, 256, 45)}
DECISION_KINDS = {"extend", "trim", "interrupt"}


def frame(state: dict, path: Path, movement: int) -> dict:
    x, y = state["block_pos"]
    angle = state["block_angle"]
    cosine, sine = math.cos(angle), math.sin(angle)
    projected = []
    for index in TRACE_VERTEX_ORDER:
        local_x, local_y = TEE[index]
        projected.append((x + cosine * local_x - sine * local_y, y + sine * local_x + cosine * local_y))

    # Confirm that the renderer's rigid transform matches the recorded geometry.
    vertices = state["block_vertices"]
    if len(vertices) != len(projected) or any(
        math.dist(actual, expected) > 1e-7 for actual, expected in zip(vertices, projected)
    ):
        raise ValueError(f"Block geometry differs at movement {movement}: {path}")

    return {
        "pusher": [round(value, 4) for value in state["pusher_pos"]],
        "block": [round(x, 4), round(y, 4), round(math.degrees(angle) % 360, 4)],
        "coverage": state["coverage"],
    }


def read_run(folder: str, method: str, scene: int, draw: int) -> tuple[dict, dict]:
    path = EVALUATION / folder / f"scene-{scene}-draw-{draw}-{method}.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    trace_path = path.with_suffix(".trace.json.gz")
    compressed = trace_path.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != run["trace_sha256"]:
        raise ValueError(f"Trace hash differs from attempt record: {path}")
    trace = json.loads(gzip.decompress(compressed))

    identity = (scene, draw, method)
    if any(tuple(record[key] for key in ("scene", "draw", "method")) != identity for record in (run, trace)):
        raise ValueError(f"Attempt identity differs: {path}")

    steps = trace["steps"]
    predictions = [step["step"] for step in steps if step["new_prediction"]]
    events = run.get("events", [])
    if (
        not steps
        or [step["step"] for step in steps] != list(range(1, run["steps"] + 1))
        or len(predictions) != run["model_queries"]
        or trace["initial"] != run["initial"]
        or not math.isclose(steps[-1]["after"]["coverage"], run["final_coverage"], rel_tol=0, abs_tol=1e-12)
        or run["success"] != (steps[-1]["after"]["coverage"] > 0.95)
        or any(event["kind"] not in DECISION_KINDS or not 1 <= event["step"] <= len(steps) for event in events)
    ):
        raise ValueError(f"Trace and attempt record disagree: {path}")

    frames = [frame(trace["initial"], path, 0)]
    frames.extend(frame(step["after"], path, step["step"]) for step in steps)
    replay = {
        "success": run["success"],
        "frames": frames,
        "predictionSteps": predictions,
        "decisions": [{"movement": event["step"], "kind": event["kind"]} for event in events],
    }
    return run, replay


def export() -> str:
    cases = []
    for scene, draw, outcomes in EXAMPLES:
        baseline_run, baseline = read_run("fixed8", "fixed8", scene, draw)
        adaptive_run, adaptive = read_run("recovery12", "adaptive12", scene, draw)
        if (
            baseline_run["initial"] != adaptive_run["initial"]
            or baseline_run["prediction_seed"] != adaptive_run["prediction_seed"]
            or (baseline["success"], adaptive["success"]) != outcomes
        ):
            raise ValueError(f"Expected a matched comparison for scene {scene}, draw {draw}")
        cases.append({"scene": scene, "draw": draw, "baseline": baseline, "adaptive": adaptive})

    return json.dumps({"arena": ARENA, "cases": cases}, separators=(",", ":"), allow_nan=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the checked-in export against the saved attempts.")
    args = parser.parse_args()
    data = export()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != data:
            parser.exit(1, "Replay data differs. Run frontend/export_replays.py to regenerate it.\n")
        print("Replay data matches all four saved attempts.")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(data, encoding="utf-8", newline="\n")
        print(f"Wrote {OUTPUT.relative_to(FRONTEND.parent)}")


if __name__ == "__main__":
    main()
