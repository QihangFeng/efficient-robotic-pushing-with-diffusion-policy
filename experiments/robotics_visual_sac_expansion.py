"""Add forty distinct scenes to the frozen visual SAC comparison."""

from __future__ import annotations

import argparse
import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
from statistics import mean
import sys

import robotics_generation_length as generation
import robotics_visual_sac_comparison as prior
import robotics_visual_sac as sac
from robotics_feasibility import WEIGHTS_SHA256


REFERENCE = sac.ROOT / "results" / "robotics-staged-extension"
OUTPUT = sac.ROOT / "results" / "robotics-visual-sac-comparison-80"
SCENES = tuple(range(110, 150))
DRAW = 40000


def read_controls(packages):
    protocol = prior.read_json(REFERENCE / "protocol.json")
    assert tuple(protocol["fresh_scenes"]) == SCENES
    assert protocol["draw_offsets"]["fresh"] == DRAW and protocol["max_movements"] == sac.MAX_MOVEMENTS
    assert protocol["checkpoint"]["weights_sha256"] == WEIGHTS_SHA256
    assert all(protocol["packages"][name] == expected for name, expected in packages.items()
               if name != "stable-baselines3")
    directory = REFERENCE / "episodes"
    document = prior.read_json(REFERENCE / "episodes.json")
    assert document["complete"]
    trials = [item["trial"] for item in document["records"] if item["trial"]["scene"] in SCENES
              and item["trial"]["method"] == "fixed8"]
    assert len(trials) == len(SCENES) and {trial["scene"] for trial in trials} == set(SCENES)
    controls = {}
    for trial in trials:
        scene = trial["scene"]
        assert trial["draw"] == DRAW
        trace = generation.verify_trial(trial, directory, SCENES, (DRAW,))
        controls[scene] = {"scene": scene, "draw": DRAW, "initial": trial["initial"],
                           "success": trial["success"], "movements": trial["steps"],
                           "final_coverage": trial["final_coverage"],
                           "peak_coverage": max(trial["initial"]["coverage"],
                                                *(step["after"]["coverage"] for step in trace["steps"]))}
    return controls


def summarize(pairs):
    metrics = prior.METRICS
    methods = {}
    for label in ("visual_sac", "diffusion_fixed8"):
        attempts = [attempt for pair in pairs for attempt in pair[label]]
        methods[label] = {"attempts": len(attempts), "successes": sum(row["success"] for row in attempts),
                          **{f"scene_mean_{key}": mean(mean(row[key] for row in pair[label]) for pair in pairs)
                             for key in metrics}}
    return {"scenes": len(pairs), "methods": methods,
            "paired_sac_minus_fixed8": {key: methods["visual_sac"][f"scene_mean_{key}"] -
                                             methods["diffusion_fixed8"][f"scene_mean_{key}"] for key in metrics}}


def run(output):
    deps = sac.ROOT / "tmp" / "visual-sac-deps"
    if deps.is_dir():
        sys.path.insert(0, str(deps))
    os.environ.setdefault("MPLCONFIGDIR", str(sac.ROOT / "tmp" / "visual-sac-matplotlib"))
    from stable_baselines3 import SAC

    earlier = prior.OUTPUT
    earlier_protocol = prior.read_json(earlier / "protocol.json")
    earlier_summary = prior.read_json(earlier / "summary.json")
    assert tuple(earlier_protocol["scenes"]) == prior.SCENES
    assert earlier_summary["protocol_sha256"] == sac.fingerprint(earlier / "protocol.json")
    assert earlier_summary["trace_sha256"] == sac.fingerprint(earlier / "sac-traces.json.gz")
    assert all(sac.fingerprint(sac.ROOT / path) == expected for path, expected in earlier_protocol["sha256"].items())
    assert all(version(name) == expected for name, expected in earlier_protocol["packages"].items())
    assert sac.torch.cuda.is_available() and sac.torch.cuda.get_device_name() == earlier_protocol["gpu"]
    checkpoint = sac.ROOT / prior.read_json(prior.TRAINING / "movements-100000.json")["checkpoint"]
    references = read_controls(earlier_protocol["packages"])
    paths = [Path(__file__).resolve(), Path(generation.__file__),
             *(earlier / name for name in ("protocol.json", "summary.json", "sac-traces.json.gz")),
             *(REFERENCE / name for name in ("protocol.json", "episodes.json")), checkpoint]
    protocol = {"new_scenes": SCENES, "reused_scenes": prior.SCENES, "new_diffusion_draw_offset": DRAW,
                "sac_inference": earlier_protocol["sac_inference"], "movement_limit": sac.MAX_MOVEMENTS,
                "success": earlier_protocol["success"],
                "statistics": "Average repeats within each scene, then weight each distinct scene equally.",
                "sha256": {path.relative_to(sac.ROOT).as_posix(): sac.fingerprint(path) for path in paths}}
    output.mkdir(parents=True, exist_ok=False)
    sac.write_json(output / "protocol.json", protocol)
    sac.torch.set_num_threads(4)
    model = SAC.load(checkpoint, device="cuda")
    fresh = []
    with sac.make_env() as env, sac.torch.inference_mode():
        for scene in SCENES:
            fresh.append(prior.evaluate_scene(model, env, scene, [references[scene]]))
            print(f"SAC {len(fresh)}/{len(SCENES)} (scene {scene}): "
                  f"success={fresh[-1]['success']}, final coverage={fresh[-1]['final_coverage']:.2%}", flush=True)
    trace_path = output / "new-sac-traces.json.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
        json.dump(fresh, stream, separators=(",", ":"), allow_nan=False)
    fresh_pairs = [{"scene": row["scene"], "visual_sac": [{k: row[k] for k in prior.METRICS}],
                    "diffusion_fixed8": [{k: references[row["scene"]][k] for k in prior.METRICS}]} for row in fresh]
    saved_pairs = [{"scene": row["scene"], "visual_sac": [{k: row[k] for k in prior.METRICS}],
                    "diffusion_fixed8": [{k: draw[k] for k in prior.METRICS} for draw in row["diffusion_fixed8"]]}
                   for row in earlier_summary["episodes"]]
    assert [row["scene"] for row in saved_pairs] == list(prior.SCENES)
    pairs = fresh_pairs + saved_pairs
    result = {"protocol_sha256": sac.fingerprint(output / "protocol.json"), "trace_sha256": sac.fingerprint(trace_path),
              "new_cohort": summarize(fresh_pairs), "previous_cohort": summarize(saved_pairs),
              "combined": summarize(pairs), "scene_pairs": pairs}
    sac.write_json(output / "summary.json", result)
    print(json.dumps(result["combined"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)
