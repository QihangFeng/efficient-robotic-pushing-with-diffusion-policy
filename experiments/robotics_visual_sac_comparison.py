"""Compare frozen visual SAC with the saved original fixed-eight PushT results."""

from __future__ import annotations

import argparse
import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time

import robotics_visual_sac as sac
from robotics_baseline_check import physical_state, states_match


ROOT = sac.ROOT
REFERENCE = ROOT / "results" / "robotics-recovery-confirmation"
TRAINING = ROOT / "results" / "robotics-visual-sac-training" / "standard-01"
OUTPUT = ROOT / "results" / "robotics-visual-sac-comparison"
SCENES = tuple(range(170, 210))
DRAWS = (60000, 70000)
METRICS = ("success", "final_coverage", "peak_coverage", "movements")
BOOTSTRAP_SEED = 20261106
BOOTSTRAP_SAMPLES = 10_000


def read_json(path):
    return json.loads(path.read_bytes())


def read_reference():
    saved = read_json(REFERENCE / "evaluation.json")
    assert saved["complete"] and saved["backbone_unchanged"]
    bindings = REFERENCE / "eval-bindings.json"
    protocol = REFERENCE / "protocol.json"
    binding = read_json(bindings)
    assert binding["inputs"]["protocol_sha256"] == sac.fingerprint(protocol)
    protocol_data = read_json(protocol)
    assert tuple(protocol_data["scenes"]) == SCENES and tuple(protocol_data["draw_offsets"]) == DRAWS
    assert protocol_data["max_steps"] == sac.MAX_MOVEMENTS
    pairs = {}
    for record in saved["records"]:
        if record["arm"] != "fixed8":
            continue
        trial = record["trial"]
        key = trial["scene"], trial["draw"]
        assert key not in pairs and trial["method"] == "fixed8"
        assert trial["execution"] == 8 and trial["horizon"] == 16
        assert record["head_digest"] == binding["original_head_digest"]
        assert record["eval_bindings_sha256"] == sac.fingerprint(bindings)
        name = f"scene-{key[0]}-draw-{key[1]}-fixed8.trace.json.gz"
        path = REFERENCE / "evaluation" / "fixed8" / name
        assert trial["trace"] == name and trial["trace_sha256"] == sac.fingerprint(path)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            trace = json.load(stream)
        assert (trace["scene"], trace["draw"], trace["method"]) == (*key, "fixed8")
        assert trace["initial"] == trial["initial"]
        coverage = [step["after"]["coverage"] for step in trace["steps"]]
        assert len(coverage) == trial["steps"] and coverage[-1] == trial["final_coverage"]
        assert trial["success"] == (coverage[-1] > .95)
        assert trial["success"] or len(coverage) == sac.MAX_MOVEMENTS
        pairs[key] = {"scene": key[0], "draw": key[1], "success": trial["success"],
                      "initial": trial["initial"], "movements": trial["steps"],
                      "final_coverage": coverage[-1], "peak_coverage": max(trial["initial"]["coverage"], *coverage)}
    assert set(pairs) == {(scene, draw) for scene in SCENES for draw in DRAWS}
    return {scene: [pairs[scene, draw] for draw in DRAWS] for scene in SCENES}


def evaluate_scene(model, env, scene, references):
    obs, _ = env.reset(seed=scene)
    initial = physical_state(env, env.previous)
    assert all(states_match(initial, row["initial"]) for row in references), f"Initial state mismatch: {scene}"
    steps = []
    started, inference = time.perf_counter(), 0.0
    for _ in range(sac.MAX_MOVEMENTS):
        tick = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        inference += time.perf_counter() - tick
        assert env.action_space.contains(action)
        obs, _, terminated, truncated, info = env.step(action)
        coverage = float(info["coverage"])
        steps.append({"action": action.tolist(), "coverage": coverage})
        if terminated or truncated:
            break
    assert bool(terminated) == (coverage > .95)
    assert terminated or (truncated and len(steps) == sac.MAX_MOVEMENTS)
    return {"scene": scene, "initial": initial, "success": bool(terminated), "truncated": bool(truncated),
            "movements": len(steps), "final_coverage": coverage,
            "peak_coverage": max(initial["coverage"], *(row["coverage"] for row in steps)),
            "wall_seconds": time.perf_counter() - started, "inference_seconds": inference, "steps": steps}


def summarize(rows, references):
    controls = [attempt for scene in SCENES for attempt in references[scene]]
    stats = {label: {"attempts": len(attempts), "successes": sum(row["success"] for row in attempts),
                     **{f"mean_{key}": mean(row[key] for row in attempts) for key in METRICS}}
             for label, attempts in (("visual_sac", rows), ("diffusion_fixed8", controls))}
    indices = sac.np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(SCENES),
                                                                size=(BOOTSTRAP_SAMPLES, len(SCENES)))
    paired = {}
    for key in METRICS:
        differences = sac.np.asarray([row[key] - mean(attempt[key] for attempt in references[row["scene"]])
                                      for row in rows])
        paired[key] = {"mean_difference": float(differences.mean()),
                       "scene_bootstrap_95_interval": sac.np.quantile(differences[indices].mean(axis=1),
                                                                     [.025, .975]).tolist()}
    return {"methods": stats, "paired_sac_minus_fixed8": paired,
            "episodes": [{**{key: value for key, value in row.items() if key != "steps"},
                          "diffusion_fixed8": [{k: v for k, v in attempt.items() if k != "initial"}
                                               for attempt in references[row["scene"]]]} for row in rows]}


def run(output):
    deps = ROOT / "tmp" / "visual-sac-deps"
    if deps.is_dir():
        sys.path.insert(0, str(deps))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "visual-sac-matplotlib"))
    from stable_baselines3 import SAC

    previous = read_json(TRAINING / "movements-100000.json")
    checkpoint = ROOT / previous["checkpoint"]
    assert previous["movements"] == 100_000 and previous["updates"] == 99_000
    assert sac.fingerprint(checkpoint) == previous["checkpoint_sha256"]
    assert version("stable-baselines3") == sac.SB3_VERSION and sac.torch.cuda.is_available()
    reference_protocol = read_json(REFERENCE / "protocol.json")
    packages = {name: version(name) for name in ("torch", "numpy", "gym-pusht", "gymnasium", "pymunk",
                                                 "shapely", "opencv-python", "pygame")}
    assert all(reference_protocol["packages"][name] == installed for name, installed in packages.items())
    references = read_reference()
    paths = [Path(__file__).resolve(), Path(sac.__file__), ROOT / "experiments" / "robotics_baseline_check.py",
             TRAINING / "protocol.json", TRAINING / "movements-100000.json",
             *(REFERENCE / name for name in ("protocol.json", "eval-bindings.json", "evaluation.json")), checkpoint]
    protocol = {"scenes": SCENES, "diffusion_draw_offsets": DRAWS, "sac_attempts_per_scene": 1,
                "sac_inference": "deterministic, CUDA, frozen checkpoint", "sac_training_seed": sac.TRAIN_SEED,
                "movement_limit": sac.MAX_MOVEMENTS, "success": "true coverage > 0.95",
                "statistics": {"unit": "paired initial scene, average both fixed-eight draws first",
                               "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_samples": BOOTSTRAP_SAMPLES},
                "packages": {**packages, "stable-baselines3": sac.SB3_VERSION},
                "gpu": sac.torch.cuda.get_device_name(),
                "sha256": {path.relative_to(ROOT).as_posix(): sac.fingerprint(path) for path in paths}}
    output.mkdir(parents=True, exist_ok=False)
    sac.write_json(output / "protocol.json", protocol)
    sac.torch.set_num_threads(4)
    model = SAC.load(checkpoint, device="cuda")
    rows = []
    with sac.make_env() as env, sac.torch.inference_mode():
        for scene in SCENES:
            row = evaluate_scene(model, env, scene, references[scene])
            rows.append(row)
            print(f"Scene {scene} ({len(rows)}/{len(SCENES)}): success={row['success']}, "
                  f"final coverage={row['final_coverage']:.2%}", flush=True)
    trace = output / "sac-traces.json.gz"
    with gzip.open(trace, "wt", encoding="utf-8") as stream:
        json.dump(rows, stream, allow_nan=False, separators=(",", ":"))
    result = {**summarize(rows, references), "protocol_sha256": sac.fingerprint(output / "protocol.json"),
              "trace_sha256": sac.fingerprint(trace)}
    sac.write_json(output / "summary.json", result)
    print(json.dumps(result["methods"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)
