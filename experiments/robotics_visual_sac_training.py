"""Run one capped visual SAC learning screen, preserving the original pilot."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

import robotics_visual_sac as pilot


CHECKPOINTS = (25_000, 50_000, 100_000)
MAX_SECONDS = 75 * 60
WARMUP = 1000
RESULTS = pilot.ROOT / "results" / "robotics-visual-sac-training"
MODELS = pilot.ROOT / "tmp" / "visual-sac-training-models"
PILOT_SOURCE = Path(pilot.__file__)


def no_progress(initial, checkpoints):
    """Stop at 50k only when neither intermediate evaluation improves."""
    if tuple(row["movements"] for row in checkpoints) != CHECKPOINTS[:2]:
        raise ValueError("The progress gate requires the 25k and 50k evaluations")
    return all(row["development"]["successes"] == 0 and
               row["development"]["mean_final_coverage"] <= initial["mean_final_coverage"]
               for row in checkpoints)


def train_segments(model, guard, record, initial):
    """Keep the learner and replay alive across evaluations and exact quotas."""
    saved = []
    for target in CHECKPOINTS:
        if guard.elapsed() >= MAX_SECONDS:
            return "time_limit", saved
        if not guard.safe():
            return "resource_guard", saved
        model.learn(total_timesteps=target - model.num_timesteps, reset_num_timesteps=False,
                    callback=guard, log_interval=None)
        if guard.reason == "resource_guard" or not guard.safe():
            return "resource_guard", saved
        if guard.reason is None and model.num_timesteps != target:
            raise RuntimeError("SAC did not reach the scheduled checkpoint")
        if not model.num_timesteps > (saved[-1]["movements"] if saved else 0):
            raise RuntimeError("SAC returned without collecting any new movements")
        saved.append(record(model))
        if guard.reason is not None:
            return guard.reason, saved
        if target == CHECKPOINTS[1] and no_progress(initial, saved):
            return "no_progress_at_50000", saved
    return "movement_limit", saved


def run(run_id):
    # Use the same isolated dependency installation as the frozen pilot.
    deps = pilot.ROOT / "tmp" / "visual-sac-deps"
    if deps.is_dir():
        sys.path.insert(0, str(deps))
    os.environ.setdefault("MPLCONFIGDIR", str(pilot.ROOT / "tmp" / "visual-sac-matplotlib"))
    import stable_baselines3 as sb3
    from stable_baselines3.common.callbacks import BaseCallback

    torch = pilot.torch
    source_sha = pilot.fingerprint(Path(__file__))
    pilot_sha = pilot.fingerprint(PILOT_SOURCE)
    reference = pilot.RESULTS / "pilot" / "summary.json"
    reference_doc = json.loads(reference.read_text(encoding="utf-8"))
    if reference_doc["source_sha256"] != pilot_sha or not reference_doc["resource_guard_passed"]:
        raise RuntimeError("The frozen pilot source or feasibility result does not match")
    if sb3.__version__ != pilot.SB3_VERSION or not torch.cuda.is_available():
        raise RuntimeError(f"Requires Stable-Baselines3 {pilot.SB3_VERSION} and CUDA")

    # run_path validates the ID; use different roots to protect earlier results.
    name = pilot.run_path(run_id).name
    output, model_dir = RESULTS / name, MODELS / name
    if output.exists() or model_dir.exists():
        raise FileExistsError(f"Run ID {name} already exists; use a new one")

    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    process = pilot.psutil.Process()
    resources = {"peak_rss_mib": 0.0, "minimum_available_ram_mib": float("inf"),
                 "minimum_free_vram_mib": float("inf")}

    def safe():
        mib = pilot.MIB
        resources["peak_rss_mib"] = max(resources["peak_rss_mib"], process.memory_info().rss / mib)
        resources["minimum_available_ram_mib"] = min(resources["minimum_available_ram_mib"],
                                                     pilot.psutil.virtual_memory().available / mib)
        resources["minimum_free_vram_mib"] = min(resources["minimum_free_vram_mib"],
                                                 torch.cuda.mem_get_info()[0] / mib)
        resources["peak_reserved_vram_mib"] = torch.cuda.max_memory_reserved() / mib
        resources["peak_allocated_vram_mib"] = torch.cuda.max_memory_allocated() / mib
        return (resources["minimum_available_ram_mib"] >= pilot.MIN_RAM_MIB
                and resources["minimum_free_vram_mib"] >= pilot.MIN_VRAM_MIB
                and resources["peak_reserved_vram_mib"] <= pilot.MAX_RESERVED_MIB)

    if not safe():
        raise RuntimeError(f"Insufficient headroom before model setup: {resources}")
    output.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    protocol = {
        "source_sha256": source_sha, "frozen_pilot_source_sha256": pilot_sha,
        "frozen_pilot_summary_sha256": pilot.fingerprint(reference),
        "packages": {name: version(name) for name in ("stable-baselines3", "torch", "gymnasium", "gym-pusht")},
        "gpu": torch.cuda.get_device_name(), "training_seed": pilot.TRAIN_SEED,
        "training_scene_range_half_open": pilot.TRAIN_SCENE_RANGE, "development_scenes": pilot.DEV_SCENES,
        "method": "From-scratch online SB3 SAC MultiInputPolicy; no demonstrations",
        "observation": "two RGB 96x96 frames and their pusher positions; no block pose",
        "action": "one absolute x-y target per movement",
        "reward": "unchanged Gym-PushT coverage reward",
        "success": "true coverage > 0.95", "max_episode_movements": pilot.MAX_MOVEMENTS,
        "buffer_size": pilot.BUFFER_SIZE, "batch_size": pilot.BATCH_SIZE, "warmup": WARMUP,
        "train_freq": 1, "gradient_steps": 1, "net_arch": [256, 256],
        "checkpoints": CHECKPOINTS, "max_elapsed_seconds_after_initial_evaluation": MAX_SECONDS,
        "early_stop": "At 50000 if both intermediate checkpoints have zero successes and neither "
                      "has mean final development coverage above initialization",
    }
    pilot.write_json(output / "protocol.json", protocol)

    class Guard(BaseCallback):
        def __init__(self):
            super().__init__()
            self.started = time.perf_counter()
            self.reason = None

        def elapsed(self):
            return time.perf_counter() - self.started

        def safe(self):
            return safe()

        def _on_step(self):
            if self.num_timesteps % 128 == 0 and not self.safe():
                self.reason = "resource_guard"
                return False
            if self.elapsed() >= MAX_SECONDS:
                self.reason = "time_limit"
                return False
            if self.num_timesteps % 1000 == 0:
                progress = {"movements": self.num_timesteps, "elapsed_seconds": self.elapsed(), **resources}
                with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(progress, allow_nan=False) + "\n")
                print(f"SAC {self.num_timesteps:,} movements in {self.elapsed():.1f}s", flush=True)
            return True

    with pilot.TrainingScenes(pilot.make_env(), pilot.TRAIN_SEED) as env:
        model = sb3.SAC("MultiInputPolicy", env, buffer_size=pilot.BUFFER_SIZE, batch_size=pilot.BATCH_SIZE,
                        learning_starts=WARMUP, train_freq=1, gradient_steps=1,
                        policy_kwargs={"net_arch": [256, 256]}, seed=pilot.TRAIN_SEED, device="cuda")
        initial = pilot.evaluate(model)
        pilot.write_json(output / "initial-development.json", initial)
        print(f"Untrained SAC: {initial['successes']}/8 successes, "
              f"{initial['mean_final_coverage']:.2%} mean final coverage", flush=True)
        guard = Guard()

        def record(current):
            torch.cuda.synchronize()
            elapsed_before_evaluation = guard.elapsed()
            losses = {key: float(current.logger.name_to_value[f"train/{key}"])
                      for key in ("actor_loss", "critic_loss")}
            if not (all(pilot.np.isfinite(v) for v in losses.values()) and
                    all(torch.isfinite(p).all().item() for p in current.policy.parameters())):
                raise RuntimeError("Nonfinite SAC model or losses")
            path = model_dir / f"movements-{current.num_timesteps:06d}.zip"
            current.save(path)
            result = {"movements": current.num_timesteps, "updates": current._n_updates,
                      "elapsed_before_evaluation_seconds": elapsed_before_evaluation,
                      "checkpoint": path.relative_to(pilot.ROOT).as_posix(),
                      "checkpoint_sha256": pilot.fingerprint(path), "losses": losses,
                      "mean_recent_training_return": float(pilot.np.mean([ep["r"] for ep in current.ep_info_buffer])),
                      "development": pilot.evaluate(current)}
            pilot.write_json(output / f"movements-{current.num_timesteps:06d}.json", result)
            scores = result["development"]
            print(f"At {current.num_timesteps:,}: {scores['successes']}/8 successes, "
                  f"{scores['mean_final_coverage']:.2%} mean final coverage", flush=True)
            return result

        stop, checkpoints = train_segments(model, guard, record, initial)
        safe()
        summary = {"stop_reason": stop, "movements": model.num_timesteps, "updates": model._n_updates,
                   "elapsed_after_initial_evaluation_seconds": guard.elapsed(),
                   "protocol_sha256": pilot.fingerprint(output / "protocol.json"),
                   "initial_development": initial, "checkpoints": checkpoints, "resources": resources}
        pilot.write_json(output / "summary.json", summary)
        print(f"SAC stopped at {model.num_timesteps:,} movements: {stop}", flush=True)
        if stop == "resource_guard":
            raise RuntimeError("Resource guard stopped training; completed checkpoints were preserved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    run(parser.parse_args().run_id)
