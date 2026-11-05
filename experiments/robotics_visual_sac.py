"""Bounded development pilot of standard Stable-Baselines3 SAC on Gym-PushT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import gymnasium as gym
import gym_pusht  # noqa: F401 -- registers the environment
import numpy as np
import psutil
import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "robotics-visual-sac-pilot"
LOCAL_MODELS = ROOT / "tmp" / "visual-sac-models"
SB3_VERSION = "2.7.1"
TRAIN_SEED = 20261105
TRAIN_SCENE_RANGE = (100_000, 1_000_000)  # upper endpoint exclusive
DEV_SCENES = tuple(range(8))  # previously inspected discovery scenes
MAX_MOVEMENTS = 300
BUFFER_SIZE = 20_000
BATCH_SIZE = 64
MIB = 2**20
MIN_RAM_MIB = 2048
MIN_VRAM_MIB = 384
MAX_RESERVED_MIB = 2560
STAGES = {"preflight": (256, 64, 90), "pilot": (20_000, 1000, 600)}


class VisualHistory(gym.Wrapper):
    """Keep the same two RGB frames and pusher positions, oldest first."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, (6, 96, 96), dtype=np.uint8),
            "pusher": gym.spaces.Box(0.0, 1.0, (4,), dtype=np.float32),
        })

    @staticmethod
    def _pack(previous, current):
        return {
            "image": np.concatenate([previous["pixels"], current["pixels"]], axis=2)
            .transpose(2, 0, 1).copy(),
            "pusher": np.concatenate([previous["agent_pos"], current["agent_pos"]])
            .astype(np.float32) / np.float32(512),
        }

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.previous = obs
        return self._pack(obs, obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        history = self._pack(self.previous, obs)
        self.previous = obs
        return history, reward, terminated, truncated, info


def make_env():
    return VisualHistory(gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                                  max_episode_steps=MAX_MOVEMENTS))


class TrainingScenes(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        scene = int(self.rng.integers(*TRAIN_SCENE_RANGE))
        return self.env.reset(seed=scene, options=options)


def evaluate(model):
    """Diagnostic only: deterministic actor on the fixed development panel."""
    episodes = []
    with make_env() as env:
        for scene in DEV_SCENES:
            obs, _ = env.reset(seed=scene)
            initial = float(env.unwrapped._get_coverage())
            coverage, peak, inference = initial, initial, 0.0
            start = time.perf_counter()
            for movement in range(1, MAX_MOVEMENTS + 1):
                tick = time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
                inference += time.perf_counter() - tick
                if not env.action_space.contains(action):
                    raise RuntimeError("Actor produced an invalid pusher target")
                obs, _, terminated, truncated, info = env.step(action)
                coverage = float(info["coverage"])
                peak = max(peak, coverage)
                if terminated or truncated:
                    break
            episodes.append(dict(scene=scene, success=bool(terminated), movements=movement,
                                 initial_coverage=initial, final_coverage=coverage, peak_coverage=peak,
                                 wall_seconds=time.perf_counter() - start, inference_seconds=inference))
    return dict(successes=sum(row["success"] for row in episodes), episodes=episodes,
                **{f"mean_{key}": float(np.mean([row[key] for row in episodes]))
                   for key in ("final_coverage", "peak_coverage", "wall_seconds", "movements")})


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_path(run_id):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", run_id):
        raise ValueError("Use a short lowercase run ID")
    return RESULTS / run_id


def run(stage, run_id, preflight_id):
    deps = ROOT / "tmp" / "visual-sac-deps"
    if deps.is_dir():
        sys.path.insert(0, str(deps))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "visual-sac-matplotlib"))
    import stable_baselines3 as sb3
    from stable_baselines3.common.callbacks import BaseCallback

    output = run_path(run_id)
    checkpoint = LOCAL_MODELS / f"{run_id}.zip"
    source_hash = fingerprint(Path(__file__))
    if stage == "pilot":
        if preflight_id is None:
            raise ValueError("Pilot requires a passing preflight")
        gate = json.loads((run_path(preflight_id) / "summary.json").read_text(encoding="utf-8"))
        if not (gate["stage"] == "preflight" and gate["passed_preflight"]
                and gate["source_sha256"] == source_hash):
            raise ValueError("The passing preflight must use the current source")
    if sb3.__version__ != SB3_VERSION or not torch.cuda.is_available():
        raise RuntimeError(f"Requires Stable-Baselines3 {SB3_VERSION} and CUDA")
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    steps, warmup, max_seconds = STAGES[stage]
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    resources = dict(peak_rss_mib=0.0, minimum_available_ram_mib=float("inf"),
                     minimum_free_vram_mib=float("inf"), peak_reserved_vram_mib=0.0,
                     peak_allocated_vram_mib=0.0)

    def sample_resources():
        resources["peak_rss_mib"] = max(resources["peak_rss_mib"], process.memory_info().rss / MIB)
        resources["minimum_available_ram_mib"] = min(resources["minimum_available_ram_mib"],
                                                     psutil.virtual_memory().available / MIB)
        resources["minimum_free_vram_mib"] = min(resources["minimum_free_vram_mib"],
                                                 torch.cuda.mem_get_info()[0] / MIB)
        resources["peak_reserved_vram_mib"] = torch.cuda.max_memory_reserved() / MIB
        resources["peak_allocated_vram_mib"] = torch.cuda.max_memory_allocated() / MIB
        return (resources["minimum_available_ram_mib"] >= MIN_RAM_MIB
                and resources["minimum_free_vram_mib"] >= MIN_VRAM_MIB
                and resources["peak_reserved_vram_mib"] <= MAX_RESERVED_MIB)

    if not sample_resources():
        raise RuntimeError(f"Insufficient headroom before model setup: {resources}")
    output.mkdir(parents=True, exist_ok=False)
    config = dict(stage=stage, source_sha256=source_hash, preflight_id=preflight_id,
                  sb3=sb3.__version__, torch=torch.__version__, gymnasium=gym.__version__,
                  gpu=torch.cuda.get_device_name(), train_seed=TRAIN_SEED,
                  train_scene_range=TRAIN_SCENE_RANGE, dev_scenes=DEV_SCENES,
                  steps=steps, warmup=warmup, max_training_seconds=max_seconds,
                  max_episode_movements=MAX_MOVEMENTS, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                  observation="two RGB 96x96 frames and their pusher x-y; no block pose",
                  reward="native Gym-PushT: min(coverage / 0.95, 1)",
                  success="true coverage > 0.95", action="one absolute x-y target per movement",
                  algorithm="SB3 SAC MultiInputPolicy, default NatureCNN and SAC hyperparameters except recorded overrides")
    write_json(output / "protocol.json", config)

    class Guard(BaseCallback):
        reason = "interaction_limit"

        def _on_step(self):
            elapsed = time.perf_counter() - started
            if self.num_timesteps % 128 == 0 and not sample_resources():
                self.reason = "resource_guard"
                return False
            if self.num_timesteps % 1000 == 0:
                progress = dict(movements=self.num_timesteps, seconds=elapsed, **resources)
                with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(progress, allow_nan=False) + "\n")
                print(f"Visual SAC: {self.num_timesteps}/{steps} movements in {elapsed:.1f}s", flush=True)
            if elapsed >= max_seconds:
                self.reason = "time_limit"
                return False
            return True

    with TrainingScenes(make_env(), TRAIN_SEED) as train_env:
        model = sb3.SAC("MultiInputPolicy", train_env, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                        learning_starts=warmup, train_freq=1, gradient_steps=1,
                        policy_kwargs={"net_arch": [256, 256]}, seed=TRAIN_SEED, device="cuda")
        before = evaluate(model) if stage == "pilot" else None
        if before is not None:
            write_json(output / "initial-development.json", before)
        targets = [model.actor.mu.weight, model.critic.q_networks[0][-1].weight]
        original = [param.detach().clone() for param in targets]
        guard = Guard()
        started = time.perf_counter()
        model.learn(total_timesteps=steps, callback=guard, log_interval=None)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        safe = sample_resources() and guard.reason != "resource_guard"
        losses = {name: float(model.logger.name_to_value[f"train/{name}"])
                  for name in ("actor_loss", "critic_loss") if f"train/{name}" in model.logger.name_to_value}
        changed = all(not torch.equal(old, new.detach()) for old, new in zip(original, targets))
        finite = (len(losses) == 2 and all(np.isfinite(value) for value in losses.values())
                  and all(torch.isfinite(param).all().item() for param in model.policy.parameters()))
        replay = model.replay_buffer
        replay_bytes = sum(array.nbytes for array in (*replay.observations.values(),
                                                      *replay.next_observations.values(), replay.actions,
                                                      replay.rewards, replay.dones, replay.timeouts))
        result = dict(stage=stage, source_sha256=source_hash, movements=model.num_timesteps,
                      updates=model._n_updates, training_seconds=train_seconds,
                      movements_per_second=model.num_timesteps / train_seconds,
                      stop_reason=guard.reason if safe else "resource_guard",
                      passed_preflight=bool(stage == "preflight" and safe and model.num_timesteps == steps
                                            and model._n_updates == steps - warmup and changed and finite),
                      actor_and_critic_updated=changed, finite_training=finite,
                      resource_guard_passed=safe, replay_bytes=replay_bytes, resources=resources,
                      losses=losses, initial_development=before)
        if stage == "pilot" and safe and finite and changed:
            LOCAL_MODELS.mkdir(parents=True, exist_ok=True)
            model.save(checkpoint)
            result.update(checkpoint=checkpoint.relative_to(ROOT).as_posix(),
                          checkpoint_sha256=fingerprint(checkpoint), final_development=evaluate(model))
        write_json(output / "summary.json", result)
        print(json.dumps({key: result[key] for key in ("stage", "movements", "updates", "training_seconds",
                                                        "stop_reason", "passed_preflight", "resources")}), flush=True)
        if not (safe and finite and changed and (stage == "pilot" or result["passed_preflight"])):
            raise RuntimeError(f"Visual SAC did not pass checks; see {output / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--preflight-id")
    args = parser.parse_args()
    run(args.stage, args.run_id, args.preflight_id)
