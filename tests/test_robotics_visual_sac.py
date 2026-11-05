"""Verify that the SAC wrapper changes observation formatting only."""

from pathlib import Path
import sys
import unittest

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_visual_sac as pilot


class VisualSACTests(unittest.TestCase):
    def test_same_sensor_history_and_simulation_under_matching_commands(self):
        with pilot.make_env() as candidate, gym.make(
            "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300
        ) as reference:
            for seed in (2, 7):
                prior, _ = reference.reset(seed=seed)
                history, _ = candidate.reset(seed=seed)
                self.assert_history(history, prior, prior)
                commands = np.random.default_rng(seed).uniform(0, 512, (32, 2)).astype(np.float32)
                for command in commands:
                    raw, reward, terminated, truncated, info = reference.step(command)
                    history, got_reward, got_terminated, got_truncated, got_info = candidate.step(command)
                    self.assertTrue(candidate.observation_space.contains(history))
                    self.assert_history(history, prior, raw)
                    self.assertEqual((got_reward, got_terminated, got_truncated), (reward, terminated, truncated))
                    self.assertEqual(got_info["coverage"], info["coverage"])
                    prior = raw
                    if terminated or truncated:
                        break

    def assert_history(self, history, old, new):
        self.assertEqual(set(history), {"image", "pusher"})
        np.testing.assert_array_equal(history["image"][:3], np.moveaxis(old["pixels"], 2, 0))
        np.testing.assert_array_equal(history["image"][3:], np.moveaxis(new["pixels"], 2, 0))
        expected = np.asarray([*old["agent_pos"], *new["agent_pos"]], dtype=np.float32)
        np.testing.assert_array_equal(history["pusher"] * 512, expected)

    def test_stationary_attempt_truncates_at_movement_300(self):
        with pilot.make_env() as env:
            history, _ = env.reset(seed=0)
            target = history["pusher"][2:] * 512
            for movement in range(1, 301):
                _, _, terminated, truncated, _ = env.step(target)
                self.assertFalse(terminated)
                self.assertEqual(truncated, movement == 300)


if __name__ == "__main__":
    unittest.main()
