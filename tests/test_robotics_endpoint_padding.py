"""Endpoint alignment, native targets, original-sample identity and resume checks."""

import copy
from collections import OrderedDict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import robotics_endpoint_padding as pilot
from test_robotics_preservation_pilot import TinyPolicy, ToyDataset

torch.set_num_threads(1)


def episode(T=25, branch=0):
    return {'episode_id': 'test', 'group': 'recovery', 'split': 'train', 'scene': 2,
            'n_actions': T, 'n_frames': T + 1, 'branch_after_step': branch}


def dataset():
    d = pilot.EndpointWindows.__new__(pilot.EndpointWindows)
    e = episode()
    pixels = np.arange(26 * 96 * 96 * 3, dtype=np.uint32).reshape(26, 96, 96, 3).astype(np.uint8)
    states = np.arange(52, dtype=np.float32).reshape(26, 2)
    actions = np.arange(50, dtype=np.float32).reshape(25, 2)
    d.episodes = {'test': e}
    d.windows = [{'episode_id': 'test', 'group': 'recovery', 'split': 'train',
                 'first_observation_index': 0, 'current_observation_index': 1,
                 'action_start_index': 0, 'action_stop_index': 16}]
    d.original_window_count = 1
    d.windows.extend(pilot.endpoint_windows(e))
    d.cache = OrderedDict(test=(pixels, states, [{'step': i + 1, 'action': a} for i, a in enumerate(actions)]))
    d.cache_size = 8
    return d, pixels, states, actions


class EndpointTests(unittest.TestCase):
    def test_seven_tail_anchors_and_zero_padding_control(self):
        self.assertEqual(list(pilot.endpoint_windows(episode(), 0)), [])
        w = list(pilot.endpoint_windows(episode()))
        self.assertEqual([x['current_observation_index'] for x in w], list(range(11, 18)))
        self.assertEqual([x['padded_action_count'] for x in w], list(range(1, 8)))

    def test_short_successful_continuation_is_included(self):
        w = list(pilot.endpoint_windows(episode(201, 193)))
        self.assertEqual(len(w), 1)
        self.assertEqual(w[0]['current_observation_index'], 193)
        self.assertEqual(w[0]['padded_action_count'], 7)

    def test_no_window_with_less_than_eight_future_actions(self):
        self.assertEqual(list(pilot.endpoint_windows(episode(200, 193))), [])
        with self.assertRaises(AssertionError):
            list(pilot.endpoint_windows(episode(), 8))

    def test_old_complete_batch_is_bitwise_unchanged(self):
        d, _, _, _ = dataset()
        old = pilot.training.PreparedWindows.batch(d, 0, torch.device('cpu'))
        actual = d.batch(0, torch.device('cpu'))
        self.assertEqual(old.keys(), actual.keys())
        for key in old:
            self.assertTrue(torch.equal(old[key], actual[key]), key)

    def test_padded_targets_observations_and_masks_have_native_alignment(self):
        d, pixels, states, actions = dataset()
        for i in range(1, len(d.windows)):
            window = d.windows[i]; j = window['first_observation_index']
            actual = d.batch(i, torch.device('cpu'))
            wanted = actions[np.minimum(np.arange(j, j + 16), len(actions) - 1)]
            np.testing.assert_array_equal(actual['action'][0].numpy(), wanted)
            np.testing.assert_array_equal(actual['observation.state'][0].numpy(), states[[j, j + 1]])
            for slot, frame in enumerate((j, j + 1)):
                self.assertTrue(torch.equal(actual['observation.image'][0, slot],
                    torch.from_numpy(pixels[frame].copy()).permute(2, 0, 1).float() / 255.))
            np.testing.assert_array_equal(actual['action_is_pad'][0].numpy(), np.arange(j, j + 16) >= len(actions))
            self.assertFalse(actual['action_is_pad'][0, :9].any())
            self.assertTrue(all(not x.requires_grad for x in actual.values()))

    def test_source_cache_actions_are_never_extended(self):
        d, _, _, actions = dataset()
        for i in range(1, len(d.windows)):
            d.batch(i, torch.device('cpu'))
        self.assertEqual(len(d.cache['test'][2]), 25)
        np.testing.assert_array_equal(np.asarray([r['action'] for r in d.cache['test'][2]]), actions)

    def test_actual_manifest_counts_and_old_prefix(self):
        d = pilot.EndpointWindows()
        self.assertEqual(d.windows[:d.original_window_count], d.manifest['windows'])
        self.assertEqual(len(d.windows), 12021)
        c = d.description()['counts']
        self.assertEqual(c['train/expert/padded'], 448)
        self.assertEqual(c['train/recovery/padded'], 129)
        self.assertEqual(c['validation/expert/padded'], 112)
        self.assertEqual(c['validation/recovery/padded'], 77)
        self.assertEqual(pilot.sample_schedule(d), pilot.sample_schedule(d))

    def test_real_endpoint_examples_match_original_array_bytes(self):
        d = pilot.EndpointWindows()
        for group in ('expert', 'recovery'):
            for split in ('train', 'validation'):
                i = next(i for i, w in enumerate(d.windows) if w['group'] == group
                         and w['split'] == split and w.get('padded_action_count') == 7)
                w = d.windows[i]; e = d.episodes[w['episode_id']]
                actual = d.batch(i, torch.device('cpu'))
                with np.load(e['file'], allow_pickle=False) as ar:
                    j = w['first_observation_index']
                    np.testing.assert_array_equal(actual['action'][0, :9].numpy(), ar['actions'][j:])
                    np.testing.assert_array_equal(actual['action'][0, 9:].numpy(), np.repeat(ar['actions'][-1:], 7, axis=0))
                    np.testing.assert_array_equal(actual['observation.state'][0].numpy(), ar['states'][[j, j + 1]])
                self.assertEqual(actual['action_is_pad'].sum().item(), 7)


class ResumeTests(unittest.TestCase):
    def test_reused_training_resume_reproduces_weights_losses_and_frozen_backbone(self):
        root = Path(__file__).resolve().parents[1] / 'tmp'
        root.mkdir(exist_ok=True)
        settings = {**pilot.SETTINGS, 'optimizer_steps': 2, 'validation_steps': [0, 1, 2],
                    'checkpoint_steps': [1, 2], 'validation_windows_per_group': 2}
        torch.manual_seed(72)
        original = TinyPolicy().eval()
        full, resumed = copy.deepcopy(original), copy.deepcopy(original)
        backbone = pilot.training.backbone_digest(original)
        with tempfile.TemporaryDirectory(dir=root, prefix='endpoint-tests-') as folder:
            folder = Path(folder); marker = folder / 'input.txt'; marker.write_text('fixed')
            bindings = {str(marker): pilot.prep.sha(marker)}
            expected = pilot.training.train(full, ToyDataset(), folder / 'full', settings, bindings)
            calls = 0
            update = pilot.training.training_update

            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('simulated interruption')
                return update(*args, **kwargs)

            with patch.object(pilot.training, 'training_update', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    pilot.training.train(resumed, ToyDataset(), folder / 'resumed', settings, bindings)
            actual = pilot.training.train(resumed, ToyDataset(), folder / 'resumed', settings, bindings)
            for left, right in zip(expected['history']['updates'], actual['history']['updates']):
                for key in ('step', 'windows', 'groups', 'loss', 'microbatch_losses', 'gradient_l2_norm'):
                    self.assertEqual(left[key], right[key])
            self.assertEqual(expected['final_head']['tensor_digest'], actual['final_head']['tensor_digest'])
            self.assertEqual(pilot.training.backbone_digest(full), backbone)
            self.assertEqual(pilot.training.backbone_digest(resumed), backbone)


if __name__ == '__main__':
    unittest.main()
