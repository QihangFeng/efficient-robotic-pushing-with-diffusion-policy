"""Prepare a proposed baseline-success rehearsal pilot without training.

Reads the immutable prior arrays and traces; creates a separate supplement and
development schedule. There is deliberately no train/evaluate entry point.
"""

import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE = Path('results/robotics-recovery-pilot')
OUTPUT = Path('results/robotics-retention-preflight')
SPLIT_KEY = '20261107/baseline-retention'
DRAWS = [91000, 92000]
ARMS = ['fixed8', 'current12', 'retention12']
GROUPS = ['expert', 'recovery', 'baseline', 'recovery']
EXCLUDED_TEST_SCENES = list(range(170, 210))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def split_scenes(scenes):
    scenes = sorted(scenes)
    assert len(scenes) == len(set(scenes)) == 20
    ranked = sorted(scenes, key=lambda s: hashlib.sha256(f'{SPLIT_KEY}/{s}'.encode()).hexdigest())
    return {'train': sorted(ranked[4:]), 'validation': sorted(ranked[:4])}


def windows(episode):
    assert episode['n_frames'] == episode['n_actions'] + 1
    return [{'episode_id': episode['episode_id'], 'group': 'baseline', 'split': episode['split'],
             'first_observation_index': i, 'current_observation_index': i + 1,
             'action_start_index': i, 'action_stop_index': i + 16}
            for i in range(episode['n_actions'] - 15)]


def schedule(scenes):
    return [{'scene': scene, 'draw': draw, 'arm': arm}
            for si, scene in enumerate(sorted(scenes)) for di, draw in enumerate(DRAWS)
            for arm in ARMS[(2 * si + di) % 3:] + ARMS[:(2 * si + di) % 3]]


def verify_arrays(episode):
    """Check all recorded RGB, own positions and action labels, without replay."""
    assert sha(episode['file']) == episode['sha256']
    assert sha(episode['source']['trace']) == episode['source']['trace_sha256']
    with gzip.open(episode['source']['trace'], 'rt', encoding='utf-8') as stream:
        trace = json.load(stream)
    steps = trace['steps']
    assert len(steps) == episode['n_actions']
    assert steps[-1]['after']['coverage'] > .95
    assert not any(row['after']['coverage'] > .95 for row in steps[:-1])
    with np.load(episode['file'], allow_pickle=False) as archive:
        assert set(archive.files) == {'pixels', 'states', 'actions'}
        pixels, states, actions = (archive[k] for k in ['pixels', 'states', 'actions'])
    assert pixels.dtype == np.uint8 and pixels.shape == (len(steps) + 1, 96, 96, 3)
    assert states.dtype == actions.dtype == np.float32
    assert states.shape == (len(steps) + 1, 2) and actions.shape == (len(steps), 2)
    assert np.isfinite(states).all() and np.isfinite(actions).all()
    np.testing.assert_array_equal(actions, np.asarray([row['action'] for row in steps], dtype=np.float32))
    recorded = [trace['initial']] + [row['after'] for row in steps]
    np.testing.assert_array_equal(states, np.asarray([r['pusher_pos'] for r in recorded], dtype=np.float32))
    for index, state in enumerate(recorded):
        assert hashlib.sha256(pixels[index].tobytes()).hexdigest() == state['image_sha256']
    return {'episode_id': episode['episode_id'], 'scene': episode['scene'],
            'rgb_frames_checked': len(recorded), 'executed_actions_checked': len(steps),
            'own_positions_checked': len(recorded), 'new_rendering_or_physics': False}


def build():
    source = read(SOURCE / 'dataset.json')
    training = read(SOURCE / 'training.json')
    assert source['complete'] and training['complete']
    assert sha(SOURCE / 'dataset.json') == training['dataset_sha256']
    assert source['ordinary_controls'] == {'episodes': 20, 'used_for_training': False}
    originals = [e for e in source['episodes'] if e['group'] == 'ordinary_control']
    splits = split_scenes([e['scene'] for e in originals])
    assert not (set(splits['train']) | set(splits['validation'])) & set(EXCLUDED_TEST_SCENES)
    recovery = source['selection']['recovery_validation_scenes']
    assert not set(splits['train']) & set(recovery)
    episodes, checks, new_windows = [], [], []
    bindings = {str(SOURCE / name): sha(SOURCE / name) for name in
                ['dataset.json', 'protocol.json', 'training.json', 'independent-audit.json']}
    for name in ['experiments/robotics_retention_preflight.py', 'tests/test_robotics_retention_preflight.py',
                 'experiments/robotics_recovery_training.py', 'experiments/robotics_recovery_dataset.py']:
        bindings[name] = sha(name)
    for original in originals:
        assert original['split'] == 'diagnostic' and original['source']['category'] == 'ordinary_successful_baseline'
        assert original['source']['scene'] == original['scene']
        checks.append(verify_arrays(original))
        for name, expected in [(original['file'], original['sha256']),
                               (original['source']['trace'], original['source']['trace_sha256'])]:
            assert bindings.setdefault(name, expected) == expected
        derived = copy.deepcopy(original)
        derived.update(group='baseline', split='validation' if original['scene'] in splits['validation'] else 'train',
                       original_group='ordinary_control', original_split='diagnostic')
        episodes.append(derived)
        new_windows.extend(windows(derived))
    assert len(new_windows) == 2686
    # Test trajectories cannot enter through the supplement or its source bindings.
    assert not any('robotics-recovery-confirmation/' in p or 'robotics-recovery-fixed8/' in p for p in bindings)
    development_scenes = sorted(splits['validation'] + recovery)
    assert len(development_scenes) == len(set(development_scenes)) == 7
    proposed_settings = copy.deepcopy(training['settings'])
    supplement = {'status': 'prepared_candidate_data_only; no training or approval implied',
        'source_dataset': {'path': str(SOURCE / 'dataset.json'), 'sha256': sha(SOURCE / 'dataset.json')},
        'group': 'baseline', 'split_key': SPLIT_KEY, 'scene_splits': splits,
        'episodes': episodes, 'windows': new_windows,
        'window_counts': dict(Counter(w['split'] for w in new_windows)),
        'source_bindings': dict(sorted(bindings.items())),
        'contract': 'Reuse existing 96x96 RGB, own pusher positions and 16 actually executed labels; no padding, block-state inputs or original-file edits.'}
    plan = {'status': 'proposal_ready_for_review; no model run',
        'training': {'settings': proposed_settings, 'microbatch_groups': GROUPS,
                     'optimizer_steps': 600, 'total_denoiser_training_calls': 2400,
                     'start_from': 'same original pretrained checkpoint, not the previous fine-tuned head',
                     'trainable_prefix': 'diffusion.unet.final_conv.', 'controller': 'unchanged milestone 8/12'},
        'comparison': {'scenes': development_scenes, 'draws': DRAWS, 'arms': ARMS,
                       'schedule': schedule(development_scenes), 'attempts': 42,
                       'max_movements': 300, 'generation_horizon': 16, 'observations': 2,
                       'denoising_steps': 100, 'success': 'true coverage strictly greater than 0.95',
                       'prediction_seed': 'scene plus draw; paired by prediction call order',
                       'excluded_test_scenes': EXCLUDED_TEST_SCENES,
                       'scope': 'Inspected discovery scenes; development evidence only, not generalization.'},
        'limits': ['No train/evaluate command exists in this preflight.',
                   'Training implementation, CPU tests and frozen execution protocol remain required before a model run.',
                   'No new model or recovery rule has been validated.']}
    audit = {'passed': True, 'baseline_success_episodes': len(episodes),
             'all_rgb_frames_checked': sum(x['rgb_frames_checked'] for x in checks),
             'all_actions_checked': sum(x['executed_actions_checked'] for x in checks),
             'complete_windows': len(new_windows), 'per_episode': checks,
             'scene_splits': splits, 'development_scenes': development_scenes,
             'test_scene_overlap': [], 'source_arrays_unchanged': True,
             'gradients_policy_predictions_and_physics_calls': 0}
    assert len(plan['comparison']['schedule']) == 42
    assert all(sha(name) == expected for name, expected in bindings.items())
    return supplement, plan, audit


def save_once(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, 'Prepared source/split changed; preserve the previous preflight.'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(output=OUTPUT):
    assert output.resolve() in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + '-repeat').resolve()}
    supplement, plan, audit = build()
    for name, value in [('baseline-supplement.json', supplement), ('plan.json', plan), ('data-audit.json', audit)]:
        save_once(output / name, value)
    return {k:audit[k] for k in ['passed', 'baseline_success_episodes', 'complete_windows', 'scene_splits', 'development_scenes']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    print(run(parser.parse_args().output_dir))
