"""Limited endpoint-padding test with the original recovery training recipe.

Append training-only windows with at most seven repeated endpoint commands.
The first eight future targets always remain observed executed commands.
Original arrays, frozen sources, model generation and controller are unchanged.
Collection/statistics below reuse the preservation pilot's audited procedure.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import robotics_preservation_pilot as reference

old = reference.old
prep, prior, training, gen, adaptive, base = (reference.prep, reference.prior,
    reference.training, reference.gen, reference.adaptive, reference.base)
OUTPUT = Path('results/robotics-endpoint-padding')
CACHE = Path('data/prepared/robotics/endpoint-padding/training')
SOURCE_DATASET = prep.SOURCE / 'dataset.json'
MAX_PADDING = 7
GROUPS = ['expert', 'recovery', 'expert', 'recovery']
SETTINGS = dict(training.SETTINGS)
ARMS = {'fixed8': 'fixed8', 'current12': 'adaptive12', 'padded12': 'adaptive12'}
SMOKE = [('before', 'fixed8'), ('current', 'current12'), ('candidate', 'padded12'), ('restored', 'fixed8')]
reference_rows = reference.reference_rows


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def endpoint_windows(episode, max_padding=MAX_PADDING):
    assert isinstance(max_padding, int) and 0 <= max_padding <= 7
    T = episode['n_actions']
    assert episode['n_frames'] in (T, T + 1)
    first = max(0, episode['branch_after_step'] - 1, T - 15)
    for j in range(first, T - 16 + max_padding + 1):
        assert j >= 0 and j + 1 < episode['n_frames']
        padding = j + 16 - T
        assert 1 <= padding <= max_padding and T - j >= 9
        yield {'episode_id': episode['episode_id'], 'group': episode['group'],
               'split': episode['split'], 'first_observation_index': j,
               'current_observation_index': j + 1, 'action_start_index': j,
               'action_stop_index': j + 16, 'padded_action_count': padding}


class EndpointWindows(training.PreparedWindows):
    def __init__(self, manifest_path=SOURCE_DATASET, cache_size=8, max_padding=MAX_PADDING):
        super().__init__(manifest_path, cache_size)
        self.windows = list(self.windows)
        self.original_window_count = len(self.windows)
        for episode in self.episodes.values():
            if episode['group'] not in training.GROUPS:
                continue
            for window in endpoint_windows(episode, max_padding):
                index = len(self.windows)
                self.windows.append(window)
                self.groups.setdefault((window['split'], window['group']), {}).setdefault(
                    episode['scene'], {}).setdefault(episode['episode_id'], []).append(index)
        self.max_padding = max_padding

    def batch(self, index, device):
        window = self.windows[index]
        if index < self.original_window_count:
            return super().batch(index, device)
        identifier = window['episode_id']
        episode = self.episodes[identifier]
        if identifier not in self.cache:
            with np.load(episode['file'], allow_pickle=False) as archive:
                pixels, states, actions = (archive[k].copy() for k in ('pixels', 'states', 'actions'))
            assert pixels.shape == (episode['n_frames'], 96, 96, 3) and pixels.dtype == np.uint8
            assert states.shape == (episode['n_frames'], 2) and states.dtype == np.float32
            assert actions.shape == (episode['n_actions'], 2) and actions.dtype == np.float32
            assert np.isfinite(states).all() and np.isfinite(actions).all()
            rows = [{'step': i + 1, 'action': a} for i, a in enumerate(actions)]
            self.cache[identifier] = pixels, states, rows
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(identifier)
        pixels, states, rows = self.cache[identifier]
        j, T = window['first_observation_index'], len(rows)
        indices = np.arange(j, j + 16)
        padded = indices >= T
        assert padded.sum() == window['padded_action_count'] <= 7 and not padded[:9].any()
        actions = np.asarray([rows[min(int(i), T - 1)]['action'] for i in indices], dtype=np.float32)
        # Endpoint copies fill training targets only; no executed trace is extended.
        sample = {'observation.image': torch.stack([torch.from_numpy(pixels[k].copy()).permute(2, 0, 1).float() / 255.
                                                   for k in (j, j + 1)]).unsqueeze(0),
                  'observation.state': torch.from_numpy(states[[j, j + 1]].copy()).unsqueeze(0),
                  'action': torch.from_numpy(actions).unsqueeze(0),
                  'action_is_pad': torch.from_numpy(padded).unsqueeze(0)}
        return {k: v.to(device) for k, v in sample.items()}

    def description(self):
        counts = Counter((w['split'], w['group'], 'padded' if i >= self.original_window_count else 'complete')
                         for i, w in enumerate(self.windows))
        return {'original_windows': self.original_window_count, 'total_windows': len(self.windows),
                'window_fingerprint': fingerprint(self.windows), 'maximum_padding': self.max_padding,
                'counts': {'/'.join(k): v for k, v in sorted(counts.items())},
                'contract': 'Two real RGB/pusher observations; >=8 real future commands; last command repeated at most7 times; native unmasked loss.'}


def sample_schedule(dataset, settings=SETTINGS):
    rng = np.random.default_rng(settings['seed'])
    return [{'step': step, 'groups': list(GROUPS),
             'windows': [dataset.choose('train', group, rng) for group in GROUPS]}
            for step in range(1, settings['optimizer_steps'] + 1)]


def check_output(output):
    assert output.resolve() in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + '-repeat').resolve()}


def artifacts(output):
    check_output(output)
    return CACHE if output.resolve() == OUTPUT.resolve() else CACHE.parent.with_name(CACHE.parent.name + '-repeat') / 'training'


def protocol_document():
    source = prep.read(reference.OUTPUT / 'protocol.json')
    training.verify_bindings(source['fingerprints'])
    dataset = EndpointWindows()
    fingerprints = {**source['fingerprints'], **dataset.bindings}
    for path in [reference.OUTPUT / n for n in ('protocol.json', 'training.json', 'summary.json', 'artifact-audit.json')]:
        fingerprints[str(path)] = prep.sha(path)
    for path in ['experiments/robotics_endpoint_padding.py', 'experiments/robotics_endpoint_padding_audit.py',
                 'tests/test_robotics_endpoint_padding.py',
                 '.venv-robotics/Lib/site-packages/lerobot/policies/diffusion/configuration_diffusion.py']:
        fingerprints[path] = prep.sha(path)
    refs = reference_rows()
    return {'scope': 'One approved limited-endpoint-padding development test; no automatic adoption.',
        'settings': SETTINGS, 'data': dataset.description(), 'training_schedule': sample_schedule(dataset),
        'comparison': {'scenes': source['comparison']['scenes'], 'draws': source['comparison']['draws'],
            'schedule': [{'scene': s, 'draw': d, 'arm': 'padded12'} for s in source['comparison']['scenes'] for d in source['comparison']['draws']],
            'reference_items': [r['item'] for r in refs], 'reference_source': str(old.OUTPUT)},
        'smoke': {'scene': 0, 'draw': 1234, 'steps': 16, 'order': [list(x) for x in SMOKE]},
        'current_head': source['current_head'], 'original_head_digest': source['original_head_digest'],
        'generation': source['generation'], 'controller_rules': source['controller_rules'],
        'statistics': source['statistics'], 'packages': source['packages'],
        'objective': 'Original mean(two expert + two recovery denoising losses); native padding loss masking remains false.',
        'initialization': 'Original pretrained head; no continued training of current or preservation checkpoints.',
        'selection': 'Final update600 only; all14 attempts collected regardless of outcomes.',
        'screen': source['screen'], 'fingerprints': dict(sorted(fingerprints.items())),
        'limitations': ['Previously inspected development scenes and one training seed.',
            'Window/episode sampling frequencies change when endpoint samples become available.',
            'Repeated endpoint targets are padding conventions, not observed post-terminal actions.',
            'Native fixed8 prefix has eight real future targets; adaptive execution may use later predicted slots.',
            'Original public expert terminal handling is unchanged; no success-based filtering.',
            'All confirmation scenes170-209 excluded from training/selection; timing descriptive across sessions.']}


def verify_protocol(output):
    check_output(output)
    saved = prep.read(output / 'protocol.json')
    assert saved == protocol_document(), 'Frozen endpoint-padding protocol or inputs changed.'
    return saved


def prepare(output=OUTPUT):
    check_output(output)
    if not (output / 'protocol.json').exists():
        assert not output.exists() or not any(output.iterdir())
        prep.save_once(output / 'protocol.json', protocol_document())
    return verify_protocol(output)


def input_bindings(output, protocol, include_feasibility=False):
    bindings = {**protocol['fingerprints'], str(output / 'protocol.json'): prep.sha(output / 'protocol.json')}
    if include_feasibility:
        gate = prep.read(output / 'feasibility.json')
        assert gate['passed'] and gate['student_state_restored'] and gate['bindings'] == bindings
        bindings[str(output / 'feasibility.json')] = prep.sha(output / 'feasibility.json')
    return bindings


def feasibility(output=OUTPUT):
    protocol = verify_protocol(output)
    bindings = input_bindings(output, protocol)
    path = output / 'feasibility.json'
    if path.exists():
        saved = prep.read(path); assert saved['passed'] and saved['bindings'] == bindings
        return saved
    guard = training.gpu_guard()
    policy, _ = adaptive.load_policy(base.MODEL)
    assert not policy.config.do_mask_loss_for_padding
    assert training.tail_digest(training.tail_state(policy)) == protocol['original_head_digest']
    snapshot = training.feasibility.snapshot_training_state(policy)
    before = training.feasibility.state_fingerprint(policy)
    dataset = EndpointWindows()
    indices = [next(i for i, w in enumerate(dataset.windows) if w['group'] == group
                    and w['split'] == 'train' and w.get('padded_action_count') == 7) for group in GROUPS]
    with adaptive.MemorySampler() as memory:
        torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        try:
            training.feasibility.configure_tail_only(policy)
            batches = [dataset.batch(i, torch.device('cuda')) for i in indices]
            assert all(int(b['action_is_pad'].sum()) == 7 for b in batches)
            with torch.no_grad(), training.fixed_noise(SETTINGS['seed'], torch.device('cuda')):
                marked = float(policy(batches[0])[0])
            unmarked_batch = {**batches[0], 'action_is_pad': torch.zeros_like(batches[0]['action_is_pad'])}
            with torch.no_grad(), training.fixed_noise(SETTINGS['seed'], torch.device('cuda')):
                unmarked = float(policy(unmarked_batch)[0])
            assert marked == unmarked
            optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=SETTINGS['lr'], weight_decay=0.)
            with training.fixed_noise(SETTINGS['seed'], torch.device('cuda')):
                update = training.training_update(policy, optimizer, batches)
            assert training.tail_digest(training.tail_state(policy)) != protocol['original_head_digest']
        finally:
            training.feasibility.restore_training_state(policy, snapshot)
        assert training.feasibility.state_fingerprint(policy) == before
        assert {n: p.requires_grad for n, p in policy.named_parameters()} == snapshot['requires_grad']
        assert {n: m.training for n, m in policy.named_modules()} == snapshot['modes']
        assert all(p.grad is None for p in policy.parameters())
        assert torch.equal(torch.get_rng_state(), snapshot['cpu_rng'])
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), snapshot['cuda_rng']))
    result = {'passed': True, 'bindings': bindings, 'student_state_restored': True,
        'padded_windows_checked': indices, 'native_padding_mask_ignored_as_configured': marked == unmarked,
        'matched_masked_unmasked_loss': marked, 'update': update,
        'wall_seconds': time.perf_counter() - started,
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_process_rss_bytes': memory.peak_rss,
        'minimum_available_bytes': memory.minimum_available, 'guard': guard,
        'scope': 'One restored real four-sample supervised update plus two native loss checks; no predictions or pushing.'}
    prep.save_once(path, result)
    return result


def train(output=OUTPUT):
    protocol = verify_protocol(output)
    bindings = input_bindings(output, protocol, True)
    path = output / 'training.json'
    if path.exists():
        saved = prep.read(path)
        assert saved['complete'] and saved['settings'] == SETTINGS and saved['bindings'] == bindings
        assert prep.sha(saved['final_head']['path']) == saved['final_head']['sha256']
        return saved
    guard = training.gpu_guard()
    policy, _ = adaptive.load_policy(base.MODEL)
    assert not policy.config.do_mask_loss_for_padding
    assert training.tail_digest(training.tail_state(policy)) == protocol['original_head_digest']
    with adaptive.MemorySampler() as memory:
        torch.cuda.reset_peak_memory_stats()
        result = training.train(policy, EndpointWindows(), artifacts(output), SETTINGS, bindings)
    assert [{k: r[k] for k in ('step', 'groups', 'windows')} for r in result['history']['updates']] == protocol['training_schedule']
    result['resources'] = {'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_process_rss_bytes': memory.peak_rss, 'minimum_available_bytes': memory.minimum_available, 'guard': guard}
    prep.save_once(path, result)
    return result


def eval_bindings(output):
    protocol = verify_protocol(output)
    learned = prep.read(output / 'training.json')
    assert learned['complete'] and learned['settings'] == SETTINGS
    assert learned['bindings'] == input_bindings(output, protocol, True)
    assert learned['original_tail_digest'] == protocol['original_head_digest']
    assert learned['final_head']['optimizer_step'] == 600
    heads = {'fixed8': {'tensor_digest': protocol['original_head_digest']},
             'current12': protocol['current_head'], 'padded12': learned['final_head']}
    assert len({v['tensor_digest'] for v in heads.values()}) == 3
    for arm in ('current12', 'padded12'):
        assert prep.sha(heads[arm]['path']) == heads[arm]['sha256']
    bindings = {'protocol_sha256': prep.sha(output / 'protocol.json'),
                'training_sha256': prep.sha(output / 'training.json'), 'heads': heads}
    prep.save_once(output / 'eval-bindings.json', bindings)
    return protocol, bindings


def expected_rows(protocol, smoke=False):
    return ([{'scene': 0, 'draw': 1234, 'arm': a, 'label': label} for label, a in SMOKE]
            if smoke else protocol['comparison']['schedule'])


def directory_for(output, item, smoke=False):
    return output / ('smoke' if smoke else 'evaluation') / item.get('label', item['arm'])


def model_binding(output, item, bindings, create=False, smoke=False):
    directory = directory_for(output, item, smoke)
    value = {'arm': item['arm'], 'inner_method': ARMS[item['arm']],
             'head_digest': bindings['heads'][item['arm']]['tensor_digest'],
             'eval_bindings_sha256': prep.sha(output / 'eval-bindings.json')}
    path = directory / 'model-binding.json'
    if path.exists(): assert prep.read(path) == value
    else:
        assert create and (not directory.exists() or not any(directory.iterdir()))
        prep.save_once(path, value)
    return value


def scan(output, protocol, bindings, smoke=False):
    expected, rows, found = expected_rows(protocol, smoke), [], set()
    folder = output / ('smoke' if smoke else 'evaluation')
    if folder.exists():
        assert all(p.is_dir() and p.name in {i.get('label', i['arm']) for i in expected} for p in folder.iterdir())
    for item in expected:
        directory = directory_for(output, item, smoke)
        path = directory / (gen.stem(item['scene'], item['draw'], ARMS[item['arm']]) + '.json')
        if path.exists():
            trial = prep.read(path)
            assert (trial['scene'], trial['draw'], trial['method']) == (item['scene'], item['draw'], ARMS[item['arm']])
            binding = model_binding(output, item, bindings, smoke=smoke)
            gen.verify_trial(trial, directory, [0] if smoke else protocol['comparison']['scenes'],
                             [1234] if smoke else protocol['comparison']['draws'], 16 if smoke else 300)
            rows.append({'item': item, 'directory': str(directory), 'binding': binding, 'trial': trial})
            found.update([path, directory / trial['trace'], directory / 'model-binding.json'])
        elif (directory / 'model-binding.json').exists():
            model_binding(output, item, bindings, smoke=smoke)
            found.add(directory / 'model-binding.json')
    if folder.exists():
        assert all(p in found for d in folder.iterdir() for p in d.iterdir()), 'Orphan/partial artifact; inspect before resuming.'
    assert [r['item'] for r in rows] == expected[:len(rows)], 'Completed trials must be a schedule prefix.'
    index = output / ('smoke.json' if smoke else 'evaluation.json')
    assert not index.with_suffix('.json.tmp').exists()
    if index.exists():
        saved = prep.read(index)
        assert saved['records'] == rows[:len(saved['records'])]
        if saved.get('complete') or saved.get('passed'): assert len(saved['records']) == len(rows) == len(expected)
    return rows


def collect(policy, item, output, bindings, heads, smoke=False):
    directory = directory_for(output, item, smoke)
    model_binding(output, item, bindings, create=True, smoke=smoke)
    path = directory / (gen.stem(item['scene'], item['draw'], ARMS[item['arm']]) + '.json')
    if not path.exists():
        assert not path.with_suffix('.trace.json.gz').exists() and not path.with_suffix('.json.tmp').exists()
        digest = bindings['heads'][item['arm']]['tensor_digest']
        prior.activate(policy, heads[item['arm']], digest)
        assert not any(p.requires_grad for p in policy.parameters())
        gen.evaluate_episode(policy, item['scene'], item['draw'], ARMS[item['arm']], directory, 16 if smoke else 300)
        assert training.tail_digest(training.tail_state(policy)) == digest


def smoke_check(output, protocol, bindings):
    rows = scan(output, protocol, bindings, True)
    assert len(rows) == 4
    first, last = [gen.read_trace(r['trial'], Path(r['directory'])) for r in [rows[0], rows[-1]]]
    historical = prep.read(old.OUTPUT / 'smoke.json')['records'][0]
    original = gen.read_trace(historical['trial'], Path(historical['directory']))
    assert len(first['steps']) == len(last['steps']) == len(original['steps']) == 16
    for a, b, c in zip(first['steps'], last['steps'], original['steps']):
        assert a['action'] == b['action'] == c['action']
        assert adaptive.states_match(a['after'], b['after']) and adaptive.states_match(a['after'], c['after'])
    return rows


def statistics(rows, protocol):
    scenes, draws = protocol['comparison']['scenes'], protocol['comparison']['draws']
    lookup = {(r['item']['scene'], r['item']['draw'], r['item']['arm']): r['trial'] for r in rows}
    assert len(rows) == len(lookup) == 42
    for scene in scenes:
        initial = lookup[scene, draws[0], 'fixed8']['initial']
        assert all(adaptive.states_match(initial, lookup[scene, d, a]['initial']) for d in draws for a in ARMS)
    metrics = ['final_coverage', 'model_queries', 'wall_seconds', 'monitor_seconds']

    def table(draw=None):
        result = {}
        for arm in ARMS:
            trials = [t for (s, d, a), t in lookup.items() if a == arm and (draw is None or d == draw)]
            result[arm] = {'attempts': len(trials), 'successes': sum(t['success'] for t in trials),
                **{f'mean_{m}': float(np.mean([t[m] for t in trials])) for m in metrics},
                'mean_prediction_seconds': float(np.mean([np.mean(t['prediction_seconds']) for t in trials])),
                'maximum_gpu_allocated_bytes': max(t['peak_gpu_allocated_bytes'] for t in trials)}
        return result

    indices = np.random.default_rng(protocol['statistics']['seed']).integers(0, len(scenes),
        size=(protocol['statistics']['bootstrap_samples'], len(scenes)))
    comparisons = []
    for method, reference in [('current12', 'fixed8'), ('padded12', 'fixed8'), ('padded12', 'current12')]:
        differences = {}
        for metric in ['success', *metrics]:
            values = np.asarray([np.mean([float(lookup[s,d,method][metric]) - float(lookup[s,d,reference][metric]) for d in draws]) for s in scenes])
            differences[metric] = {'mean_difference': float(values.mean()),
                'paired_scene_bootstrap_95_interval': np.quantile(values[indices].mean(1), [.025,.975]).tolist()}
        cases = lambda a, b: [{'scene': s, 'draw': d} for s in scenes for d in draws
                             if lookup[s,d,a]['success'] and not lookup[s,d,b]['success']]
        comparisons.append({'method': method, 'reference': reference,
            'gained_successes': cases(method, reference), 'lost_successes': cases(reference, method), 'differences': differences})
    recovery = {a: sum(lookup[s,d,a]['success'] for s in [15,16,18] for d in draws) for a in ARMS}
    tables = table()
    screen = (len(comparisons[1]['lost_successes']) < protocol['screen']['original_baseline_losses_below']
              and tables['padded12']['successes'] >= protocol['screen']['successes_at_least']
              and recovery['padded12'] >= protocol['screen']['recovery_successes_at_least'])
    return {'methods': tables, 'per_draw': {str(d): table(d) for d in draws}, 'comparisons': comparisons,
            'recovery_validation_successes_of_six': recovery, 'development_screen_passed': screen}


def summarize(output, protocol, bindings):
    index = prep.read(output / 'evaluation.json')
    assert index['complete'] and index['backbone_unchanged']
    smoke_check(output, protocol, bindings)
    candidates = scan(output, protocol, bindings)
    assert len(candidates) == 14
    refs = reference_rows(protocol)
    result = {'complete': True, 'scope': 'Development only; 14 new candidate attempts and 28 reused references; no automatic adoption.',
              **statistics(refs + candidates, protocol), 'statistics': protocol['statistics'],
              'new_scored_attempts': 14, 'reused_scored_attempts': 28,
              'protocol_sha256': prep.sha(output / 'protocol.json'),
              'eval_bindings_sha256': prep.sha(output / 'eval-bindings.json')}
    prep.save_once(output / 'summary.json', result)
    return result


def evaluate(output=OUTPUT):
    protocol, bindings = eval_bindings(output)
    if (output / 'summary.json').exists(): return summarize(output, protocol, bindings)
    rows = scan(output, protocol, bindings); scan(output, protocol, bindings, True)
    reference_rows(protocol)
    training.gpu_guard()
    before, started, memory = len(rows), time.perf_counter(), adaptive.MemorySampler()
    resource = output / f'evaluation-resources-from-{before:03d}-{time.time_ns()}.json'
    policy, heads = None, None
    try:
        with memory:
            policy, _ = adaptive.load_policy(base.MODEL)
            heads = {'fixed8': training.tail_state(policy), **{a: training.read_tail(Path(bindings['heads'][a]['path']), bindings['heads'][a]['sha256'])
                                                            for a in ('current12', 'padded12')}}
            assert training.tail_digest(heads['fixed8']) == protocol['original_head_digest']
            backbone = training.backbone_digest(policy)
            smoke_rows = scan(output, protocol, bindings, True)
            for item in expected_rows(protocol, True)[len(smoke_rows):]:
                collect(policy, item, output, bindings, heads, True)
                smoke_rows = scan(output, protocol, bindings, True)
                old.save_progress(output / 'smoke.json', {'passed': False, 'records': smoke_rows})
            smoke_check(output, protocol, bindings)
            old.save_progress(output / 'smoke.json', {'passed': True, 'records': smoke_rows})
            for item in expected_rows(protocol)[len(rows):]:
                collect(policy, item, output, bindings, heads)
                rows = scan(output, protocol, bindings)
                old.save_progress(output / 'evaluation.json', {'complete': False, 'records': rows})
                print(f'Endpoint-padding comparison {len(rows)}/14: {item}', flush=True)
            assert training.backbone_digest(policy) == backbone
            old.save_progress(output / 'evaluation.json', {'complete': True, 'backbone_unchanged': True, 'records': rows})
            assert prep.sha(base.MODEL / 'model.safetensors') == base.WEIGHTS_SHA256
            result = summarize(output, protocol, bindings)
        return result
    finally:
        if policy is not None and heads is not None: prior.activate(policy, heads['fixed8'], protocol['original_head_digest'])
        prep.save_once(resource, {'before': before, 'after': len(rows), 'wall_seconds': time.perf_counter() - started,
            'peak_process_rss_bytes': memory.peak_rss, 'minimum_available_bytes': memory.minimum_available,
            'scope': 'After initial CPU input verification; includes model/head loading, smoke, collection, final checks and restoration. Per-attempt times stay separate.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['prepare','verify','feasibility','train','evaluate'])
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    function = {'prepare': prepare, 'verify': verify_protocol, 'feasibility': feasibility, 'train': train, 'evaluate': evaluate}[args.stage]
    result = function(args.output_dir)
    print({'stage': args.stage, 'passed': result.get('passed', True), 'complete': result.get('complete', False),
           'output': str(args.output_dir)}, flush=True)
