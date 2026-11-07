"""Isolated baseline-success rehearsal pilot; explicit stages, no automatic run.

The existing training update, final block, physics, policy inference and 8/12
controller are reused. Preparation and CPU tests do not load a robot policy.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch

import robotics_retention_preflight as prep
import robotics_recovery_confirmation as prior

training, gen, adaptive, base = prior.training, prior.gen, prior.adaptive, prior.base
OUTPUT = Path('results/robotics-retention-pilot')
CACHE = Path('data/prepared/robotics/retention-pilot/training')
BASE_GROUPS = ['expert', 'recovery', 'expert', 'recovery']
GROUPS = ['expert', 'recovery', 'baseline', 'recovery']
VALIDATION_GROUPS = ['expert', 'recovery', 'baseline']
ARMS = {'fixed8': 'fixed8', 'current12': 'adaptive12', 'retention12': 'adaptive12'}
SMOKE = [('before', 'fixed8'), ('current', 'current12'), ('candidate', 'retention12'), ('restored', 'fixed8')]
SETTINGS = {**training.SETTINGS, 'baseline_seed': 51002, 'microbatch_groups': GROUPS}


class RetentionWindows(training.PreparedWindows):
    """Reuse the original sample constructor with a separate baseline supplement."""

    def __init__(self, source=prep.SOURCE / 'dataset.json', supplement=prep.OUTPUT / 'baseline-supplement.json'):
        super().__init__(source)
        extra = prep.read(supplement)
        assert extra['source_dataset'] == {'path': str(source), 'sha256': prep.sha(source)}
        self.source_window_count = len(self.windows)
        self.windows = list(self.windows)
        for episode in extra['episodes']:
            original = self.episodes[episode['episode_id']]
            assert original['group'] == 'ordinary_control' and original['split'] == 'diagnostic'
            assert episode['original_group'] == original['group'] and episode['original_split'] == original['split']
            assert episode['file'] == original['file'] and episode['sha256'] == original['sha256']
            assert prep.sha(episode['file']) == episode['sha256']
            assert episode['group'] == 'baseline' and episode['split'] in ('train', 'validation')
            self.episodes[episode['episode_id']] = copy.deepcopy(episode)
        for window in extra['windows']:
            episode = self.episodes[window['episode_id']]
            assert window['group'] == episode['group'] == 'baseline'
            assert window['split'] == episode['split']
            i = window['first_observation_index']
            assert 0 <= i and window['current_observation_index'] == i + 1 < episode['n_frames']
            assert window['action_start_index'] == i and window['action_stop_index'] == i + 16 <= episode['n_actions']
            index = len(self.windows)
            self.windows.append(dict(window))
            self.groups.setdefault((window['split'], 'baseline'), {}).setdefault(
                episode['scene'], {}).setdefault(episode['episode_id'], []).append(index)
        for group in VALIDATION_GROUPS:
            assert self.groups['train', group] and self.groups['validation', group]
            assert not set(self.groups['train', group]) & set(self.groups['validation', group])
        self.bindings[str(supplement)] = prep.sha(supplement)


def sample_schedule(dataset, settings=SETTINGS, reference=None):
    """Keep the original two recovery samples and first expert sample unchanged."""
    old_rng = np.random.default_rng(settings['seed'])
    baseline_rng = np.random.default_rng(settings['baseline_seed'])
    rows = []
    for step in range(1, settings['optimizer_steps'] + 1):
        original = [dataset.choose('train', group, old_rng) for group in BASE_GROUPS]
        if reference is not None:
            assert reference[step - 1]['step'] == step and reference[step - 1]['windows'] == original, \
                'Unchanged data draws do not reproduce the saved original training schedule.'
        actual = list(original)
        actual[2] = dataset.choose('train', 'baseline', baseline_rng)
        rows.append({'step': step, 'groups': GROUPS, 'windows': actual, 'original_reference_windows': original})
    return rows


def validation_plan(dataset, settings):
    rng = np.random.default_rng(settings['validation_seed'])
    return [{'group': group, 'window': dataset.choose('validation', group, rng),
             'noise_seed': settings['validation_seed'] + 1000 + index + gi * 1000}
            for gi, group in enumerate(VALIDATION_GROUPS)
            for index in range(settings['validation_windows_per_group'])]


def validate(policy, dataset, plan):
    device = next(policy.parameters()).device
    losses = {g: [] for g in VALIDATION_GROUPS}
    assert all(not m.training for m in policy.modules())
    with torch.no_grad():
        for row in plan:
            with training.fixed_noise(row['noise_seed'], device):
                loss, _ = policy(dataset.batch(row['window'], device))
            assert loss.ndim == 0 and bool(torch.isfinite(loss))
            losses[row['group']].append(float(loss))
    assert all(losses.values())
    return {'groups': {g:float(np.mean(v)) for g,v in losses.items()},
            'windows_per_group': {g:len(v) for g,v in losses.items()}}


def check_output(output):
    assert output.resolve() in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + '-repeat').resolve()}, \
        'Use the isolated retention-pilot output; preserve earlier experiment evidence.'


def artifacts(output):
    check_output(output)
    return CACHE if output.resolve() == OUTPUT.resolve() else CACHE.parent.with_name(CACHE.parent.name + '-repeat') / 'training'


def protocol_document():
    prep.run()
    prior.input_document()  # Verify the original completed training/checkpoint provenance.
    source_training = prep.read(prep.SOURCE / 'training.json')
    dataset = RetentionWindows()
    fingerprint = {**source_training['bindings'], **dataset.bindings}
    fingerprint[str(prep.SOURCE / 'training.json')] = prep.sha(prep.SOURCE / 'training.json')
    fingerprint[source_training['final_head']['path']] = source_training['final_head']['sha256']
    for name in ['experiments/robotics_retention_pilot.py', 'tests/test_robotics_retention_pilot.py',
                 prep.OUTPUT / 'plan.json', prep.OUTPUT / 'data-audit.json']:
        fingerprint[str(name)] = prep.sha(name)
    training.verify_bindings(fingerprint)
    return {'scope': 'Bounded development pilot; no generalization or automatic promotion claim',
            'settings': SETTINGS, 'training_schedule': sample_schedule(dataset, SETTINGS, source_training['history']['updates']),
            'comparison': prep.read(prep.OUTPUT / 'plan.json')['comparison'],
            'current_head': source_training['final_head'], 'original_head_digest': source_training['original_tail_digest'],
            'smoke': {'scene': 0, 'draw': 1234, 'steps': 16, 'order': [list(x) for x in SMOKE]},
            'generation': {'horizon': 16, 'observations': 2, 'denoising_steps': 100},
            'controller_rules': adaptive.RULES, 'statistics': {'scene_clusters': 7, 'draws': 2, 'bootstrap_samples': 10000, 'seed': 20261107},
            'packages': {n:importlib.metadata.version(n) for n in ['torch', 'lerobot', 'gym-pusht', 'numpy', 'diffusers']},
            'fingerprints': dict(sorted(fingerprint.items())),
            'selection': 'Final step600 only; complete all42 scheduled attempts regardless of interim outcomes.',
            'limitations': ['Inspected discovery scenes; development evidence only.', 'One training seed and one prescribed data-mix change.',
                'No scenes170-209 or their failed trajectories used for training or selection.',
                'Fresh unseen evaluation requires a separate agreed protocol.', 'No automatic replacement of the milestone or current checkpoint.']}


def verify_protocol(output):
    check_output(output)
    saved = prep.read(output / 'protocol.json')
    assert saved == protocol_document(), 'Prepared retention protocol or its inputs changed.'
    return saved


def prepare(output=OUTPUT):
    check_output(output)
    path = output / 'protocol.json'
    if not path.exists():
        assert not output.exists() or not any(output.iterdir()), 'Do not retroactively freeze model results.'
        prep.save_once(path, protocol_document())
    return verify_protocol(output)


def fit(policy, dataset, settings, schedule, directory, bindings):
    """Only the existing output block changes; all examples are fixed before fitting."""
    training.verify_bindings(bindings)
    selected = training.feasibility.configure_tail_only(policy)
    backbone = training.backbone_digest(policy)
    original_tail = training.tail_digest(training.tail_state(policy))
    directory.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=settings['lr'], weight_decay=0.)
    plan = validation_plan(dataset, settings)
    device = next(policy.parameters()).device
    torch.manual_seed(settings['seed'])
    if device.type == 'cuda': torch.cuda.manual_seed(settings['seed'])
    history = {'updates': [], 'validation': []}
    start, resume = 0, directory / 'resume.pt'

    def checkpoint(step):
        assert all(p.grad is None for p in policy.parameters())
        training.atomic_torch_save(resume, {'format': 1, 'step': step, 'settings': settings, 'bindings': bindings,
            'schedule': schedule, 'plan': plan, 'tail': training.tail_state(policy), 'optimizer': optimizer.state_dict(),
            'history': history, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None})

    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        assert saved['format'] == 1 and saved['settings'] == settings and saved['bindings'] == bindings
        assert saved['schedule'] == schedule and saved['plan'] == plan
        training.load_tail(policy, saved['tail'])
        optimizer.load_state_dict(saved['optimizer'])
        assert all(g['lr'] == settings['lr'] and g['weight_decay'] == 0 for g in optimizer.param_groups)
        torch.set_rng_state(saved['torch_rng'])
        if saved['cuda_rng'] is not None:
            assert device.type == 'cuda'
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start, history = saved['step'], saved['history']
        assert 0 <= start <= settings['optimizer_steps'] and len(history['updates']) == start
        for recorded, expected in zip(history['updates'], schedule[:start]):
            assert all(recorded[k] == expected[k] for k in ['step', 'groups', 'windows', 'original_reference_windows'])
    else:
        assert not any(directory.iterdir()), 'Unbound or partial training output; do not overwrite.'
        history['validation'].append({'step': 0, **validate(policy, dataset, plan)})
        checkpoint(0)
    for sample in schedule[start:]:
        assert sample['groups'] == GROUPS and len(sample['windows']) == 4
        batch = [dataset.batch(index, device) for index in sample['windows']]
        update = training.training_update(policy, optimizer, batch)
        history['updates'].append({**sample, **update})
        step = sample['step']
        if step in settings['validation_steps']:
            assert training.backbone_digest(policy) == backbone
            history['validation'].append({'step': step, **validate(policy, dataset, plan)})
            training.verify_bindings(bindings)
            checkpoint(step)
        if step % 20 == 0: print(f'Retention training {step}/{settings["optimizer_steps"]}', flush=True)
    assert len(history['updates']) == settings['optimizer_steps']
    assert sum(r['denoiser_forward_calls'] for r in history['updates']) == settings['optimizer_steps'] * 4
    assert [r['step'] for r in history['validation']] == settings['validation_steps']
    assert training.backbone_digest(policy) == backbone
    state = training.tail_state(policy)
    assert training.tail_digest(state) != original_tail
    final = directory / 'final-tail.pt'
    if final.exists(): assert training.tail_digest(training.read_tail(final, prep.sha(final))) == training.tail_digest(state)
    else: training.atomic_torch_save(final, state)
    return {'complete': True, 'settings': settings, 'bindings': bindings, 'trainable_names': selected,
            'frozen_backbone_and_normalization_unchanged': True, 'original_tail_digest': original_tail,
            'backbone_digest': backbone, 'history': history, 'validation_plan': plan,
            'final_head': {'path': str(final), 'sha256': prep.sha(final), 'tensor_digest': training.tail_digest(state),
                           'optimizer_step': settings['optimizer_steps']}}


def train(output=OUTPUT):
    protocol = verify_protocol(output)
    path = output / 'training.json'
    bindings = {**protocol['fingerprints'], str(output / 'protocol.json'): prep.sha(output / 'protocol.json')}
    if path.exists():
        saved = prep.read(path)
        assert saved['complete'] and saved['bindings'] == bindings and saved['settings'] == SETTINGS
        assert prep.sha(saved['final_head']['path']) == saved['final_head']['sha256']
        return saved
    dataset = RetentionWindows()
    guard = training.gpu_guard()
    policy, _ = adaptive.load_policy(base.MODEL)
    assert training.tail_digest(training.tail_state(policy)) == protocol['original_head_digest']
    with adaptive.MemorySampler() as memory:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = fit(policy, dataset, SETTINGS, protocol['training_schedule'], artifacts(output), bindings)
        result['wall_seconds'] = time.perf_counter() - started
    result['resources'] = {'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_process_rss_bytes': memory.peak_rss, 'minimum_available_bytes': memory.minimum_available, 'guard': guard}
    prep.save_once(path, result)
    return result


def eval_bindings(output):
    protocol = verify_protocol(output)
    learned = prep.read(output / 'training.json')
    assert learned['complete'] and learned['settings'] == SETTINGS and learned['frozen_backbone_and_normalization_unchanged']
    expected_train_bindings = {**protocol['fingerprints'], str(output / 'protocol.json'): prep.sha(output / 'protocol.json')}
    assert learned['bindings'] == expected_train_bindings
    assert learned['original_tail_digest'] == protocol['original_head_digest']
    assert learned['final_head']['optimizer_step'] == 600
    heads = {'fixed8': {'tensor_digest': protocol['original_head_digest']},
             'current12': protocol['current_head'], 'retention12': learned['final_head']}
    assert len({h['tensor_digest'] for h in heads.values()}) == 3
    for arm in ['current12', 'retention12']:
        assert prep.sha(heads[arm]['path']) == heads[arm]['sha256']
    bindings = {'protocol_sha256': prep.sha(output / 'protocol.json'),
                'training_sha256': prep.sha(output / 'training.json'), 'heads': heads}
    prep.save_once(output / 'eval-bindings.json', bindings)
    return protocol, bindings


def expected_rows(protocol, smoke=False):
    return ([{'scene': 0, 'draw': 1234, 'arm': a, 'label': label} for label,a in SMOKE]
            if smoke else protocol['comparison']['schedule'])


def save_progress(path, value):
    """Atomic LF JSON, preserving byte fingerprints across Windows checkouts."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    assert not temporary.exists(), 'Inspect partial progress output before resuming.'
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
    temporary.replace(path)


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
    expected = expected_rows(protocol, smoke)
    rows, found = [], set()
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
        assert all(p in found for d in folder.iterdir() for p in d.iterdir()), 'Orphan/partial artifact; inspect rather than overwrite.'
    assert [r['item'] for r in rows] == expected[:len(rows)], 'Completed trials must be a schedule prefix.'
    index = output / ('smoke.json' if smoke else 'evaluation.json')
    assert not index.with_suffix('.json.tmp').exists()
    if index.exists():
        saved = prep.read(index)
        assert saved['records'] == rows[:len(saved['records'])]
        if saved.get('complete') or saved.get('passed'): assert len(saved['records']) == len(rows) == len(expected)
    return rows


def collect(policy, item, output, protocol, bindings, heads, smoke=False):
    directory = directory_for(output, item, smoke)
    model_binding(output, item, bindings, create=True, smoke=smoke)
    path = directory / (gen.stem(item['scene'], item['draw'], ARMS[item['arm']]) + '.json')
    if not path.exists():
        assert not path.with_suffix('.trace.json.gz').exists() and not path.with_suffix('.json.tmp').exists()
        digest = bindings['heads'][item['arm']]['tensor_digest']
        prior.activate(policy, heads[item['arm']], digest)
        gen.evaluate_episode(policy, item['scene'], item['draw'], ARMS[item['arm']], directory, 16 if smoke else 300)
        assert training.tail_digest(training.tail_state(policy)) == digest


def smoke_check(output, protocol, bindings):
    rows = scan(output, protocol, bindings, True)
    assert len(rows) == len(SMOKE)
    first, last = [gen.read_trace(r['trial'], Path(r['directory'])) for r in [rows[0], rows[-1]]]
    historical = prep.read(prior.OUTPUT / 'smoke.json')['records'][0]
    old = gen.read_trace(historical['trial'], Path(historical['directory']))
    assert len(first['steps']) == len(last['steps']) == 16
    for left,right,reference in zip(first['steps'],last['steps'],old['steps']):
        assert left['action'] == right['action'] == reference['action']
        assert adaptive.states_match(left['after'],right['after']) and adaptive.states_match(left['after'],reference['after'])
    return rows


def summarize(output, protocol, bindings):
    index = prep.read(output / 'evaluation.json')
    assert index['complete'] and index['backbone_unchanged']
    smoke_check(output, protocol, bindings)
    rows = scan(output, protocol, bindings)
    assert len(rows) == 42
    lookup = {(r['item']['scene'],r['item']['draw'],r['item']['arm']):r['trial'] for r in rows}
    scenes, draws = protocol['comparison']['scenes'], protocol['comparison']['draws']
    for scene in scenes:
        reference = lookup[scene,draws[0],'fixed8']['initial']
        assert all(adaptive.states_match(lookup[scene,d,a]['initial'],reference) for d in draws for a in ARMS)
    metrics = ['final_coverage','model_queries','wall_seconds','monitor_seconds']
    def table(draw=None):
        result = {}
        for arm in ARMS:
            trials = [r['trial'] for r in rows if r['item']['arm'] == arm and (draw is None or r['item']['draw'] == draw)]
            result[arm] = {'attempts':len(trials),'successes':sum(t['success'] for t in trials),
                **{f'mean_{m}':float(np.mean([t[m] for t in trials])) for m in metrics},
                'mean_prediction_seconds':float(np.mean([np.mean(t['prediction_seconds']) for t in trials])),
                'maximum_gpu_allocated_bytes':max(t['peak_gpu_allocated_bytes'] for t in trials)}
        return result
    tables = table()
    comparisons = []
    indices = np.random.default_rng(20261107).integers(0,len(scenes),size=(10000,len(scenes)))
    for method,reference in [('current12','fixed8'),('retention12','fixed8'),('retention12','current12')]:
        differences = {}
        for m in ['success',*metrics]:
            values = np.asarray([np.mean([float(lookup[s,d,method][m])-float(lookup[s,d,reference][m]) for d in draws]) for s in scenes])
            differences[m] = {'mean_difference':float(values.mean()),'paired_scene_bootstrap_95_interval':np.quantile(values[indices].mean(1),[.025,.975]).tolist()}
        cases = lambda a,b:[{'scene':s,'draw':d} for s in scenes for d in draws if lookup[s,d,a]['success'] and not lookup[s,d,b]['success']]
        comparisons.append({'method':method,'reference':reference,'gained_successes':cases(method,reference),
                            'lost_successes':cases(reference,method),'differences':differences})
    recovery_scenes = prep.read(prep.SOURCE / 'dataset.json')['selection']['recovery_validation_scenes']
    recovery = {a:sum(lookup[s,d,a]['success'] for s in recovery_scenes for d in draws) for a in ARMS}
    result = {'complete':True,'scope':'Development only; no automatic adoption or generalization claim.',
              'methods':tables,'per_draw':{str(d):table(d) for d in draws},
              'comparisons':comparisons,'recovery_validation_successes_of_six':recovery,
              'statistics':protocol['statistics'],'protocol_sha256':prep.sha(output/'protocol.json'),
              'eval_bindings_sha256':prep.sha(output/'eval-bindings.json')}
    prep.save_once(output / 'summary.json', result)
    return result


def evaluate(output=OUTPUT):
    protocol, bindings = eval_bindings(output)
    if (output / 'summary.json').exists(): return summarize(output, protocol, bindings)
    rows = scan(output, protocol, bindings)
    scan(output, protocol, bindings, True)
    training.gpu_guard()
    policy, _ = adaptive.load_policy(base.MODEL)
    heads = {'fixed8':training.tail_state(policy), **{a:training.read_tail(Path(bindings['heads'][a]['path']),bindings['heads'][a]['sha256'])
                                                   for a in ['current12','retention12']}}
    assert training.tail_digest(heads['fixed8']) == protocol['original_head_digest']
    backbone = training.backbone_digest(policy)
    before, started, memory = len(rows), time.perf_counter(), adaptive.MemorySampler()
    resource = output / f'evaluation-resources-from-{before:03d}-{time.time_ns()}.json'
    try:
        with memory:
            smoke_rows = scan(output, protocol, bindings, True)
            for item in expected_rows(protocol, True)[len(smoke_rows):]:
                collect(policy,item,output,protocol,bindings,heads,True)
                smoke_rows = scan(output,protocol,bindings,True)
                save_progress(output/'smoke.json',{'passed':False,'records':smoke_rows})
            smoke_check(output,protocol,bindings)
            save_progress(output/'smoke.json',{'passed':True,'records':smoke_rows})
            for item in expected_rows(protocol)[len(rows):]:
                collect(policy,item,output,protocol,bindings,heads)
                rows = scan(output,protocol,bindings)
                save_progress(output/'evaluation.json',{'complete':False,'records':rows})
                print(f'Retention comparison {len(rows)}/42: {item}',flush=True)
            assert training.backbone_digest(policy) == backbone
            save_progress(output/'evaluation.json',{'complete':True,'backbone_unchanged':True,'records':rows})
        assert prep.sha(base.MODEL/'model.safetensors') == base.WEIGHTS_SHA256
        return summarize(output,protocol,bindings)
    finally:
        prior.activate(policy,heads['fixed8'],protocol['original_head_digest'])
        prep.save_once(resource,{'before':before,'after':len(rows),'wall_seconds':time.perf_counter()-started,
            'peak_process_rss_bytes':memory.peak_rss,'minimum_available_bytes':memory.minimum_available,
            'scope':'This invocation, including load/smoke/verification; per-attempt times stay separate.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['prepare','verify','train','evaluate'])
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    function = {'prepare':prepare,'verify':verify_protocol,'train':train,'evaluate':evaluate}[args.stage]
    result = function(args.output_dir)
    print({'stage':args.stage,'complete':result.get('complete',False),'output':str(args.output_dir)},flush=True)
