"""Approved output-preservation pilot; frozen old code and evidence are reused.

Two expert and two recovery losses retain the original sample/noise stream.
A fifth, separately seeded forward anchors the student's denoising output to
the original final block. No teacher is deployed and no full model is copied.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import robotics_retention_pilot as old

prep, prior, training, gen, adaptive, base = old.prep, old.prior, old.training, old.gen, old.adaptive, old.base
OUTPUT = Path('results/robotics-preservation-pilot')
CACHE = Path('data/prepared/robotics/preservation-pilot/training')
GROUPS = ['expert', 'recovery', 'expert', 'recovery']
ARMS = {'fixed8': 'fixed8', 'current12': 'adaptive12', 'preserved12': 'adaptive12'}
SMOKE = [('before', 'fixed8'), ('current', 'current12'), ('candidate', 'preserved12'), ('restored', 'fixed8')]
SETTINGS = {**training.SETTINGS, 'baseline_seed': 51002, 'anchor_noise_seed': 51003,
            'preservation_weight': 1., 'microbatch_groups': GROUPS,
            'auxiliary_group': 'baseline', 'auxiliary_supervised_loss': False}


def teacher_digest(teacher):
    return training.tail_digest({training.TAIL + k: v for k, v in teacher.state_dict().items()})


def make_teacher(policy):
    teacher = copy.deepcopy(policy.diffusion.unet.final_conv).eval().requires_grad_(False)
    assert teacher_digest(teacher) == training.tail_digest(training.tail_state(policy))
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    return teacher


def preservation_loss(policy, teacher, batch, seed):
    """Compare both heads on exactly the same frozen features/noised input.

    The regular supervised loss returned by this auxiliary forward is ignored.
    Its random draws are restored, so it cannot shift later supervised noise.
    """
    assert all(not m.training for m in policy.modules()) and not teacher.training
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert batch['action'].shape == (1, 16, 2) and all(not v.requires_grad for v in batch.values())
    captured = []

    def capture(_module, inputs, output):
        assert len(inputs) == 1 and not inputs[0].requires_grad
        assert output.shape[0] == 1 and output.numel() == 32
        captured.append((inputs[0].detach(), output))

    hook = policy.diffusion.unet.final_conv.register_forward_hook(capture)
    device = next(policy.parameters()).device
    try:
        with training.fixed_noise(seed, device):
            ignored, _ = policy(batch)
            del ignored
            assert len(captured) == 1
            features, student = captured[0]
            with torch.no_grad():
                target = teacher(features)
            assert target.shape == student.shape and torch.isfinite(target).all()
            loss = F.mse_loss(student, target)
            assert loss.ndim == 0 and torch.isfinite(loss)
            return loss
    finally:
        hook.remove()


def training_update(policy, teacher, optimizer, batches, anchor_batch, noise_seed, weight):
    assert len(batches) == 4 and weight >= 0 and math.isfinite(weight)
    assert all(not m.training for m in policy.modules())
    selected = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    assert selected and all(n.startswith(training.TAIL) for n, _ in selected)
    parameters = [p for _, p in selected]
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in parameters}
    versions = tuple(p._version for p in teacher.parameters())
    calls, teacher_calls, losses = [], [], []
    hook = policy.diffusion.unet.register_forward_hook(lambda *_: calls.append(1))
    teacher_hook = teacher.register_forward_hook(lambda *_: teacher_calls.append(1))
    started = time.perf_counter()
    try:
        optimizer.zero_grad(set_to_none=True)
        for batch in batches:
            assert batch['action'].shape == (1, 16, 2)
            assert all(not v.requires_grad for v in batch.values())
            loss, _ = policy(batch)
            assert loss.ndim == 0 and torch.isfinite(loss)
            losses.append(float(loss.detach()))
            (loss / 4).backward()
        anchor = preservation_loss(policy, teacher, anchor_batch, noise_seed)
        anchor_value = float(anchor.detach())
        (weight * anchor).backward()
        assert len(calls) == 5 and len(teacher_calls) == 1
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        assert all(p.grad is None for _, p in policy.named_parameters() if not p.requires_grad)
        assert tuple(p._version for p in teacher.parameters()) == versions
        assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
        norm = float(torch.sqrt(sum(p.grad.detach().float().square().sum() for p in parameters)))
        assert math.isfinite(norm) and norm > 0
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in parameters)
        supervised = float(np.mean(losses))
        return {'loss': supervised + weight * anchor_value, 'supervised_loss': supervised,
                'preservation_loss': anchor_value, 'microbatch_losses': losses,
                'gradient_l2_norm': norm, 'denoiser_forward_calls': len(calls),
                'teacher_head_calls': len(teacher_calls), 'teacher_versions_unchanged': True,
                'adam_state_count': len(optimizer.state), 'seconds': time.perf_counter() - started}
    finally:
        hook.remove(); teacher_hook.remove()
        optimizer.zero_grad(set_to_none=True)


def sample_schedule(dataset, settings=SETTINGS, reference=None):
    rng, anchor_rng = np.random.default_rng(settings['seed']), np.random.default_rng(settings['baseline_seed'])
    rows = []
    for step in range(1, settings['optimizer_steps'] + 1):
        windows = [dataset.choose('train', group, rng) for group in GROUPS]
        if reference is not None:
            assert reference[step - 1]['step'] == step and reference[step - 1]['windows'] == windows
        rows.append({'step': step, 'groups': GROUPS, 'windows': windows,
                     'anchor_window': dataset.choose('train', 'baseline', anchor_rng),
                     'anchor_noise_seed': settings['anchor_noise_seed'] + step})
    return rows


def validate(policy, teacher, dataset, plan):
    result = old.validate(policy, dataset, plan)
    device = next(policy.parameters()).device
    with torch.no_grad():
        values = [float(preservation_loss(policy, teacher, dataset.batch(r['window'], device), r['noise_seed']))
                  for r in plan if r['group'] == 'baseline']
    assert values
    return {**result, 'preservation_mean_loss': float(np.mean(values)), 'preservation_windows': len(values),
            'teacher_digest': teacher_digest(teacher)}


def check_output(output):
    assert output.resolve() in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + '-repeat').resolve()}, \
        'Use isolated preservation outputs; preserve earlier sources and evidence.'


def artifacts(output):
    check_output(output)
    return CACHE if output.resolve() == OUTPUT.resolve() else CACHE.parent.with_name(CACHE.parent.name + '-repeat') / 'training'


def reference_rows(protocol=None):
    """Read the 28 scored references; never collect or rewrite old attempts."""
    index = prep.read(old.OUTPUT / 'evaluation.json')
    assert index['complete'] and index['backbone_unchanged'] and len(index['records']) == 42
    source_bindings = prep.read(old.OUTPUT / 'eval-bindings.json')
    rows = [r for r in index['records'] if r['item']['arm'] in ('fixed8', 'current12')]
    assert len(rows) == 28
    for row in rows:
        item, trial, directory = row['item'], row['trial'], Path(row['directory'])
        assert item['arm'] in ARMS and trial['method'] == ARMS[item['arm']]
        assert prep.read(directory / (gen.stem(item['scene'], item['draw'], trial['method']) + '.json')) == trial
        assert prep.read(directory / 'model-binding.json') == row['binding']
        assert row['binding']['head_digest'] == source_bindings['heads'][item['arm']]['tensor_digest']
        assert row['binding']['eval_bindings_sha256'] == prep.sha(old.OUTPUT / 'eval-bindings.json')
        gen.verify_trial(trial, directory, [1, 5, 10, 15, 16, 18, 23], [91000, 92000], 300)
    if protocol is not None:
        assert [r['item'] for r in rows] == protocol['comparison']['reference_items']
    return rows


def protocol_document():
    source = old.verify_protocol(old.OUTPUT)
    import robotics_retention_pilot_audit as old_audit
    assert old_audit.complete_audit()['passed']
    dataset = old.RetentionWindows()
    source_training = prep.read(prep.SOURCE / 'training.json')
    refs = reference_rows()
    fingerprints = dict(source['fingerprints'])
    for path in [old.OUTPUT / n for n in ['protocol.json', 'eval-bindings.json', 'evaluation.json', 'smoke.json',
                                         'summary.json', 'artifact-audit.json']]:
        fingerprints[str(path)] = prep.sha(path)
    for row in refs + prep.read(old.OUTPUT / 'smoke.json')['records']:
        trial, directory, item = row['trial'], Path(row['directory']), row['item']
        for path in [directory / (gen.stem(item['scene'], item['draw'], trial['method']) + '.json'),
                     directory / trial['trace'], directory / 'model-binding.json']:
            fingerprints[str(path)] = prep.sha(path)
    for name in ['experiments/robotics_preservation_pilot.py', 'experiments/robotics_preservation_pilot_audit.py',
                 'tests/test_robotics_preservation_pilot.py', 'experiments/robotics_retention_pilot_audit.py',
                 '.venv-robotics/Lib/site-packages/lerobot/policies/diffusion/modeling_diffusion.py']:
        fingerprints[name] = prep.sha(name)
    comparison = source['comparison']
    return {'scope': 'Approved bounded development test; no automatic adoption or generalization claim.',
            'settings': SETTINGS, 'training_schedule': sample_schedule(dataset, SETTINGS, source_training['history']['updates']),
            'comparison': {'scenes': comparison['scenes'], 'draws': comparison['draws'],
                'schedule': [{'scene': s, 'draw': d, 'arm': 'preserved12'} for s in comparison['scenes'] for d in comparison['draws']],
                'reference_items': [r['item'] for r in refs], 'reference_source': str(old.OUTPUT)},
            'smoke': {'scene': 0, 'draw': 1234, 'steps': 16, 'order': [list(x) for x in SMOKE]},
            'current_head': source['current_head'], 'original_head_digest': source['original_head_digest'],
            'generation': source['generation'], 'controller_rules': source['controller_rules'],
            'statistics': source['statistics'], 'packages': source['packages'],
            'objective': 'mean(two expert + two recovery denoising losses) + 1.0 * original-teacher output MSE on one baseline window',
            'teacher': 'Frozen original final block on shared frozen features; no second full model or sampling chain.',
            'selection': 'Final step600 only; collect all14 candidate attempts regardless of outcomes.',
            'screen': {'original_baseline_losses_below': 2, 'successes_at_least': 11, 'recovery_successes_at_least': 3,
                       'automatic_adoption': False},
            'limitations': ['Seven previously inspected development scenes; two draws per scene.',
                'One fixed penalty weight/training seed; not exact KL or a forgetting diagnosis.',
                'All scenes170-209 excluded from training/selection.', 'Reference timings are from an earlier session.'],
            'fingerprints': dict(sorted(fingerprints.items()))}


def verify_protocol(output):
    check_output(output)
    saved = prep.read(output / 'protocol.json')
    assert saved == protocol_document(), 'Frozen preservation protocol or inputs changed.'
    return saved


def prepare(output=OUTPUT):
    check_output(output)
    if not (output / 'protocol.json').exists():
        assert not output.exists() or not any(output.iterdir()), 'Do not retroactively freeze results.'
        prep.save_once(output / 'protocol.json', protocol_document())
    return verify_protocol(output)


def input_bindings(output, protocol, include_feasibility=False):
    result = {**protocol['fingerprints'], str(output / 'protocol.json'): prep.sha(output / 'protocol.json')}
    if include_feasibility:
        gate = prep.read(output / 'feasibility.json')
        assert gate['passed'] and gate['bindings'] == result and gate['student_state_restored']
        result[str(output / 'feasibility.json')] = prep.sha(output / 'feasibility.json')
    return result


def fit(policy, dataset, settings, schedule, directory, bindings):
    training.verify_bindings(bindings)
    selected = training.feasibility.configure_tail_only(policy)
    backbone, original = training.backbone_digest(policy), training.tail_digest(training.tail_state(policy))
    teacher = make_teacher(policy)
    assert teacher_digest(teacher) == original
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=settings['lr'], weight_decay=0.)
    plan = old.validation_plan(dataset, settings)
    device = next(policy.parameters()).device
    torch.manual_seed(settings['seed'])
    if device.type == 'cuda': torch.cuda.manual_seed(settings['seed'])
    history, start = {'updates': [], 'validation': []}, 0
    directory.mkdir(parents=True, exist_ok=True)
    resume = directory / 'resume.pt'

    def checkpoint(step):
        assert all(p.grad is None for p in policy.parameters())
        assert teacher_digest(teacher) == original
        training.atomic_torch_save(resume, {'format': 1, 'step': step, 'settings': settings, 'bindings': bindings,
            'schedule': schedule, 'plan': plan, 'teacher_digest': original, 'tail': training.tail_state(policy),
            'optimizer': optimizer.state_dict(), 'history': history, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None})

    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        assert saved['format'] == 1 and saved['settings'] == settings and saved['bindings'] == bindings
        assert saved['schedule'] == schedule and saved['plan'] == plan and saved['teacher_digest'] == original
        training.load_tail(policy, saved['tail']); optimizer.load_state_dict(saved['optimizer'])
        assert all(g['lr'] == settings['lr'] and g['weight_decay'] == 0 for g in optimizer.param_groups)
        torch.set_rng_state(saved['torch_rng'])
        if saved['cuda_rng'] is not None:
            assert device.type == 'cuda'; torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start, history = saved['step'], saved['history']
        assert 0 <= start <= settings['optimizer_steps'] and len(history['updates']) == start
        for actual, frozen in zip(history['updates'], schedule[:start]):
            assert all(actual[k] == frozen[k] for k in ['step', 'groups', 'windows', 'anchor_window', 'anchor_noise_seed'])
    else:
        assert not any(directory.iterdir()), 'Unbound or partial training artifact; inspect before proceeding.'
        history['validation'].append({'step': 0, **validate(policy, teacher, dataset, plan)})
        checkpoint(0)
    for sample in schedule[start:]:
        assert sample['groups'] == GROUPS
        batches = [dataset.batch(i, device) for i in sample['windows']]
        anchor = dataset.batch(sample['anchor_window'], device)
        update = training_update(policy, teacher, optimizer, batches, anchor, sample['anchor_noise_seed'], settings['preservation_weight'])
        history['updates'].append({**sample, **update})
        step = sample['step']
        if step in settings['validation_steps']:
            assert training.backbone_digest(policy) == backbone and teacher_digest(teacher) == original
            history['validation'].append({'step': step, **validate(policy, teacher, dataset, plan)})
            training.verify_bindings(bindings); checkpoint(step)
        if step % 20 == 0:
            print(f'Preservation training {step}/{settings["optimizer_steps"]}: anchor={update["preservation_loss"]:.6f}', flush=True)
    assert len(history['updates']) == settings['optimizer_steps']
    assert sum(r['denoiser_forward_calls'] for r in history['updates']) == 5 * settings['optimizer_steps']
    assert sum(r['teacher_head_calls'] for r in history['updates']) == settings['optimizer_steps']
    assert [r['step'] for r in history['validation']] == settings['validation_steps']
    assert training.backbone_digest(policy) == backbone and teacher_digest(teacher) == original
    state = training.tail_state(policy)
    assert training.tail_digest(state) != original
    final = directory / 'final-tail.pt'
    if final.exists(): assert training.tail_digest(training.read_tail(final, prep.sha(final))) == training.tail_digest(state)
    else: training.atomic_torch_save(final, state)
    return {'complete': True, 'settings': settings, 'bindings': bindings, 'trainable_names': selected,
            'frozen_backbone_and_normalization_unchanged': True, 'teacher_unchanged': True,
            'teacher_digest': original, 'original_tail_digest': original, 'backbone_digest': backbone,
            'history': history, 'validation_plan': plan,
            'final_head': {'path': str(final), 'sha256': prep.sha(final), 'tensor_digest': training.tail_digest(state),
                           'optimizer_step': settings['optimizer_steps']}}


def feasibility(output=OUTPUT):
    protocol = verify_protocol(output)
    bindings = input_bindings(output, protocol)
    path = output / 'feasibility.json'
    if path.exists():
        saved = prep.read(path); assert saved['passed'] and saved['bindings'] == bindings
        return saved
    guard = training.gpu_guard()
    policy, _ = adaptive.load_policy(base.MODEL)
    assert training.tail_digest(training.tail_state(policy)) == protocol['original_head_digest']
    snapshot = training.feasibility.snapshot_training_state(policy)
    before = training.feasibility.state_fingerprint(policy)
    dataset = old.RetentionWindows(); sample = protocol['training_schedule'][0]
    with adaptive.MemorySampler() as memory:
        torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        try:
            training.feasibility.configure_tail_only(policy); teacher = make_teacher(policy)
            teacher_before = teacher_digest(teacher)
            batches = [dataset.batch(i, torch.device('cuda')) for i in sample['windows']]
            anchor = dataset.batch(sample['anchor_window'], torch.device('cuda'))
            optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=SETTINGS['lr'], weight_decay=0.)
            with torch.no_grad(): first = float(preservation_loss(policy, teacher, anchor, sample['anchor_noise_seed']))
            assert first == 0.
            torch.manual_seed(SETTINGS['seed']); torch.cuda.manual_seed(SETTINGS['seed'])
            update = training_update(policy, teacher, optimizer, batches, anchor, sample['anchor_noise_seed'], 1.)
            original_update = prep.read(prep.SOURCE / 'training.json')['history']['updates'][0]
            assert update['microbatch_losses'] == original_update['microbatch_losses'], \
                'Initial supervised sample/noise stream does not reproduce original training.'
            with torch.no_grad(): after = float(preservation_loss(policy, teacher, anchor, sample['anchor_noise_seed']))
            assert after > 0 and teacher_digest(teacher) == teacher_before
        finally:
            training.feasibility.restore_training_state(policy, snapshot)
        assert training.feasibility.state_fingerprint(policy) == before
        assert {n: p.requires_grad for n, p in policy.named_parameters()} == snapshot['requires_grad']
        assert {n: m.training for n, m in policy.named_modules()} == snapshot['modes']
        assert all(p.grad is None for p in policy.parameters())
        assert torch.equal(torch.get_rng_state(), snapshot['cpu_rng'])
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), snapshot['cuda_rng']))
        seconds = time.perf_counter() - started
    result = {'passed': True, 'bindings': bindings, 'student_state_restored': True,
              'original_teacher_digest': teacher_before, 'initial_preservation_loss': first,
              'post_update_preservation_loss': after, 'update': update, 'wall_seconds': seconds,
              'original_first_supervised_losses_match': True,
              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_process_rss_bytes': memory.peak_rss,
              'minimum_available_bytes': memory.minimum_available, 'guard': guard,
              'scope': 'One real supervised-plus-preservation update and two loss checks; original student restored; no policy predictions or pushing.'}
    prep.save_once(path, result)
    return result


def train(output=OUTPUT):
    protocol = verify_protocol(output)
    bindings = input_bindings(output, protocol, True)
    path = output / 'training.json'
    if path.exists():
        saved = prep.read(path)
        assert saved['complete'] and saved['bindings'] == bindings and saved['settings'] == SETTINGS
        assert prep.sha(saved['final_head']['path']) == saved['final_head']['sha256']
        return saved
    guard = training.gpu_guard()
    policy, _ = adaptive.load_policy(base.MODEL)
    assert training.tail_digest(training.tail_state(policy)) == protocol['original_head_digest']
    with adaptive.MemorySampler() as memory:
        torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        result = fit(policy, old.RetentionWindows(), SETTINGS, protocol['training_schedule'], artifacts(output), bindings)
        result['wall_seconds'] = time.perf_counter() - started
    result['resources'] = {'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_process_rss_bytes': memory.peak_rss, 'minimum_available_bytes': memory.minimum_available, 'guard': guard}
    prep.save_once(path, result)
    return result


def eval_bindings(output):
    protocol = verify_protocol(output)
    learned = prep.read(output / 'training.json')
    assert learned['complete'] and learned['settings'] == SETTINGS and learned['teacher_unchanged']
    assert learned['bindings'] == input_bindings(output, protocol, True)
    assert learned['original_tail_digest'] == learned['teacher_digest'] == protocol['original_head_digest']
    assert learned['final_head']['optimizer_step'] == 600
    heads = {'fixed8': {'tensor_digest': protocol['original_head_digest']},
             'current12': protocol['current_head'], 'preserved12': learned['final_head']}
    assert len({v['tensor_digest'] for v in heads.values()}) == 3
    for arm in ('current12', 'preserved12'):
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
    for method, reference in [('current12', 'fixed8'), ('preserved12', 'fixed8'), ('preserved12', 'current12')]:
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
              and tables['preserved12']['successes'] >= protocol['screen']['successes_at_least']
              and recovery['preserved12'] >= protocol['screen']['recovery_successes_at_least'])
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
                                                            for a in ('current12', 'preserved12')}}
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
                print(f'Preservation comparison {len(rows)}/14: {item}', flush=True)
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
