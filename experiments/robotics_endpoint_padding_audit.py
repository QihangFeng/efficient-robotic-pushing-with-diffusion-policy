"""Independent CPU audit of endpoint-padding data, training and comparison.

Reuses the preceding pilot's trace/statistics arithmetic below. No neural
inference, physics, gradients or extra trials are repeated by this auditor.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

OUTPUT = Path('results/robotics-endpoint-padding')
TAIL = 'diffusion.unet.final_conv.'
ARMS = {'fixed8':'fixed8','current12':'adaptive12','padded12':'adaptive12'}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def tensor_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        h=hashlib.sha256(f'{value.dtype}:{tuple(value.shape)}:'.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
        digest.update(name.encode());digest.update(h.hexdigest().encode())
    return digest.hexdigest()


def save_once(path,value):
    if path.exists(): assert read(path)==value
    else:
        with path.open('x',encoding='utf-8',newline='\n') as f:
            f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def training_audit(output=OUTPUT):
    assert output.resolve() in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + '-repeat').resolve()}
    protocol, result, gate = [read(output / n) for n in ['protocol.json', 'training.json', 'feasibility.json']]
    source = read('results/robotics-recovery-pilot/training.json')
    dataset = read('results/robotics-recovery-pilot/dataset.json')
    for path, expected in protocol['fingerprints'].items():
        assert sha(path) == expected, path
    initial = {**protocol['fingerprints'], str(output / 'protocol.json'): sha(output / 'protocol.json')}
    assert gate['passed'] and gate['student_state_restored'] and gate['bindings'] == initial
    assert gate['native_padding_mask_ignored_as_configured'] and gate['update']['denoiser_forward_calls'] == 4
    assert result['complete'] and result['bindings'] == {**initial, str(output / 'feasibility.json'): sha(output / 'feasibility.json')}
    assert result['settings'] == protocol['settings'] == source['settings']
    assert result['frozen_backbone_and_normalization_unchanged']
    assert result['backbone_digest'] == source['backbone_digest']
    assert result['original_tail_digest'] == source['original_tail_digest'] == protocol['original_head_digest']
    assert not read('data/raw/robotics/diffusion_pusht/config.json')['do_mask_loss_for_padding']
    episodes = {e['episode_id']: e for e in dataset['episodes']}
    windows = list(dataset['windows'])
    for episode in dataset['episodes']:
        if episode['group'] not in ('expert', 'recovery'):
            continue
        T = episode['n_actions']
        # Independently enumerate conditioning states with 8..14 real future moves.
        for t in range(max(1, episode['branch_after_step'], T - 14), T - 7):
            j = t - 1
            windows.append({'episode_id': episode['episode_id'], 'group': episode['group'],
                'split': episode['split'], 'first_observation_index': j,
                'current_observation_index': t, 'action_start_index': j,
                'action_stop_index': j + 16, 'padded_action_count': t + 15 - T})
    assert len(windows) == 12021 and fingerprint(windows) == protocol['data']['window_fingerprint']
    groups = {}
    for i, window in enumerate(windows):
        e = episodes[window['episode_id']]
        assert window['group'] == e['group'] and window['split'] == e['split']
        j, T = window['first_observation_index'], e['n_actions']
        assert window['current_observation_index'] == j + 1 < e['n_frames']
        assert window['action_stop_index'] == j + 16
        assert window.get('padded_action_count', 0) == max(0, j + 16 - T) <= 7
        assert T - j >= 9
        if e['group'] == 'recovery':
            assert e['scene'] in ([2, 3, 4, 6, 7] if e['split'] == 'train' else [15, 16, 18])
        groups.setdefault((window['split'], window['group']), {}).setdefault(e['scene'], {}).setdefault(e['episode_id'], []).append(i)
    updates = result['history']['updates']
    assert len(updates) == len(protocol['training_schedule']) == 600
    counts, padded, rng = Counter(), Counter(), np.random.default_rng(51000)
    for step, (actual, frozen) in enumerate(zip(updates, protocol['training_schedule']), 1):
        assert actual['step'] == frozen['step'] == step
        assert actual['groups'] == frozen['groups'] == ['expert', 'recovery', 'expert', 'recovery']
        chosen = []
        for group in actual['groups']:
            scenes = groups['train', group]
            scene = sorted(scenes)[int(rng.integers(len(scenes)))]
            eps = scenes[scene]
            ep = sorted(eps)[int(rng.integers(len(eps)))]
            indices = eps[ep]
            chosen.append(indices[int(rng.integers(len(indices)))])
        assert actual['windows'] == frozen['windows'] == chosen
        for index, group in zip(chosen, actual['groups']):
            w = windows[index]
            assert w['split'] == 'train' and w['group'] == group
            counts[group] += 1
            if w.get('padded_action_count', 0):
                padded[group] += 1
        assert actual['denoiser_forward_calls'] == 4 and actual['adam_state_count'] == 6
        assert len(actual['microbatch_losses']) == 4 and np.isfinite(actual['microbatch_losses']).all()
        assert abs(actual['loss'] - np.mean(actual['microbatch_losses'])) < 1e-12
        assert actual['gradient_l2_norm'] > 0 and np.isfinite(actual['gradient_l2_norm'])
    assert counts == {'expert': 1200, 'recovery': 1200} and all(padded[g] > 0 for g in counts)
    assert len(result['validation_plan']) == 64
    for v in result['validation_plan']:
        assert windows[v['window']]['split'] == 'validation' and windows[v['window']]['group'] == v['group']
    assert [v['step'] for v in result['history']['validation']] == [0, 200, 400, 600]
    head = result['final_head']
    assert head['optimizer_step'] == 600 and sha(head['path']) == head['sha256']
    state = torch.load(head['path'], map_location='cpu', weights_only=True)
    assert len(state) == 6 and all(k.startswith(TAIL) for k in state)
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in state.values())
    assert sum(v.numel() for v in state.values()) == result['trainable_parameters'] == 1313282
    assert tensor_digest(state) == head['tensor_digest']
    assert len({head['tensor_digest'], source['final_head']['tensor_digest'], source['original_tail_digest']}) == 3
    audit = {'passed': True, 'optimizer_updates': 600, 'student_training_denoiser_calls': 2400,
        'actual_group_counts': dict(counts), 'sampled_endpoint_windows': dict(padded),
        'complete_plus_padded_windows_verified': len(windows), 'updated_parameters': 1313282,
        'frozen_backbone_matches_original': True, 'native_padding_loss_masking': False,
        'test_scenes_used_for_training': False, 'training_sha256': sha(output / 'training.json'),
        'protocol_sha256': sha(output / 'protocol.json'), 'auditor_sha256': sha(__file__),
        'scope': 'CPU source/provenance/schedule arithmetic and saved head tensors; no repeated gradients.'}
    save_once(output / 'training-audit.json', audit)
    return audit


def same_state(a, b):
    assert a.keys() == b.keys()
    assert all(abs(a[k] - b[k]) <= 1e-12 if k == 'coverage' else a[k] == b[k] for k in a)


def trace_check(row):
    trial, item, directory = row['trial'], row['item'], Path(row['directory'])
    stem = f"scene-{item['scene']}-draw-{item['draw']}-{ARMS[item['arm']]}"
    assert read(directory / (stem + '.json')) == trial
    assert trial['trace'] == stem + '.trace.json.gz' and sha(directory / trial['trace']) == trial['trace_sha256']
    with gzip.open(directory / trial['trace'], 'rt', encoding='utf-8') as stream: trace = json.load(stream)
    assert trace['initial'] == trial['initial']
    rows = trace['steps']; limit = 16 if 'label' in item else 300
    assert len(rows) == trial['steps'] <= limit
    assert trial['success'] == (rows[-1]['after']['coverage'] > .95)
    assert not any(r['after']['coverage'] > .95 for r in rows[:-1])
    assert trial['success'] or (trial['truncated'] and len(rows) == limit)
    assert rows[-1]['after']['coverage'] == trial['final_coverage']
    assert trial['prediction_seed'] == item['scene'] + item['draw']
    assert trial['horizon'] == 16 and trial['generated_shapes'] == [[1,16,2]] * trial['model_queries']
    events = {e['step']: e for e in trial['events']}; pending = age = 0; lengths = []; seconds = []
    for index, row in enumerate(rows, 1):
        assert row['step'] == index and row['new_prediction'] == (pending == 0)
        if pending == 0:
            if age: lengths.append(age)
            age = 0; pending = 8 if trial['method'] == 'fixed8' else 12
            commands = row['predicted_commands']; assert len(commands) == pending
            seconds.append(row['prediction_seconds'])
        age += 1; pending -= 1
        assert row['action'] == commands[age - 1] and row['plan_age'] == age and row['pending_before'] == pending
        if index in events:
            event = events[index]; assert event['plan_age'] == age
            if event['kind'] == 'extend': assert age == 8 and pending == 4 and event['discarded'] == 0
            elif event['kind'] == 'trim': assert age == 8 and pending == 4 and event['discarded'] == 4
            else: assert event['kind'] == 'interrupt' and age >= 4 and pending >= 2 and event['discarded'] == pending
            pending -= event['discarded']
        assert row['pending_after'] == pending
    lengths.append(age)
    assert lengths == trial['unit_lengths'] and len(lengths) == trial['model_queries']
    assert seconds == trial['prediction_seconds'] and sum(seconds) <= trial['wall_seconds']
    return trace


def complete_audit(output=OUTPUT):
    train = training_audit(output)
    protocol, index, smoke, bindings, summary = [read(output / n) for n in
        ['protocol.json','evaluation.json','smoke.json','eval-bindings.json','summary.json']]
    assert index['complete'] and index['backbone_unchanged'] and smoke['passed']
    assert [r['item'] for r in index['records']] == protocol['comparison']['schedule']
    assert len(index['records']) == 14 and len(smoke['records']) == 4
    assert bindings['protocol_sha256'] == sha(output / 'protocol.json') and bindings['training_sha256'] == sha(output / 'training.json')
    original = read('results/robotics-retention-pilot/evaluation.json')
    refs = [r for r in original['records'] if r['item']['arm'] in ('fixed8','current12')]
    assert len(refs) == 28 and [r['item'] for r in refs] == protocol['comparison']['reference_items']
    movements, traces = 0, {}
    for row in index['records'] + smoke['records']:
        binding = read(Path(row['directory']) / 'model-binding.json'); item = row['item']
        assert binding == row['binding'] and binding['arm'] == item['arm']
        assert binding['head_digest'] == bindings['heads'][item['arm']]['tensor_digest']
        assert binding['eval_bindings_sha256'] == sha(output / 'eval-bindings.json')
        assert row['trial']['method'] == binding['inner_method'] == ARMS[item['arm']]
        trace = trace_check(row); movements += len(trace['steps'])
        if 'label' in item: traces[item['label']] = trace
    for row in refs: movements += len(trace_check(row)['steps'])
    for a, b in zip(traces['before']['steps'], traces['restored']['steps']):
        assert a['action'] == b['action']; same_state(a['after'], b['after'])
    all_rows = refs + index['records']
    lookup = {(r['item']['scene'],r['item']['draw'],r['item']['arm']): r['trial'] for r in all_rows}
    scenes, draws = protocol['comparison']['scenes'], protocol['comparison']['draws']
    assert len(lookup) == 42 and len(scenes) == 7 and len(draws) == 2
    for scene in scenes:
        initial = lookup[scene,draws[0],'fixed8']['initial']
        for draw in draws:
            for arm in ARMS: same_state(initial, lookup[scene,draw,arm]['initial'])
    for draw in [None,*draws]:
        table = summary['methods'] if draw is None else summary['per_draw'][str(draw)]
        for arm in ARMS:
            trials = [t for (s,d,a), t in lookup.items() if a == arm and (draw is None or draw == d)]
            assert table[arm]['attempts'] == len(trials) and table[arm]['successes'] == sum(t['success'] for t in trials)
            for metric in ['final_coverage','model_queries','wall_seconds','monitor_seconds']:
                assert abs(table[arm]['mean_' + metric] - sum(t[metric] for t in trials) / len(trials)) < 1e-10
    indices = np.random.default_rng(20261107).integers(0,7,size=(10000,7))
    for comparison in summary['comparisons']:
        a, b = comparison['method'], comparison['reference']
        for metric, saved in comparison['differences'].items():
            values = np.array([sum(float(lookup[s,d,a][metric]) - float(lookup[s,d,b][metric]) for d in draws) / 2 for s in scenes])
            np.testing.assert_allclose(saved['mean_difference'], values.mean(), atol=1e-12, rtol=0)
            np.testing.assert_allclose(saved['paired_scene_bootstrap_95_interval'], np.quantile(values[indices].mean(1), [.025,.975]), atol=1e-12, rtol=0)
        for name,x,y in [('gained_successes',a,b),('lost_successes',b,a)]:
            assert comparison[name] == [{'scene':s,'draw':d} for s in scenes for d in draws
                                       if lookup[s,d,x]['success'] and not lookup[s,d,y]['success']]
    recovery = {a: sum(lookup[s,d,a]['success'] for s in [15,16,18] for d in draws) for a in ARMS}
    assert summary['recovery_validation_successes_of_six'] == recovery
    assert summary['development_screen_passed'] == (len(summary['comparisons'][1]['lost_successes']) < 2
        and summary['methods']['padded12']['successes'] >= 11 and recovery['padded12'] >= 3)
    assert summary['new_scored_attempts'] == 14 and summary['reused_scored_attempts'] == 28
    result = {'passed': True, 'new_scored_attempts': 14, 'reused_scored_attempts': 28, 'new_smoke_attempts': 4,
              'independent_scene_clusters': 7, 'recorded_movements_checked': movements, 'training_checked': train['passed'],
              'model_identities_and_queue_arithmetic_match': True, 'summary_and_cluster_statistics_match': True,
              'summary_sha256': sha(output / 'summary.json'), 'auditor_sha256': sha(__file__),
              'scope': 'CPU development artifact audit; no new inference or physics.'}
    save_once(output / 'artifact-audit.json', result)
    return result



if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['training','complete'],required=True)
    args=parser.parse_args()
    print(training_audit() if args.stage=='training' else complete_audit())
