"""CPU artifact audit of the preservation pilot, with independent arithmetic.

Loads only small output-block tensors. No policy, inference, physics or gradient
computation is constructed or repeated by this auditor.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

OUTPUT = Path('results/robotics-preservation-pilot')
TAIL = 'diffusion.unet.final_conv.'
ARMS = {'fixed8': 'fixed8', 'current12': 'adaptive12', 'preserved12': 'adaptive12'}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        h = hashlib.sha256(f'{value.dtype}:{tuple(value.shape)}:'.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
        digest.update(name.encode()); digest.update(h.hexdigest().encode())
    return digest.hexdigest()


def save_once(path, value):
    if path.exists(): assert read(path) == value, 'Preserve previous audit evidence; inputs changed.'
    else:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def training_audit(output=OUTPUT):
    assert output.resolve() in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + '-repeat').resolve()}
    protocol, result, gate = [read(output / n) for n in ['protocol.json', 'training.json', 'feasibility.json']]
    source = read('results/robotics-recovery-pilot/training.json')
    dataset = read('results/robotics-recovery-pilot/dataset.json')
    supplement = read('results/robotics-retention-preflight/baseline-supplement.json')
    for path, expected in protocol['fingerprints'].items(): assert sha(path) == expected, path
    initial = {**protocol['fingerprints'], str(output / 'protocol.json'): sha(output / 'protocol.json')}
    assert gate['passed'] and gate['student_state_restored'] and gate['bindings'] == initial
    assert gate['original_first_supervised_losses_match'] and gate['initial_preservation_loss'] == 0
    assert np.isfinite(gate['post_update_preservation_loss']) and gate['post_update_preservation_loss'] > 0
    assert gate['update']['microbatch_losses'] == source['history']['updates'][0]['microbatch_losses']
    assert result['complete'] and result['bindings'] == {**initial, str(output / 'feasibility.json'): sha(output / 'feasibility.json')}
    assert result['settings'] == protocol['settings']
    assert result['settings']['preservation_weight'] == 1. and not result['settings']['auxiliary_supervised_loss']
    assert result['frozen_backbone_and_normalization_unchanged'] and result['teacher_unchanged']
    assert result['backbone_digest'] == source['backbone_digest']
    original = protocol['original_head_digest']
    assert result['teacher_digest'] == result['original_tail_digest'] == gate['original_teacher_digest'] == source['original_tail_digest'] == original
    episodes = {e['episode_id']: e for e in dataset['episodes']}
    episodes.update({e['episode_id']: e for e in supplement['episodes']})
    windows = dataset['windows'] + supplement['windows']
    updates = result['history']['updates']
    assert len(updates) == len(protocol['training_schedule']) == 600
    counts = {'expert': 0, 'recovery': 0, 'baseline_anchor': 0}
    for step, (actual, frozen, previous) in enumerate(zip(updates, protocol['training_schedule'], source['history']['updates']), 1):
        assert actual['step'] == frozen['step'] == previous['step'] == step
        assert actual['groups'] == ['expert', 'recovery', 'expert', 'recovery']
        assert actual['windows'] == previous['windows'] == frozen['windows']
        for key in ['anchor_window', 'anchor_noise_seed']: assert actual[key] == frozen[key]
        assert actual['anchor_noise_seed'] == protocol['settings']['anchor_noise_seed'] + step
        for index, group in zip(actual['windows'], actual['groups']):
            window = windows[index]; episode = episodes[window['episode_id']]
            assert window['group'] == episode['group'] == group and window['split'] == episode['split'] == 'train'
            assert window['action_stop_index'] - window['action_start_index'] == 16
            counts[group] += 1
        window = windows[actual['anchor_window']]; episode = episodes[window['episode_id']]
        assert window['group'] == episode['group'] == 'baseline' and window['split'] == episode['split'] == 'train'
        assert episode['scene'] in supplement['scene_splits']['train']
        assert episode['scene'] not in [1,5,10,15,16,18,23,*range(170,210)]
        counts['baseline_anchor'] += 1
        assert actual['denoiser_forward_calls'] == 5 and actual['teacher_head_calls'] == 1
        assert actual['teacher_versions_unchanged'] and actual['adam_state_count'] == 6
        assert len(actual['microbatch_losses']) == 4 and np.isfinite(actual['microbatch_losses']).all()
        assert abs(actual['supervised_loss'] - np.mean(actual['microbatch_losses'])) < 1e-12
        assert actual['preservation_loss'] >= 0 and np.isfinite(actual['preservation_loss'])
        assert abs(actual['loss'] - actual['supervised_loss'] - actual['preservation_loss']) < 1e-12
        assert actual['gradient_l2_norm'] > 0 and np.isfinite(actual['gradient_l2_norm'])
    assert counts == {'expert': 1200, 'recovery': 1200, 'baseline_anchor': 600}
    assert updates[0]['preservation_loss'] == 0
    assert updates[0]['microbatch_losses'] == source['history']['updates'][0]['microbatch_losses']
    assert any(r['preservation_loss'] > 0 for r in updates[1:])
    assert result['validation_plan'][:64] == source['validation_plan'] and len(result['validation_plan']) == 96
    for row in result['validation_plan']:
        assert windows[row['window']]['split'] == 'validation' and windows[row['window']]['group'] == row['group']
    assert [v['step'] for v in result['history']['validation']] == [0,200,400,600]
    for v in result['history']['validation']:
        assert v['teacher_digest'] == original and v['preservation_windows'] == 32
        assert np.isfinite(v['preservation_mean_loss']) and v['preservation_mean_loss'] >= 0
    assert result['history']['validation'][0]['preservation_mean_loss'] == 0
    head = result['final_head']
    assert head['optimizer_step'] == 600 and sha(head['path']) == head['sha256']
    state = torch.load(head['path'], map_location='cpu', weights_only=True)
    assert len(state) == 6 and all(k.startswith(TAIL) for k in state)
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in state.values())
    assert sum(v.numel() for v in state.values()) == 1313282 and tensor_digest(state) == head['tensor_digest']
    assert len({head['tensor_digest'], source['final_head']['tensor_digest'], original}) == 3
    result = {'passed': True, 'optimizer_updates': 600, 'student_training_denoiser_calls': 3000,
              'teacher_training_final_block_calls': 600, 'actual_group_counts': counts,
              'unchanged_supervised_sample_slots_checked': 2400, 'original_first_supervised_losses_match': True,
              'validation_samples': 96, 'updated_parameters': 1313282, 'frozen_backbone_matches_original': True,
              'original_teacher_preserved': True, 'test_scenes_used_for_training': False,
              'training_sha256': sha(output / 'training.json'), 'protocol_sha256': sha(output / 'protocol.json'),
              'auditor_sha256': sha(__file__), 'scope': 'CPU artifact/provenance arithmetic; does not repeat gradients.'}
    save_once(output / 'training-audit.json', result)
    return result


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
        and summary['methods']['preserved12']['successes'] >= 11 and recovery['preserved12'] >= 3)
    assert summary['new_scored_attempts'] == 14 and summary['reused_scored_attempts'] == 28
    result = {'passed': True, 'new_scored_attempts': 14, 'reused_scored_attempts': 28, 'new_smoke_attempts': 4,
              'independent_scene_clusters': 7, 'recorded_movements_checked': movements, 'training_checked': train['passed'],
              'model_identities_and_queue_arithmetic_match': True, 'summary_and_cluster_statistics_match': True,
              'summary_sha256': sha(output / 'summary.json'), 'auditor_sha256': sha(__file__),
              'scope': 'CPU development artifact audit; no new inference or physics.'}
    save_once(output / 'artifact-audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['training','complete'], required=True)
    args = parser.parse_args()
    print(training_audit() if args.stage == 'training' else complete_audit())
