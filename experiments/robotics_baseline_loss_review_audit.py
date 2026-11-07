"""Independent arithmetic/source audit of the saved baseline-loss review."""

import gzip
import hashlib
import json
import math
from pathlib import Path

OUTPUT = Path('results/robotics-baseline-loss-review')
ARMS = ['fixed8', 'milestone12', 'recovery8', 'recovery12']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def drawdown(values):
    peak, result = values[0], 0.
    for value in values:
        peak = max(peak, value)
        result = max(result, peak - value)
    return result


def audit(output=OUTPUT):
    assert output.resolve() in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + '-repeat').resolve()}
    summary = read(output / 'summary.json')
    for name, expected in summary['source_bindings'].items():
        assert sha(name) == expected, name
    rows = read('results/robotics-recovery-confirmation/evaluation.json')['records']
    rows += read('results/robotics-recovery-fixed8/evaluation.json')['records']
    lookup = {(r['trial']['scene'], r['trial']['draw'], r['arm']):r for r in rows}
    assert len(rows) == len(lookup) == 320
    traces = {}
    for key, row in lookup.items():
        path = Path(row['directory']) / row['trial']['trace']
        assert sha(path) == row['trial']['trace_sha256']
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            traces[key] = json.load(stream)
    loss_keys = [(s,d) for s in range(170,210) for d in [60000,70000]
                 if lookup[s,d,'fixed8']['trial']['success'] and not lookup[s,d,'recovery12']['trial']['success']]
    assert loss_keys == [(c['scene'],c['draw']) for c in summary['losses']]
    assert len(loss_keys) == 9 and len({s for s,d in loss_keys}) == 7
    patterns = {}
    for case in summary['losses']:
        scene, draw = case['scene'],case['draw']
        a = lookup[scene,draw,'milestone12']['trial']['success']
        f = lookup[scene,draw,'recovery8']['trial']['success']
        label = f'original_adaptive_{int(a)}_trained_fixed8_{int(f)}'
        patterns[label] = patterns.get(label,0) + 1
        fixed, adaptive = traces[scene,draw,'recovery8'], traces[scene,draw,'recovery12']
        assert fixed['steps'][:8] != []
        for x,y in zip(fixed['steps'][:8],adaptive['steps'][:8]):
            assert x['action']==y['action'] and x['new_prediction']==y['new_prediction']
            for k in x['after']:
                assert (abs(x['after'][k]-y['after'][k])<=1e-12 if k=='coverage' else x['after'][k]==y['after'][k])
        assert fixed['steps'][8]['new_prediction'] and not adaptive['steps'][8]['new_prediction']
        assert case['trained_schedule_divergence']['first_different_movement']==9
        for arm in ARMS:
            raw=traces[scene,draw,arm]; actual=case['methods'][arm];steps=raw['steps']
            values=[raw['initial']['coverage']]+[r['after']['coverage'] for r in steps]
            assert actual['peak_coverage']==max(values)
            assert actual['first_peak_after_movement']==values.index(max(values))
            assert actual['final_coverage']==values[-1]
            end=len(steps)
            while end>0:
                prev=raw['initial'] if end==1 else steps[end-2]['after'];cur=steps[end-1]['after']
                distance=math.dist(prev['block_pos'],cur['block_pos'])
                angle=abs((prev['block_angle']-cur['block_angle']+math.pi)%(2*math.pi)-math.pi)
                if distance>1e-9 or angle>1e-9: break
                end-=1
            stationary=actual['terminal_block_stationarity']
            assert stationary['after_movement']==end
            assert stationary['movements']==len(steps)-end
            assert stationary['further_predictions']==sum(r['new_prediction'] for r in steps[end:])
    assert patterns==summary['patterns']
    totals={k:{'attempts':0,'extensions':0,'extension_suffix_drawdowns_over_three_pp':0,
        'attempts_with_extension_suffix_drop_over_three_pp':0,'fresh_prefixes':0,
        'fresh_prefix_drawdowns_over_three_pp':0,'attempts_with_fresh_prefix_drop_over_three_pp':0,
        'attempts_without_interruptions':0} for k in ['retained','gained','lost','shared_failure']}
    for s in range(170,210):
        for d in [60000,70000]:
            original=lookup[s,d,'fixed8']['trial']['success'];new=lookup[s,d,'recovery12']['trial']
            category=('retained' if new['success'] else 'lost') if original else ('gained' if new['success'] else 'shared_failure')
            raw=traces[s,d,'recovery12']; steps=raw['steps']; total=totals[category]
            total['attempts']+=1; total['attempts_without_interruptions']+=not any(e['kind']=='interrupt' for e in new['events'])
            ext_bad=0
            for e in new['events']:
                if e['kind']!='extend': continue
                suffix=[]
                for r in steps[e['step']:e['step']+4]:
                    if r['new_prediction']: break
                    suffix.append(r['after']['coverage'])
                ext_bad+=drawdown([steps[e['step']-1]['after']['coverage']]+suffix)>.03
                total['extensions']+=1
            total['extension_suffix_drawdowns_over_three_pp']+=ext_bad
            total['attempts_with_extension_suffix_drop_over_three_pp']+=ext_bad>0
            starts=[i for i,r in enumerate(steps) if r['new_prediction']]
            prefix_bad=0
            for i,j in zip(starts,starts[1:]+[len(steps)]):
                prior=raw['initial']['coverage'] if i==0 else steps[i-1]['after']['coverage']
                prefix_bad+=drawdown([prior]+[r['after']['coverage'] for r in steps[i:min(j,i+8)]])>.03
                total['fresh_prefixes']+=1
            total['fresh_prefix_drawdowns_over_three_pp']+=prefix_bad
            total['attempts_with_fresh_prefix_drop_over_three_pp']+=prefix_bad>0
    assert totals==summary['combined_method_comparison_groups']
    timing_counts={}
    for s in range(170,210):
        for d in [60000,70000]:
            a,b=traces[s,d,'recovery8']['steps'],traces[s,d,'recovery12']['steps']
            first=next((i+1 for i,(x,y) in enumerate(zip(a,b)) if x['new_prediction']!=y['new_prediction']),None)
            timing_counts[str(first)]=timing_counts.get(str(first),0)+1
    assert timing_counts==summary['trained_first_schedule_difference_counts']
    dataset_path=Path('results/robotics-recovery-pilot/dataset.json')
    dataset=read(dataset_path)
    controls=[e for e in dataset['episodes'] if e['group']=='ordinary_control']
    assert len(controls)==20 and not dataset['ordinary_controls']['used_for_training']
    for e in controls:
        assert e['split']=='diagnostic' and sha(e['file'])==e['sha256']
    available={'episodes':len(controls),'scene_ids':[e['scene'] for e in controls],
        'complete_native_windows':sum(max(0,min(e['n_frames']-1,e['n_actions']-15)) for e in controls),
        'used_in_current_training':False,'manifest_sha256':sha(dataset_path),
        'scope':'Availability for a proposed future retention experiment; not new training data or a trained candidate.'}
    result={'passed':True,'attempts':320,'losses':9,'distinct_lost_scenes':7,
        'movements':sum(len(t['steps']) for t in traces.values()),
        'source_bindings_verified':len(summary['source_bindings']),
        'component_patterns_and_peaks_and_stationarity_match':True,'all_control_group_drop_counts_match':True,
        'trained_first_schedule_difference_counts':timing_counts,'available_baseline_success_rehearsal':available,
        'summary_sha256':sha(output/'summary.json'),'auditor_sha256':sha(__file__),
        'scope':'Independent saved-JSON/trace arithmetic; no policy import, inference, training or physics.'}
    path=output/'artifact-audit.json'
    if path.exists(): assert read(path)==result
    else:
        with path.open('x',encoding='utf-8',newline='\n') as stream: stream.write(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    print(audit())
