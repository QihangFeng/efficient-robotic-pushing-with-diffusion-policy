"""Independently audit retention training and completed development rollouts.

No policy is constructed and no physics, inference or gradients are run. Tensor
loading is CPU-only and restricted to the small learned output block.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

OUTPUT = Path('results/robotics-retention-pilot')
TAIL = 'diffusion.unet.final_conv.'
ARMS = {'fixed8':'fixed8','current12':'adaptive12','retention12':'adaptive12'}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def tensor_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        tensor=hashlib.sha256(f'{value.dtype}:{tuple(value.shape)}:'.encode())
        tensor.update(value.detach().cpu().contiguous().numpy().tobytes())
        digest.update(name.encode());digest.update(tensor.hexdigest().encode())
    return digest.hexdigest()


def save_once(path,value):
    if path.exists(): assert read(path)==value,'Audit inputs changed; preserve previous evidence.'
    else:
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def training_audit(output=OUTPUT):
    assert output.resolve() in {OUTPUT.resolve(),OUTPUT.with_name(OUTPUT.name+'-repeat').resolve()}
    protocol=read(output/'protocol.json');result=read(output/'training.json')
    source=read('results/robotics-recovery-pilot/training.json')
    data=read('results/robotics-recovery-pilot/dataset.json')
    supplement=read('results/robotics-retention-preflight/baseline-supplement.json')
    for name,expected in protocol['fingerprints'].items(): assert sha(name)==expected,name
    expected={**protocol['fingerprints'],str(output/'protocol.json'):sha(output/'protocol.json')}
    assert result['bindings']==expected and result['settings']==protocol['settings']
    assert result['complete'] and result['frozen_backbone_and_normalization_unchanged']
    assert result['backbone_digest']==source['backbone_digest']
    assert result['original_tail_digest']==source['original_tail_digest']==protocol['original_head_digest']
    episodes={e['episode_id']:e for e in data['episodes']}
    episodes.update({e['episode_id']:e for e in supplement['episodes']})
    windows=data['windows']+supplement['windows']
    updates=result['history']['updates']
    assert len(updates)==len(protocol['training_schedule'])==600
    counts={'expert':0,'recovery':0,'baseline':0}
    for step,(actual,frozen,old) in enumerate(zip(updates,protocol['training_schedule'],source['history']['updates']),1):
        assert actual['step']==frozen['step']==old['step']==step
        assert actual['groups']==['expert','recovery','baseline','recovery']
        assert actual['original_reference_windows']==old['windows']
        assert actual['windows']==frozen['windows'] and actual['groups']==frozen['groups']
        assert actual['denoiser_forward_calls']==4 and actual['adam_state_count']==6
        assert np.isfinite(actual['loss']) and np.isfinite(actual['gradient_l2_norm']) and actual['gradient_l2_norm']>0
        for slot,(index,group) in enumerate(zip(actual['windows'],actual['groups'])):
            window=windows[index];episode=episodes[window['episode_id']]
            assert window['group']==episode['group']==group and window['split']==episode['split']=='train'
            assert window['action_stop_index']-window['action_start_index']==16
            if group=='baseline':
                assert episode['scene'] in supplement['scene_splits']['train']
                assert episode['scene'] not in supplement['scene_splits']['validation']
            if slot in (0,1,3): assert index==old['windows'][slot]
            counts[group]+=1
    assert counts=={'expert':600,'recovery':1200,'baseline':600}
    assert result['validation_plan'][:64]==source['validation_plan']
    assert len(result['validation_plan'])==96
    for sample in result['validation_plan']:
        assert windows[sample['window']]['split']=='validation'
        assert windows[sample['window']]['group']==sample['group']
    assert [v['step'] for v in result['history']['validation']]==[0,200,400,600]
    head=result['final_head']
    assert head['optimizer_step']==600 and sha(head['path'])==head['sha256']
    state=torch.load(head['path'],map_location='cpu',weights_only=True)
    assert len(state)==6 and all(k.startswith(TAIL) for k in state)
    assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in state.values())
    assert sum(v.numel() for v in state.values())==1313282
    assert tensor_digest(state)==head['tensor_digest']
    assert len({head['tensor_digest'],source['final_head']['tensor_digest'],source['original_tail_digest']})==3
    audit={'passed':True,'optimizer_updates':600,'denoiser_training_calls':2400,'actual_group_counts':counts,
           'unchanged_training_sample_slots_checked':1800,'validation_samples':96,'updated_parameters':1313282,
           'frozen_backbone_matches_original':True,'test_scenes_used_for_training':False,
           'training_sha256':sha(output/'training.json'),'protocol_sha256':sha(output/'protocol.json'),
           'auditor_sha256':sha(__file__),'scope':'CPU artifact/provenance checks; does not repeat gradients.'}
    save_once(output/'training-audit.json',audit)
    return audit


def same_state(left,right):
    assert left.keys()==right.keys()
    assert all(abs(left[k]-right[k])<=1e-12 if k=='coverage' else left[k]==right[k] for k in left)


def trace_check(row):
    trial=row['trial'];directory=Path(row['directory']);item=row['item']
    stem=f"scene-{item['scene']}-draw-{item['draw']}-{ARMS[item['arm']]}"
    assert read(directory/(stem+'.json'))==trial
    assert trial['trace']==stem+'.trace.json.gz' and sha(directory/trial['trace'])==trial['trace_sha256']
    with gzip.open(directory/trial['trace'],'rt',encoding='utf-8') as stream: trace=json.load(stream)
    assert trace['initial']==trial['initial']
    rows=trace['steps'];limit=16 if 'label' in item else 300
    assert len(rows)==trial['steps']<=limit
    assert trial['success']==(rows[-1]['after']['coverage']>.95)
    assert not any(r['after']['coverage']>.95 for r in rows[:-1])
    assert trial['success'] or (trial['truncated'] and len(rows)==limit)
    assert rows[-1]['after']['coverage']==trial['final_coverage']
    assert trial['prediction_seed']==item['scene']+item['draw']
    assert trial['horizon']==16 and trial['generated_shapes']==[[1,16,2]]*trial['model_queries']
    events={e['step']:e for e in trial['events']};pending=0;age=0;lengths=[];seconds=[]
    for index,r in enumerate(rows,1):
        assert r['step']==index and r['new_prediction']==(pending==0)
        if pending==0:
            if age: lengths.append(age)
            age=0;pending=8 if trial['method']=='fixed8' else 12
            commands=r['predicted_commands'];assert len(commands)==pending
            seconds.append(r['prediction_seconds'])
        age+=1;pending-=1
        assert r['action']==commands[age-1] and r['plan_age']==age and r['pending_before']==pending
        if index in events:
            event=events[index];assert event['plan_age']==age
            if event['kind']=='extend': assert age==8 and pending==4 and event['discarded']==0
            elif event['kind']=='trim': assert age==8 and pending==4 and event['discarded']==4
            else: assert event['kind']=='interrupt' and age>=4 and pending>=2 and event['discarded']==pending
            pending-=event['discarded']
        assert r['pending_after']==pending
    lengths.append(age)
    assert lengths==trial['unit_lengths'] and len(lengths)==trial['model_queries']
    assert seconds==trial['prediction_seconds'] and sum(seconds)<=trial['wall_seconds']
    return trace


def complete_audit(output=OUTPUT):
    train=training_audit(output)
    protocol=read(output/'protocol.json');bindings=read(output/'eval-bindings.json');summary=read(output/'summary.json')
    index=read(output/'evaluation.json');smoke=read(output/'smoke.json')
    assert index['complete'] and index['backbone_unchanged'] and smoke['passed']
    assert [r['item'] for r in index['records']]==protocol['comparison']['schedule']
    assert len(index['records'])==42 and len(smoke['records'])==4
    assert bindings['protocol_sha256']==sha(output/'protocol.json') and bindings['training_sha256']==sha(output/'training.json')
    traces={};movements=0
    for row in index['records']+smoke['records']:
        item=row['item'];binding=read(Path(row['directory'])/'model-binding.json')
        assert binding==row['binding'] and binding['arm']==item['arm']
        assert binding['head_digest']==bindings['heads'][item['arm']]['tensor_digest']
        assert binding['eval_bindings_sha256']==sha(output/'eval-bindings.json')
        assert row['trial']['method']==binding['inner_method']==ARMS[item['arm']]
        trace=trace_check(row);movements+=len(trace['steps'])
        traces[item.get('label',(item['scene'],item['draw'],item['arm']))]=trace
    for a,b in zip(traces['before']['steps'],traces['restored']['steps']):
        assert a['action']==b['action'];same_state(a['after'],b['after'])
    lookup={(r['item']['scene'],r['item']['draw'],r['item']['arm']):r['trial'] for r in index['records']}
    scenes=protocol['comparison']['scenes'];draws=protocol['comparison']['draws']
    assert len(scenes)==7 and len(draws)==2
    for scene in scenes:
        reference=lookup[scene,draws[0],'fixed8']['initial']
        for d in draws:
            for a in ARMS: same_state(reference,lookup[scene,d,a]['initial'])
    for draw in [None,*draws]:
        table=summary['methods'] if draw is None else summary['per_draw'][str(draw)]
        for arm in ARMS:
            trials=[t for (s,d,a),t in lookup.items() if a==arm and (draw is None or draw==d)]
            assert table[arm]['attempts']==len(trials) and table[arm]['successes']==sum(t['success'] for t in trials)
            for metric in ['final_coverage','model_queries','wall_seconds','monitor_seconds']:
                assert abs(table[arm]['mean_'+metric]-sum(t[metric] for t in trials)/len(trials))<1e-10
    indices=np.random.default_rng(20261107).integers(0,7,size=(10000,7))
    for comparison in summary['comparisons']:
        a,b=comparison['method'],comparison['reference']
        for metric,saved in comparison['differences'].items():
            values=np.array([sum(float(lookup[s,d,a][metric])-float(lookup[s,d,b][metric]) for d in draws)/2 for s in scenes])
            np.testing.assert_allclose(saved['mean_difference'],values.mean(),atol=1e-12,rtol=0)
            np.testing.assert_allclose(saved['paired_scene_bootstrap_95_interval'],np.quantile(values[indices].mean(1),[.025,.975]),atol=1e-12,rtol=0)
        for name,x,y in [('gained_successes',a,b),('lost_successes',b,a)]:
            assert comparison[name]==[{'scene':s,'draw':d} for s in scenes for d in draws if lookup[s,d,x]['success'] and not lookup[s,d,y]['success']]
    audit={'passed':True,'scored_attempts':42,'smoke_attempts':4,'independent_scene_clusters':7,
           'recorded_movements_checked':movements,'training_checked':train['passed'],
           'model_identities_and_action_queue_arithmetic_match':True,'summary_and_cluster_statistics_match':True,
           'summary_sha256':sha(output/'summary.json'),'auditor_sha256':sha(__file__),
           'scope':'Development artifact audit; no new neural inference or physics.'}
    save_once(output/'artifact-audit.json',audit)
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['training','complete'],required=True)
    args=parser.parse_args()
    print(training_audit() if args.stage=='training' else complete_audit())
