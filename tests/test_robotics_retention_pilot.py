"""CPU tests of the isolated retention experiment; no pretrained policy load."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import robotics_retention_pilot as pilot
from test_robotics_recovery_training import RandomLossPolicy, training_batch
from test_robotics_recovery_confirmation import ToyPolicy


def temporary():
    root=Path(__file__).resolve().parents[1]/'tmp'
    root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root,prefix='retention-pilot-tests-')


class ToyDataset:
    def choose(self,split,group,rng):
        return {'expert':0,'recovery':100,'baseline':200}[group]+int(rng.integers(10))
    def batch(self,index,device):
        value=training_batch()
        value['action']=value['action']+index/100.
        return {k:v.to(device) for k,v in value.items()}


def small_settings():
    return {**pilot.SETTINGS,'optimizer_steps':2,'validation_steps':[0,1,2],
            'checkpoint_steps':[1,2],'validation_windows_per_group':2}


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset=pilot.RetentionWindows()
        cls.reference=pilot.prep.read(pilot.prep.SOURCE/'training.json')['history']['updates']
        cls.schedule=pilot.sample_schedule(cls.dataset,reference=cls.reference)

    def test_all_three_unchanged_draws_match_original_six_hundred_updates(self):
        self.assertEqual(len(self.schedule),600)
        for row,old in zip(self.schedule,self.reference):
            self.assertEqual(row['original_reference_windows'],old['windows'])
            for slot in [0,1,3]: self.assertEqual(row['windows'][slot],old['windows'][slot])
            self.assertEqual(row['groups'],['expert','recovery','baseline','recovery'])

    def test_replacement_is_always_baseline_training_and_never_validation(self):
        for row in self.schedule:
            window=self.dataset.windows[row['windows'][2]]
            self.assertEqual((window['group'],window['split']),('baseline','train'))
            episode=self.dataset.episodes[window['episode_id']]
            self.assertNotIn(episode['scene'],[1,5,10,23,*range(170,210)])

    def test_original_dataset_is_not_relabelled_by_adapter(self):
        original=pilot.prep.read(pilot.prep.SOURCE/'dataset.json')
        controls=[e for e in original['episodes'] if e['group']=='ordinary_control']
        self.assertEqual(len(controls),20)
        self.assertTrue(all(e['split']=='diagnostic' for e in controls))
        self.assertFalse(original['ordinary_controls']['used_for_training'])

    def test_real_baseline_sample_has_native_inputs_and_sixteen_unpadded_labels(self):
        index=self.schedule[0]['windows'][2]
        value=self.dataset.batch(index,torch.device('cpu'))
        self.assertEqual(set(value),{'observation.image','observation.state','action','action_is_pad'})
        self.assertEqual(tuple(value['observation.image'].shape),(1,2,3,96,96))
        self.assertEqual(tuple(value['observation.state'].shape),(1,2,2))
        self.assertEqual(tuple(value['action'].shape),(1,16,2))
        self.assertFalse(bool(value['action_is_pad'].any()))

    def test_reference_schedule_change_is_not_silently_accepted(self):
        changed=copy.deepcopy(self.reference)
        changed[0]['windows'][0]=-1
        with self.assertRaises(AssertionError): pilot.sample_schedule(self.dataset,reference=changed)


class TrainingTests(unittest.TestCase):
    def test_toy_updates_only_final_block_and_preserves_backbone(self):
        with temporary() as folder:
            policy=RandomLossPolicy()
            dataset=ToyDataset(); settings=small_settings()
            source=Path(folder)/'input.txt';source.write_text('fixed')
            bindings={str(source):pilot.prep.sha(source)}
            before=pilot.training.backbone_digest(policy)
            schedule=pilot.sample_schedule(dataset,settings)
            result=pilot.fit(policy,dataset,settings,schedule,Path(folder)/'weights',bindings)
            self.assertTrue(result['complete'])
            self.assertEqual(pilot.training.backbone_digest(policy),before)
            self.assertEqual(sum(r['denoiser_forward_calls'] for r in result['history']['updates']),8)
            self.assertEqual(result['final_head']['optimizer_step'],2)
            self.assertEqual(set(result['history']['validation'][0]['groups']),{'expert','recovery','baseline'})
            self.assertTrue(all(name.startswith(pilot.training.TAIL) for name in result['trainable_names']))

    def test_interrupted_toy_training_resumes_to_identical_weights_and_losses(self):
        with temporary() as folder:
            first=RandomLossPolicy();initial=copy.deepcopy(first.state_dict())
            settings=small_settings();dataset=ToyDataset();schedule=pilot.sample_schedule(dataset,settings)
            source=Path(folder)/'input.txt';source.write_text('fixed');bindings={str(source):pilot.prep.sha(source)}
            uninterrupted=pilot.fit(first,dataset,settings,schedule,Path(folder)/'full',bindings)
            second=RandomLossPolicy();second.load_state_dict(initial)
            actual_update=pilot.training.training_update;calls=[]
            def interrupted(*args):
                calls.append(1)
                if len(calls)==2: raise RuntimeError('deliberate interruption before next update')
                return actual_update(*args)
            with patch.object(pilot.training,'training_update',side_effect=interrupted), self.assertRaises(RuntimeError):
                pilot.fit(second,dataset,settings,schedule,Path(folder)/'resume',bindings)
            resumed=RandomLossPolicy();resumed.load_state_dict(initial)
            result=pilot.fit(resumed,dataset,settings,schedule,Path(folder)/'resume',bindings)
            self.assertEqual(pilot.training.tail_digest(pilot.training.tail_state(first)),pilot.training.tail_digest(pilot.training.tail_state(resumed)))
            self.assertEqual([r['loss'] for r in uninterrupted['history']['updates']],[r['loss'] for r in result['history']['updates']])

    def test_partial_training_artifact_is_never_overwritten(self):
        with temporary() as folder:
            target=Path(folder)/'weights';target.mkdir();partial=target/'resume.pt.tmp';partial.write_bytes(b'partial')
            source=Path(folder)/'input.txt';source.write_text('fixed')
            with self.assertRaisesRegex(AssertionError,'Unbound'):
                pilot.fit(RandomLossPolicy(),ToyDataset(),small_settings(),pilot.sample_schedule(ToyDataset(),small_settings()),
                          target,{str(source):pilot.prep.sha(source)})
            self.assertEqual(partial.read_bytes(),b'partial')


def evaluation_context(folder):
    output=Path(folder)/'output';output.mkdir()
    policy=ToyPolicy();original=pilot.training.tail_state(policy)
    heads={'fixed8':original,'current12':{k:v+1 for k,v in original.items()},'retention12':{k:v+2 for k,v in original.items()}}
    bindings={'heads':{a:{'tensor_digest':pilot.training.tail_digest(v)} for a,v in heads.items()}}
    pilot.prep.save_once(output/'eval-bindings.json',bindings)
    protocol={'comparison':{'scenes':[1],'draws':[91000],
                           'schedule':[{'scene':1,'draw':91000,'arm':a} for a in pilot.ARMS]}}
    return output,policy,heads,bindings,protocol


class EvaluationTests(unittest.TestCase):
    def test_summary_pairs_both_draws_within_seven_scenes(self):
        with temporary() as folder:
            output=Path(folder)
            scenes=[1,5,10,15,16,18,23];draws=[91000,92000]
            protocol={'comparison':{'scenes':scenes,'draws':draws},
                      'statistics':{'scene_clusters':7,'draws':2}}
            rows=[]
            for scene in scenes:
                for draw in draws:
                    for arm in pilot.ARMS:
                        candidate=arm=='retention12'
                        rows.append({'item':{'scene':scene,'draw':draw,'arm':arm},'trial':{
                            'initial':{'coverage':0.,'image_sha256':str(scene)},
                            'success':candidate or scene in scenes[:2],
                            'final_coverage':.96 if candidate else .9,'model_queries':10 if candidate else 12,
                            'wall_seconds':50. if candidate else 60.,'monitor_seconds':1.,
                            'prediction_seconds':[5.],'peak_gpu_allocated_bytes':100}})
            pilot.prep.save_once(output/'evaluation.json',{'complete':True,'backbone_unchanged':True})
            pilot.prep.save_once(output/'protocol.json',protocol);pilot.prep.save_once(output/'eval-bindings.json',{})
            with patch.object(pilot,'scan',return_value=rows),patch.object(pilot,'smoke_check'):
                result=pilot.summarize(output,protocol,{})
            self.assertEqual(result['methods']['retention12']['successes'],14)
            self.assertTrue(all(result['per_draw'][str(d)]['retention12']['attempts']==7 for d in draws))
            self.assertEqual(result['statistics']['scene_clusters'],7)
            comparison=result['comparisons'][1]
            self.assertEqual(len(comparison['gained_successes']),10)
            self.assertEqual(comparison['lost_successes'],[])
            self.assertAlmostEqual(comparison['differences']['success']['mean_difference'],5/7)

    def test_progress_is_atomic_lf_json_and_refuses_an_orphan_temporary(self):
        with temporary() as folder:
            path=Path(folder)/'progress.json'
            pilot.save_progress(path,{'records':[1]})
            self.assertNotIn(b'\r\n',path.read_bytes())
            path.with_suffix('.json.tmp').write_bytes(b'partial')
            with self.assertRaises(AssertionError): pilot.save_progress(path,{'records':[2]})
            self.assertEqual(pilot.prep.read(path),{'records':[1]})

    def test_candidate_collector_uses_candidate_weights_and_unchanged_adaptive_method(self):
        with temporary() as folder:
            output,policy,heads,bindings,protocol=evaluation_context(folder)
            item={'scene':1,'draw':91000,'arm':'retention12'}
            def evaluate(actual,scene,draw,method,directory,limit):
                self.assertIs(actual,policy)
                self.assertEqual((scene,draw,method,limit),(1,91000,'adaptive12',300))
                self.assertEqual(pilot.training.tail_digest(pilot.training.tail_state(actual)),bindings['heads']['retention12']['tensor_digest'])
                self.assertTrue(all(not p.requires_grad for p in policy.parameters()))
            with patch.object(pilot.gen,'evaluate_episode',side_effect=evaluate) as call:
                pilot.collect(policy,item,output,protocol,bindings,heads)
            self.assertEqual(call.call_count,1)

    def test_existing_sidecar_does_not_make_another_prediction(self):
        with temporary() as folder:
            output,policy,heads,bindings,protocol=evaluation_context(folder)
            item=protocol['comparison']['schedule'][0]
            pilot.model_binding(output,item,bindings,create=True)
            directory=pilot.directory_for(output,item)
            pilot.prep.save_once(directory/(pilot.gen.stem(1,91000,'fixed8')+'.json'),{'saved':True})
            with patch.object(pilot.gen,'evaluate_episode') as evaluator,patch.object(pilot.prior,'activate') as activate:
                pilot.collect(policy,item,output,protocol,bindings,heads)
            evaluator.assert_not_called();activate.assert_not_called()

    def test_missing_weight_binding_cannot_be_created_during_verification(self):
        with temporary() as folder:
            output,_,_,bindings,protocol=evaluation_context(folder)
            with self.assertRaises(AssertionError): pilot.model_binding(output,protocol['comparison']['schedule'][0],bindings)

    def test_scan_rejects_orphan_trace(self):
        with temporary() as folder:
            output,_,_,bindings,protocol=evaluation_context(folder)
            item=protocol['comparison']['schedule'][0]
            pilot.model_binding(output,item,bindings,create=True)
            directory=pilot.directory_for(output,item)
            (directory/'orphan.trace.json.gz').write_bytes(b'partial')
            with self.assertRaisesRegex(AssertionError,'Orphan'):
                pilot.scan(output,protocol,bindings)

    def test_schedule_gap_cannot_be_resumed_as_a_completed_prefix(self):
        with temporary() as folder:
            output,_,_,bindings,protocol=evaluation_context(folder)
            item=protocol['comparison']['schedule'][1]
            pilot.model_binding(output,item,bindings,create=True)
            directory=pilot.directory_for(output,item)
            trial={'scene':1,'draw':91000,'method':'adaptive12','trace':'second.trace.json.gz'}
            pilot.prep.save_once(directory/(pilot.gen.stem(1,91000,'adaptive12')+'.json'),trial)
            (directory/trial['trace']).write_bytes(b'fake')
            with patch.object(pilot.gen,'verify_trial'),self.assertRaisesRegex(AssertionError,'prefix'):
                pilot.scan(output,protocol,bindings)

    def test_previous_results_and_checkpoints_are_protected(self):
        for output in [pilot.prior.OUTPUT,pilot.prep.SOURCE,Path('results')]:
            with self.assertRaises(AssertionError): pilot.check_output(output)
        self.assertNotEqual(pilot.artifacts(pilot.OUTPUT).resolve(),pilot.artifacts(pilot.OUTPUT.with_name(pilot.OUTPUT.name+'-repeat')).resolve())


if __name__=='__main__': unittest.main()
