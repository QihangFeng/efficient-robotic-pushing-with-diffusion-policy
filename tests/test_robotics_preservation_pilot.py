"""CPU checks of the preservation objective, data, resume and collection."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import robotics_preservation_pilot as pilot
import robotics_preservation_pilot_audit as audit
from test_robotics_recovery_feasibility import training_batch

torch.set_num_threads(1)


class TinyUnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Conv1d(2, 4, 1)
        self.final_conv = torch.nn.Sequential(torch.nn.Conv1d(4, 4, 3, padding=1),
            torch.nn.GroupNorm(1, 4), torch.nn.Mish(), torch.nn.Conv1d(4, 2, 1))

    def forward(self, actions, timesteps):
        return self.final_conv(self.trunk(actions.transpose(1, 2))).transpose(1, 2)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion = torch.nn.Module()
        self.diffusion.unet = TinyUnet()
        self.register_buffer('frozen_normalization', torch.tensor([3.]))

    def forward(self, batch):
        action = batch['action'] / 100.
        noise = torch.randn_like(action)
        timestep = torch.randint(0, 100, (1,))
        prediction = self.diffusion.unet(action + noise * .1, timestep)
        return torch.nn.functional.mse_loss(prediction, noise), None


class ToyDataset:
    def choose(self, split, group, rng):
        return {'expert': 0, 'recovery': 100, 'baseline': 200}[group] + int(rng.integers(10))

    def batch(self, index, device):
        batch = training_batch()
        batch['action'] = batch['action'] + index / 100.
        return {k: v.to(device) for k, v in batch.items()}


def settings():
    return {**pilot.SETTINGS, 'optimizer_steps': 2, 'validation_steps': [0,1,2],
            'checkpoint_steps': [1,2], 'validation_windows_per_group': 2}


def temporary():
    root = Path(__file__).resolve().parents[1] / 'tmp'
    root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root, prefix='preservation-tests-')


def configure(policy):
    pilot.training.feasibility.configure_tail_only(policy)
    return torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.)


class ObjectiveTests(unittest.TestCase):
    def test_teacher_is_exact_separate_frozen_original_head(self):
        student = TinyPolicy(); configure(student); teacher = pilot.make_teacher(student)
        self.assertEqual(pilot.teacher_digest(teacher), pilot.training.tail_digest(pilot.training.tail_state(student)))
        for a, b in zip(student.diffusion.unet.final_conv.parameters(), teacher.parameters()):
            self.assertTrue(torch.equal(a, b)); self.assertNotEqual(a.data_ptr(), b.data_ptr())
            self.assertFalse(b.requires_grad)

    def test_penalty_is_zero_for_teacher_and_gradient_pulls_output_bias_back(self):
        student = TinyPolicy(); configure(student); teacher = pilot.make_teacher(student)
        batch = training_batch()
        self.assertEqual(float(pilot.preservation_loss(student, teacher, batch, 25)), 0.)
        teacher_before = pilot.teacher_digest(teacher)
        with torch.no_grad(): student.diffusion.unet.final_conv[-1].bias.add_(.2)
        value = pilot.preservation_loss(student, teacher, batch, 25)
        self.assertAlmostEqual(float(value.detach()), .04, places=6)
        value.backward()
        np.testing.assert_allclose(student.diffusion.unet.final_conv[-1].bias.grad.numpy(), [.2,.2], atol=1e-6)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(p.grad is None for n, p in student.named_parameters() if not n.startswith(pilot.training.TAIL)))
        self.assertEqual(pilot.teacher_digest(teacher), teacher_before)

    def test_auxiliary_forward_preserves_rng_and_removes_hook_on_failure(self):
        student = TinyPolicy(); configure(student); teacher = pilot.make_teacher(student)
        before = torch.get_rng_state().clone()
        pilot.preservation_loss(student, teacher, training_batch(), 81)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with patch.object(student, 'forward', side_effect=RuntimeError('test')), self.assertRaises(RuntimeError):
            pilot.preservation_loss(student, teacher, training_batch(), 81)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(student.diffusion.unet.final_conv._forward_hooks)

    def test_first_update_matches_original_four_sample_update_exactly(self):
        student = TinyPolicy(); reference = copy.deepcopy(student)
        opt = configure(student); old_opt = configure(reference); teacher = pilot.make_teacher(student)
        batches = [ToyDataset().batch(i, torch.device('cpu')) for i in [0,100,1,101]]
        torch.manual_seed(51000)
        old = pilot.training.training_update(reference, old_opt, batches)
        old_rng = torch.get_rng_state().clone()
        torch.manual_seed(51000)
        new = pilot.training_update(student, teacher, opt, batches, training_batch(), 51004, 1.)
        self.assertEqual(new['preservation_loss'], 0.)
        self.assertEqual(new['microbatch_losses'], old['microbatch_losses'])
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        self.assertEqual(pilot.training.tail_digest(pilot.training.tail_state(student)),
                         pilot.training.tail_digest(pilot.training.tail_state(reference)))
        self.assertEqual((new['denoiser_forward_calls'], new['teacher_head_calls']), (5,1))

    def test_zero_weight_keeps_later_supervised_noise_and_updates_unchanged(self):
        student = TinyPolicy(); reference = copy.deepcopy(student)
        opt = configure(student); old_opt = configure(reference); teacher = pilot.make_teacher(student)
        batches = [training_batch() for _ in range(4)]
        original_rng = torch.get_rng_state().clone()
        for _ in range(2): pilot.training.training_update(reference, old_opt, batches)
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(original_rng)
        for step in range(2): pilot.training_update(student, teacher, opt, batches, training_batch(), 900+step, 0.)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(pilot.training.tail_digest(pilot.training.tail_state(student)),
                         pilot.training.tail_digest(pilot.training.tail_state(reference)))

    def test_update_refuses_unfrozen_backbone_or_trainable_teacher(self):
        student = TinyPolicy(); opt = configure(student); teacher = pilot.make_teacher(student)
        batches = [training_batch() for _ in range(4)]
        student.diffusion.unet.trunk.weight.requires_grad_(True)
        with self.assertRaises(AssertionError):
            pilot.training_update(student, teacher, opt, batches, training_batch(), 100, 1.)
        student.diffusion.unet.trunk.weight.requires_grad_(False)
        next(teacher.parameters()).requires_grad_(True)
        with self.assertRaises(AssertionError): pilot.preservation_loss(student, teacher, training_batch(), 100)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = pilot.old.RetentionWindows()
        cls.reference = pilot.prep.read(pilot.prep.SOURCE / 'training.json')['history']['updates']
        cls.schedule = pilot.sample_schedule(cls.dataset, reference=cls.reference)

    def test_all_2400_supervised_slots_match_original_and_anchors_match_prescribed_draws(self):
        old = pilot.prep.read(pilot.old.OUTPUT / 'protocol.json')['training_schedule']
        self.assertEqual(len(self.schedule), 600)
        for row, previous, replaced in zip(self.schedule, self.reference, old):
            self.assertEqual(row['windows'], previous['windows'])
            self.assertEqual(row['groups'], ['expert','recovery','expert','recovery'])
            self.assertEqual(row['anchor_window'], replaced['windows'][2])
            self.assertEqual(row['anchor_noise_seed'], 51003 + row['step'])

    def test_anchors_are_training_only_with_native_two_frame_inputs(self):
        for row in self.schedule:
            window = self.dataset.windows[row['anchor_window']]
            self.assertEqual((window['group'], window['split']), ('baseline','train'))
            episode = self.dataset.episodes[window['episode_id']]
            self.assertNotIn(episode['scene'], [1,5,10,15,16,18,23,*range(170,210)])
        batch = self.dataset.batch(self.schedule[0]['anchor_window'], torch.device('cpu'))
        self.assertEqual(tuple(batch['observation.image'].shape), (1,2,3,96,96))
        self.assertEqual(tuple(batch['observation.state'].shape), (1,2,2))
        self.assertEqual(tuple(batch['action'].shape), (1,16,2))
        self.assertFalse(bool(batch['action_is_pad'].any()))

    def test_validation_keeps_original_64_windows_and_has_no_training_windows(self):
        plan = pilot.old.validation_plan(self.dataset, pilot.SETTINGS)
        previous = pilot.prep.read(pilot.prep.SOURCE / 'training.json')['validation_plan']
        self.assertEqual(plan[:64], previous)
        self.assertEqual(len(plan), 96)
        self.assertTrue(all(self.dataset.windows[r['window']]['split'] == 'validation' for r in plan))

    def test_real_protocol_collects_fourteen_and_reuses_twenty_eight_without_loading_policy(self):
        with patch.object(pilot.adaptive, 'load_policy', side_effect=AssertionError('No real model in CPU preparation')):
            protocol = pilot.protocol_document()
        self.assertEqual(len(protocol['comparison']['schedule']), 14)
        self.assertEqual(len(protocol['comparison']['reference_items']), 28)
        self.assertEqual({r['arm'] for r in protocol['comparison']['schedule']}, {'preserved12'})
        self.assertEqual(protocol['settings']['preservation_weight'], 1.)
        self.assertFalse(protocol['screen']['automatic_adoption'])

    def test_independent_auditor_reads_real_fixed_and_adaptive_reference_queues(self):
        rows = pilot.reference_rows()
        for arm in ('fixed8', 'current12'):
            row = next(r for r in rows if r['item']['arm'] == arm)
            trace = audit.trace_check(row)
            self.assertEqual(len(trace['steps']), row['trial']['steps'])
        state = pilot.training.tail_state(TinyPolicy())
        self.assertEqual(audit.tensor_digest(state), pilot.training.tail_digest(state))


class FitTests(unittest.TestCase):
    def test_toy_fit_preserves_teacher_backbone_and_uses_five_forwards(self):
        with temporary() as folder:
            root = Path(folder); source = root / 'source'; source.write_text('fixed')
            student = TinyPolicy(); before = pilot.training.backbone_digest(student)
            data = ToyDataset(); plan = pilot.sample_schedule(data, settings())
            result = pilot.fit(student, data, settings(), plan, root / 'weights', {str(source): pilot.prep.sha(source)})
            self.assertTrue(result['complete'] and result['teacher_unchanged'])
            self.assertEqual(pilot.training.backbone_digest(student), before)
            self.assertEqual(sum(r['denoiser_forward_calls'] for r in result['history']['updates']), 10)
            self.assertEqual(result['history']['validation'][0]['preservation_mean_loss'], 0.)
            self.assertTrue(all(v['teacher_digest'] == result['teacher_digest'] for v in result['history']['validation']))

    def test_interrupted_fit_resumes_to_identical_weights_objectives_and_rng(self):
        with temporary() as folder:
            root = Path(folder); source = root / 'source'; source.write_text('fixed')
            bindings = {str(source): pilot.prep.sha(source)}
            first = TinyPolicy(); initial = copy.deepcopy(first.state_dict())
            data = ToyDataset(); plan = pilot.sample_schedule(data, settings())
            full = pilot.fit(first, data, settings(), plan, root / 'full', bindings)
            full_rng = torch.get_rng_state().clone()
            second = TinyPolicy(); second.load_state_dict(initial)
            update = pilot.training_update; calls = []
            def interrupt(*args):
                calls.append(1)
                if len(calls) == 2: raise RuntimeError('stop before second update')
                return update(*args)
            with patch.object(pilot, 'training_update', side_effect=interrupt), self.assertRaises(RuntimeError):
                pilot.fit(second, data, settings(), plan, root / 'resumed', bindings)
            resumed = TinyPolicy(); resumed.load_state_dict(initial)
            result = pilot.fit(resumed, data, settings(), plan, root / 'resumed', bindings)
            self.assertEqual(pilot.training.tail_digest(pilot.training.tail_state(first)),
                             pilot.training.tail_digest(pilot.training.tail_state(resumed)))
            for key in ('loss','supervised_loss','preservation_loss'):
                self.assertEqual([r[key] for r in full['history']['updates']], [r[key] for r in result['history']['updates']])
            self.assertTrue(torch.equal(full_rng, torch.get_rng_state()))

    def test_partial_training_artifact_is_preserved(self):
        with temporary() as folder:
            root = Path(folder); target = root / 'weights'; target.mkdir()
            partial = target / 'resume.pt.tmp'; partial.write_bytes(b'partial')
            source = root / 'source'; source.write_text('fixed'); data = ToyDataset()
            with self.assertRaisesRegex(AssertionError, 'Unbound'):
                pilot.fit(TinyPolicy(), data, settings(), pilot.sample_schedule(data, settings()), target, {str(source): pilot.prep.sha(source)})
            self.assertEqual(partial.read_bytes(), b'partial')


def context(folder):
    output = Path(folder) / 'output'; output.mkdir()
    student = TinyPolicy(); original = pilot.training.tail_state(student)
    heads = {'fixed8': original, 'current12': {k:v+1 for k,v in original.items()},
             'preserved12': {k:v+2 for k,v in original.items()}}
    bindings = {'heads': {a: {'tensor_digest': pilot.training.tail_digest(s)} for a,s in heads.items()}}
    pilot.prep.save_once(output / 'eval-bindings.json', bindings)
    protocol = {'comparison': {'scenes': [1], 'draws': [91000],
                'schedule': [{'scene':1,'draw':91000,'arm':'preserved12'}]}}
    return output, student, heads, bindings, protocol


class EvaluationTests(unittest.TestCase):
    def test_candidate_collector_activates_only_candidate_head_and_unchanged_controller(self):
        with temporary() as folder:
            output, student, heads, bindings, protocol = context(folder)
            def evaluate(actual, scene, draw, method, directory, limit):
                self.assertIs(actual, student)
                self.assertEqual((scene,draw,method,limit), (1,91000,'adaptive12',300))
                self.assertEqual(pilot.training.tail_digest(pilot.training.tail_state(actual)), bindings['heads']['preserved12']['tensor_digest'])
                self.assertTrue(all(not p.requires_grad for p in actual.parameters()))
            with patch.object(pilot.gen, 'evaluate_episode', side_effect=evaluate) as call:
                pilot.collect(student, protocol['comparison']['schedule'][0], output, bindings, heads)
            self.assertEqual(call.call_count, 1)

    def test_completed_sidecar_is_not_rerun(self):
        with temporary() as folder:
            output, student, heads, bindings, protocol = context(folder); item = protocol['comparison']['schedule'][0]
            pilot.model_binding(output, item, bindings, create=True)
            directory = pilot.directory_for(output, item)
            pilot.prep.save_once(directory / (pilot.gen.stem(1,91000,'adaptive12') + '.json'), {'saved':True})
            with patch.object(pilot.gen, 'evaluate_episode') as evaluator, patch.object(pilot.prior, 'activate') as activate:
                pilot.collect(student, item, output, bindings, heads)
            evaluator.assert_not_called(); activate.assert_not_called()

    def test_verification_refuses_missing_binding_and_orphan_trace(self):
        with temporary() as folder:
            output, _, _, bindings, protocol = context(folder); item = protocol['comparison']['schedule'][0]
            with self.assertRaises(AssertionError): pilot.model_binding(output, item, bindings)
            pilot.model_binding(output, item, bindings, create=True)
            (pilot.directory_for(output, item) / 'orphan.trace.json.gz').write_bytes(b'partial')
            with self.assertRaisesRegex(AssertionError, 'Orphan'): pilot.scan(output, protocol, bindings)

    def test_statistics_pair_two_draws_within_seven_scenes_and_check_preservation_gate(self):
        scenes = [1,5,10,15,16,18,23]; draws = [91000,92000]
        protocol = {'comparison': {'scenes':scenes,'draws':draws},
                    'statistics': {'seed':20261107,'bootstrap_samples':10000},
                    'screen': {'original_baseline_losses_below':2,'successes_at_least':11,'recovery_successes_at_least':3}}
        rows = []
        for scene in scenes:
            for draw in draws:
                for arm in pilot.ARMS:
                    candidate = arm == 'preserved12'
                    rows.append({'item': {'scene':scene,'draw':draw,'arm':arm}, 'trial': {
                        'initial': {'coverage':0.,'image_sha256':str(scene)}, 'success':candidate or scene in scenes[:2],
                        'final_coverage':.96 if candidate else .9, 'model_queries':10 if candidate else 12,
                        'wall_seconds':50. if candidate else 60., 'monitor_seconds':1.,
                        'prediction_seconds':[5.], 'peak_gpu_allocated_bytes':100}})
        result = pilot.statistics(rows, protocol)
        self.assertEqual(result['methods']['preserved12']['successes'], 14)
        self.assertEqual(result['recovery_validation_successes_of_six']['preserved12'], 6)
        self.assertTrue(result['development_screen_passed'])
        self.assertTrue(all(result['per_draw'][str(d)]['preserved12']['attempts'] == 7 for d in draws))
        comparison = result['comparisons'][2]
        self.assertEqual(len(comparison['gained_successes']), 10)
        self.assertEqual(comparison['lost_successes'], [])
        self.assertAlmostEqual(comparison['differences']['success']['mean_difference'], 5/7)

    def test_previous_outputs_and_models_are_protected(self):
        for output in [pilot.old.OUTPUT, pilot.prep.SOURCE, Path('results')]:
            with self.assertRaises(AssertionError): pilot.check_output(output)
        self.assertNotEqual(pilot.artifacts(pilot.OUTPUT).resolve(),
                            pilot.artifacts(pilot.OUTPUT.with_name(pilot.OUTPUT.name + '-repeat')).resolve())

    def test_success_total_alone_cannot_pass_when_two_baseline_successes_are_lost(self):
        scenes = [1,5,10,15,16,18,23]; draws = [91000,92000]
        protocol = {'comparison': {'scenes':scenes,'draws':draws},
                    'statistics': {'seed':20261107,'bootstrap_samples':10000},
                    'screen': {'original_baseline_losses_below':2,'successes_at_least':11,'recovery_successes_at_least':3}}
        rows = []
        for scene in scenes:
            for draw in draws:
                for arm in pilot.ARMS:
                    success = not (arm == 'preserved12' and scene in (1,5) and draw == 91000)
                    rows.append({'item': {'scene':scene,'draw':draw,'arm':arm}, 'trial': {
                        'initial': {'coverage':0.,'image_sha256':str(scene)}, 'success':success,
                        'final_coverage':.96 if success else .9, 'model_queries':10, 'wall_seconds':50.,
                        'monitor_seconds':1., 'prediction_seconds':[5.], 'peak_gpu_allocated_bytes':100}})
        result = pilot.statistics(rows, protocol)
        self.assertEqual(result['methods']['preserved12']['successes'], 12)
        self.assertFalse(result['development_screen_passed'])
        row = next(r for r in rows if r['item'] == {'scene':1,'draw':91000,'arm':'preserved12'})
        row['trial']['success'] = True; row['trial']['final_coverage'] = .96
        self.assertTrue(pilot.statistics(rows, protocol)['development_screen_passed'])


if __name__ == '__main__': unittest.main()
