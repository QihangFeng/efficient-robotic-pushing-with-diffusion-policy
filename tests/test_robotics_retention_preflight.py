"""No model updates: whole-scene splitting and exact sample provenance."""

import ast
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import robotics_retention_preflight as preflight


class SplitAndScheduleTests(unittest.TestCase):
    def test_split_is_order_independent_and_does_not_split_same_scene(self):
        scenes=[0,1,5,8,10,12,13,14,17,19,20,21,22,23,24,25,26,27,28,29]
        first=preflight.split_scenes(scenes)
        self.assertEqual(first,preflight.split_scenes(list(reversed(scenes))))
        self.assertEqual(len(first['train']),16)
        self.assertEqual(len(first['validation']),4)
        self.assertFalse(set(first['train'])&set(first['validation']))

    def test_duplicate_scene_cannot_receive_two_split_assignments(self):
        with self.assertRaises(AssertionError):
            preflight.split_scenes([0]*20)

    def test_native_h16_windows_have_two_observations_and_no_padding(self):
        episode={'episode_id':'baseline-0','n_actions':17,'n_frames':18,'split':'train'}
        rows=preflight.windows(episode)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['current_observation_index'],1)
        self.assertEqual(rows[0]['action_start_index'],0)
        self.assertEqual(rows[-1]['action_stop_index'],17)
        self.assertTrue(all(w['action_stop_index']-w['action_start_index']==16 for w in rows))

    def test_schedule_contains_every_scene_draw_arm_once_with_rotated_order(self):
        scenes=[1,5,10,15,16,18,23]
        rows=preflight.schedule(scenes)
        self.assertEqual(len(rows),42)
        self.assertEqual(len({(r['scene'],r['draw'],r['arm']) for r in rows}),42)
        self.assertNotEqual([r['arm'] for r in rows[:3]],[r['arm'] for r in rows[3:6]])

    def test_no_policy_or_simulator_library_is_imported(self):
        tree=ast.parse(Path(preflight.__file__).read_text())
        modules=[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        modules += [n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(set(modules)&{'torch','gymnasium','gym_pusht','lerobot'})


class PreservationTests(unittest.TestCase):
    def test_source_results_are_protected(self):
        with self.assertRaises(AssertionError):
            preflight.run(preflight.SOURCE)

    def test_existing_preparation_cannot_be_overwritten(self):
        root=Path(__file__).resolve().parents[1]/'tmp'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            path=Path(folder)/'data.json'
            preflight.save_once(path,{'value':1})
            before=path.read_bytes()
            preflight.save_once(path,{'value':1})
            with self.assertRaises(AssertionError): preflight.save_once(path,{'value':2})
            self.assertEqual(path.read_bytes(),before)


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.plan,cls.audit=preflight.build()

    def test_every_original_control_is_verified_and_split(self):
        self.assertEqual(self.audit['baseline_success_episodes'],20)
        self.assertEqual(self.audit['complete_windows'],2686)
        self.assertEqual(len(self.data['episodes']),20)
        self.assertEqual(sum(self.data['window_counts'].values()),2686)
        self.assertTrue(all(e['original_split']=='diagnostic' for e in self.data['episodes']))
        original=preflight.read(preflight.SOURCE/'dataset.json')
        self.assertFalse(original['ordinary_controls']['used_for_training'])

    def test_proposal_has_no_test_scenes_and_equal_training_budget(self):
        self.assertFalse(set(self.plan['comparison']['scenes'])&set(range(170,210)))
        self.assertEqual(self.plan['training']['optimizer_steps'],600)
        self.assertEqual(len(self.plan['training']['microbatch_groups']),4)
        self.assertEqual(self.plan['training']['microbatch_groups'].count('recovery'),2)
        self.assertEqual(self.audit['gradients_policy_predictions_and_physics_calls'],0)

    def test_validation_is_disjoint_for_every_complete_window(self):
        scenes={e['episode_id']:e['scene'] for e in self.data['episodes']}
        train={scenes[w['episode_id']] for w in self.data['windows'] if w['split']=='train'}
        validation={scenes[w['episode_id']] for w in self.data['windows'] if w['split']=='validation'}
        self.assertFalse(train&validation)
        self.assertEqual(validation,set(self.data['scene_splits']['validation']))


if __name__=='__main__': unittest.main()
