"""Sequential selection, complete cycles, and cancellation without any inference."""
import itertools
import json
import unittest
from unittest.mock import patch

import app
import randomtest


def make_plan(cases, **kwargs):
    args = dict(rounds=5, cases=cases, readers=['reader'], extractors=[''],
                text_models=['reader'], details=['low', 'medium'],
                modes=['single', 'agentic'], seed=7, strategy='sequential')
    args.update(kwargs)
    return randomtest.plan(**args)


class SequentialPlanTests(unittest.TestCase):
    def test_whole_selected_corpus_in_stable_order_even_above_fifty(self):
        cases = [f'sol{i:03}' for i in range(100, 0, -1)]
        plan = make_plan(cases, rounds=1, history={cases[0]: 900})
        self.assertEqual([r['case'] for r in plan['rounds']], sorted(cases))
        self.assertTrue(plan['continuous'])

    def test_exclusions_remove_files_from_every_cycle(self):
        pools = randomtest.pools([{'name': 'reader', 'vision': True}], ['c', 'b', 'a'])
        selected = randomtest.apply_exclusions(pools, {'cases': ['b']})
        plan = make_plan(selected['cases'])
        rounds = list(itertools.islice(randomtest.iter_rounds(plan), 7))
        self.assertEqual([r['case'] for r in rounds], ['a', 'c', 'a', 'c', 'a', 'c', 'a'])
        self.assertEqual([r['cycle'] for r in rounds], [1, 1, 2, 2, 3, 3, 4])
        self.assertEqual([r['position'] for r in rounds], [1, 2, 1, 2, 1, 2, 1])

    def test_empty_selection_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'Every document is excluded'):
            randomtest.apply_exclusions({'cases': ['a']}, {'cases': ['a']})

    def test_one_file_repeats_and_keeps_model_settings(self):
        plan = make_plan(['a'], history={'a': 10}, lock={'reader': 'reader', 'mode': 'single'})
        first, second = itertools.islice(randomtest.iter_rounds(plan), 2)
        self.assertEqual(first['case'], second['case'])
        self.assertEqual(second['seen'], 11)
        for key in ('reader', 'detail', 'extractor', 'mode'):
            self.assertEqual(first[key], second[key])
        self.assertNotIn('cycle', plan['rounds'][0])
        self.assertEqual(plan['rounds'][0]['seen'], 10)

    def test_supported_scopes_and_locks(self):
        for scope in randomtest.SCOPES:
            with self.subTest(scope=scope):
                plan = make_plan(['b', 'a'], scope=scope, lock={'case': 'b'})
                self.assertEqual([r['case'] for r in plan['rounds']], ['b'])
                self.assertEqual(plan, make_plan(['b', 'a'], scope=scope, lock={'case': 'b'}))

    def test_existing_strategies_stay_finite(self):
        for strategy in ('balanced', 'uniform'):
            plan = make_plan(['a', 'b'], strategy=strategy, rounds=8)
            self.assertFalse(plan['continuous'])
            self.assertEqual(list(randomtest.iter_rounds(plan)), plan['rounds'])
            self.assertEqual(len(plan['rounds']), 8)


class SequentialRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()
        self.pool = dict(cases=['c', 'a', 'b'], readers=['reader'],
                         extractors=[''], text_models=['reader'])
        self.patches = [
            patch.object(app, '_random_pools', return_value=self.pool),
            patch.object(app.runlog, 'case_counts', return_value={}),
            patch.object(app.randomtest, 'summarise_round', side_effect=lambda r: r),
            # Any unexpected model/server traffic must fail instead of running.
            patch('requests.sessions.Session.send', side_effect=AssertionError('No network in tests')),
        ]
        for mocked in self.patches:
            mocked.start()
            self.addCleanup(mocked.stop)
        self.addCleanup(app.end_sweep)

    def test_preview_respects_file_selection_and_rejects_empty_selection(self):
        body = dict(strategy='sequential', scope='ocr', exclude={'cases': ['b']})
        response = self.client.post('/api/randomtest', json=body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual([r['case'] for r in response.json['rounds']], ['a', 'c'])
        body['exclude']['cases'] = ['a', 'b', 'c']
        self.assertEqual(self.client.post('/api/randomtest', json=body).status_code, 400)

    def test_full_cycle_above_fifty_then_wraps_and_stops(self):
        self.pool['cases'] = [f'sol{i:03}' for i in range(100, 0, -1)]
        ran = []

        def fake_round(round_, cancel):
            ran.append(round_['case'])
            if len(ran) == 205:
                self.client.post('/api/randomtest/stop')
            return {'status': 'ok'}

        with patch.object(app, '_run_round', side_effect=fake_round):
            response = self.client.post('/api/randomtest/stream', json={'strategy': 'sequential'})
            events = [json.loads(line) for line in response.get_data(as_text=True).splitlines()]
        expected = sorted(self.pool['cases'])
        self.assertEqual(ran, expected * 2 + expected[:5])
        starts = [e for e in events if e['event'] == 'round_start']
        self.assertEqual(starts[-1]['plan']['cycle'], 3)
        self.assertEqual(starts[-1]['plan']['position'], 5)
        self.assertEqual(events[-1]['completed'], 205)
        self.assertEqual(events[-1]['failed'], 0)
        self.assertIsNone(events[-1]['total'])
        self.assertTrue(events[-1]['stopped'])
        self.assertFalse(app.cancel_sweep())

    def test_a_failed_file_does_not_stop_the_sequence(self):
        ran = []

        def fake_round(round_, cancel):
            ran.append(round_['case'])
            if len(ran) == 1:
                raise OSError('fixture failure')
            if len(ran) == 5:
                cancel.set()
            return {'status': 'ok'}

        with patch.object(app, '_run_round', side_effect=fake_round):
            response = self.client.post('/api/randomtest/stream',
                                        json={'strategy': 'sequential', 'exclude': {'cases': ['b']}})
            events = [json.loads(line) for line in response.get_data(as_text=True).splitlines()]
        self.assertEqual(ran, ['a', 'c', 'a', 'c', 'a'])
        self.assertEqual(events[-1]['failed'], 1)
        self.assertEqual(events[-1]['completed'], 4)

    def test_cancel_during_round_is_not_a_failure(self):
        def fake_round(round_, cancel):
            cancel.set()
            raise app.SweepCancelled('stopped')

        with patch.object(app, '_run_round', side_effect=fake_round):
            response = self.client.post('/api/randomtest/stream', json={'strategy': 'sequential'})
            events = [json.loads(line) for line in response.get_data(as_text=True).splitlines()]
        self.assertTrue(events[-1]['stopped'])
        self.assertEqual(events[-1]['failed'], 0)
        self.assertEqual(events[-1]['completed'], 0)

    def test_closing_the_stream_releases_the_sweep(self):
        with patch.object(app, '_run_round', side_effect=AssertionError('No round expected')):
            response = self.client.post('/api/randomtest/stream',
                                        json={'strategy': 'sequential'}, buffered=False)
            response.close()
        self.assertFalse(app.cancel_sweep())


if __name__ == '__main__':
    unittest.main()
