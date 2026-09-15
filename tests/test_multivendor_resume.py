"""Resume, archive and freeze guarantees for the three-vendor runner.

These tests use a synthetic directory and make no API calls. They exist because the expensive
property to protect is "an interrupted or failed round must not redo the completed work, and must
not silently mix epochs".
"""
import json, os, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_multivendor_formal as R


def job(key, provider='glm', **extra):
    j = {'id': 'Who&When/Algorithm-Generated/1.json', 'provider': provider, 'policy': 'full',
         'budget': None, 'repeat': 0, 'model': 'm'}
    j.update(extra)
    j['key'] = key
    return j


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        (self.out / 'records').mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, key, status, **extra):
        row = job(key, **extra); row['status'] = status
        (self.out / 'records' / (key + '.json')).write_text(
            json.dumps(row, ensure_ascii=False), encoding='utf-8')

    # --- what gets re-run -------------------------------------------------

    def test_completed_records_are_skipped_and_missing_ones_are_run(self):
        self.write('a', 'completed')
        jobs = [job('a'), job('b')]
        self.assertEqual([j['key'] for j in R.pending_jobs(self.out, jobs)], ['b'])

    def test_all_non_completed_statuses_are_scheduled_again(self):
        statuses = ['truncated', 'empty_answer', 'parse_error', 'api_failure',
                    'request_error', 'missing_key', 'aborted']
        for i, s in enumerate(statuses):
            self.write('k%d' % i, s)
        jobs = [job('k%d' % i) for i in range(len(statuses))]
        self.assertEqual(len(R.pending_jobs(self.out, jobs)), len(statuses))

    def test_a_successful_but_wrong_prediction_is_not_retried(self):
        # A wrong answer is a result, not a failure. Re-running it would bias accuracy.
        self.write('a', 'completed', correct=False, prediction={'step': 3, 'agent': 'x'})
        self.assertEqual(R.pending_jobs(self.out, [job('a')]), [])

    def test_partial_progress_resumes_without_redoing_finished_work(self):
        # 10 jobs, 4 already done -> exactly the other 6 are scheduled, in order.
        for i in range(4):
            self.write('k%d' % i, 'completed')
        jobs = [job('k%d' % i) for i in range(10)]
        self.assertEqual([j['key'] for j in R.pending_jobs(self.out, jobs)],
                         ['k4', 'k5', 'k6', 'k7', 'k8', 'k9'])

    # --- archiving --------------------------------------------------------

    def test_archive_moves_only_failures_and_keeps_successes(self):
        self.write('good', 'completed')
        self.write('bad', 'truncated')
        self.write('alsobad', 'api_failure')
        moved = R.archive_failed(self.out)
        self.assertEqual(moved, 2)
        self.assertTrue((self.out / 'records' / 'good.json').exists())
        self.assertFalse((self.out / 'records' / 'bad.json').exists())
        archived = list((self.out / 'failed_archive').rglob('*.json'))
        self.assertEqual(len(archived), 2, 'failed attempts must be preserved, not deleted')

    def test_archiving_preserves_the_prior_attempt_for_audit(self):
        self.write('bad', 'truncated', error='finish_reason=length')
        R.archive_failed(self.out)
        kept = json.loads(next((self.out / 'failed_archive').rglob('*.json')).read_text(encoding='utf-8'))
        self.assertEqual(kept['status'], 'truncated')
        self.assertEqual(kept['error'], 'finish_reason=length')

    def test_second_archive_of_the_same_directory_is_safe(self):
        self.write('bad', 'truncated')
        R.archive_failed(self.out)
        self.assertEqual(R.archive_failed(self.out), 0, 'must not re-archive or lose records')

    # --- frozen configuration --------------------------------------------

    def _fp(self, split='test'):
        import argparse
        inputs = self.out / 'synthetic_inputs.json'
        inputs.write_text('[]', encoding='utf-8')
        return R.fingerprint(argparse.Namespace(split=split),
                             inputs)

    def test_freeze_check_accepts_an_unchanged_configuration(self):
        (self.out / 'manifest.json').write_text(
            json.dumps({'fingerprint': self._fp()}), encoding='utf-8')
        R.check_freeze(self.out, self._fp(), allow=False)  # must not raise

    def test_freeze_check_refuses_when_the_protocol_moved(self):
        stale = self._fp(); stale['parser_version'] = 'mv-0.0'
        (self.out / 'manifest.json').write_text(
            json.dumps({'fingerprint': stale}), encoding='utf-8')
        with self.assertRaises(SystemExit):
            R.check_freeze(self.out, self._fp(), allow=False)

    def test_freeze_check_can_be_overridden_explicitly(self):
        stale = self._fp(); stale['split'] = 'development'
        (self.out / 'manifest.json').write_text(
            json.dumps({'fingerprint': stale}), encoding='utf-8')
        R.check_freeze(self.out, self._fp(), allow=True)  # must not raise

    # --- concurrency ------------------------------------------------------

    def test_lock_refuses_a_second_runner_in_the_same_directory(self):
        (self.out / 'run.lock').write_text(
            json.dumps({'pid': os.getpid(), 'started': 'now'}), encoding='utf-8')
        with self.assertRaises(SystemExit):
            R.acquire_lock(self.out)

    def test_lock_is_taken_over_when_the_recorded_process_is_gone(self):
        (self.out / 'run.lock').write_text(
            json.dumps({'pid': 999999999, 'started': 'long ago'}), encoding='utf-8')
        lock = R.acquire_lock(self.out)
        self.assertTrue(lock.exists())

    def test_lock_is_released_after_a_run(self):
        lock = R.acquire_lock(self.out)
        R.release_lock(lock)
        self.assertFalse((self.out / 'run.lock').exists())


if __name__ == '__main__':
    unittest.main()
