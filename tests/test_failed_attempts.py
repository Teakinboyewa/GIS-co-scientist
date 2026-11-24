"""Results of a crashed attempt must not reach the manuscript.

Run dd5922b8: the first attempt wrote tables/global_moransI_pm25_and_ols_residuals.csv
and registered it, then crashed. The attempt that succeeded never wrote that
file, yet the manuscript reported it as Table 6 -- a residual Moran's I from a
model fit that is not the one the paper describes.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock


class TheCrashedAttemptsFilesLeaveTheResults(unittest.TestCase):

    def setUp(self):
        self.out = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.out, 'tables'))
        self.earlier = os.path.join(self.out, 'merged.gpkg')     # an earlier step's output
        open(self.earlier, 'w').write('good')

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def _new(self, rel, text='x'):
        p = os.path.join(self.out, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'w').write(text)
        return p

    def test_new_files_move_and_earlier_outputs_stay(self):
        from utils.failed_attempts import quarantine_new_files
        before = {self.earlier}
        table = self._new('tables/morans_i.csv')
        open(self.earlier, 'w').write('overwritten by the crash')  # modified, not new
        moved = quarantine_new_files(self.out, before, [table, self.earlier], 'ValueError: boom')
        self.assertFalse(os.path.exists(table))
        self.assertTrue(os.path.exists(self.earlier), 'an earlier step\'s file must never be moved')
        self.assertEqual(len(moved), 1)
        self.assertTrue(moved[0].replace('\\', '/').endswith('tables/morans_i.csv'))
        self.assertIn('_failed_attempts', moved[0])
        err = os.path.join(os.path.dirname(os.path.dirname(moved[0])), 'ERROR.txt')
        self.assertEqual(open(err).read(), 'ValueError: boom')

    def test_bookkeeping_is_left_alone(self):
        from utils.failed_attempts import quarantine_new_files
        files = [self._new('manifest.json', '{}'), self._new('generated_code/op_1.py')]
        self.assertEqual(quarantine_new_files(self.out, set(), files), [])
        self.assertTrue(all(os.path.exists(f) for f in files))

    def test_quarantined_table_drops_out_of_the_figure_catalog(self):
        """The Table 6 failure, end to end."""
        from agents.result_analysis_agent import figure_catalog_v2 as fc
        from utils.failed_attempts import quarantine_new_files
        kept = self._new('tables/ols_coefficients.csv', 'a,b\n1,2\n')
        fc.record_table(output_dir=self.out, file_path=kept, caption='OLS', role='promoted')
        stale = self._new('tables/global_moransI.csv', 'a,b\n1,2\n')
        fc.record_table(output_dir=self.out, file_path=stale, caption='Moran', role='promoted')
        quarantine_new_files(self.out, {kept, self.earlier}, [stale], 'KeyError')
        with mock.patch('utils.config.resolve_output_dir', lambda ctx: self.out):
            paths = [os.path.basename(e['path']) for e in fc.build_catalog({})]
        self.assertIn('ols_coefficients.csv', paths)
        self.assertNotIn('global_moransI.csv', paths)

    def test_single_agent_failed_script_is_quarantined_and_reported(self):
        from agents.single_agent.single_agent import execute_python
        ctx = {'output_dir': self.out}
        code = ("import os\n"
                "open(os.path.join(OUTPUT_DIR, 'tables', 'partial.csv'), 'w').write('a\\n1\\n')\n"
                "raise KeyError('ci')\n")
        r = execute_python(ctx, code)
        self.assertEqual(r['status'], 'error')
        self.assertEqual(r['created_files'], [])
        self.assertFalse(os.path.exists(os.path.join(self.out, 'tables', 'partial.csv')))
        self.assertEqual(len(r['quarantined_files']), 1)

    def test_every_experiment_executor_quarantines(self):
        for path in ['agents/modeling_agent/agm_modeling_agent.py',
                     'agents/data_agent/agm_data_agent.py',
                     'agents/task_manager/debugger.py',
                     'utils/agm_helper.py',
                     'agents/single_agent/single_agent.py']:
            self.assertIn('quarantine_new_files(', open(path, encoding='utf-8').read(), path)


if __name__ == '__main__':
    unittest.main()
