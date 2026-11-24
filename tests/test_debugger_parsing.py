"""The debugger's repaired script must be the script, not a quoted snippet.

Run dd5922b8: the model wrote inline <code>...</code> in its diagnosis, the
parser took the first <code> match, and a one-line fragment was executed as the
"fixed" program (NameError: gpd). A correct repair was thrown away and the step
only survived through the regenerate-from-scratch fallback.

Also pins the executor contract the debugger depends on: the figure-catalog
helpers resolve when repaired code runs (run 7c29e713 lost a correct repair to
NameError: record_table).
"""

import os
import shutil
import tempfile
import unittest

SCRIPT = """import geopandas as gpd
gdf = gpd.GeoDataFrame({'a': [1]})
print(len(gdf))"""

INLINE_FIRST = f"""<diagnosis>
  <failed_line>Line 12: <code>sp_gdf = gpd.GeoDataFrame(sp_df, geometry="geometry")</code></failed_line>
  <error_type>ValueError</error_type>
  <root_cause>The frame used to create <code>sp_gdf</code> has no <code>geometry</code> column.</root_cause>
</diagnosis>
<fix>
  <description>Restore the geometry column before building <code>sp_gdf</code>.</description>
  <code>
```python
{SCRIPT}
```
  </code>
  <changed_lines>12</changed_lines>
</fix>
<validation><check>runs</check><risk>None</risk></validation>"""


class TheRepairIsTheWholeScript(unittest.TestCase):

    def test_inline_code_in_the_diagnosis_is_not_taken_as_the_fix(self):
        from agents.task_manager.debugger import _parse_xml_debug_response
        parsed = _parse_xml_debug_response(INLINE_FIRST)
        self.assertEqual(parsed['fixed_code'], SCRIPT)
        self.assertIn('ValueError', parsed['error_type'])

    def test_plain_response_still_parses(self):
        from agents.task_manager.debugger import _parse_xml_debug_response
        content = f"<fix><description>d</description><code>\n{SCRIPT}\n</code></fix>"
        self.assertEqual(_parse_xml_debug_response(content)['fixed_code'], SCRIPT)

    def test_fenced_answer_without_tags_still_parses(self):
        from agents.task_manager.debugger import _parse_xml_debug_response
        self.assertEqual(_parse_xml_debug_response(f"Fixed:\n```python\n{SCRIPT}\n```")['fixed_code'], SCRIPT)


class RepairedCodeCanRegisterFigures(unittest.TestCase):

    def test_record_helpers_resolve_in_the_repair_executor(self):
        from unittest import mock
        import agents.task_manager.debugger as dbg
        out = tempfile.mkdtemp()
        try:
            # The output-path guard only allows Outputs/; it is not under test here.
            with mock.patch.object(dbg, 'install_output_path_guard',
                                   lambda *a: __import__('contextlib').nullcontext()):
                r = dbg._execute_fixed_code(
                    "print(callable(record_figure), callable(record_table), callable(record_data_to_viz))",
                    output_dir=out)
            self.assertEqual(r['status'], 'completed', r.get('error'))
            self.assertIn('True True True', r['output'])
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
