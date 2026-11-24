"""Generated-code filenames must not inherit path characters from the plan.

Found on pilot task T07 (B1): the operation "filter invalid/empty
geometries..." became op_filter invalid/empty geometrie_<id>.py, the "/" was
read as a folder, the save failed, and the five steps that depended on that
step were skipped.
"""

import os
import re
import unittest

AGENT_FILES = ("agents/data_agent/agm_data_agent.py",
               "agents/modeling_agent/agm_modeling_agent.py")
SANITIZE = re.compile(r'safe_op = re\.sub\(r"\[\^A-Za-z0-9\._-\]\+", "_", str\(operation_name\)\)')


class GeneratedCodeFilenames(unittest.TestCase):

    def test_both_agents_sanitize_the_operation_name(self):
        for path in AGENT_FILES:
            with self.subTest(path=path):
                src = open(path, encoding="utf-8").read()
                self.assertRegex(src, SANITIZE)
                self.assertNotIn('f"op_{operation_name', src)

    def test_the_t07_operation_name_becomes_a_plain_filename(self):
        name = "filter invalid/empty geometries and complete-case selection"
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:30].strip("._") or "step"
        self.assertEqual(os.path.basename(safe), safe)
        for bad in '/\\:*?"<>| ':
            self.assertNotIn(bad, safe)


if __name__ == "__main__":
    unittest.main()
