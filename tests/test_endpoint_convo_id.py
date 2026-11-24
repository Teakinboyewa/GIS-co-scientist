"""Every app.py endpoint that uses convo_id defines it.

The geoprocessing-workflow feedback endpoint passed convo_id to
track_followup_usage without ever setting it, so every Spatial Analysis plan
feedback failed with "name 'convo_id' is not defined" (added in cbe2e61).
Python only reports this when the line runs, so a static check is the
cheapest guard.
"""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _assigned_names(fn):
    names = {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
    for node in ast.walk(fn):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.With)):
            targets = [getattr(node, "target", None)] + [i.optional_vars for i in getattr(node, "items", [])]
        for t in targets:
            for n in ast.walk(t) if t is not None else []:
                if isinstance(n, ast.Name):
                    names.add(n.id)
    return names


class EndpointsDefineConvoId(unittest.TestCase):
    def test_no_endpoint_uses_convo_id_without_setting_it(self):
        tree = ast.parse((ROOT / "WebUI/app.py").read_text(encoding="utf-8"))
        module_names = {t.id for n in tree.body if isinstance(n, ast.Assign)
                        for t in n.targets if isinstance(t, ast.Name)}
        missing = []
        for fn in tree.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            if "convo_id" in used and "convo_id" not in _assigned_names(fn) | module_names:
                missing.append(f"{fn.name} (line {fn.lineno})")
        self.assertEqual(missing, [], "these functions read convo_id but never set it")

    def test_geoprocessing_feedback_endpoint_sets_it(self):
        tree = ast.parse((ROOT / "WebUI/app.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "revise_geoprocessing_workflow_endpoint")
        self.assertIn("convo_id", _assigned_names(fn))


if __name__ == "__main__":
    unittest.main()
