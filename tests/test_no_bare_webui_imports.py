"""Project code must import WebUI modules through the WebUI package.

PythonAnywhere loads the app with pythonanywhere_wsgi.py, which puts only the
project root on sys.path. `from agm_workflow import _active_reruns` in the
re-run analysis endpoint therefore raised ModuleNotFoundError there, and the
button returned "rerun-analysis HTTP 500". Locally it worked, because
`python WebUI/app.py` puts WebUI/ on sys.path, but it loaded a second copy of
agm_workflow whose _active_reruns set the re-runs never registered in.

Imports inside a try block are optional fallbacks and are allowed. Standalone
command-line scripts (a __main__ block, imported by nothing) set up their own
sys.path and are not checked.
"""
import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBUI_MODULES = {p.stem for p in (ROOT / "WebUI").glob("*.py")}
SOURCES = [p for d in ("WebUI", "agents", "utils") for p in (ROOT / d).rglob("*.py")
           if "__pycache__" not in p.parts and not p.name.endswith(("_v1.py", "_v2.py"))]
ALL_TEXT = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in SOURCES + list(ROOT.glob("*.py")))


def _standalone_script(path, src):
    if not re.search(r"__name__\s*==\s*['\"]__main__['\"]", src):
        return False
    return not re.search(rf"\bimport\s+{path.stem}\b|\b{path.stem}\s+import\b|\.{path.stem}\b", ALL_TEXT)


def _in_try(tree, node):
    return any(isinstance(t, ast.Try) and any(node is n for b in t.body for n in ast.walk(b))
               for t in ast.walk(tree))


class NoBareWebUIImports(unittest.TestCase):
    def test_webui_modules_imported_through_the_package(self):
        problems = []
        for path in SOURCES:
            src = path.read_text(encoding="utf-8", errors="replace")
            if _standalone_script(path, src):
                continue
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                else:
                    continue
                for name in names:
                    if name.split(".")[0] in WEBUI_MODULES and not _in_try(tree, node):
                        problems.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}")
        self.assertEqual(problems, [])

    def test_wsgi_puts_only_the_project_root_on_the_path(self):
        # The premise of the test above: if this changes, revisit it.
        wsgi = (ROOT / "pythonanywhere_wsgi.py").read_text(encoding="utf-8")
        self.assertIn("from WebUI.app import app as application", wsgi)
        self.assertNotIn("'WebUI'", wsgi)


if __name__ == "__main__":
    unittest.main()
