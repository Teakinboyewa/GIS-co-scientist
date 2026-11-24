"""Post-hoc actions (feedback, step delete, re-run, ...) must not fail on plumbing.

Found by exercising every post-hoc endpoint with a stubbed LLM:
- Deleting a step from a research plan always returned 500: the repair
  function imported utils.gibd_llm (no such module), langchain.schema (gone in
  LangChain 1.x) and utils.helper (no such module). Function-level imports
  only fail when the function runs, so nothing caught them.
- When step routing could not use the LLM's answer it fell back to the first
  registered agent, the Research Question Understanding Agent, which cannot
  execute steps, so every step failed with an AttributeError.
"""
import ast
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ("utils", "agents", "WebUI")


def _sources():
    for d in LOCAL:
        for p in (ROOT / d).rglob("*.py"):
            if "__pycache__" not in p.parts and not p.name.endswith(("_v1.py", "_v2.py")):
                yield p


def _module_file(mod):
    base = ROOT.joinpath(*mod.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.exists():
            return cand
    return None


def _top_level_names(path):
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8", errors="replace")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.Try, ast.If)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    names.update((a.asname or a.name).split(".")[0] for a in sub.names)
                elif isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                    names.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    names.update(n.id for t in sub.targets for n in ast.walk(t) if isinstance(n, ast.Name))
    return names


def _guarded(tree, node):
    """True when the import sits inside a try block (an intentional optional import)."""
    for t in ast.walk(tree):
        if isinstance(t, ast.Try) and any(node is n for b in t.body for n in ast.walk(b)):
            return True
    return False


class ProjectImportsResolve(unittest.TestCase):
    def test_every_import_of_a_project_module_resolves(self):
        problems = []
        for path in _sources():
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                        and node.module.split(".")[0] in LOCAL) or _guarded(tree, node):
                    continue
                where = f"{path.relative_to(ROOT)}:{node.lineno}"
                target = _module_file(node.module)
                if target is None and ROOT.joinpath(*node.module.split(".")).is_dir():
                    names = set()          # namespace package (no __init__.py): names are submodules
                elif target is None:
                    problems.append(f"{where} from {node.module}: no such module")
                    continue
                else:
                    names = _top_level_names(target)
                for a in node.names:
                    if a.name != "*" and a.name not in names and _module_file(f"{node.module}.{a.name}") is None:
                        problems.append(f"{where} from {node.module} import {a.name}: not defined there")
        self.assertEqual(problems, [])

    def test_no_langchain_schema(self):
        # langchain.schema was removed in LangChain 1.x; use langchain_core.messages.
        found = [f"{p.relative_to(ROOT)}:{n.lineno}" for p in _sources()
                 for n in ast.walk(ast.parse(p.read_text(encoding="utf-8", errors="replace")))
                 if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("langchain.schema")]
        self.assertEqual(found, [])


class RoutingFallback(unittest.TestCase):
    def test_fallback_is_an_agent_that_executes_steps(self):
        from agents.task_manager import objective_executor as oe

        class Runner:
            def handle_generate_and_execute_code_for_simple_operation(self, *a):
                pass

        registry = {"RQUnderstandingAgent": {"instance": object()},
                    "DataAgent": {"instance": Runner()},
                    "ModelingAgent": {"instance": Runner()}}
        with patch.object(oe.task_manager, "agent_registry", registry):
            self.assertEqual(oe._fallback_agent(), "ModelingAgent")
        registry.pop("ModelingAgent")
        with patch.object(oe.task_manager, "agent_registry", registry):
            self.assertEqual(oe._fallback_agent(), "DataAgent")


if __name__ == "__main__":
    unittest.main()
