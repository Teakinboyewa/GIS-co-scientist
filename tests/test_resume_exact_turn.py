"""Proceed / Provide Feedback resume exactly the turn whose card was clicked.

Seen on PythonAnywhere (conversation 22071797...): after a Spatial Analysis
centroid turn, a buffer turn paused at Task Breakdown and Proceed ran the
Research planner on the centroid question. get_workflow_state returned the
worker's in-memory copy of an earlier state before reading the file another
worker had written, and with no workflow_type in it the resume defaulted to
Research. tests/test_stale_ctx_across_workers.py faked get_workflow_state, so
it never exercised that cache; these tests use the real function on a real file.

Loads the real functions from WebUI/agm_workflow.py (via ast) so the app's
GIS/LLM dependencies are not imported.
"""
import ast
import json
import os
import tempfile
import uuid
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
FUNCS = {"get_workflow_state", "_restore_rq_breakdown_ctx", "_rq_breakdown_get_context",
         "_rq_breakdown_store_context", "_is_current_run", "_paused_breakdown"}


class Worker:
    """One server process: its own in-memory caches, the shared state file."""

    def __init__(self, state_file):
        tree = ast.parse((ROOT / "WebUI/agm_workflow.py").read_text(encoding="utf-8"))
        body = [n for n in tree.body
                if (isinstance(n, ast.FunctionDef) and n.name in FUNCS)
                or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "STALE_BREAKDOWN"
                                                      for t in n.targets))]
        self.ns = {"os": os, "json": json, "print": lambda *a: None,
                   "workflow_states": {}, "_rq_breakdown_contexts": {},
                   "_get_state_file": lambda task_id: state_file,
                   "_cancel_event": object(), "settings": Mock(output_dir=os.path.dirname(state_file))}
        exec(compile(ast.Module(body=body, type_ignores=[]), "agm_workflow.py", "exec"), self.ns)

    def __getattr__(self, name):
        return self.ns[name]


def turn(rq, run_id=None):
    return {"rq": rq, "task_id": "conv1", "output_dir": "x", "logs": [], "settings": {},
            "artifacts": {"rq_breakdown": {"data_requests": None}}, "run_id": run_id or uuid.uuid4().hex}


class ResumeTheClickedTurn(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.file = os.path.join(self.dir.name, "workflow_state.json")
        self.a, self.b = Worker(self.file), Worker(self.file)
        self.centroid = turn("Create centroid across all texas counties")
        self.buffer = turn("Create 100 meters buffer around the centroids")

    def write(self, state):
        with open(self.file, "w", encoding="utf-8") as f:
            json.dump(state, f)

    def paused(self, ctx, workflow_type="spatial"):
        return {"ctx": ctx, "workflow_phase": "rq_breakdown_paused", "workflow_type": workflow_type}

    def test_the_pythonanywhere_case(self):
        # Worker B cached the earlier turn's state (e.g. when the conversation was opened)
        self.write({"ctx": self.centroid})
        self.b.get_workflow_state("conv1")
        # Worker A ran the buffer turn and paused at Task Breakdown
        self.write(self.paused(self.buffer))
        # Proceed reaches worker B
        ctx, _, wtype, error = self.b._paused_breakdown("conv1", self.buffer["run_id"], "spatial")
        self.assertIsNone(error)
        self.assertEqual(ctx["rq"], self.buffer["rq"])
        self.assertEqual(wtype, "spatial")

    def test_a_card_from_an_earlier_turn_is_refused(self):
        self.write(self.paused(self.buffer))
        ctx, _, _, error = self.a._paused_breakdown("conv1", self.centroid["run_id"], "spatial")
        self.assertIsNone(ctx)
        self.assertIn("no longer waiting", error)

    def test_state_that_moved_past_the_breakdown_is_refused_even_without_a_run_id(self):
        # e.g. a plan is now awaiting approval; an old card (no run_id) must not restart it
        self.write({"ctx": self.centroid})
        _, _, _, error = self.a._paused_breakdown("conv1")
        self.assertIn("no longer waiting", error)

    def test_pipeline_comes_from_the_saved_state_not_a_default(self):
        self.write(self.paused(self.buffer, "spatial"))
        _, _, wtype, _ = self.a._paused_breakdown("conv1", self.buffer["run_id"])
        self.assertEqual(wtype, "spatial")
        _, _, _, error = self.a._paused_breakdown("conv1", self.buffer["run_id"], "research")
        self.assertIn("no longer waiting", error)

    def test_research_pause_saved_before_pipelines_were_recorded(self):
        self.write({"ctx": self.buffer, "workflow_phase": "rq_breakdown_paused"})
        _, _, wtype, error = self.a._paused_breakdown("conv1", self.buffer["run_id"])
        self.assertIsNone(error)
        self.assertEqual(wtype, "research")

    def test_saved_file_wins_over_the_worker_cache(self):
        self.write({"ctx": self.centroid})
        self.b.get_workflow_state("conv1")
        self.write(self.paused(self.buffer))
        self.assertEqual(self.b.get_workflow_state("conv1")["ctx"]["rq"], self.buffer["rq"])

    def test_cache_is_used_only_when_the_file_is_gone(self):
        self.write(self.paused(self.buffer))
        self.a.get_workflow_state("conv1")
        os.remove(self.file)
        self.assertEqual(self.a.get_workflow_state("conv1")["ctx"]["rq"], self.buffer["rq"])


class FrontendSendsTheTurn(unittest.TestCase):
    """The card keeps its turn and sends it; app.py passes it on and saves it with the card."""

    def test_wiring(self):
        js = (ROOT / "WebUI/script.js").read_text(encoding="utf-8")
        app = (ROOT / "WebUI/app.py").read_text(encoding="utf-8")
        wf = (ROOT / "WebUI/agm_workflow.py").read_text(encoding="utf-8")
        self.assertIn("card.dataset.rqRunId = options.runId", js)
        self.assertEqual(js.count("run_id: card"), 3)                 # Proceed + both Feedback paths
        self.assertIn("workflow_type: card.dataset.rqWorkflowType", js)
        self.assertIn("run_id=card_run_id, workflow_type=card_workflow_type", app)
        self.assertIn("revise_rq_breakdown_stream(task_id, feedback, run_id=card_run_id)", app)
        self.assertIn("'run_id': (update.get('data') or {}).get('run_id')", app)
        self.assertEqual(wf.count('"run_id": ctx.get("run_id"), "workflow_type"'), 4)
        self.assertNotIn('saved_state.get("workflow_type", "research")', wf)


if __name__ == "__main__":
    unittest.main()
