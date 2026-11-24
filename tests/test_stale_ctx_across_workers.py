"""A resumed turn must not pick up a cached ctx from an earlier turn.

Seen on PythonAnywhere: obesity data was downloaded by a Data Retriever turn,
then a Spatial Analysis turn in the same conversation was judged ready with
the data available, yet Proceed downloaded the obesity data again. The ctx
caches are per process and keyed by conversation id; the Proceed request
reached a worker still holding the Data Retriever turn's ctx, and the resume
preferred that cached ctx over the newer saved state.

Loads the real functions from WebUI/agm_workflow.py (via ast) so the app's
GIS/LLM dependencies are not imported.
"""
import ast
import os
import uuid
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"_is_current_run", "_plan_context", "_restore_rq_breakdown_ctx",
         "_rq_breakdown_get_context", "_rq_breakdown_store_context", "init_workflow_context"}


def _load(disk, memory, plan_memory):
    tree = ast.parse((ROOT / "WebUI/agm_workflow.py").read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    assert {n.name for n in body} == NAMES
    scope = {
        "os": os, "uuid": uuid, "print": lambda *a: None,
        "_rq_breakdown_contexts": memory,
        "get_workflow_state": lambda task_id: disk.get(task_id),
        "_cancel_event": object(),
        "settings": Mock(output_dir="/tmp"),
        "research_plan_generator": Mock(get_context=lambda t: plan_memory.get(t),
                                        store_context=lambda t, c: plan_memory.__setitem__(t, c)),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "agm_workflow.py", "exec"), scope)
    return scope


def _turn(rq, requests):
    """A ctx as one turn of conversation conv1 leaves it at the Task Breakdown pause."""
    return {"rq": rq, "task_id": "conv1", "output_dir": "/tmp/conv1", "logs": [], "settings": {},
            "artifacts": {"rq_breakdown": {"data_requests": requests}}, "run_id": uuid.uuid4().hex}


class CachedContextFromAnEarlierTurn(unittest.TestCase):
    def setUp(self):
        self.disk, self.memory, self.plan_memory = {}, {}, {}
        self.s = _load(self.disk, self.memory, self.plan_memory)
        # Worker A handled the Data Retriever turn and still caches its ctx.
        self.old = _turn("Download obesity data for SC", ["Adult obesity prevalence, SC, 2023"])
        self.memory["conv1"] = self.old
        self.plan_memory["conv1"] = self.old
        # Worker B handled the newer Spatial Analysis turn and saved it to disk.
        self.new = _turn("Create county centroids", None)
        self.disk["conv1"] = {"ctx": dict(self.new), "workflow_phase": "rq_breakdown_paused"}

    def test_resume_on_the_stale_worker_uses_the_latest_turn(self):
        ctx, _ = self.s["_restore_rq_breakdown_ctx"]("conv1")
        self.assertEqual(ctx["run_id"], self.new["run_id"])
        self.assertIsNone(ctx["artifacts"]["rq_breakdown"]["data_requests"])
        # and the stale cache is replaced, so later calls on this worker agree
        self.assertEqual(self.memory["conv1"]["run_id"], self.new["run_id"])

    def test_resume_on_the_current_worker_keeps_the_live_ctx(self):
        self.memory["conv1"] = self.new          # same turn: the live object is preferred
        ctx, _ = self.s["_restore_rq_breakdown_ctx"]("conv1")
        self.assertIs(ctx, self.new)

    def test_saved_state_without_a_run_id_keeps_the_old_behaviour(self):
        self.disk["conv1"]["ctx"].pop("run_id")
        ctx, _ = self.s["_restore_rq_breakdown_ctx"]("conv1")
        self.assertIs(ctx, self.old)

    def test_cached_ctx_from_before_this_fix_is_treated_as_stale(self):
        self.old.pop("run_id")
        ctx, _ = self.s["_restore_rq_breakdown_ctx"]("conv1")
        self.assertEqual(ctx["run_id"], self.new["run_id"])

    def test_plan_context_ignores_an_earlier_turn(self):
        self.assertIsNone(self.s["_plan_context"]("conv1"))
        self.plan_memory["conv1"] = self.new
        self.assertIs(self.s["_plan_context"]("conv1"), self.new)
        self.disk.clear()                         # nothing saved: the cache is all there is
        self.plan_memory["conv1"] = self.old
        self.assertIs(self.s["_plan_context"]("conv1"), self.old)

    def test_each_turn_gets_its_own_run_id(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.s["settings"].output_dir = d
            a = self.s["init_workflow_context"]("q1", [], conversation_id="conv1")
            b = self.s["init_workflow_context"]("q2", [], conversation_id="conv1")
        self.assertEqual(a["task_id"], b["task_id"])
        self.assertNotEqual(a["run_id"], b["run_id"])


if __name__ == "__main__":
    unittest.main()
