"""A Task Breakdown card whose turn has moved on must not look actionable.

Seen locally (conversation a475253a...): the breakdown was proceeded at 12:23
and its workflow finished at 12:35. After a reload the card still offered
Provide Feedback; the server correctly refused the feedback as stale, but the
card only logged the error and stayed on "Revising Task Breakdown..." forever.

breakdown_waiting is what /api/rq-breakdown/status answers with; the script.js
checks make sure the card is redrawn when no revision comes back and that
reloaded cards ask the server before offering their buttons.
"""
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_resume_exact_turn as base  # noqa: E402
from test_resume_exact_turn import ROOT, Worker, turn  # noqa: E402

base.FUNCS = base.FUNCS | {"breakdown_waiting"}


class BreakdownWaiting(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.file = os.path.join(self.dir.name, "workflow_state.json")
        self.w = Worker(self.file)
        self.ctx = turn("Select Texas counties smaller than 50 km2")

    def write(self, phase):
        with open(self.file, "w", encoding="utf-8") as f:
            json.dump({"ctx": self.ctx, "workflow_phase": phase}, f)

    def test_paused_turn_is_waiting(self):
        self.write("rq_breakdown_paused")
        self.assertTrue(self.w.breakdown_waiting("conv1", self.ctx["run_id"]))
        self.assertTrue(self.w.breakdown_waiting("conv1"))

    def test_the_reported_case_finished_workflow(self):
        self.write("operations_complete")
        self.assertFalse(self.w.breakdown_waiting("conv1", self.ctx["run_id"]))

    def test_card_from_another_turn(self):
        self.write("rq_breakdown_paused")
        self.assertFalse(self.w.breakdown_waiting("conv1", "someotherrun"))

    def test_no_state(self):
        self.assertFalse(self.w.breakdown_waiting("conv1", self.ctx["run_id"]))


class CardHandling(unittest.TestCase):
    js = (ROOT / "WebUI/script.js").read_text(encoding="utf-8")

    def feedback_handler(self):
        start = self.js.index("/api/rq-breakdown/revise-stream/${activeTaskId}")
        return self.js[start - 2500:start + 4000]

    def test_card_redrawn_when_no_revision_arrives(self):
        h = self.feedback_handler()
        self.assertIn("if (!revised) restoreCard(revisionError);", h)
        self.assertIn("restoreCard(err.name === 'AbortError'", h)

    def test_reloaded_cards_ask_the_server(self):
        self.assertIn("/api/rq-breakdown/status/", self.js)
        self.assertTrue(re.search(r"s\.waiting === false[\s\S]{0,120}_setRqBreakdownNote", self.js))

    def test_route_exists(self):
        app = (ROOT / "WebUI/app.py").read_text(encoding="utf-8")
        self.assertIn("@app.route('/api/rq-breakdown/status/<task_id>')", app)
        self.assertIn("breakdown_waiting(task_id,", app)


if __name__ == "__main__":
    unittest.main()
