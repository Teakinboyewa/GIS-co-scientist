"""A source with no data for the request is not a bug to debug.

Before this existed the retrieval loop knew two failures that no code change
can fix -- a bad credential and a rate limit -- and treated everything else
as a bug: "no data for the requested date range" went to the LLM debugger,
which obligingly changed the dates until a call returned 200, and when it
did not, the handbook was revised, the source re-researched, and the whole
request retried three times over. Tens of executions to rediscover that the
data was never there, and a real chance of analysing data nobody asked for.

These pin the three ways the loop now recognises that condition (the
error text, the debugger declining, the one-off mid-budget triage) and the
stopping rule at every layer above it: the repair ladder, and the outer
per-request retry.
"""

import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.data_agent import DataRetrieverAgent as agent_module
from agents.data_agent import agm_data_agent
from agents.data_agent.DataRetrieverAgent import (
    DataRetrieverAgent, DataUnavailableError,
    _looks_like_data_unavailable_error, _data_unavailable_reason,
    _data_unavailable_verdict)
from tests.test_retrieval_repair_ladder import (
    LadderAgent, HANDBOOK_TOML)


def _http_error(status, body, host="api.example.invalid"):
    """An exception shaped like requests' HTTPError, without requests."""
    resp = types.SimpleNamespace(status_code=status, text=body,
                                 url=f"https://{host}/v1/data")
    err = Exception(f"{status} Client Error: Bad Request for url")
    err.response = resp
    return err


class DetectorTests(unittest.TestCase):
    def test_marker_raised_by_the_generated_code(self):
        err = RuntimeError("DATA_UNAVAILABLE: dataset ends 2023-12-31, "
                           "task asked for 2026")
        self.assertTrue(_looks_like_data_unavailable_error(err))
        self.assertEqual(_data_unavailable_reason(err),
                         "dataset ends 2023-12-31, task asked for 2026")

    def test_provider_body_names_the_coverage_gap(self):
        err = _http_error(
            400, '{"error": "No data available for the requested date range"}')
        self.assertTrue(_looks_like_data_unavailable_error(err))

    def test_wrapped_error_is_seen_through_the_chain(self):
        inner = _http_error(404, "Requested period is not available")
        try:
            try:
                raise inner
            except Exception as e:
                raise RuntimeError("download failed") from e
        except RuntimeError as outer:
            self.assertTrue(_looks_like_data_unavailable_error(outer))
            # The provider's words, not the wrapper's.
            self.assertIn("not available", _data_unavailable_reason(outer))

    def test_ordinary_bugs_are_not_mistaken_for_missing_data(self):
        for err in (KeyError("date"),
                    FileNotFoundError("output dir does not exist"),
                    TypeError("unsupported operand"),
                    _http_error(400, "Invalid parameter 'bbox'"),
                    ValueError("could not convert string to float")):
            with self.subTest(err=err):
                self.assertFalse(_looks_like_data_unavailable_error(err))

    def test_geocoder_helper_saying_no_results_is_not_the_source(self):
        """Nominatim finding nothing is a place-name problem the debugger
        can fix (see _NON_SOURCE_HELPER_HOSTS); it says nothing about the
        actual source's coverage."""
        err = _http_error(404, "No results found",
                          host="nominatim.openstreetmap.org")
        self.assertFalse(_looks_like_data_unavailable_error(err))


class DebuggerVerdictTests(unittest.TestCase):
    def test_marker_line_without_code_is_a_verdict(self):
        self.assertEqual(
            _data_unavailable_verdict(
                "DATA_UNAVAILABLE: the API only serves 2000-2023; "
                "the task asks for 2026."),
            "the API only serves 2000-2023; the task asks for 2026.")
        self.assertEqual(
            _data_unavailable_verdict(
                "Explanation first.\n**DATA_UNAVAILABLE** — no records for "
                "this county"),
            "no records for this county")

    def test_reply_that_still_ships_code_is_a_fix_not_a_verdict(self):
        reply = ("DATA_UNAVAILABLE: maybe\n```python\nprint('try again')\n```")
        self.assertIsNone(_data_unavailable_verdict(reply))

    def test_plain_fix_is_not_a_verdict(self):
        self.assertIsNone(_data_unavailable_verdict(
            "Explanation: wrong param.\n```python\npass\n```"))


def _reply(text):
    """One streamed chunk carrying ``text``, shaped like the OpenAI client's."""
    return [types.SimpleNamespace(
        usage=None,
        choices=[types.SimpleNamespace(
            delta=types.SimpleNamespace(content=text))])]


class ExecuteLoopTests(unittest.TestCase):
    """execute_complete_program run for real (exec of tiny programs), with
    only the LLM debugger stubbed."""

    def setUp(self):
        self.agent = DataRetrieverAgent(api_key="sk-test")
        self.debug_replies = []

        def fake_stream(**kwargs):
            self.debug_calls += 1
            return _reply(self.debug_replies.pop(0))

        self.debug_calls = 0
        patcher = mock.patch.object(agent_module, "_stream_with_usage",
                                    side_effect=fake_stream)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_code(self, code, try_cnt=5, triage=None):
        return self.agent.execute_complete_program(
            code=code, try_cnt=try_cnt, task="t", model_name="m",
            handbook_str="h", attempt_timeout=30, triage=triage)

    def test_marker_from_the_code_stops_before_any_debugging(self):
        with self.assertRaises(DataUnavailableError) as ctx:
            self.run_code("raise RuntimeError('DATA_UNAVAILABLE: no 2026 data')")
        self.assertEqual(str(ctx.exception), "no 2026 data")
        self.assertEqual(self.debug_calls, 0)
        self.assertEqual(self.agent.last_execution_report["attempts_used"], 1)

    def test_provider_wording_stops_before_any_debugging(self):
        with self.assertRaises(DataUnavailableError):
            self.run_code(
                "raise ValueError('Server said: no data available for the "
                "requested period')")
        self.assertEqual(self.debug_calls, 0)

    def test_debugger_declining_ends_the_loop(self):
        """A generic error the detector cannot read, but the debugger can:
        its DATA_UNAVAILABLE reply is honoured instead of extracting an
        empty program and running that."""
        self.debug_replies = ["DATA_UNAVAILABLE: the dataset stops at 2023."]
        with self.assertRaises(DataUnavailableError) as ctx:
            self.run_code("raise ValueError('400 Bad Request')")
        self.assertEqual(str(ctx.exception), "the dataset stops at 2023.")
        self.assertEqual(self.debug_calls, 1)
        self.assertEqual(self.agent.last_execution_report["attempts_used"], 1)

    def test_early_triage_runs_once_and_can_end_the_loop(self):
        """After the debugger's first fix also fails, the classifier gets
        exactly one look. 'request_infeasible' stops with the user-facing
        error; the verdict is kept on the report for the caller."""
        self.debug_replies = ["fix\n```python\nraise ValueError('still 400 B')\n```",
                              "fix\n```python\nraise ValueError('still 400 C')\n```",
                              "fix\n```python\nraise ValueError('still 400 D')\n```"]
        triage_calls = []

        def triage(note, code):
            triage_calls.append(note)
            return {"failure_category": "request_infeasible",
                    "summary": "County has no stations in this network."}

        with self.assertRaises(DataUnavailableError) as ctx:
            self.run_code("raise ValueError('400 A')", triage=triage)
        self.assertEqual(str(ctx.exception),
                         "County has no stations in this network.")
        self.assertEqual(len(triage_calls), 1)
        self.assertIn("still 400 B", triage_calls[0])
        report = self.agent.last_execution_report
        self.assertEqual(report["attempts_used"], 2)
        self.assertEqual(report["triage"]["analysis"]["failure_category"],
                         "request_infeasible")
        self.assertTrue(report["triage"]["error_signature"].endswith("still 400 B"))

    def test_early_triage_external_stops_debugging_without_raising(self):
        self.debug_replies = ["fix\n```python\nraise ValueError('503 B')\n```",
                              "fix\n```python\nraise ValueError('503 C')\n```"]
        code = self.run_code(
            "raise ValueError('503 A')",
            triage=lambda note, code: {"failure_category": "external",
                                       "summary": "outage"})
        self.assertIn("503 B", code)          # the code that ran last
        self.assertEqual(self.debug_calls, 1)
        self.assertEqual(self.agent.last_execution_report["attempts_used"], 2)

    def test_early_triage_fails_open(self):
        """A handbook_deficiency verdict, an empty one, or a classifier that
        throws all mean 'keep debugging' -- triage is an optimisation, never
        a gate that can strand a fixable bug."""
        for triage in (lambda n, c: {"failure_category": "handbook_deficiency"},
                       lambda n, c: {},
                       mock.Mock(side_effect=RuntimeError("classifier down"))):
            with self.subTest(triage=triage):
                self.debug_calls = 0
                self.debug_replies = [
                    "fix\n```python\nraise ValueError('e2')\n```",
                    "fix\n```python\nraise ValueError('e3')\n```",
                    "fix\n```python\nprint('ok')\n```"]
                self.run_code("raise ValueError('e1')", try_cnt=5,
                              triage=triage)
                self.assertTrue(self.agent.last_execution_report["success"])
                self.assertEqual(self.debug_calls, 3)

    def test_no_triage_callback_means_the_old_loop(self):
        self.debug_replies = ["fix\n```python\nraise ValueError('e2')\n```",
                              "fix\n```python\nprint('ok')\n```"]
        self.run_code("raise ValueError('e1')")
        self.assertTrue(self.agent.last_execution_report["success"])
        self.assertNotIn("triage", self.agent.last_execution_report)


class LadderStopTests(unittest.TestCase):
    """The rungs above the execute loop never run for a request the source
    cannot serve: no revision, no re-research, and the result says why."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.books = os.path.join(self.tmp.name, "books")
        self.outputs = os.path.join(self.tmp.name, "out")
        os.makedirs(self.books)
        os.makedirs(self.outputs)
        with open(os.path.join(self.books, "test_source.toml"), "w",
                  encoding="utf-8") as fh:
            fh.write(HANDBOOK_TOML)
        self.addCleanup(self.tmp.cleanup)

    def _agent(self, cls, **attrs):
        agent = cls(api_key="sk-test", extra_handbook_dirs=[self.books],
                    script=[])
        agent.output_dir = self.outputs
        for k, v in attrs.items():
            setattr(agent, k, v)
        return agent

    def test_execute_loop_verdict_skips_every_rung(self):
        class Unavailable(LadderAgent):
            def execute_complete_program(self, code, try_cnt, task, model_name,
                                         handbook_str, **kwargs):
                self.calls.append("execute:unavailable")
                self.last_execution_report = {"success": False,
                                              "traceback": "x"}
                raise DataUnavailableError("no records after 2023")

        agent = self._agent(Unavailable)
        with mock.patch.object(agent_module, "HANDBOOK_REFINE_ATTEMPTS", 1):
            result = agent.LLM_Find("Download the test data.")
        self.assertEqual(agent.calls, ["execute:unavailable"])
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["data_unavailable"])
        self.assertIn("no records after 2023", result["error"])
        self.assertEqual(result["source_name"], "Test Source")

    def test_post_loop_triage_infeasible_stops_the_ladder(self):
        """The classifier's verdict after an exhausted loop is honoured the
        same way as the loop's own: stop, and say the request is the
        problem, not the skill."""
        class Exhausted(LadderAgent):
            def execute_complete_program(self, code, try_cnt, task, model_name,
                                         handbook_str, **kwargs):
                self.calls.append("execute:fail")
                self.last_execution_report = {
                    "success": False,
                    "traceback": "Traceback...\nHTTPError: 400 Bad Request"}
                return code

        agent = self._agent(
            Exhausted,
            triage_verdict={"failure_category": "request_infeasible",
                            "summary": "Task asks for 2026; series ends 2023."})
        with mock.patch.object(agent_module, "HANDBOOK_REFINE_ATTEMPTS", 1):
            result = agent.LLM_Find("Download the test data.")
        self.assertEqual(agent.calls, ["execute:fail", "triage"])
        self.assertTrue(result["data_unavailable"])
        self.assertIn("series ends 2023", result["error"])

    def test_early_verdict_is_reused_when_the_error_did_not_change(self):
        """The loop already paid for a classification of this exact error;
        the post-loop ladder must not buy a second one."""
        class Exhausted(LadderAgent):
            def execute_complete_program(self, code, try_cnt, task, model_name,
                                         handbook_str, **kwargs):
                self.calls.append("execute:fail")
                self.last_execution_report = {
                    "success": False,
                    "traceback": "Traceback...\nHTTPError: 503",
                    "triage": {"analysis": {"failure_category": "external",
                                            "summary": "down"},
                               "error_signature": "HTTPError: 503"}}
                return code

        agent = self._agent(Exhausted)
        with mock.patch.object(agent_module, "HANDBOOK_REFINE_ATTEMPTS", 1):
            try:
                agent.LLM_Find("Download the test data.")
            except Exception:
                pass
        self.assertEqual(agent.calls, ["execute:fail"])  # no second triage

    def test_early_verdict_is_not_reused_for_a_different_error(self):
        class Exhausted(LadderAgent):
            def execute_complete_program(self, code, try_cnt, task, model_name,
                                         handbook_str, **kwargs):
                self.calls.append("execute:fail")
                self.last_execution_report = {
                    "success": False,
                    "traceback": "Traceback...\nKeyError: 'features'",
                    "triage": {"analysis": {"failure_category": "external",
                                            "summary": "down"},
                               "error_signature": "HTTPError: 503"}}
                return code

        agent = self._agent(Exhausted)
        with mock.patch.object(agent_module, "HANDBOOK_REFINE_ATTEMPTS", 1):
            try:
                agent.LLM_Find("Download the test data.")
            except Exception:
                pass
        self.assertIn("triage", agent.calls)


class OuterRetryTests(unittest.TestCase):
    """data_download retries a failed request up to three times with a fresh
    source selection. For a request the source cannot serve, that is three
    rediscoveries of the same gap -- or a nudge toward data nobody asked
    for -- so it stops after the first, like it already does for a bad key."""

    def _run(self, results):
        calls = []

        class FakeDownloader:
            def __init__(self, **kwargs):
                pass

            def LLM_Find(self, data_request, **kwargs):
                calls.append(data_request)
                return results.pop(0)

        ctx = {"artifacts": {"rq_breakdown": {"data_requests": ["req one"]}}}
        stub_self = types.SimpleNamespace(api_key="sk-test")
        with mock.patch.object(agm_data_agent, "DataRetrieverAgent",
                               FakeDownloader):
            out = agm_data_agent.DataAgent.data_download(stub_self, ctx)
        return calls, out

    def test_data_unavailable_is_not_retried(self):
        calls, out = self._run([
            {"status": "failed", "error": "The data source has no data for "
             "this request: series ends 2023", "data_unavailable": True}])
        self.assertEqual(len(calls), 1)
        self.assertTrue(out[0]["data_unavailable"])

    def test_ordinary_failures_still_get_their_retries(self):
        calls, _ = self._run([
            {"status": "failed", "error": "boom"},
            {"status": "failed", "error": "boom"},
            {"status": "failed", "error": "boom"}])
        self.assertEqual(len(calls), 3)
        # The failure note rides along into the retry prompt as before.
        self.assertIn("boom", calls[1])


if __name__ == "__main__":
    unittest.main()
