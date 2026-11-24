"""Token accounting for the Research / Spatial Analysis / Data Retrieval modes.

These tests load the shipped source of each function under test directly, so
they exercise the real code without needing the project's heavy runtime
dependencies (numpy, flask, langchain) to be installed.

What they pin down:

* a call is attributed to the phase that was running when it was made
* a call whose backend reported no usage is COUNTED and disclosed, never
  estimated -- an invented number here would be indistinguishable from a
  measured one, and the whole point of the figure is that it is measured
* worker threads record into the run that spawned them, and only that run
* the browser receives cumulative phase totals but only newly-added call rows
"""
import ast
import importlib.util
import io
import sys
import threading
import types
import unittest


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_functions(path, names, namespace):
    """Exec the named top-level functions from *path* into *namespace*."""
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    wanted = [node for node in tree.body
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
              and node.name in names]
    missing = set(names) - {node.name for node in wanted}
    assert not missing, "not found in %s: %s" % (path, missing)
    exec(compile(ast.Module(body=wanted, type_ignores=[]), path, "exec"), namespace)
    return namespace


def _module_constant(path, name):
    """Evaluate one module-level constant assignment from *path*."""
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("%s not found in %s" % (name, path))


def _module_function(path, name):
    """Load one top-level function, resolving its module-level dependencies."""
    return load_functions(path, {name}, {})[name]


ledger_mod = load_module("agents/data_agent/usage_ledger.py", "usage_ledger_under_test")


class LedgerTests(unittest.TestCase):
    def test_phase_attribution_follows_set_phase(self):
        with ledger_mod.collect() as ledger:
            ledger.set_phase("data_audit")
            ledger_mod.record("gpt-6-luna", {"prompt_tokens": 100, "completion_tokens": 20})
            ledger_mod.record("gpt-6-luna", {"input_tokens": 50, "output_tokens": 5})
            ledger.set_phase("rq_breakdown")
            ledger_mod.record("gpt-6-luna", {"input_tokens": 7, "output_tokens": 3})
            totals = ledger.totals()
        self.assertEqual(totals["calls"], 3)
        self.assertEqual(totals["total_tokens"], 120 + 55 + 10)
        self.assertEqual(totals["by_phase"]["data_audit"]["total_tokens"], 175)
        self.assertEqual(totals["by_phase"]["rq_breakdown"]["total_tokens"], 10)

    def test_call_without_usage_is_counted_not_estimated(self):
        with ledger_mod.collect() as ledger:
            ledger_mod.record("gpt-6-luna", {"input_tokens": 7, "output_tokens": 3})
            ledger_mod.record("gpt-6-luna", {})     # backend returned nothing
            totals = ledger.totals()
        self.assertEqual(totals["calls"], 2)
        self.assertEqual(totals["calls_missing_usage"], 1)
        self.assertEqual(totals["total_tokens"], 10)   # not inflated by a guess

    def test_records_since_returns_only_new_records(self):
        with ledger_mod.collect() as ledger:
            for i in range(3):
                ledger_mod.record("m", {"input_tokens": i, "output_tokens": 0})
            first, offset = ledger.records_since(0)
            self.assertEqual((len(first), offset), (3, 3))
            ledger_mod.record("m", {"input_tokens": 9, "output_tokens": 1})
            new, offset = ledger.records_since(offset)
        self.assertEqual((len(new), offset), (1, 4))
        self.assertEqual(new[0]["input_tokens"], 9)

    def test_workflow_phase_wins_over_a_callers_own_label(self):
        # The retrieval agent labels its calls "retrieval" and the handbook
        # generator labels its own "generation". In the chat modes the phases
        # are workflow CARDS, so those labels match nothing on screen and a
        # whole download run showed no tokens on any card. The card wins; the
        # caller's label survives as `stage`.
        with ledger_mod.collect() as ledger:
            ledger.set_phase("data_download_req_1")
            ledger_mod.record("gpt-6-luna", {"input_tokens": 900, "output_tokens": 100},
                              phase="retrieval", source="generate_code")
            totals = ledger.totals()
            entry = ledger.records[0]
        self.assertEqual(entry["phase"], "data_download_req_1")
        self.assertEqual(entry["stage"], "retrieval")
        self.assertEqual(totals["by_phase"]["data_download_req_1"]["total_tokens"], 1000)
        self.assertNotIn("retrieval", totals["by_phase"])

    def test_a_callers_label_still_wins_when_no_phase_is_set(self):
        # Generate & Test opens a ledger per phase and never calls set_phase,
        # so its own labels must keep working exactly as before.
        with ledger_mod.collect("generation") as ledger:
            ledger_mod.record("gpt-6-luna", {"input_tokens": 10, "output_tokens": 2},
                              phase="retrieval", source="generate_code")
            ledger_mod.record("gpt-6-luna", {"input_tokens": 10, "output_tokens": 2})
            totals = ledger.totals()
        self.assertEqual(totals["by_phase"]["retrieval"]["total_tokens"], 12)
        self.assertEqual(totals["by_phase"]["generation"]["total_tokens"], 12)

    def test_no_active_ledger_is_inert(self):
        # Notebooks, the CLI and the test suite call the agents with no ledger
        # open; instrumenting a call site must not change behaviour there.
        self.assertIsNone(ledger_mod.record("m", {"input_tokens": 1}))


class ThreadAdoptionTests(unittest.TestCase):
    """agm_workflow._with_flask_ctx must carry the ledger onto worker threads.

    Every agent in these workflows runs on a thread spawned through it, so
    without adoption they record into nothing and a run that plainly cost money
    reports zero tokens.
    """

    def setUp(self):
        pkg = types.ModuleType("agents.data_agent")
        pkg.usage_ledger = ledger_mod
        agents = types.ModuleType("agents")
        agents.data_agent = pkg
        sys.modules["agents"] = agents
        sys.modules["agents.data_agent"] = pkg
        self.ns = load_functions(
            "WebUI/agm_workflow.py",
            {"_with_flask_ctx", "_with_usage_ledger"},
            {"usage_ledger": ledger_mod})

    def test_worker_thread_records_into_the_spawning_ledger(self):
        with ledger_mod.collect() as ledger:
            ledger.set_phase("execution")

            def work():
                ledger_mod.record("gpt-6-luna", {"input_tokens": 9, "output_tokens": 1})

            thread = threading.Thread(target=self.ns["_with_flask_ctx"](work))
            thread.start()
            thread.join()
            totals = ledger.totals()
        self.assertEqual(totals["calls"], 1)
        self.assertEqual(totals["by_phase"]["execution"]["total_tokens"], 10)

    def test_no_ledger_at_spawn_time_leaves_the_target_untouched(self):
        def work():
            pass
        self.assertIs(self.ns["_with_flask_ctx"](work), work)

    def test_ledger_does_not_leak_into_an_unrelated_thread(self):
        # Two sessions run concurrently in the Flask app; their tokens must
        # never pool. Thread-locality is what prevents that, so a thread not
        # spawned through the wrapper must see no ledger at all.
        seen = []
        with ledger_mod.collect():
            def bare():
                seen.append(ledger_mod.active())
            thread = threading.Thread(target=bare)
            thread.start()
            thread.join()
        self.assertEqual(seen, [None])


class HelperCaptureTests(unittest.TestCase):
    """GIBDChat's usage extraction, for each shape a provider returns."""

    def setUp(self):
        self.ns = load_functions(
            "utils/agm_helper.py",
            {"_record_llm_usage", "_usage_from_lc_message",
             "_record_chat_result", "_record_stream_chunk"},
            {"_usage_ledger": lambda: ledger_mod})

    def test_reads_usage_metadata(self):
        msg = types.SimpleNamespace(
            usage_metadata={"input_tokens": 11, "output_tokens": 2},
            response_metadata={})
        self.assertEqual(self.ns["_usage_from_lc_message"](msg),
                         {"input_tokens": 11, "output_tokens": 2})

    def test_falls_back_to_response_metadata_token_usage(self):
        # Older langchain_openai reports only here.
        msg = types.SimpleNamespace(
            usage_metadata=None,
            response_metadata={"token_usage": {"prompt_tokens": 8, "completion_tokens": 1}})
        self.assertEqual(self.ns["_usage_from_lc_message"](msg),
                         {"prompt_tokens": 8, "completion_tokens": 1})

    def test_records_a_delegated_chat_result(self):
        msg = types.SimpleNamespace(
            usage_metadata={"input_tokens": 30, "output_tokens": 4},
            response_metadata={})
        result = types.SimpleNamespace(
            generations=[types.SimpleNamespace(message=msg)], llm_output=None)
        with ledger_mod.collect() as ledger:
            self.ns["_record_chat_result"](result, "gpt-4o", source="openai")
            totals = ledger.totals()
        self.assertEqual(totals["calls"], 1)
        self.assertEqual(totals["total_tokens"], 34)

    def test_a_result_with_no_usage_still_counts_the_call(self):
        msg = types.SimpleNamespace(usage_metadata=None, response_metadata={})
        result = types.SimpleNamespace(
            generations=[types.SimpleNamespace(message=msg)], llm_output={})
        with ledger_mod.collect() as ledger:
            self.ns["_record_chat_result"](result, "gpt-4o")
            totals = ledger.totals()
        self.assertEqual(totals["calls"], 1)
        self.assertEqual(totals["calls_missing_usage"], 1)

    def test_stream_chunks_without_usage_are_ignored(self):
        # A streamed response reports usage on one trailing chunk. Recording
        # per chunk would multiply one call into hundreds.
        bare = types.SimpleNamespace(
            message=types.SimpleNamespace(usage_metadata=None, response_metadata={}))
        final = types.SimpleNamespace(
            message=types.SimpleNamespace(
                usage_metadata={"input_tokens": 5, "output_tokens": 2},
                response_metadata={}))
        with ledger_mod.collect() as ledger:
            self.assertFalse(self.ns["_record_stream_chunk"](bare, "m"))
            self.assertFalse(self.ns["_record_stream_chunk"](bare, "m"))
            self.assertTrue(self.ns["_record_stream_chunk"](final, "m"))
            totals = ledger.totals()
        self.assertEqual(totals["calls"], 1)
        self.assertEqual(totals["total_tokens"], 7)


class RawCompletionCaptureTests(unittest.TestCase):
    """The second LLM family: helper.client_chat_completion[_stream].

    GIBDChat is not the only path. The RQ understanding agent (which produces
    the Task Breakdown in every mode), the task manager, the spatial analysis
    executor and the modeling agent all call these raw helpers instead, so a
    run whose work happens there records nothing unless they are instrumented
    too.
    """

    def setUp(self):
        self.routed = []

        def fake_route(model, messages, user_key=None, base_url=None, **kwargs):
            self.routed.append(kwargs)
            return types.SimpleNamespace(usage={"prompt_tokens": 60,
                                                "completion_tokens": 12})

        def fake_route_stream(model, messages, user_key=None, base_url=None, **kwargs):
            self.routed.append(kwargs)
            if kwargs.get("stream_options") and self.reject:
                raise RuntimeError("stream_options not supported by this backend")
            yield types.SimpleNamespace(usage=None, text="a")
            yield types.SimpleNamespace(usage=None, text="b")
            if kwargs.get("stream_options"):
                yield types.SimpleNamespace(
                    usage={"prompt_tokens": 30, "completion_tokens": 5})

        self.reject = False
        self.ns = load_functions(
            "utils/agm_helper.py",
            {"_usage_from_chunk", "_record_llm_usage",
             "client_chat_completion", "client_chat_completion_stream"},
            {"_usage_ledger": lambda: ledger_mod,
             "_route_chat_completion": fake_route,
             "_route_chat_completion_stream": fake_route_stream,
             "print": lambda *a, **k: None})

    def test_non_streaming_call_is_recorded(self):
        with ledger_mod.collect() as ledger:
            self.ns["client_chat_completion"]("gpt-4o", [{"role": "user", "content": "x"}])
            totals = ledger.totals()
        self.assertEqual(totals["calls"], 1)
        self.assertEqual(totals["total_tokens"], 72)

    def test_streamed_call_is_recorded_once_from_the_trailing_chunk(self):
        with ledger_mod.collect() as ledger:
            chunks = list(self.ns["client_chat_completion_stream"](
                "gpt-4o", [{"role": "user", "content": "x"}]))
            totals = ledger.totals()
        self.assertEqual(totals["calls"], 1)        # one call, not one per chunk
        self.assertEqual(totals["total_tokens"], 35)
        self.assertEqual(self.routed[0].get("stream_options"),
                         {"include_usage": True})
        self.assertEqual([c.text for c in chunks if getattr(c, "text", None)],
                         ["a", "b"])

    def test_a_backend_rejecting_stream_options_still_streams(self):
        # The router is a generator, so the rejection surfaces only when the
        # first chunk is pulled. Content must survive the fallback intact.
        self.reject = True
        with ledger_mod.collect() as ledger:
            chunks = list(self.ns["client_chat_completion_stream"](
                "gpt-4o", [{"role": "user", "content": "x"}]))
            totals = ledger.totals()
        self.assertEqual([c.text for c in chunks if getattr(c, "text", None)],
                         ["a", "b"])
        self.assertIsNone(self.routed[1].get("stream_options"))
        self.assertEqual(totals["calls"], 0)   # accounting lost, stream intact

    def test_record_usage_false_opts_a_self_accounting_caller_out(self):
        # DataRetrieverAgent records each call itself; recording here too
        # would count every retrieval call twice.
        with ledger_mod.collect() as ledger:
            list(self.ns["client_chat_completion_stream"](
                "gpt-4o", [{"role": "user", "content": "x"}], record_usage=False))
            self.ns["client_chat_completion"](
                "gpt-4o", [{"role": "user", "content": "x"}], record_usage=False)
            totals = ledger.totals()
        self.assertEqual(totals["calls"], 0)

    def test_caller_supplied_stream_options_passes_through_unrecorded(self):
        # handbook_generator asks for usage itself and records in its own
        # consume(); this must not add a second record for the same call.
        with ledger_mod.collect() as ledger:
            list(self.ns["client_chat_completion_stream"](
                "gpt-4o", [{"role": "user", "content": "x"}],
                stream_options={"include_usage": True}))
            totals = ledger.totals()
        self.assertEqual(totals["calls"], 0)
        self.assertEqual(len(self.routed), 1)   # no probe-then-retry


class TrackUsageTests(unittest.TestCase):
    """app.track_usage: what the browser actually receives."""

    def setUp(self):
        def _get_phase(step):
            return (step or "general").split(":")[0]

        pkg = types.ModuleType("agents.data_agent")
        pkg.usage_ledger = ledger_mod
        sys.modules["agents.data_agent"] = pkg
        self.ns = load_functions(
            "WebUI/app.py",
            {"_usage_call_row", "_usage_run_total", "track_usage", "usage_summary_event"},
            {"_get_phase": _get_phase,
             # Module-level constant the loaded functions close over.
             "_USAGE_CALL_FIELDS": _module_constant("WebUI/app.py", "_USAGE_CALL_FIELDS")})

    def _workflow(self):
        yield {"type": "llm_update", "step": "data_audit", "content": "auditing"}
        ledger_mod.record("gpt-6-luna", {"input_tokens": 100, "output_tokens": 20})
        yield {"type": "llm_update", "step": "data_audit", "content": "audited"}
        yield {"type": "llm_update", "step": "rq_breakdown", "content": "breaking down"}
        ledger_mod.record("gpt-6-luna", {"input_tokens": 7, "output_tokens": 3})
        yield {"type": "result", "step": "rq_breakdown", "response": "done"}

    def test_emits_cumulative_phase_totals_and_delta_calls(self):
        holder = []
        events = list(self.ns["track_usage"](self._workflow(), holder))
        usage_events = [e for e in events if e.get("type") == "token_usage"]
        self.assertEqual(len(usage_events), 2)

        first, second = usage_events
        self.assertEqual(first["phase"], "data_audit")
        self.assertEqual(first["usage"]["total_tokens"], 120)
        self.assertEqual(len(first["calls"]), 1)
        self.assertEqual(first["run_total"]["total_tokens"], 120)

        # Only the NEW call is sent, but the run total covers both. The card
        # renderer assigns rather than accumulates, so `usage` has to stay
        # cumulative even though `calls` is a delta.
        self.assertEqual(second["phase"], "rq_breakdown")
        self.assertEqual(len(second["calls"]), 1)
        self.assertEqual(second["usage"]["total_tokens"], 10)
        self.assertEqual(second["run_total"]["total_tokens"], 130)
        self.assertEqual(second["run_total"]["calls"], 2)

    def test_workflow_events_pass_through_unchanged_and_in_order(self):
        events = list(self.ns["track_usage"](self._workflow(), []))
        passthrough = [e for e in events if e.get("type") != "token_usage"]
        self.assertEqual([e["type"] for e in passthrough],
                         ["llm_update", "llm_update", "llm_update", "result"])
        self.assertEqual(passthrough[0]["content"], "auditing")

    def test_summary_carries_the_whole_ledger(self):
        holder = []
        list(self.ns["track_usage"](self._workflow(), holder))
        summary = self.ns["usage_summary_event"](holder[0])
        self.assertEqual(summary["type"], "usage_summary")
        self.assertEqual(summary["run_total"]["calls"], 2)
        self.assertEqual(len(summary["calls"]), 2)
        self.assertEqual({c["phase"] for c in summary["calls"]},
                         {"data_audit", "rq_breakdown"})

    def test_a_run_with_no_llm_calls_produces_no_summary(self):
        # Chat-only turns must not persist an all-zero summary, which would
        # render as a meter claiming the run happened for free.
        holder = []
        list(self.ns["track_usage"](iter([{"type": "result", "step": "chat"}]), holder))
        self.assertIsNone(self.ns["usage_summary_event"](holder[0]))
        self.assertIsNone(self.ns["usage_summary_event"](None))


class CardAttributionTests(unittest.TestCase):
    """A token_usage event must land on the card for its own phase.

    _get_phase maps steps to phases but is not reversible: it sends both
    "execution" and "setup" to "general". Deriving a phase back out of a step
    would therefore file those two phases' tokens onto the wrong card, so the
    event names its phase outright and the card path reads that.
    """

    def setUp(self):
        real_get_phase = _module_function("WebUI/app.py", "_get_phase")
        self.ns = load_functions(
            "WebUI/app.py",
            {"_touch_workflow_card_metrics", "_workflow_usage_from_update",
             "_as_nonnegative_int", "_workflow_card_for_persistence"},
            {"_get_phase": real_get_phase, "time": __import__("time")})

    def test_explicit_phase_wins_over_step_derivation(self):
        cards = {"execution": {"status": "", "body": "", "step": "exec"},
                 "general": {"status": "", "body": "", "step": "general"}}
        event = {"type": "token_usage", "phase": "execution",
                 "usage": {"input_tokens": 40, "output_tokens": 8, "total_tokens": 48}}
        self.ns["_touch_workflow_card_metrics"](cards, event, "general")
        self.assertEqual(cards["execution"]["_metrics_runtime"]["total_tokens"], 48)
        self.assertNotIn("_metrics_runtime", cards["general"])

    def test_cumulative_totals_replace_rather_than_accumulate(self):
        # The event carries the phase's running total, so a later event must
        # overwrite the earlier one -- adding them would double-count.
        cards = {"data_audit": {"status": "", "body": "", "step": "data_audit"}}
        for total in (120, 175):
            self.ns["_touch_workflow_card_metrics"](
                cards,
                {"type": "token_usage", "phase": "data_audit",
                 "usage": {"total_tokens": total}},
                None)
        self.assertEqual(cards["data_audit"]["_metrics_runtime"]["total_tokens"], 175)

    def test_metrics_reach_the_persisted_card(self):
        cards = {"rq_breakdown": {"status": "", "body": "", "step": "rq_breakdown"}}
        self.ns["_touch_workflow_card_metrics"](
            cards,
            {"type": "token_usage", "phase": "rq_breakdown",
             "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}},
            None)
        saved = self.ns["_workflow_card_for_persistence"](cards["rq_breakdown"])
        self.assertEqual(saved["metrics"]["total_tokens"], 10)
        self.assertNotIn("_metrics_runtime", saved)


if __name__ == "__main__":
    unittest.main(verbosity=2)
