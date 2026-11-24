import json
import os
import tempfile
import unittest
from unittest import mock

from WebUI import experiment2_runner as runner
from agents.data_agent import handbook_generator


def exhaust(generator):
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as done:
            return events, done.value


class Experiment2CorrectionTests(unittest.TestCase):
    def test_windows_execution_root_is_short(self):
        root = runner.experiment_output_roots(
            "u" * 64, "12345678-1234-1234-1234-123456789012")[0]
        if os.name == "nt":
            self.assertTrue(root.casefold().startswith(
                os.path.abspath(os.path.join(
                    tempfile.gettempdir(), "GISCoE2")).casefold()))
        self.assertLess(len(root), 120)

    def test_short_redaction_values_do_not_destroy_trace(self):
        trace = "A request failed while parsing a spatial response."
        redacted = runner._redact_text(trace, ["a", "s"])
        self.assertEqual(redacted, trace)
        self.assertEqual(
            runner._redact_text("token=abcd&other=ok", ["abcd"]),
            "token=[REDACTED]&other=ok",
        )

    def test_only_genuine_long_path_failure_is_platform_failure(self):
        missing_local = runner._failure_routing({
            "status": "failed",
            "error": "FileNotFoundError: 'missing/result.geojson'",
        })
        self.assertEqual(missing_local["origin"], "retrieval")
        self.assertTrue(missing_local["refinable"])

        long_path = "C:\\" + ("nested\\" * 40) + "result.geojson"
        platform = runner._failure_routing({
            "status": "failed",
            "error": f"FileNotFoundError: '{long_path}'",
        })
        self.assertEqual(platform["origin"], "platform")
        self.assertFalse(platform["refinable"])

    def test_mechanisms_have_independent_sources_and_versions(self):
        study = {
            "source_name": "Example source",
            "documentation_url": "https://example.test/docs",
            "retrieval_task": "Retrieve features as GeoJSON.",
            "sources": [],
            "tasks": [],
        }
        rest_source, rest_task = runner._trial_source_and_task(study, "rest")
        bulk_source, bulk_task = runner._trial_source_and_task(study, "bulk")
        self.assertNotEqual(rest_source["id"], bulk_source["id"])
        self.assertEqual({item["mechanism"] for item in study["sources"]},
                         {"rest", "bulk"})
        self.assertEqual(rest_task["expected_format"], "GeoJSON")
        self.assertEqual(bulk_task["expected_format"], "GeoJSON")
        self.assertEqual(runner._version_name("rest", 0), "REST-H0")
        self.assertEqual(runner._version_name("bulk", 1), "BULK-H1")

    def test_semantic_validation_enforces_required_spatial_evidence(self):
        model_result = {
            "task_completed": True,
            "confidence": 0.9,
            "summary": "The file exists but spatial evidence is absent.",
            "mechanism_correctness": "pass",
            "parameter_correctness": "pass",
            "spatial_query_correctness": "unknown",
            "temporal_query_correctness": "not_applicable",
            "output_correctness": "pass",
            "checks": [],
            "failure_category": "Spatial validation",
            "should_refine_handbook": True,
            "handbook_gap": "Explain how to validate the requested county.",
        }
        with mock.patch.object(
                handbook_generator, "_call_model",
                return_value=json.dumps(model_result)):
            result = handbook_generator.evaluate_retrieval_result(
                {"data_source_name": "Example"},
                "Retrieve features within Centre County as GeoJSON.",
                {"status": "passed", "output_evidence": []},
                mechanism="REST API",
            )
        self.assertFalse(result["task_completed"])

    def test_controlled_execution_uses_one_attempt(self):
        calls = []

        class FakeRetriever:
            def __init__(self, **kwargs):
                self.model = ""

            def run_controlled_handbook_trial(self, **kwargs):
                calls.append(kwargs)
                return {
                    "status": "failed",
                    "error": "HTTP 400",
                    "execution": {"attempts_used": 1},
                    "downloaded_files": [],
                    "http_requests": [],
                }

        study = {"controls": {"model": "test-model"}}
        version = {
            "version": "REST-H0",
            "source_id": "example-source",
            "handbook_snapshot": {"data_source_name": "Example"},
        }
        task = {
            "description": "Retrieve one feature.",
            "expected_format": "GeoJSON",
        }
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as output_root:
            with mock.patch.object(runner, "DataRetrieverAgent", FakeRetriever), \
                    mock.patch.object(
                        runner, "_study_output_root",
                        return_value=output_root):
                _, result = exhaust(runner._execute_version(
                    "user123", "api-key",
                    "12345678-1234-1234-1234-123456789012",
                    study, version, task, {}, "rest_retrieval_h0",
                ))
        self.assertEqual(calls[0]["try_count"], 1)
        self.assertEqual(result["execution"]["attempts_used"], 1)

    def test_live_pipeline_runs_each_mechanism_as_its_own_h0(self):
        generation_queries = []
        retrieval_calls = []

        def fake_generate(query, **kwargs):
            generation_queries.append(query)
            mechanism = (
                "Bulk Download" if "bulk download" in query.casefold()
                else "REST API"
            )
            return {
                "data_source_name": f"Example {mechanism}",
                "brief_description": "Test handbook",
                "handbook": f"Use only the {mechanism} workflow.",
                "code_example": "print('example')",
                "website": "https://example.test/docs",
                "requires_key": False,
                "key_name": "",
                "_verification": {
                    "verified": True,
                    "status": "passed",
                    "attempts": 1,
                    "files": [],
                    "note": "Verified.",
                },
            }

        def fake_semantic(*args, **kwargs):
            return {
                "task_completed": True,
                "confidence": 1,
                "summary": "The mechanism-specific output satisfies the task.",
                "mechanism_correctness": "pass",
                "parameter_correctness": "pass",
                "spatial_query_correctness": "not_applicable",
                "temporal_query_correctness": "not_applicable",
                "output_correctness": "pass",
                "checks": [],
                "failure_category": "",
                "should_refine_handbook": False,
                "handbook_gap": "",
            }

        class FakeRetriever:
            def __init__(self, **kwargs):
                self.model = ""

            def run_controlled_handbook_trial(self, **kwargs):
                retrieval_calls.append(kwargs)
                os.makedirs(kwargs["output_dir"], exist_ok=True)
                path = os.path.join(kwargs["output_dir"], "result.geojson")
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump({"type": "FeatureCollection", "features": [
                        {"type": "Feature", "properties": {"id": 1},
                         "geometry": None}
                    ]}, handle)
                return {
                    "status": "passed",
                    "error": "",
                    "execution": {"attempts_used": 1, "success": True},
                    "downloaded_files": [{
                        "name": "result.geojson",
                        "path": path,
                        "size_bytes": os.path.getsize(path),
                    }],
                    "http_requests": [{
                        "attempt": 1, "method": "GET",
                        "url": "https://example.test/data",
                    }],
                    "generated_code": "print('retrieval')",
                }

        user_id = "f" * 64
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_root:
            study_root = os.path.join(temp_root, "studies")
            output_root = os.path.join(temp_root, "outputs")
            with mock.patch.object(runner.store, "STORE_ROOT", study_root), \
                    mock.patch.object(runner, "OUTPUT_ROOT", output_root), \
                    mock.patch.object(
                        runner, "DataRetrieverAgent", FakeRetriever), \
                    mock.patch.object(
                        runner.user_data_sources, "write_user_source",
                        side_effect=lambda _user, source: source[
                            "data_source_name"].lower().replace(" ", "-")), \
                    mock.patch.object(
                        handbook_generator, "generate_handbook",
                        side_effect=fake_generate), \
                    mock.patch.object(
                        handbook_generator, "evaluate_retrieval_result",
                        side_effect=fake_semantic):
                study = runner.store.new_study("Mechanism isolation")
                study.update({
                    "source_name": "Example source",
                    "documentation_url": "https://example.test/docs",
                    "retrieval_task": "Retrieve features as GeoJSON.",
                    "mechanisms": ["rest", "bulk"],
                })
                study["controls"]["max_refinements"] = 0
                saved = runner.store.save_study(user_id, study)
                events, _ = exhaust(runner.run_experiment_pipeline(
                    user_id, "api-key", saved["id"], restart=True,
                    confirm_mechanisms=["rest", "bulk"]))
                completed = runner.store.get_study(user_id, saved["id"])

        self.assertEqual(
            [item["version"] for item in completed["handbook_versions"]],
            ["REST-H0", "BULK-H0"],
        )
        self.assertEqual(
            [item["mechanism"] for item in completed["trials"]],
            ["rest", "bulk"],
        )
        self.assertEqual(
            [item["outcome"] for item in
             completed["pipeline_state"]["conditions"]],
            ["passed", "passed"],
            completed["pipeline_state"],
        )
        self.assertEqual(completed["pipeline_state"]["outcome"], "passed")
        self.assertEqual(len(generation_queries), 2)
        self.assertIn("REST API", generation_queries[0])
        self.assertIn("Bulk Download", generation_queries[1])
        self.assertTrue(all(call["try_count"] == 1
                            for call in retrieval_calls))
        self.assertTrue(any(event["type"] == "condition_started"
                            for event in events))

    def test_pipeline_pauses_for_retrieval_confirmation(self):
        retrieval_calls = []

        def fake_generate(query, **kwargs):
            return {
                "data_source_name": "Example REST API",
                "brief_description": "Test handbook",
                "handbook": "Use only the REST API workflow.",
                "code_example": "print('example')",
                "website": "https://example.test/docs",
                "requires_key": False,
                "key_name": "",
                "_verification": {
                    "verified": True, "status": "passed", "attempts": 1,
                    "files": [], "note": "Verified.",
                },
            }

        def fake_semantic(*args, **kwargs):
            return {
                "task_completed": True, "confidence": 1,
                "summary": "Satisfied.", "mechanism_correctness": "pass",
                "parameter_correctness": "pass",
                "spatial_query_correctness": "not_applicable",
                "temporal_query_correctness": "not_applicable",
                "output_correctness": "pass", "checks": [],
                "failure_category": "", "should_refine_handbook": False,
                "handbook_gap": "",
            }

        class FakeRetriever:
            def __init__(self, **kwargs):
                self.model = ""

            def run_controlled_handbook_trial(self, **kwargs):
                retrieval_calls.append(kwargs)
                os.makedirs(kwargs["output_dir"], exist_ok=True)
                path = os.path.join(kwargs["output_dir"], "result.geojson")
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump({"type": "FeatureCollection", "features": []},
                              handle)
                return {
                    "status": "passed", "error": "",
                    "execution": {"attempts_used": 1, "success": True},
                    "downloaded_files": [{
                        "name": "result.geojson", "path": path,
                        "size_bytes": os.path.getsize(path),
                    }],
                    "http_requests": [],
                    "generated_code": "print('retrieval')",
                }

        user_id = "e" * 64
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_root:
            study_root = os.path.join(temp_root, "studies")
            output_root = os.path.join(temp_root, "outputs")
            with mock.patch.object(runner.store, "STORE_ROOT", study_root), \
                    mock.patch.object(runner, "OUTPUT_ROOT", output_root), \
                    mock.patch.object(
                        runner, "DataRetrieverAgent", FakeRetriever), \
                    mock.patch.object(
                        runner.user_data_sources, "write_user_source",
                        side_effect=lambda _user, source: source[
                            "data_source_name"].lower().replace(" ", "-")), \
                    mock.patch.object(
                        handbook_generator, "generate_handbook",
                        side_effect=fake_generate), \
                    mock.patch.object(
                        handbook_generator, "evaluate_retrieval_result",
                        side_effect=fake_semantic):
                study = runner.store.new_study("Confirmation gate")
                study.update({
                    "source_name": "Example source",
                    "documentation_url": "https://example.test/docs",
                    "retrieval_task": "Retrieve features as GeoJSON.",
                    "mechanisms": ["rest"],
                })
                study["controls"]["max_refinements"] = 0
                saved = runner.store.save_study(user_id, study)

                # No confirmation supplied: the pipeline must stop right
                # after H0 and never call the retrieval executor.
                paused_events, _ = exhaust(runner.run_experiment_pipeline(
                    user_id, "api-key", saved["id"], restart=True))
                paused_study = runner.store.get_study(user_id, saved["id"])

                self.assertTrue(any(
                    event["type"] == "retrieval_confirmation_required"
                    and event["mechanism"] == "rest"
                    for event in paused_events))
                self.assertEqual(retrieval_calls, [])
                self.assertEqual(
                    paused_study["pipeline_state"]["conditions"][0]["status"],
                    "awaiting_test_confirmation")
                self.assertEqual(
                    [item["version"]
                     for item in paused_study["handbook_versions"]],
                    ["REST-H0"])

                # Resuming with confirmation for "rest" must now run the
                # retrieval test and complete the condition.
                resumed_events, _ = exhaust(runner.run_experiment_pipeline(
                    user_id, "api-key", saved["id"],
                    confirm_mechanisms=["rest"]))
                resumed_study = runner.store.get_study(user_id, saved["id"])

        self.assertEqual(len(retrieval_calls), 1)
        self.assertEqual(
            resumed_study["pipeline_state"]["conditions"][0]["outcome"],
            "passed")
        self.assertTrue(any(
            event["type"] == "condition_finished" for event in resumed_events))


if __name__ == "__main__":
    unittest.main()
