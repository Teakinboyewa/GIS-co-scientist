"""The spatial-integration conditions must differ only in delivery.

The study attributes differences between B1, B2 and B3 to *how* knowledge
reaches the agent. That holds only if (1) every delivery carries every piece
of the shared knowledge base, (2) B0 is the published system untouched, and
(3) the condition a run claims is the condition it ran.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from agents.spatial_reasoning import conditions, knowledge_base as kb


def _content_fields(entry):
    """Every substantive sentence of an entry, whitespace-normalized."""
    d, r = entry["diagnostic"], entry["response"]
    return [kb._clean(x) for x in (entry["title"], entry["implication"], d["name"],
                                   d["procedure"], r["if_detected"], r["if_not_detected"])]


class TheKnowledgeBaseIsComplete(unittest.TestCase):

    def test_every_concept_and_assumption_field_is_covered(self):
        self.assertEqual({e["concept"] for e in kb.entries()}, set(kb.CONCEPTS))
        self.assertEqual({e["assumption_field"] for e in kb.entries()}, set(kb.ASSUMPTION_FIELDS))

    def test_fingerprint_is_stable(self):
        self.assertEqual(kb.fingerprint(), kb.fingerprint())
        self.assertEqual(len(kb.fingerprint()), 64)


class EveryDeliveryCarriesTheSameContent(unittest.TestCase):

    def test_prompt_checklist_contains_every_field(self):
        text = kb.render_prompt_checklist()
        for e in kb.entries():
            for field in _content_fields(e):
                self.assertIn(field, text, f"B1 is missing part of {e['id']}")
            # The condition is rephrased as 'when ...', so check its body.
            self.assertIn(kb._clean(e["condition"])[1:], text)

    def test_skill_files_contain_every_field(self):
        text = "\n".join(kb.render_skill(c) for c in kb.CONCEPTS)
        for e in kb.entries():
            for field in _content_fields(e) + [kb._clean(e["condition"])]:
                self.assertIn(field, text, f"B3 is missing part of {e['id']}")

    def test_retrieval_can_reach_every_entry_and_returns_it_whole(self):
        for e in kb.entries():
            hits = kb.retrieve(e["title"])
            self.assertIn(e["id"], [h["id"] for h in hits],
                          f"B2 cannot retrieve {e['id']} by its own title")
            text = kb.entry_text(e)
            for field in _content_fields(e) + [kb._clean(e["condition"])]:
                self.assertIn(field, text)

    def test_a_pronoun_does_not_match_morans_i(self):
        """'I' in a question used to match the 'I' of Moran's I and outrank GWR guidance."""
        self.assertEqual(kb.retrieve("should I use GWR")[0]["id"], "unwarranted_local_model")

    def test_retrieval_is_deterministic(self):
        q = "residual autocorrelation in a county regression"
        self.assertEqual([h["id"] for h in kb.retrieve(q)], [h["id"] for h in kb.retrieve(q)])

    def test_retrieval_and_skill_tools_log_access(self):
        log = []
        _, tools = kb.planner_delivery("retrieval", log)
        tools[0].invoke({"query": "Moran's I residuals"})
        _, tools = kb.planner_delivery("skill", log)
        tools[0].invoke({"name": "spatial-heterogeneity"})
        tools[0].invoke({"name": "nonsense"})
        self.assertEqual([c["tool"] for c in log],
                         ["retrieve_spatial_knowledge", "load_spatial_skill", "load_spatial_skill"])
        self.assertTrue(log[0]["returned"])
        self.assertEqual(log[2]["returned"], [])

    def test_b0_adds_nothing(self):
        self.assertEqual(kb.planner_delivery("none", []), ("", []))


class ConditionsResolveOrRefuse(unittest.TestCase):

    def test_unset_is_b0(self):
        self.assertEqual(conditions.resolve(None)["name"], "B0")
        self.assertEqual(conditions.condition_of({})["name"], "B0")

    def test_unknown_name_is_refused_not_defaulted(self):
        with self.assertRaises(conditions.UnknownCondition):
            conditions.resolve("b1")

    def test_unbuilt_components_are_refused(self):
        """A condition needing a component that is not built must not run
        without it (all are built now; simulate one missing)."""
        built = conditions.IMPLEMENTED_COMPONENTS - {"critic"}
        with mock.patch.object(conditions, "IMPLEMENTED_COMPONENTS", built):
            for name in ("P", "P-no-structure", "P-no-diagnostics"):
                with self.assertRaises(conditions.ConditionNotImplemented, msg=name):
                    conditions.resolve(name)
            self.assertEqual(conditions.resolve("P-no-critic")["name"], "P-no-critic")

    def test_text_level_baselines_resolve(self):
        for name, delivery in (("B0", "none"), ("B1", "prompt"), ("B2", "retrieval"), ("B3", "skill")):
            self.assertEqual(conditions.resolve(name)["kb_delivery"], delivery)


class TheConditionIsRecordedAndReachesThePlanner(unittest.TestCase):

    def setUp(self):
        self.out = tempfile.mkdtemp()
        self.ctx = {
            "rq": "Is park access associated with obesity across tracts?",
            "task_id": "t1", "output_dir": self.out, "logs": [], "settings": {},
            "artifacts": {"rq_breakdown": {"intent": "association", "scale": {}},
                          "data_registry": []},
        }

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def _apply(self, name):
        with mock.patch.object(conditions, "resolve_output_dir", lambda ctx: self.out):
            return conditions.apply_to_ctx(self.ctx, name)

    def test_apply_writes_the_condition_to_the_output_folder(self):
        self._apply("B2")
        saved = json.load(open(os.path.join(self.out, "spatial_condition.json"), encoding="utf-8"))
        self.assertEqual(saved["name"], "B2")
        self.assertEqual(saved["kb_sha256"], kb.fingerprint())
        self.assertEqual(conditions.condition_of(self.ctx)["name"], "B2")

    def test_a_run_on_out_of_date_code_is_flagged(self):
        """Pilot P runs on T04/T09 used a critic fixed on disk a day earlier:
        the web UI had not been restarted. Each run now records it."""
        import agents.spatial_reasoning as pkg
        self._apply("B0")
        self.assertFalse(conditions.condition_of(self.ctx)["code"]["stale"])
        loaded = dict(pkg._LOADED)
        first = next(iter(loaded))
        with mock.patch.dict(pkg._LOADED, {first: "0" * 64}):
            self._apply("P")
        record = json.load(open(os.path.join(self.out, "spatial_condition.json"), encoding="utf-8"))
        self.assertTrue(record["code"]["stale"])
        self.assertEqual(record["code"]["changed_since_load"], [first])
        self.assertTrue(any("out-of-date" in l or "old version" in l for l in self.ctx["logs"]))

    def test_condition_survives_serialization(self):
        """Both pause points restore ctx from JSON-serialized state."""
        self._apply("B3")
        restored = json.loads(json.dumps(self.ctx))
        self.assertEqual(conditions.condition_of(restored)["kb_delivery"], "skill")

    def _planner_inputs(self, name):
        from agents.task_manager import research_plan_generator as rpg
        import utils.prompts as prompts
        self._apply(name)
        seen = {}

        def fake_loop(llm, messages, tools, max_iterations, **kw):
            seen["system"] = messages[0].content
            seen["tools"] = [t.name for t in tools]
            return '{"objectives": []}'

        fake_llm = mock.MagicMock()
        fake_llm.bind_tools.side_effect = lambda tools: fake_llm
        with mock.patch.object(rpg, "GIBDChat", return_value=fake_llm), \
             mock.patch.object(rpg, "_build_research_design_tools", return_value=[]), \
             mock.patch.object(rpg, "_run_agent_loop", fake_loop), \
             mock.patch.object(rpg, "resolve_output_dir", lambda ctx: self.out):
            rpg.generate_research_plan({"ctx": self.ctx})
        return seen, prompts.generate_research_plan_system_prompt

    def test_b0_planner_prompt_is_the_published_one(self):
        seen, published = self._planner_inputs("B0")
        self.assertEqual(seen["system"], published)
        self.assertEqual(seen["tools"], [])

    def test_b1_planner_prompt_carries_the_checklist(self):
        seen, published = self._planner_inputs("B1")
        self.assertTrue(seen["system"].startswith(published))
        self.assertIn(kb.render_prompt_checklist(), seen["system"])

    def test_b2_and_b3_get_their_tools(self):
        seen, _ = self._planner_inputs("B2")
        self.assertEqual(seen["tools"], ["retrieve_spatial_knowledge"])
        seen, _ = self._planner_inputs("B3")
        self.assertEqual(seen["tools"], ["load_spatial_skill"])
        self.assertIn("spatial-dependence", seen["system"])
        access = json.load(open(os.path.join(self.out, "spatial_knowledge_access.json"),
                                encoding="utf-8"))
        self.assertEqual(access["condition"], "B3")


class TheConditionsPageShowsWhatThePlannerGets(TheConditionIsRecordedAndReachesThePlanner):
    """The WebUI page and the downloads must be the planner's real input."""

    def test_page_prompt_is_the_prompt_sent(self):
        from agents.spatial_reasoning import inspection
        for name in ("B0", "B1", "B2", "B3"):
            with self.subTest(name=name):
                seen, _ = self._planner_inputs(name)
                shown = inspection.describe_condition(name)
                self.assertEqual(shown["system_prompt"], seen["system"])
                self.assertEqual([t["name"] for t in shown["tools"]], seen["tools"])

    def test_each_run_saves_its_planner_input(self):
        from agents.spatial_reasoning import inspection
        seen, _ = self._planner_inputs("B1")
        with open(os.path.join(self.out, "planner_system_prompt.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), seen["system"])
        root, task = os.path.split(self.out)
        rec = inspection.run_record(root, task)
        self.assertEqual(rec["condition"]["name"], "B1")
        self.assertEqual({f["key"] for f in rec["files"]},
                         {"condition", "system_prompt", "user_prompt", "tools", "knowledge_access"})
        self.assertEqual(rec["assumptions"], [])          # baselines state no assumptions
        self.assertIsNotNone(inspection.run_zip(root, task))

    def test_unbuilt_conditions_are_described_without_a_prompt(self):
        from agents.spatial_reasoning import inspection
        with mock.patch.object(conditions, "IMPLEMENTED_COMPONENTS", set()):
            d = inspection.describe_condition("P")
            self.assertFalse(d["built"])
            self.assertNotIn("system_prompt", d)
            self.assertIn("Not built yet", inspection.condition_markdown("P"))

    def test_p_is_described_by_its_components(self):
        from agents.spatial_reasoning import inspection
        d = inspection.describe_condition("P")
        self.assertEqual(d["addition"], "")                # the planner prompt is the published one
        self.assertEqual([c["name"] for c in d["components"]],
                         ["(a) Structured assumptions", "(b) Grounded diagnostics",
                          "Plan revision", "(c) Spatial critic"])
        self.assertEqual(inspection.describe_condition("B1")["components"], [])

    def test_run_downloads_reject_unsafe_task_ids(self):
        from agents.spatial_reasoning import inspection
        for bad in ("..", "../x", "a/b", ""):
            with self.assertRaises(ValueError, msg=bad):
                inspection.run_record(self.out, bad)
        self.assertIsNone(inspection.run_file(self.out, "t1", "not_a_key"))


if __name__ == "__main__":
    unittest.main()
