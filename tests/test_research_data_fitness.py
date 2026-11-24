"""The Research planner and reviewer judge which datasets fit the question.

Same check as Spatial Analysis (tests/test_geoprocessing_data_fitness.py): the
planner gets the breakdown's relevant-data list and sufficiency verdict, a
fitness profile of every candidate dataset, and is told to plan only with
datasets that fit and to record each one in a data_fitness list, which the
final plan keeps. The LLM call itself is replaced, so these run offline.
"""
import json
import os
import shutil
import unittest
import uuid
from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import box

import utils.prompts as prompts
from utils.config import load_configurations
from utils.toolbase import dataset_profile
from agents.task_manager import research_plan_generator as rpg


class ResearchPlannerDataFitness(unittest.TestCase):
    def setUp(self):
        dataset_profile._profile.cache_clear()
        self.task_id = f"zz_test_{uuid.uuid4().hex[:8]}"
        self.out = os.path.join(load_configurations().output_dir, self.task_id)
        os.makedirs(self.out)
        self.addCleanup(shutil.rmtree, self.out, True)
        pa = os.path.join(self.out, "pa_counties.geojson")
        gpd.GeoDataFrame({"STATEFP": ["42"]}, geometry=[box(-80, 40, -79, 41)], crs="EPSG:4326").to_file(
            pa, driver="GeoJSON")
        self.ctx = {
            "rq": "PM2.5 by California county, 2019", "task_id": self.task_id, "output_dir": self.out,
            "logs": [], "settings": {},
            "artifacts": {
                "data_registry": [{"file": "pa_counties", "path": pa, "source": "downloaded", "shape": [1, 2]}],
                "rq_breakdown": {
                    "intent": "Map PM2.5 by California county", "category": "Descriptive",
                    "scale": {"spatial_scale": "County level: California", "temporal_scale": "2019"},
                    "data_requirement": ["California county polygons"],
                    "data_available": ["none of the inputs cover California"],
                    "data_available_sufficiency": "No - no California data is provided.",
                }}}

    def _planner_request(self):
        seen = {}

        def fake_loop(llm_with_tools, messages, tools, max_iterations, **kw):
            seen["user"] = messages[-1].content
            seen["tools"] = [t.name for t in tools]
            return json.dumps({"objectives": [], "data_fitness": [
                {"source_id": "pa_counties", "used": False, "reason": "Pennsylvania, not California"}]})

        with patch.object(rpg, "_run_agent_loop", fake_loop):
            state = rpg.generate_research_plan({"ctx": self.ctx, "generated_research_plan": "",
                                                "review_result": "", "final_research_plan": ""})
        return seen, state

    def test_planner_sees_the_breakdown_verdict_and_dataset_profiles(self):
        seen, _ = self._planner_request()
        user = seen["user"]
        self.assertIn("none of the inputs cover California", user)
        self.assertIn("Sufficiency verdict: No - no California data is provided.", user)
        self.assertIn("### pa_counties", user)
        self.assertIn("extent (lon/lat): -80.0, 40.0 to -79.0, 41.0", user)
        self.assertIn("CANDIDATES, not requirements", user)
        self.assertIn('"data_fitness"', user)
        self.assertNotIn("**Available datasets:**", user)
        self.assertIn("profile_dataset", seen["tools"])

    def test_final_plan_keeps_the_data_fitness_list(self):
        _, state = self._planner_request()
        state["review_result"] = {"objectives": [], "review_summary": {}}   # reviewer dropped it
        rpg.finalize_research_plan(state)
        final = self.ctx["artifacts"]["research_plan_final"]
        self.assertEqual(final["data_fitness"][0]["source_id"], "pa_counties")
        self.assertFalse(final["data_fitness"][0]["used"])
        with open(os.path.join(self.out, "research_plan_final.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["data_fitness"], final["data_fitness"])

    def test_reviewer_checks_data_fitness_with_the_profile_tool(self):
        self.assertIn("**Data fitness**", prompts.review_research_plan_system_prompt)
        self.assertIn("profile_dataset", prompts.review_research_plan_system_prompt)
        from utils.toolbase.research_plan_review_tools import _build_research_plan_review_tools
        tools = {t.name: t for t in _build_research_plan_review_tools(self.ctx)}
        self.assertIn("extent (lon/lat)", tools["profile_dataset"].invoke({"dataset_name": "pa_counties"}))

    def test_both_modes_share_one_set_of_checks(self):
        from agents.task_manager import geoprocessing_workflow_generator as g
        self.assertIn(dataset_profile.DATA_FITNESS_CHECKS, g.DATA_FITNESS_INSTRUCTIONS)
        self.assertIn(dataset_profile.DATA_FITNESS_CHECKS, rpg.RESEARCH_DATA_FITNESS_INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
