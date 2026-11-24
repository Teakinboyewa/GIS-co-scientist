"""The Spatial Analysis workflow designer judges which datasets fit the question.

It used to see only file names and row counts, and its prompt told it to give
every dataset a node and a branch, so layers left over from earlier requests
ended up in the workflow. It now gets the task breakdown, a fitness profile of
each dataset (extent, CRS, period, categories, missing values) and is told to
leave out datasets that do not fit.
"""
import json
import os
import tempfile
import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from utils.toolbase import dataset_profile
from utils.toolbase.dataset_profile import format_profile, profile_dataset
import utils.prompts as prompts
from agents.task_manager import geoprocessing_workflow_generator as g


class DatasetProfile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        dataset_profile._profile.cache_clear()

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def test_vector_extent_is_reported_in_lon_lat_whatever_the_crs(self):
        # Two squares in UTM zone 17N (around central Pennsylvania)
        gdf = gpd.GeoDataFrame(
            {"NAME": ["Centre", "Blair"], "STATEFP": ["42", "42"], "pop": [158000, None]},
            geometry=[box(250000, 4500000, 260000, 4510000), box(240000, 4490000, 250000, 4500000)],
            crs="EPSG:32617")
        p = self.path("pa.gpkg")
        gdf.to_file(p)
        prof = profile_dataset(p)
        self.assertEqual(prof["kind"], "vector")
        self.assertEqual(prof["rows"], 2)
        self.assertIn("32617", prof["crs"])
        self.assertIn("Polygon", prof["geometry_type"])
        from pyproj import Transformer
        expected = Transformer.from_crs("EPSG:32617", "EPSG:4326", always_xy=True).transform_bounds(
            240000, 4490000, 260000, 4510000)
        for got, want in zip(prof["extent_lonlat"], expected):
            self.assertAlmostEqual(got, want, places=3)
        self.assertTrue(-85 < prof["extent_lonlat"][0] < -83, "should be degrees, not metres")
        cols = prof["columns"]
        self.assertTrue(cols["NAME"]["unique"])
        self.assertIn("42 (2)", cols["STATEFP"]["top"])
        self.assertEqual(cols["pop"]["missing"], "1/2")

    def test_epoch_times_become_a_date_range(self):
        ms = [pd.Timestamp("2026-10-01", tz="UTC").value // 10**6,
              pd.Timestamp("2026-10-31", tz="UTC").value // 10**6]
        gdf = gpd.GeoDataFrame({"time": ms, "mag": [1.2, 3.4]},
                               geometry=[Point(-120, 36), Point(-118, 34)], crs="EPSG:4326")
        p = self.path("quakes.geojson")
        gdf.to_file(p, driver="GeoJSON")
        col = profile_dataset(p)["columns"]["time"]
        self.assertEqual(col["range"], ["2026-10-01", "2026-10-31"])
        self.assertIn("epoch ms", col["dtype"])

    def test_table_with_coordinates_gets_an_extent_and_date_range(self):
        p = self.path("pm25.csv")
        pd.DataFrame({"latitude": [32.6, 41.7], "longitude": [-124.2, -115.5],
                      "date_local": ["2019-01-01", "2019-12-31"], "value": [5.0, 12.5]}).to_csv(p, index=False)
        prof = profile_dataset(p)
        self.assertEqual(prof["kind"], "table")
        self.assertEqual(prof["extent_lonlat"], [-124.2, 32.6, -115.5, 41.7])
        self.assertEqual(prof["columns"]["date_local"]["range"], ["2019-01-01", "2019-12-31"])
        self.assertEqual(prof["columns"]["value"]["range"], [5, 12.5])

    def test_unreadable_files_never_raise(self):
        p = self.path("broken.geojson")
        with open(p, "w") as f:
            f.write("not json")
        self.assertIn("error", profile_dataset(p))
        self.assertIn("error", profile_dataset(self.path("missing.gpkg")))
        self.assertIn("could not profile", format_profile("broken", profile_dataset(p)))

    def test_format_is_compact_and_names_the_coverage(self):
        gdf = gpd.GeoDataFrame({"NAME": ["A", "B"]}, geometry=[Point(-120, 36), Point(-118, 34)],
                               crs="EPSG:4326")
        p = self.path("pts.geojson")
        gdf.to_file(p, driver="GeoJSON")
        text = format_profile("pts", profile_dataset(p))
        self.assertIn("extent (lon/lat): -120.0, 34.0 to -118.0, 36.0", text)
        self.assertIn("EPSG:4326", text)
        self.assertLess(len(text), 600)


class WorkflowDesignerPrompt(unittest.TestCase):
    def setUp(self):
        dataset_profile._profile.cache_clear()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        p = os.path.join(self.dir.name, "pa_counties.geojson")
        gpd.GeoDataFrame({"STATEFP": ["42"]}, geometry=[box(-80, 40, -79, 41)], crs="EPSG:4326").to_file(
            p, driver="GeoJSON")
        self.ctx = {"artifacts": {
            "data_registry": [{"file": "pa_counties", "path": p, "source": "downloaded", "shape": [1, 2]}],
            "rq_breakdown": {
                "intent": "Map PM2.5 by California county",
                "scale": {"spatial_scale": "County level: California", "temporal_scale": "2019"},
                "data_requirement": ["California county polygons"],
                "data_available": [],
                "data_available_sufficiency": "No - no California data is provided.",
            }}}

    def test_breakdown_reaches_the_designer(self):
        text = g._breakdown_context(self.ctx)
        for expected in ("Map PM2.5 by California county", "County level: California", "2019",
                         "California county polygons", "No - no California data"):
            self.assertIn(expected, text)
        self.assertEqual(g._breakdown_context({"artifacts": {}}), "(no task breakdown available)")

    def test_candidates_carry_profiles_and_paths(self):
        text = g._candidate_datasets(self.ctx)
        self.assertIn("### pa_counties", text)
        self.assertIn("extent (lon/lat): -80.0, 40.0 to -79.0, 41.0", text)
        self.assertIn("STATEFP", text)
        self.assertIn("path: ", text)

    def test_designer_is_told_to_leave_out_unfit_data(self):
        self.assertIn("CANDIDATES, not requirements", g.DATA_FITNESS_INSTRUCTIONS)
        self.assertIn("Leave every other dataset out", g.DATA_FITNESS_INSTRUCTIONS)
        rules = " ".join(prompts.geoprocessing_workflow_generator_requirement)
        self.assertIn("Use only the provided datasets that fit the question", rules)
        self.assertNotIn("When multiple independent data sources are provided", rules)
        self.assertNotIn("NO explanation or conversation outside the code block", rules)
        self.assertIn("Use only the provided datasets that fit", prompts.geoprocessing_workflow_generator_task_prefix)
        system_prompt, _ = g._build_geoproecessing_workflow_generation_prompt(
            {"rq": "q", "task_id": "t", "output_dir": self.dir.name, **self.ctx})
        self.assertNotIn("each data is a node", system_prompt)

    def test_profile_tool_is_available(self):
        from utils.toolbase.geoprocessing_workflow_generator_tools import (
            _build_geoprocessing_workflow_design_tools)
        tools = {t.name: t for t in _build_geoprocessing_workflow_design_tools(self.ctx)}
        self.assertIn("profile_dataset", tools)
        self.assertIn("extent (lon/lat)", tools["profile_dataset"].invoke({"dataset_name": "pa_counties"}))
        self.assertIn("not found", tools["profile_dataset"].invoke({"dataset_name": "nope"}))


if __name__ == "__main__":
    unittest.main()
