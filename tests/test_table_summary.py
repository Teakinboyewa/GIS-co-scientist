"""Output-file summaries must report the file's real size and statistics.

Found on pilot task T01: every model used 900 blocks, but the manuscript
said "200 blocks" and its descriptive statistics came from the first 200
rows, because the summary read a capped frame and reported its shape.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd


class TableSummaries(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        # First 200 rows have a different mean from the rest, so a capped
        # read gives visibly wrong statistics.
        values = np.concatenate([np.zeros(200), np.full(700, 10.0)]) + rng.normal(0, 0.1, 900)
        self.df = pd.DataFrame({"block_id": range(900), "lst_c": values})
        self.csv = os.path.join(self.dir, "blocks.csv")
        self.df.to_csv(self.csv, index=False)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _gpkg(self):
        import geopandas as gpd
        from shapely.geometry import Point
        path = os.path.join(self.dir, "blocks.gpkg")
        gpd.GeoDataFrame(self.df, geometry=[Point(i, 0) for i in range(900)],
                         crs="EPSG:26917").to_file(path, driver="GPKG")
        return path

    def test_csv_reports_all_rows_and_full_statistics(self):
        from utils.table_summary import load_table, size_fields
        frame, total, sampled = load_table(self.csv)
        self.assertEqual(total, 900)
        self.assertFalse(sampled)
        self.assertAlmostEqual(frame["lst_c"].mean(), self.df["lst_c"].mean())
        self.assertEqual(size_fields(frame, total, sampled)["shape"], [900, 2])

    def test_spatial_reports_all_rows(self):
        from utils.table_summary import load_table
        frame, total, sampled = load_table(self._gpkg())
        self.assertEqual((len(frame), total, sampled), (900, 900, False))

    def test_a_capped_read_says_it_is_a_sample(self):
        from utils.table_summary import load_table, size_fields
        for path in (self.csv, self._gpkg()):
            with self.subTest(path=os.path.basename(path)):
                frame, total, sampled = load_table(path, max_rows=200)
                fields = size_fields(frame, total, sampled)
                self.assertTrue(sampled)
                self.assertEqual(fields["shape"][0], 900)
                self.assertIn("first 200 rows only", fields["statistics_note"])
                self.assertIn("900", fields["statistics_note"])

    def test_manuscript_brief_uses_true_size(self):
        from agents.reporting_agent import manuscript_tools as mt
        stats = mt._extract_file_stats({self.csv, self._gpkg()})
        self.assertEqual(stats["blocks.csv"]["shape"], [900, 2])
        self.assertEqual(stats["blocks.gpkg"]["shape"][0], 900)
        self.assertNotIn("statistics_note", stats["blocks.csv"])
        # describe() is computed on all rows: the mean reflects the 700 rows of 10s.
        self.assertIn("7.7", stats["blocks.csv"]["describe"])

    def test_result_analysis_file_readers_use_true_size(self):
        import json
        for module in ("result_discussion", "result_presentation"):
            with self.subTest(module=module):
                src = open(os.path.join("agents", "result_analysis_agent", f"{module}.py"),
                           encoding="utf-8").read()
                self.assertNotIn("nrows=100", src)
                self.assertNotIn("rows=100", src)
                self.assertIn("load_table(file_path)", src)


if __name__ == "__main__":
    unittest.main()
