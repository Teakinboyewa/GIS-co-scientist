"""Result tables must reach Result Analysis and the Manuscript whole.

Seen in conversation ae911bea...: a re-run regression step wrote OLS, spatial
lag (SAR) and spatial error (SEM) estimates to one 16-row, 20-column CSV. The
manuscript brief showed it as df.head(5).to_string()[:1500]; the padding of
20 columns used up the 1500 characters before row 4, so the writer saw only
the OLS rows and reported that the spatial models gave no estimates.
"""
import ast
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.table_rows import table_rows_text, FULL_TABLE_MAX_ROWS  # noqa: E402

COLS = ["record_type", "model", "term", "estimate", "std_error", "z_statistic", "p_value",
        "ci95_lower", "ci95_upper", "n", "aic", "bic", "log_likelihood", "r_squared_or_pseudo_r_squared",
        "residual_morans_I", "residual_morans_p_permutation", "selected_model", "interpretation",
        "weights_specification", "sample_size"]


def regression_table():
    rows = []
    for model, param in (("OLS", None), ("Spatial lag (SAR)", "rho"), ("Spatial error (SEM)", "lambda")):
        for term in ("CONSTANT", "annual_mean_pm25"):
            rows.append({"record_type": "coefficient", "model": model, "term": term,
                         "estimate": 0.2656430787568001, "std_error": 0.26262337369329225})
        if param:
            rows.append({"record_type": "spatial_parameter", "model": model, "term": param, "estimate": 0.0830121})
        rows.append({"record_type": "model_diagnostics", "model": model, "term": "fit",
                     "aic": 133.3346904616265, "bic": 135.60567889348482})
    for t in ("LM lag", "LM error", "Robust LM lag", "Robust LM error", "scope"):
        rows.append({"record_type": "test", "model": "OLS", "term": t, "p_value": 0.78})
    return pd.DataFrame(rows, columns=COLS)


class TableRowsText(unittest.TestCase):
    def test_the_reported_table_is_passed_whole(self):
        df = regression_table()
        self.assertEqual(len(df), 16)
        text = table_rows_text(df)
        for needle in ("Spatial lag (SAR)", "Spatial error (SEM)", "lambda", "rho", "Robust LM error"):
            self.assertIn(needle, text)
        self.assertIn("[all 16 rows]", text)
        # what the old preview showed
        old = df.head(5).to_string()[:1500]
        self.assertNotIn("Spatial error (SEM)", old)

    def test_large_tables_keep_a_labelled_preview(self):
        df = pd.DataFrame({"a": range(500), "b": range(500)})
        text = table_rows_text(df)
        self.assertIn("[showing the first 5 of 500 rows]", text)
        self.assertEqual(text.count("\n"), 1 + 5 + 1)

    def test_partially_loaded_file_reports_true_count(self):
        df = pd.DataFrame({"a": range(10)})
        self.assertIn("[showing the first 5 of 9000 rows]", table_rows_text(df, total_rows=9000))

    def test_char_cap_cuts_on_a_row_boundary_and_says_so(self):
        df = pd.DataFrame({"t": ["x" * 300] * FULL_TABLE_MAX_ROWS})
        text = table_rows_text(df, max_chars=1000)
        body = text.rsplit("[", 1)[0]
        self.assertTrue(body.endswith("\n"))
        self.assertRegex(text, r"\[showing the first \d of 60 rows\]")

    def test_geometry_is_left_out(self):
        df = pd.DataFrame({"a": [1], "geometry": ["POLYGON ((0 0, 1 0, 1 1, 0 0))"]})
        self.assertNotIn("POLYGON", table_rows_text(df))


class ManuscriptBrief(unittest.TestCase):
    def test_brief_carries_every_row(self):
        from agents.reporting_agent import manuscript_tools as mt
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "county_air_obesity_spatial_regression.csv")
            regression_table().to_csv(fp, index=False)
            st = mt._extract_file_stats([fp])["county_air_obesity_spatial_regression.csv"]
        self.assertIn("Spatial error (SEM)", st["head"])
        self.assertIn("lambda", st["head"])


class ResultAnalysisReaders(unittest.TestCase):
    def test_no_head5_previews_left(self):
        for rel in ("agents/result_analysis_agent/result_presentation.py",
                    "agents/result_analysis_agent/result_discussion.py",
                    "agents/reporting_agent/manuscript_tools.py"):
            src = (ROOT / rel).read_text(encoding="utf-8")
            ast.parse(src)
            self.assertNotIn(".head(5).to_string()", src, rel)
            self.assertIn("table_rows_text(", src, rel)


if __name__ == "__main__":
    unittest.main()
