"""DataRetrieverAgent.resolve_source_name: map the model's source pick onto a
handbook, tolerating echoed list numbers and handbook IDs, but never guessing
between ambiguous entries."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.data_agent.DataRetrieverAgent import DataRetrieverAgent  # noqa: E402

CENSUS = "U.S. Census Bureau APIs (Census Data API + TIGERweb / TIGER/Line)"
CATALOG = {
    CENSUS: {"ID": "US_Census_APIs"},
    "OpenStreetMap": {"ID": "OpenStreetMap"},
    "USGS Earthquake Catalog": {"ID": "USGS_Earthquake"},
    "TIGER/Line Shapefiles (2025)": {"ID": "TIGER_2025"},
    "TIGER/Line Shapefiles (2025).": {"ID": "TIGER_2025_dup"},
    "Unknown": {"ID": "Unknown"},
}


def resolve(name):
    # resolve_source_name only uses the static _normalize_source_name, so the
    # class can stand in for an instance (avoids the agent's heavy __init__).
    return DataRetrieverAgent.resolve_source_name(DataRetrieverAgent, name, CATALOG)


class ResolveSourceNameTest(unittest.TestCase):
    def test_exact_name(self):
        self.assertEqual(resolve(CENSUS), (CENSUS, "US_Census_APIs"))

    def test_echoed_list_number_and_full_stop(self):
        # The case from conversation ddf64f5d: "8. <name>."
        self.assertEqual(resolve(f"8. {CENSUS}."), (CENSUS, "US_Census_APIs"))
        self.assertEqual(resolve(f"12) {CENSUS}"), (CENSUS, "US_Census_APIs"))

    def test_case_and_spacing(self):
        self.assertEqual(resolve("  openstreetmap "), ("OpenStreetMap", "OpenStreetMap"))

    def test_handbook_id(self):
        self.assertEqual(resolve("USGS_Earthquake"),
                         ("USGS Earthquake Catalog", "USGS_Earthquake"))
        self.assertEqual(resolve("usgs earthquake"),
                         ("USGS Earthquake Catalog", "USGS_Earthquake"))

    def test_unknown_and_missing(self):
        self.assertEqual(resolve("Unknown")[1], "Unknown")
        self.assertEqual(resolve("USGS ComCat API")[1], "Unknown")
        self.assertEqual(resolve("")[1], "Unknown")

    def test_ambiguous_stays_unrecognized(self):
        # Two entries normalize to the same key: never guess between them.
        self.assertEqual(resolve("tiger/line shapefiles (2025)")[1], "Unknown")

    def test_leading_year_not_stripped(self):
        # Only "N." / "N)" list markers are removed, not a name's own number.
        catalog = {"2020 Decennial Census": {"ID": "Census2020"}, "Unknown": {"ID": "Unknown"}}
        got = DataRetrieverAgent.resolve_source_name(
            DataRetrieverAgent, "2020 decennial census", catalog)
        self.assertEqual(got, ("2020 Decennial Census", "Census2020"))


if __name__ == "__main__":
    unittest.main()
