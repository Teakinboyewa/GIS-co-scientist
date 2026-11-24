"""classify_delivered_files: only the requested result counts as a dataset;
copies, raw archive extracts and scratch files are supporting."""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.data_agent.DataRetrieverAgent import (  # noqa: E402
    SCRATCH_DIR_NAME, classify_delivered_files)


class ClassifyDeliveredFilesTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def make(self, rel, content="x"):
        path = os.path.join(self.dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path

    def names(self, paths):
        return sorted(os.path.relpath(p, self.dir).replace("\\", "/") for p in paths)

    def test_tiger_county_run_from_conversation(self):
        # The files one real run wrote (conversation ddf64f5d).
        files = [
            self.make("tigerline_2025_county_CA.geojson"),
            self.make("tigerline_2025_county_CA.gpkg"),
            self.make("tigerline_2025_county_CA_metadata.json", json.dumps({"title": "meta"})),
            self.make("tl_2025_us_county.zip"),
        ] + [self.make(f"tl_2025_us_county_extracted/tl_2025_us_county{e}")
             for e in (".shp", ".dbf", ".shx", ".prj", ".cpg", ".shp.iso.xml", ".shp.ea.iso.xml")]
        datasets, supporting = classify_delivered_files(files)
        self.assertEqual(self.names(datasets), ["tigerline_2025_county_CA.geojson"])
        self.assertIn("tigerline_2025_county_CA.gpkg", self.names(supporting))
        self.assertIn("tl_2025_us_county_extracted/tl_2025_us_county.shp", self.names(supporting))
        self.assertIn("tl_2025_us_county.zip", self.names(supporting))
        self.assertEqual(len(datasets) + len(supporting), len(files))

    def test_earthquake_run_metadata_json_is_supporting(self):
        files = [
            self.make("usgs_earthquakes_california_2026-10.geojson", '{"type": "FeatureCollection"}'),
            self.make("usgs_earthquakes_california_2026-10.metadata.json", json.dumps({"title": "m"})),
        ]
        datasets, _ = classify_delivered_files(files)
        self.assertEqual(self.names(datasets), ["usgs_earthquakes_california_2026-10.geojson"])

    def test_extract_alone_is_the_result(self):
        files = [self.make("tl_2025_us_county.zip"),
                 self.make("tl_2025_us_county_extracted/tl_2025_us_county.shp")]
        datasets, _ = classify_delivered_files(files)
        self.assertEqual(self.names(datasets), ["tl_2025_us_county_extracted/tl_2025_us_county.shp"])

    def test_scratch_folder_is_supporting(self):
        files = [self.make("roads.gpkg"),
                 self.make(f"{SCRATCH_DIR_NAME}/page_1.geojson", '{"type": "FeatureCollection"}'),
                 self.make(f"{SCRATCH_DIR_NAME}/raw.csv")]
        datasets, supporting = classify_delivered_files(files)
        self.assertEqual(self.names(datasets), ["roads.gpkg"])
        self.assertEqual(len(supporting), 2)

    def test_distinct_datasets_kept(self):
        files = [self.make("counties.geojson"), self.make("tracts.gpkg"),
                 self.make("dem.tif"), self.make("dem.tiff")]
        datasets, _ = classify_delivered_files(files)
        # Different stems stay; rasters are never merged as format copies.
        self.assertEqual(self.names(datasets),
                         ["counties.geojson", "dem.tif", "dem.tiff", "tracts.gpkg"])

    def test_same_stem_in_different_folders_kept(self):
        files = [self.make("a/data.geojson"), self.make("b/data.gpkg")]
        datasets, _ = classify_delivered_files(files)
        self.assertEqual(len(datasets), 2)


if __name__ == "__main__":
    unittest.main()
