"""A result added to the map is an input to the next step of the conversation.

Seen on PythonAnywhere: Spatial Analysis created texas_county_centroids.gpkg,
then "create 100 m buffers around the centroids" recomputed the centroids
from the county polygons. The centroid layer was in the Data Layers panel,
but layers added via "Add to Map" were never sent with the next message, so
the step's data registry did not contain it. The frontend now sends them by
name, and the server finds them among the conversation's own outputs.
"""
import ast
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _load_finder(output_dir):
    tree = ast.parse((ROOT / "WebUI/app.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_find_conversation_output")
    ns = {"os": os}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "app.py", "exec"), ns)
    config = types.ModuleType("utils.config")
    config.load_configurations = lambda: types.SimpleNamespace(output_dir=output_dir)
    return ns["_find_conversation_output"], config


class FindConversationOutput(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.find, config = _load_finder(self.dir.name)
        p = patch.dict(sys.modules, {"utils.config": config})
        p.start()
        self.addCleanup(p.stop)

    def put(self, *parts):
        path = os.path.join(self.dir.name, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("x")
        return os.path.realpath(path)

    def test_finds_a_result_in_the_conversations_outputs(self):
        path = self.put("conv1", "final_outputs", "texas_county_centroids.gpkg")
        self.assertEqual(self.find("conv1", "texas_county_centroids.gpkg"), path)

    def test_newest_copy_wins(self):
        old = self.put("conv1", "run_a", "result.gpkg")
        new = self.put("conv1", "final_outputs", "result.gpkg")
        os.utime(old, (time.time() - 100, time.time() - 100))
        self.assertEqual(self.find("conv1", "result.gpkg"), new)

    def test_never_reaches_another_conversation(self):
        self.put("conv2", "final_outputs", "secret.gpkg")
        self.assertIsNone(self.find("conv1", "secret.gpkg"))
        for name in ("../conv2/final_outputs/secret.gpkg", "final_outputs/secret.gpkg", ""):
            self.assertIsNone(self.find("conv1", name), name)

    def test_missing_file_or_conversation(self):
        self.put("conv1", "a.gpkg")
        self.assertIsNone(self.find("conv1", "b.gpkg"))
        self.assertIsNone(self.find("nope", "a.gpkg"))
        self.assertIsNone(self.find("", "a.gpkg"))


class Wiring(unittest.TestCase):
    def test_result_layers_are_sent_and_resolved(self):
        js = (ROOT / "WebUI/script.js").read_text(encoding="utf-8")
        loop = js[js.index("// Append all loaded files"):]
        loop = loop[:loop.index("requestOptions.body = formData")]
        self.assertIn("layer._fromArtifact && layer.fileName", loop)
        self.assertEqual(loop.count("formData.append('existing_files', layer.fileName)"), 2)
        app = (ROOT / "WebUI/app.py").read_text(encoding="utf-8")
        block = app[app.index("existing_files = request.form.getlist('existing_files')"):]
        block = block[:block.index("except Exception as e:")]
        self.assertIn("_find_conversation_output(conversation_id, safe)", block)


if __name__ == "__main__":
    unittest.main()
