"""A breakdown that calls the provided data sufficient must not trigger downloads.

The model sometimes returns a "sufficient" verdict together with data_requests
for data the user already has; with Auto Data on, Proceed then downloaded it
again. These load the helpers straight from WebUI/agm_workflow.py (via ast) so
the app's GIS/LLM dependencies are not imported.
"""
import ast
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'_INSUFFICIENT_RE', '_SUFFICIENT_RE', '_is_data_sufficient',
         '_drop_requests_when_sufficient', '_dropped_requests_note'}


def _load():
    tree = ast.parse((ROOT / 'WebUI/agm_workflow.py').read_text(encoding='utf-8'))
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in NAMES)
            or (isinstance(n, ast.Assign) and any(getattr(t, 'id', None) in NAMES for t in n.targets))]
    scope = {'re': re, 'json': json, 'os': os, 'task_manager': Mock()}
    exec(compile(ast.Module(body=body, type_ignores=[]), 'agm_workflow.py', 'exec'), scope)
    return scope


def _ctx(sufficiency, requests, datasets=('quakes.geojson',), retriever=False, path=None):
    artifacts = {'rq_breakdown': {'data_available_sufficiency': sufficiency, 'data_requests': requests}}
    if path:
        artifacts['rq_breakdown_path'] = path
    return {'datasets': [{'path': d} for d in datasets],
            'settings': {'data_retriever_only': retriever},
            'artifacts': artifacts}


class SufficientDataIsNotDownloadedAgain(unittest.TestCase):
    def setUp(self):
        self.drop = _load()['_drop_requests_when_sufficient']

    def test_sufficient_verdict_clears_requests(self):
        # Shapes taken from real runs that re-downloaded data already present.
        for verdict in (
            'Yes - the supplied wildfire points cover the report.',
            'The data provided is sufficient for generating a detailed descriptive report on wildfires in Oregon.',
            "The data provided appears to be sufficient. The dataset 'census_tract_boundaries.gpkg' includes geometry.",
        ):
            with self.subTest(verdict=verdict):
                ctx = _ctx(verdict, ['Active wildfire data for Oregon, USA'])
                self.assertEqual(self.drop(ctx), ['Active wildfire data for Oregon, USA'])
                rq = ctx['artifacts']['rq_breakdown']
                self.assertIsNone(rq['data_requests'])
                self.assertEqual(rq['dropped_data_requests'], ['Active wildfire data for Oregon, USA'])

    def test_missing_data_keeps_requests(self):
        for verdict in (
            'No - county boundaries are needed to summarise events by county.',
            'The available data provides flood zones, but lacks population data.',
            'Partially sufficient; population data is missing.',
        ):
            with self.subTest(verdict=verdict):
                ctx = _ctx(verdict, ['California county boundaries'])
                self.assertEqual(self.drop(ctx), [])
                self.assertEqual(ctx['artifacts']['rq_breakdown']['data_requests'],
                                 ['California county boundaries'])

    def test_unclear_verdict_keeps_requests(self):
        ctx = _ctx('Data availability is not applicable to this question.', ['Some dataset'])
        self.assertEqual(self.drop(ctx), [])

    def test_data_retriever_mode_is_untouched(self):
        ctx = _ctx('Yes - already provided.', ['Road network for Akure'], retriever=True)
        self.assertEqual(self.drop(ctx), [])
        self.assertEqual(ctx['artifacts']['rq_breakdown']['data_requests'], ['Road network for Akure'])

    def test_no_datasets_means_sufficient_cannot_be_right(self):
        ctx = _ctx('Yes - sufficient.', ['PM2.5 for California counties'], datasets=())
        self.assertEqual(self.drop(ctx), [])

    def test_nothing_to_drop(self):
        for requests in (None, []):
            with self.subTest(requests=requests):
                self.assertEqual(self.drop(_ctx('Yes - covered.', requests)), [])

    def test_saved_breakdown_file_matches(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'rq_breakdown.json')
            ctx = _ctx('Yes - covered.', ['Earthquakes in California, Oct 2026'], path=path)
            self.drop(ctx)
            with open(path, encoding='utf-8') as f:
                saved = json.load(f)
        self.assertIsNone(saved['data_requests'])
        self.assertEqual(saved['dropped_data_requests'], ['Earthquakes in California, Oct 2026'])


if __name__ == '__main__':
    unittest.main()
