"""Exercise baseline routing without importing the app's GIS/LLM dependencies."""
import ast
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class WebUIBaselineRouting(unittest.TestCase):
    def setUp(self):
        # Load the real registry without its unrelated output-directory import.
        registry_tree = ast.parse((ROOT / 'agents/spatial_reasoning/conditions.py').read_text(encoding='utf-8'))
        registry_tree.body = [node for node in registry_tree.body
                              if not isinstance(node, ast.ImportFrom)]
        registry = types.ModuleType('agents.spatial_reasoning.conditions')
        exec(compile(registry_tree, 'conditions.py', 'exec'), registry.__dict__)
        helper = types.ModuleType('utils.agm_helper')
        helper.set_ui_model = Mock()
        self.modules = patch.dict('sys.modules', {
            'agents.spatial_reasoning.conditions': registry,
            'utils.agm_helper': helper,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        tree = ast.parse((ROOT / 'WebUI/agm_workflow.py').read_text(encoding='utf-8'))
        entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'smart_message_handler_stream')
        self.workflow = Mock(return_value=iter(()))
        scope = {'reset_cancellation': Mock(), 'agm_workflow_handler': self.workflow,
                 'print': lambda *args: None}
        exec(compile(ast.Module(body=[entry], type_ignores=[]), 'agm_workflow.py', 'exec'), scope)
        self.handler = scope[entry.name]

    def test_each_baseline_reaches_research_with_uploaded_data(self):
        for baseline in ('B0', 'B1', 'B2', 'B3'):
            with self.subTest(baseline=baseline):
                self.workflow.return_value = iter(())
                list(self.handler('question', file_paths=['data.gpkg'],
                                  intent_mode='research', spatial_condition=baseline))
                self.assertEqual(self.workflow.call_args.kwargs['spatial_condition'], baseline)

    def test_baseline_survives_research_auto_download(self):
        list(self.handler('question', intent_mode='research',
                          data_download_mode=True, spatial_condition='B2'))
        self.assertEqual(self.workflow.call_args.kwargs['spatial_condition'], 'B2')

    def test_invalid_and_unbuilt_conditions_never_start_workflow(self):
        registry = __import__('sys').modules['agents.spatial_reasoning.conditions']
        events = list(self.handler('question', spatial_condition='typo'))
        self.assertEqual(events[0]['type'], 'error')
        # Every component is built now; simulate one that is not.
        with patch.object(registry, 'IMPLEMENTED_COMPONENTS', set()):
            events = list(self.handler('question', spatial_condition='P'))
        self.assertEqual(events[0]['type'], 'error')
        self.workflow.assert_not_called()

    def test_the_proposed_method_reaches_research(self):
        list(self.handler('question', file_paths=['data.gpkg'],
                          intent_mode='research', spatial_condition='P'))
        self.assertEqual(self.workflow.call_args.kwargs['spatial_condition'], 'P')

    def test_other_modes_cannot_silently_ignore_a_baseline(self):
        events = list(self.handler('question', intent_mode='single_agent', spatial_condition='B1'))
        self.assertEqual(events[0]['type'], 'error')
        self.workflow.assert_not_called()

    def test_omitted_baseline_defaults_to_control(self):
        list(self.handler('question', file_paths=['data.gpkg']))
        self.assertEqual(self.workflow.call_args.kwargs['spatial_condition'], 'B0')


if __name__ == '__main__':
    unittest.main()
