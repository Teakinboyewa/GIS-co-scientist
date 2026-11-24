"""The no-planning ablation must remove the Planning Agent and nothing else.

Requested in review: remove the Planning Agent and have the Modeling Agent
generate code directly. If the ablation also dropped the RQ breakdown, or let
the router send the step elsewhere, the measured difference would no longer be
attributable to planning.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock


class ThePlaceholderPlanCarriesNoPlanning(unittest.TestCase):

    def setUp(self):
        self.out = tempfile.mkdtemp()
        self.ctx = {
            'rq': 'How does mobility mediate SES and depression in SC counties?',
            'task_id': 't1', 'output_dir': self.out, 'logs': [],
            'settings': {'ablation': 'no_planning'},
            'artifacts': {
                'rq_breakdown': {'title': 'Mobility mediation', 'intent': 'test mediation',
                                 'scale': {'spatial_scale': 'county'}},
                'data_registry': [
                    {'file': 'SC_ACS_data.csv', 'path': '/up/SC_ACS_data.csv'},
                    {'file': 'SC_counties.gpkg', 'path': '/up/SC_counties.gpkg'},
                ],
            },
        }

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def test_one_objective_one_step_covering_the_whole_question(self):
        from agents.task_manager import ablation, research_plan_generator
        # The output-path guard only accepts folders under Outputs/.
        with mock.patch.object(ablation, 'resolve_output_dir', lambda ctx: self.out), \
             mock.patch.object(research_plan_generator, 'resolve_output_dir', lambda ctx: self.out):
            plan = ablation.build_no_planning_plan(self.ctx)
        self.assertEqual(len(plan['objectives']), 1)
        steps = plan['objectives'][0]['steps']
        self.assertEqual(len(steps), 1)
        step = steps[0]
        self.assertIn(self.ctx['rq'], step['description'])
        # The RQ breakdown still reaches code generation, as it does via the plan.
        self.assertIn('test mediation', step['description'])
        self.assertIn('county', step['description'])
        # Every uploaded dataset is an input, with its path resolved.
        self.assertEqual(step['input_node_paths'],
                         {'SC_ACS_data.csv': '/up/SC_ACS_data.csv',
                          'SC_counties.gpkg': '/up/SC_counties.gpkg'})
        self.assertEqual(step['output_node_path'], os.path.join(self.out, 'analysis_results.csv'))
        # No planning decisions leak in.
        self.assertEqual(plan['objectives'][0]['hypothesis'], '')
        self.assertIs(self.ctx['artifacts']['research_plan_final'], plan)
        saved = json.load(open(os.path.join(self.out, 'research_plan_final.json'), encoding='utf-8'))
        self.assertEqual(saved['ablation'], 'no_planning')

    def test_routing_is_pinned_to_the_modeling_agent(self):
        from agents.task_manager.ablation import pinned_routing
        routing = pinned_routing([{'description': 'd', 'operation': 'o'}])['routing']
        self.assertEqual([r['agent_name'] for r in routing], ['ModelingAgent'])

    def test_full_research_mode_has_no_ablation(self):
        from agents.task_manager.ablation import ablation_of
        self.assertIsNone(ablation_of({'settings': {'interactive_code': False}}))
        self.assertIsNone(ablation_of({}))


class TheModeReachesTheResearchWorkflow(unittest.TestCase):

    def _route(self, intent_mode):
        import WebUI.agm_workflow as w
        seen = {}

        def fake_handler(*args, **kwargs):
            seen.update(kwargs, intent=args[1])
            return iter(())
        with mock.patch.object(w, 'agm_workflow_handler', fake_handler):
            list(w.smart_message_handler_stream('q', file_paths=['/a.csv'], file_names=['a.csv'],
                                                intent_mode=intent_mode))
        return seen

    def test_no_planning_mode_runs_research_with_the_flag(self):
        seen = self._route('research_no_planning')
        self.assertEqual(seen['ablation'], 'no_planning')
        self.assertEqual(seen['intent']['intent'], 'research')

    def test_research_mode_is_unchanged(self):
        seen = self._route('research')
        self.assertIsNone(seen['ablation'])

    def test_no_plan_is_shown_or_approved(self):
        """A plan card or Approve button in a no-planning run reads as planning."""
        import WebUI.agm_workflow as w
        from agents.task_manager import ablation
        ctx = {'task_id': 't1', 'settings': {'ablation': 'no_planning'},
               'artifacts': {'research_plan_final': {'objectives': []}}}
        calls = []

        def fake_continue(task_id):
            calls.append(task_id)
            yield {'type': 'step_result', 'step': 'exec_obj1_step1'}
        with mock.patch.object(ablation, 'build_no_planning_plan', lambda c: None), \
             mock.patch.object(w, 'continue_after_approval_stream', fake_continue), \
             mock.patch.object(w, 'save_workflow_state', lambda *a, **k: None), \
             mock.patch.object(w.research_plan_generator, 'store_context', lambda *a: None):
            events = list(w._no_planning_handoff(ctx, {'intent': 'research'}, [], 0))
        self.assertEqual(calls, ['t1'], 'execution must start without an approval click')
        types = [e['type'] for e in events]
        self.assertNotIn('research_plan_viz', types)
        self.assertNotIn('result', types, 'no pending-approval pause')
        self.assertFalse(any('research_plan' in (e.get('step') or '') for e in events))
        self.assertEqual(events[0]['step'], 'ablation_no_planning')

    def test_plan_stage_and_router_consult_the_flag(self):
        src = open('WebUI/agm_workflow.py', encoding='utf-8').read()
        self.assertIn('yield from _no_planning_handoff(', src)
        self.assertIn('build_no_planning_plan(ctx)', src)
        self.assertIn('ctx["settings"]["ablation"] = ablation', src)
        ex = open('agents/task_manager/objective_executor.py', encoding='utf-8').read()
        self.assertIn('if ablation_of(ctx) == NO_PLANNING:', ex)


if __name__ == '__main__':
    unittest.main()
