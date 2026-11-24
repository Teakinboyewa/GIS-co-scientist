"""The single-agent baseline must receive the specialist agents' own prompts.

The baseline answers "what does the multi-agent orchestration add?" -- which
it only does if the instructions are held fixed. A stage prompt edited in one
arm and not the other turns the comparison into a prompt comparison without
anyone noticing, so these tests pin the prompt text, then the loop's behaviour.
"""

import ast
import json
import os
import re
import shutil
import tempfile
import unittest
from unittest import mock

from langchain_core.messages import AIMessage


def _modeling_codegen_source():
    """The ModelingAgent's inline code-generation prompt, read from its source
    with the per-operation f-string slots swapped for the single agent's
    markers, then evaluated as a plain string literal."""
    src = open('agents/modeling_agent/agm_modeling_agent.py', encoding='utf-8').read()
    start = src.index('system_prompt = f"""You are an expert Geo-information scientist')
    body_start = src.index('"""', start) + 3
    body_end = src.index('"""', body_start)
    body = src[body_start:body_end]
    for slot, marker in [
        ('- {output_node}: {output_node_path}', '- <<OUTPUT_NODE>>'),
        ('`{output_node_path}`', '`<<OUTPUT_NODE_PATH>>`'),
        ('{operation_description}', '<<OPERATION>>'),
        ('{input_paths_summary}', '<<INPUT_PATHS>>'),
        ('{resolve_output_dir(ctx)}', '<<OUTPUT_DIR>>'),
    ]:
        assert slot in body, f"slot {slot!r} no longer in the ModelingAgent prompt"
        body = body.replace(slot, marker)
    body = body.replace('{{', '{').replace('}}', '}')
    return ast.literal_eval('"""' + body + '"""')


class TheBaselineUsesTheSameInstructions(unittest.TestCase):

    def test_codegen_copy_matches_the_modeling_agent_prompt(self):
        from agents.single_agent.single_agent_prompts import CODEGEN_GUIDANCE
        self.assertEqual(CODEGEN_GUIDANCE, _modeling_codegen_source(),
                         "CODEGEN_GUIDANCE has drifted from "
                         "ModelingAgent._generate_code_for_operation")

    def test_every_stage_prompt_is_included_verbatim(self):
        import utils.prompts as prompts
        from agents.reporting_agent import section_prompts as sp
        from agents.single_agent.single_agent_prompts import build_system_prompt
        system = build_system_prompt('/out/task')
        for text in [prompts.data_summary_system_prompt,
                     prompts.rq_breakdown_system_prompt,
                     prompts.generate_research_plan_system_prompt,
                     sp.TITLE_ABSTRACT_PROMPT, sp.INTRODUCTION_PROMPT,
                     sp.METHODOLOGY_PROMPT, sp.RESULTS_PROMPT,
                     sp.DISCUSSION_PROMPT, sp.CONCLUSION_PROMPT,
                     sp.REFERENCES_PROMPT]:
            self.assertIn(text.strip(), system)

    def test_codegen_slots_are_all_filled(self):
        from agents.single_agent.single_agent_prompts import build_system_prompt
        system = build_system_prompt('/out/task')
        self.assertNotIn('<<', system)
        self.assertIn('Save secondary outputs (plots, reports) to: `/out/task`', system)

    def test_research_question_is_passed_as_typed(self):
        from agents.single_agent.single_agent_prompts import build_user_message
        rq = "Is flood exposure higher in low-income tracts of Harris County?"
        msg = build_user_message(rq, [{'name': 'a.csv', 'path': '/x/a.csv'}], '/out')
        self.assertIn(rq, msg)
        self.assertIn('/x/a.csv', msg)


class _ScriptedLLM:
    """Stands in for GIBDChat: replays one AIMessage per turn."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def bind_tools(self, tools):
        self.tool_names = [t.name for t in tools]
        return self

    def stream(self, messages):
        self.seen.append(list(messages))
        yield self.turns.pop(0)


def _call(name, args, i):
    return {'name': name, 'args': args, 'id': f'call_{i}', 'type': 'tool_call'}


class TheLoopRunsTheWorkflow(unittest.TestCase):

    def setUp(self):
        self.out = tempfile.mkdtemp()
        self.ctx = {'rq': 'Does X explain Y?', 'datasets': [], 'task_id': 't1',
                    'output_dir': self.out, 'artifacts': {'data_registry': []}, 'logs': []}

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def _run(self, turns):
        from agents.single_agent import single_agent as sa
        llm = _ScriptedLLM(turns)
        events = []
        with mock.patch.object(sa, 'GIBDChat', lambda **kw: llm), \
             mock.patch.object(sa, '_artifacts', lambda files: [{'filename': os.path.basename(f)} for f in files]):
            result = sa.run_single_agent(self.ctx, events.append)
        return result, events, llm

    def test_full_run_executes_code_and_saves_the_manuscript(self):
        code = ("import pandas as pd, os\n"
                "pd.DataFrame({'a': [1, 2]}).to_csv(os.path.join(OUTPUT_DIR, 'step1.csv'), index=False)\n"
                "print('rows', 2)\n")
        result, events, llm = self._run([
            AIMessage(content='Planning.', tool_calls=[
                _call('write_file', {'relative_path': 'rq_breakdown.json', 'content': '{"title": "t"}'}, 1)]),
            AIMessage(content='', tool_calls=[
                _call('run_python', {'code': code, 'purpose': 'Objective 1, step 1'}, 2)]),
            AIMessage(content='', tool_calls=[
                _call('write_file', {'relative_path': 'manuscript.md', 'content': '# T\n\nBody (A, 2020) [REF_1].'}, 3)]),
            AIMessage(content='Done: X explains Y.'),
        ])
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['final_message'], 'Done: X explains Y.')
        self.assertTrue(os.path.exists(os.path.join(self.out, 'step1.csv')))
        step_results = [e for e in events if e['type'] == 'step_result']
        self.assertEqual(len(step_results), 1)
        self.assertEqual(step_results[0]['status'], 'completed')
        self.assertIn('rows 2', step_results[0]['output'])
        self.assertEqual(step_results[0]['step'], 'single_agent_exec_1')
        # The step's output reaches the registry, so list_datasets can see it.
        self.assertTrue(any(s['path'].endswith('step1.csv')
                            for s in self.ctx['artifacts']['data_registry']))
        # The tool result went back to the model.
        tool_msgs = [m for m in llm.seen[2] if m.type == 'tool']
        self.assertIn('rows 2', tool_msgs[-1].content)
        # The exact prompt and every turn are kept with the run.
        self.assertTrue(os.path.exists(os.path.join(self.out, 'single_agent', 'system_prompt.md')))
        lines = open(os.path.join(self.out, 'single_agent', 'transcript.jsonl'), encoding='utf-8').read().splitlines()
        self.assertEqual(json.loads(lines[-1])['status'], 'completed')

    def test_failing_code_returns_the_traceback_to_the_agent(self):
        result, events, llm = self._run([
            AIMessage(content='', tool_calls=[
                _call('run_python', {'code': 'import pandas as pd\npd.read_csv("missing.csv")', 'purpose': 's'}, 1)]),
            AIMessage(content='', tool_calls=[
                _call('write_file', {'relative_path': 'manuscript.md', 'content': 'x'}, 2)]),
            AIMessage(content='end'),
        ])
        step = [e for e in events if e['type'] == 'step_result'][0]
        self.assertEqual(step['status'], 'error')
        self.assertIn('FileNotFoundError', step['error'])
        fed_back = [m for m in llm.seen[1] if m.type == 'tool'][-1].content
        self.assertIn('traceback', fed_back)

    def test_fenced_code_runs_like_the_pipeline_extracts_it(self):
        """Run 171cb9d8 lost its first script to NameError: name 'python'."""
        from agents.single_agent.single_agent import execute_python, strip_code_fence
        self.assertEqual(strip_code_fence("```python\nprint(1)\n```"), "print(1)")
        self.assertEqual(strip_code_fence("print(1)"), "print(1)")
        result = execute_python(self.ctx, "```python\nprint('fenced ok')\n```\n")
        self.assertEqual(result['status'], 'completed')
        self.assertIn('fenced ok', result['output'])

    def test_stopping_without_a_manuscript_is_nudged_then_reported(self):
        result, events, llm = self._run([AIMessage(content='I am done.')] * 3)
        self.assertEqual(result['status'], 'no_manuscript')
        self.assertEqual(len(llm.seen), 3)
        self.assertIn('manuscript.md', llm.seen[1][-1].content)

    def test_write_file_cannot_escape_the_output_dir(self):
        result, events, llm = self._run([
            AIMessage(content='', tool_calls=[
                _call('write_file', {'relative_path': '../escape.txt', 'content': 'x'}, 1)]),
            AIMessage(content='', tool_calls=[
                _call('write_file', {'relative_path': 'manuscript.md', 'content': 'x'}, 2)]),
            AIMessage(content='end'),
        ])
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(self.out), 'escape.txt')))
        self.assertIn('outside the output directory', [m for m in llm.seen[1] if m.type == 'tool'][-1].content)

    def test_cancellation_stops_before_the_next_turn(self):
        from agents.single_agent import single_agent as sa
        llm = _ScriptedLLM([AIMessage(content='never')])
        with mock.patch.object(sa, 'GIBDChat', lambda **kw: llm):
            result = sa.run_single_agent(self.ctx, lambda e: None, is_cancelled=lambda: True)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(llm.seen, [])

    def test_finalize_applies_the_pipeline_cleanup_and_keeps_the_raw_file(self):
        from agents.single_agent.single_agent import finalize_manuscript
        raw = "# Title\n\nAs shown (Smith, 2020) [REF_1].\n| a | b |\n|---|---|\n| 1 | 2 |\n"
        with open(os.path.join(self.out, 'manuscript.md'), 'w', encoding='utf-8') as f:
            f.write(raw)
        md, path = finalize_manuscript(self.ctx)
        self.assertNotIn('[REF_1]', md)
        self.assertIn('(Smith, 2020)', md)
        self.assertIn('(Smith, 2020).\n\n| a | b |', md)
        self.assertEqual(open(os.path.join(self.out, 'manuscript_agent_raw.md'), encoding='utf-8').read(), raw)


class TheUiOffersTheMode(unittest.TestCase):

    def test_menu_entry_and_phase_mapping(self):
        html = open('WebUI/index.html', encoding='utf-8').read()
        self.assertIn('value="single_agent"', html)
        self.assertIn("selectModeFromDropdown('single_agent'", html)
        js = open('WebUI/script.js', encoding='utf-8').read()
        self.assertIn("if (step.startsWith('single_agent')) return step;", js)

    def test_backend_routes_the_mode(self):
        src = open('WebUI/agm_workflow.py', encoding='utf-8').read()
        self.assertTrue(re.search(r'intent_mode in \([^)]*"single_agent"', src))
        self.assertIn('yield from single_agent_workflow_handler(', src)


if __name__ == '__main__':
    unittest.main()
