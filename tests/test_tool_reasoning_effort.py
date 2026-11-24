"""gpt-6-luna's Chat Completions endpoint refuses function tools with any
reasoning level but "none" (400, first benchmark run). Every tool-using agent
goes through GIBDChat, so GIBDChat sends reasoning_effort="none" on those
calls and leaves calls without tools, and other models, as they were."""
import unittest
from unittest import mock

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from utils import agm_helper
from utils.agm_helper import GIBDChat, tool_call_reasoning_effort

TOOLS = [{"type": "function", "function": {"name": "f", "description": "d",
                                           "parameters": {"type": "object", "properties": {}}}}]


class FakeChatOpenAI:
    made = []

    def __init__(self, **kwargs):
        FakeChatOpenAI.made.append(kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content="ok"))


class ToolCallsRunWithoutReasoning(unittest.TestCase):

    def test_the_rule(self):
        self.assertEqual(tool_call_reasoning_effort("gpt-6-luna", True, "low"), "none")
        self.assertEqual(tool_call_reasoning_effort("gpt-6-luna", False, "low"), "low")
        self.assertIsNone(tool_call_reasoning_effort("gpt-6-luna", False, None))
        self.assertEqual(tool_call_reasoning_effort("gpt-5.2", True, "low"), "low")

    def _direct_call(self, model, tools, stream):
        FakeChatOpenAI.made = []
        llm = GIBDChat(model_name=model, gibd_user_key="sk-test", reasoning_effort="low")
        kwargs = {"tools": TOOLS} if tools else {}
        msgs = [HumanMessage(content="hi")]
        with mock.patch("langchain_openai.ChatOpenAI", FakeChatOpenAI), \
             mock.patch.object(agm_helper, "ALLOW_OPENAI", True):
            if stream:
                list(llm._stream(msgs, **kwargs))
            else:
                llm._generate(msgs, **kwargs)
        return FakeChatOpenAI.made[-1]

    def test_direct_openai_tool_calls_send_none(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.assertEqual(self._direct_call("gpt-6-luna", True, stream).get("reasoning_effort"), "none")
                # Unchanged elsewhere: direct OpenAI calls never sent a level before.
                self.assertNotIn("reasoning_effort", self._direct_call("gpt-6-luna", False, stream))
                self.assertNotIn("reasoning_effort", self._direct_call("gpt-5.2", True, stream))


if __name__ == "__main__":
    unittest.main()
