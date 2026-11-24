"""The experimental condition must survive the trip from session to agent.

A control session's use_handbook=False was recorded, shown in the UI, and then
dropped at the HTTP boundary: the endpoint defaulted a missing body field to
True, and the runner only consults the session when it receives None. Fifteen
runs executed the treatment path while the UI displayed "no handbook", which
is undetectable in the results -- the arm simply is not the arm it claims.

These tests pin the decision itself, not the plumbing around it.
"""

import unittest


def endpoint_decision(body):
    """The expression from app.py's test-stream endpoint."""
    return (None if body.get('use_handbook') is None
            else body.get('use_handbook') is not False)


def runner_decision(passed, session_value):
    """The fallback from handbook_studio_runner.run_test_stream."""
    if passed is None:
        return session_value is not False
    return passed


class ConditionSurvivesTheRequest(unittest.TestCase):

    def test_control_session_stays_control_when_body_omits_the_flag(self):
        """The regression that cost 15 runs."""
        decided = endpoint_decision({'retrieval_task': 'x'})
        self.assertIsNone(decided, "a missing flag must defer to the session")
        self.assertFalse(runner_decision(decided, session_value=False))

    def test_treatment_session_stays_treatment_when_body_omits_the_flag(self):
        decided = endpoint_decision({'retrieval_task': 'x'})
        self.assertTrue(runner_decision(decided, session_value=True))

    def test_explicit_false_is_honoured(self):
        decided = endpoint_decision({'use_handbook': False})
        self.assertFalse(decided)
        self.assertFalse(runner_decision(decided, session_value=True))

    def test_explicit_true_is_honoured(self):
        decided = endpoint_decision({'use_handbook': True})
        self.assertTrue(decided)
        self.assertTrue(runner_decision(decided, session_value=False))

    def test_a_session_with_no_setting_defaults_to_treatment(self):
        decided = endpoint_decision({})
        self.assertTrue(runner_decision(decided, session_value=None))


class TheUiSendsTheCondition(unittest.TestCase):

    def test_start_test_payload_carries_use_handbook(self):
        """startTest computed `control` but only used it to pick an endpoint."""
        import re
        source = open('WebUI/handbook_studio_mode.js', encoding='utf-8').read()
        start = source.index('async function startTest()')
        payload = source[start:start + 1400]
        self.assertIn('use_handbook: !control', payload,
                      "startTest must send the condition, not just route on it")


if __name__ == '__main__':
    unittest.main()
