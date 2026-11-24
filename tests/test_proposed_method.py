"""The proposed method (P) and its ablations.

What must hold for the comparison with the baselines to mean anything:
  - each condition runs exactly its components, and baselines run none;
  - hypotheses cannot be stated with a missing assumption field, and no
    agent can change them once locked;
  - diagnostics see only exploration data, and the analysis is pointed at
    the confirmation data;
  - plan changes made in response to evidence are marked exploratory;
  - the critic cannot request diagnostics when none can run.
The LLM is replaced by scripted answers; diagnostics run for real.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from agents.spatial_reasoning import (assumptions, conditions, critic, diagnostics,
                                      exploration, llm, revision)
from agents.spatial_reasoning.llm import InvalidResponse


def _grid(n_side, seed=0, lam=0.0):
    """n_side x n_side square units with y = 1 + 2 x1 - x2 + u, u = (I - lam W)^-1 e."""
    import geopandas as gpd
    from shapely.geometry import box
    from libpysal.weights import lat2W
    rng = np.random.default_rng(seed)
    n = n_side * n_side
    w = lat2W(n_side, n_side, rook=False)
    w.transform = "r"
    W, _ = w.full()
    e = rng.standard_normal(n)
    u = np.linalg.solve(np.eye(n) - lam * W, e)
    x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
    cells = [box(c, r, c + 1, r + 1) for r in range(n_side) for c in range(n_side)]
    return gpd.GeoDataFrame({"y": 1 + 2 * x1 - x2 + u, "x1": x1, "x2": x2},
                            geometry=cells, crs="EPSG:3857")


HYPOTHESIS = {
    "id": "H1", "statement": "Higher x1 is associated with higher y across cells.",
    "dataset": "cells", "outcome": "y", "predictors": ["x1", "x2"],
    "assumptions": {
        "scale_and_zoning": {"analysis_unit": "grid cell",
                             "question_unit": "area", "data_unit": "area",
                             "assumes_results_hold_across_aggregations": True,
                             "justification": "Cells are the unit of both question and data."},
        "independence": {"assumes_independent_observations": True,
                         "justification": "No theoretical reason to expect spillovers."},
        "global_or_local": {"assumes": "global",
                            "justification": "One process governs the whole study area."},
    },
}


class _Ctx(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp()
        self.path = os.path.join(self.out, "cells.gpkg")
        _grid(10, lam=0.8).to_file(self.path, driver="GPKG")
        self.ctx = {"rq": "Is x1 associated with y?", "output_dir": self.out, "logs": [],
                    "settings": {}, "artifacts": {
                        "rq_breakdown": {"intent": "association", "scale": {}},
                        "data_registry": [{
                            "file": "cells", "path": self.path, "source": "data_audit_record",
                            "shape": [100, 4],
                            "columns": {"y": "float64", "x1": "float64", "x2": "float64",
                                        "geometry": "geometry"}}]}}

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)


class EachConditionRunsExactlyItsComponents(unittest.TestCase):

    def test_every_condition_is_buildable(self):
        for name in conditions.CONDITIONS:
            self.assertEqual(conditions.resolve(name)["name"], name)

    def test_graph_nodes_follow_the_switches(self):
        from agents.task_manager.research_plan_generator import plan_graph_nodes
        base = ["generate_research_plan", "review_research_plan", "finalize_research_plan"]
        expected = {
            "B0": base, "B1": base, "B2": base, "B3": base,
            "P": ["state_assumptions", *base[:2], "spatial_grounding", base[2]],
            "P-no-structure": [*base[:2], "spatial_grounding", base[2]],
            "P-no-diagnostics": ["state_assumptions", *base[:2], "spatial_grounding", base[2]],
            "P-no-critic": ["state_assumptions", *base[:2], "spatial_grounding", base[2]],
            "B3+structure": ["state_assumptions", *base],
        }
        for name, nodes in expected.items():
            with self.subTest(name=name):
                ctx = {"settings": {"spatial_condition": conditions.resolve(name)}}
                self.assertEqual([n for n, _ in plan_graph_nodes(ctx)], nodes)


class HypothesesNeedEveryField(_Ctx):

    def _valid(self, h):
        return assumptions.validate_hypotheses({"hypotheses": [h]}, assumptions.input_datasets(self.ctx))

    def test_a_complete_hypothesis_passes(self):
        self.assertEqual(self._valid(json.loads(json.dumps(HYPOTHESIS)))[0]["id"], "H1")

    def test_missing_or_wrong_fields_are_refused(self):
        def broken(path, value):
            h = json.loads(json.dumps(HYPOTHESIS))
            target = h
            for key in path[:-1]:
                target = target[key]
            if value is KeyError:
                del target[path[-1]]
            else:
                target[path[-1]] = value
            return h
        cases = {
            "missing field": broken(["assumptions", "independence"], KeyError),
            "no justification": broken(["assumptions", "global_or_local", "justification"], ""),
            "not a boolean": broken(["assumptions", "independence",
                                     "assumes_independent_observations"], "yes"),
            "bad global/local": broken(["assumptions", "global_or_local", "assumes"], "both"),
            "no question unit": broken(["assumptions", "scale_and_zoning", "question_unit"], KeyError),
            "bad data unit": broken(["assumptions", "scale_and_zoning", "data_unit"], "tract"),
            "unknown column": broken(["predictors"], ["x9"]),
            "unknown dataset": broken(["dataset"], "other"),
        }
        for label, h in cases.items():
            with self.subTest(label), self.assertRaises(InvalidResponse):
                self._valid(h)

    def test_a_question_about_individuals_on_area_data_is_an_ecological_inference(self):
        """Pilot T05: no condition, P included, raised the ecological fallacy;
        no diagnostic can, so it is a comparison of two stated units."""
        h = json.loads(json.dumps(HYPOTHESIS))
        self.assertEqual(assumptions.unit_of_inference(h)[0], "supported")
        h["assumptions"]["scale_and_zoning"]["question_unit"] = "individual"
        status, why = assumptions.unit_of_inference(h)
        self.assertEqual(status, "contradicted")
        self.assertIn("ecological fallacy", why)
        self.assertIn("ecological fallacy", assumptions.planner_instructions([h]))
        self.assertNotIn("ecological fallacy", assumptions.planner_instructions([HYPOTHESIS]))
        from agents.spatial_reasoning.pipeline import statuses
        st = statuses({}, [h])["H1"]
        self.assertEqual(st[assumptions.UNIT_CHECK]["status"], "contradicted")
        old = json.loads(json.dumps(HYPOTHESIS))
        del old["assumptions"]["scale_and_zoning"]["question_unit"]
        self.assertIsNone(assumptions.unit_of_inference(old))     # records from before the field

    def test_an_invalid_answer_is_sent_back_and_corrected(self):
        answers = iter(['{"hypotheses": []}', json.dumps({"hypotheses": [HYPOTHESIS]})])
        seen = []

        def answer(messages):
            seen.append(messages[-1].content)
            return next(answers)
        hyps = assumptions.state_hypotheses(self.ctx, answer=answer)
        self.assertEqual(hyps[0]["id"], "H1")
        self.assertIn("non-empty list", seen[1])

    def test_planning_cannot_start_without_valid_hypotheses(self):
        with self.assertRaises(InvalidResponse):
            assumptions.state_hypotheses(self.ctx, answer=lambda m: '{"hypotheses": []}')


class TheLockHolds(_Ctx):

    def test_agents_cannot_change_locked_hypotheses(self):
        plan = {"objectives": [
            {"number": 1, "hypothesis_ids": ["H1"], "hypothesis": "rewritten by the reviewer"},
            {"number": 2, "hypothesis": "an extra one"}]}
        report = assumptions.enforce(plan, [HYPOTHESIS])
        self.assertTrue(all(o["hypothesis"] == HYPOTHESIS["statement"] for o in plan["objectives"]))
        self.assertEqual([r["reason"] for r in report["restored"]],
                         ["hypothesis text was changed",
                          "hypothesis_ids missing; assigned by position", "hypothesis text was changed"])

    def test_ids_the_reviewer_dropped_are_recovered_not_guessed_by_position(self):
        """Pilot T07: the reviewer dropped the ids and objective 2 (testing H3)
        was labelled H2 by position, leaving H3 'untested'."""
        h = [dict(HYPOTHESIS, id=f"H{i}", statement=s, predictors=p) for i, (s, p) in enumerate([
            ("Road density is associated with higher PM2.5 concentrations.", ["x1"]),
            ("Industrial share is associated with higher PM2.5 concentrations.", ["x2"]),
            ("Controlling for industrial share, road density remains associated with PM2.5.",
             ["x1", "x2"])], start=1)]
        reviewed = {"objectives": [
            {"number": 1, "title": "Road density and industrial share each relate to PM2.5"},
            {"number": 2, "title": "Road density controlling for industrial share"}]}
        draft = {"objectives": [{"number": 1, "hypothesis_ids": ["H1", "H2"]}]}
        report = assumptions.enforce(reviewed, h, draft=draft)
        self.assertEqual(reviewed["objectives"][0]["hypothesis_ids"], ["H1", "H2"])
        self.assertEqual(reviewed["objectives"][1]["hypothesis_ids"], ["H3"])
        self.assertEqual(report["untested_hypotheses"], [])
        self.assertEqual([r["reason"] for r in report["restored"] if r["reason"].startswith("hypothesis_ids")],
                         ["hypothesis_ids recovered from the planner's draft",
                          "hypothesis_ids missing; matched by content"])

    def test_user_edits_at_the_checkpoint_are_recorded(self):
        assumptions.lock(self.ctx, [HYPOTHESIS])
        self.ctx["artifacts"]["research_plan_final"] = {"objectives": [
            {"number": 1, "hypothesis_ids": ["H1"], "hypothesis": "edited by the user"}]}
        record = assumptions.record_checkpoint_edits(self.ctx)
        self.assertTrue(record["edited_by_user"])
        self.assertTrue(os.path.exists(os.path.join(self.out, assumptions.CHECKPOINT_FILE)))


class DiagnosticsAreRealAndThresholded(unittest.TestCase):

    def test_planted_dependence_is_detected_and_its_absence_is_not(self):
        dep = diagnostics.residual_dependence(_grid(12, seed=1, lam=0.85), "y", ["x1", "x2"])
        null = diagnostics.residual_dependence(_grid(12, seed=2, lam=0.0), "y", ["x1", "x2"])
        self.assertTrue(dep["dependence_detected"])
        self.assertFalse(null["dependence_detected"])
        self.assertEqual(null["suggested_specification"], "none")

    def test_a_varying_coefficient_is_detected_and_a_constant_one_is_not(self):
        """Pilot T04/T09: MGWR found the planted local effect, but OLS AICc
        was computed without the likelihood constants, so MGWR never 'beat'
        OLS and every relationship was reported as global."""
        g = _grid(14, seed=3)
        col = np.array([c.centroid.x for c in g.geometry])
        g["y"] = 1 + (col / 13 * 4 - 2) * g["x1"] - g["x2"] + 0.5 * np.random.default_rng(4).standard_normal(len(g))
        varying = diagnostics.local_variation(g, "y", ["x1", "x2"])
        self.assertTrue(varying["covariates"]["x1"]["local"])
        self.assertFalse(varying["covariates"]["x2"]["local"])
        constant = diagnostics.local_variation(_grid(14, seed=5), "y", ["x1", "x2"])
        self.assertFalse(constant["local_variation_detected"])

    def test_strong_lm_evidence_counts_as_dependence_without_moran(self):
        """Pilot T05 replicate: Moran's I p = 0.07, LM-lag p < 0.001 under
        both weights. T03: LM-lag p = 0.02 under both, no planted dependence."""
        def w(moran_p, lag_p, err_p=0.5, rlag=(9.0, 0.001), rerr=(1.0, 0.3)):
            return {"weights": "queen", "moran_I": 0.08, "moran_p": moran_p, "lm_tests": {
                "lm_lag": {"stat": 0, "p": lag_p}, "lm_error": {"stat": 0, "p": err_p},
                "robust_lm_lag": {"stat": rlag[0], "p": rlag[1]},
                "robust_lm_error": {"stat": rerr[0], "p": rerr[1]}}}
        t05 = diagnostics.dependence_decision([w(0.07, 0.0001), w(0.048, 0.0002)])
        self.assertTrue(t05["dependence_detected"])
        self.assertEqual(t05["detected_by"], "lm_lag")
        self.assertEqual(t05["suggested_specification"], "lag")
        t03 = diagnostics.dependence_decision([w(0.5, 0.023), w(0.5, 0.023)])
        self.assertFalse(t03["dependence_detected"])
        one_weights_only = diagnostics.dependence_decision([w(0.2, 0.2), w(0.04, 0.001)])
        self.assertFalse(one_weights_only["dependence_detected"])     # T08 replicate pattern
        self.assertFalse(one_weights_only["robust_across_weights"])
        status, why = diagnostics.assumption_status(
            "independence", HYPOTHESIS["assumptions"]["independence"],
            {"residual_dependence": t05 | {"by_weights": [w(0.07, 0.0001)]}})
        self.assertEqual(status, "contradicted")       # HYPOTHESIS assumed independence
        self.assertIn("LM-lag p < 0.01", why)

    def test_a_noisy_coefficient_change_is_not_scale_sensitivity(self):
        """Pilot T05: population (|b|/se < 1) moved by 290% under aggregation
        and the data were reported as scale-sensitive."""
        g = _grid(12, seed=7)
        g["noise"] = np.random.default_rng(8).standard_normal(len(g))
        r = diagnostics.aggregation_sensitivity(g, "y", ["x1", "noise"])
        self.assertTrue(r["tested"])
        self.assertFalse(r["scale_sensitive"])
        changes = [a["coefficients"]["noise"] for a in r["aggregations"] if "coefficients" in a]
        self.assertTrue(all(c["z_difference"] < diagnostics.Z_DIFF for c in changes))

    def test_a_sign_reversal_under_aggregation_is_scale_sensitivity(self):
        """T05's process: x = smooth core + local variation; y = 0.5 x - 1.2 core.
        The unit slope is positive, the slope between area means negative."""
        g = _grid(20, seed=9)
        rng = np.random.default_rng(10)
        cx = np.array([c.centroid.x for c in g.geometry]) / 19
        core = 4 * cx
        g["x"] = core + 2.5 * rng.standard_normal(len(g))
        g["y"] = 0.5 * g["x"] - 1.2 * core + 0.3 * rng.standard_normal(len(g))
        r = diagnostics.aggregation_sensitivity(g, "y", ["x"])
        self.assertTrue(r["scale_sensitive"])
        flips = [a["coefficients"]["x"]["sign_flip"] for a in r["aggregations"] if "coefficients" in a]
        self.assertTrue(any(flips))

    def test_too_few_units_for_mgwr_is_untested_not_a_failure(self):
        """Pilot T05: the exploration half held 13 districts and MGWR crashed."""
        small = diagnostics.local_variation(_grid(4, seed=6), "y", ["x1", "x2"])
        self.assertIn("skipped", small)
        status, why = diagnostics.assumption_status(
            "global_or_local", HYPOTHESIS["assumptions"]["global_or_local"], {"local_variation": small})
        self.assertEqual(status, "untested")
        self.assertIn("16 units", why)

    def test_assumption_status(self):
        results = {"residual_dependence": {"dependence_detected": True, "by_weights": [
            {"moran_I": 0.3, "moran_p": 0.001, "weights": "queen"}]}}
        status, _ = diagnostics.assumption_status(
            "independence", HYPOTHESIS["assumptions"]["independence"], results)
        self.assertEqual(status, "contradicted")
        results["local_variation"] = {"local_variation_detected": True, "covariates": {}}
        status, why = diagnostics.assumption_status(
            "global_or_local", HYPOTHESIS["assumptions"]["global_or_local"], results)
        self.assertEqual(status, "inconclusive")          # dependence and local variation confound
        self.assertEqual(diagnostics.assumption_status("scale_and_zoning", {}, {})[0], "untested")


class DiagnosticsSeeOnlyExplorationData(_Ctx):

    def test_blocked_split_partitions_units_and_points_analysis_at_the_other_half(self):
        record = exploration.prepare(self.ctx)
        self.assertEqual(record["mode"], "blocked_split")
        units = record["units"]["cells"]
        self.assertEqual(units["exploration"] + units["confirmation"], 100)
        self.assertGreater(min(units.values()), 0)
        exploration.point_analysis_at_confirmation(self.ctx, record)
        src = self.ctx["artifacts"]["data_registry"][0]
        self.assertEqual(src["path"], record["confirmation"]["cells"])
        self.assertEqual(src["original_path"], self.path)

    def test_a_supplied_replicate_is_used_and_the_data_stay_put(self):
        self.ctx["settings"]["exploration_data"] = {"cells": "replicate.gpkg"}
        record = exploration.prepare(self.ctx)
        self.assertEqual(record["mode"], "replicate")
        self.assertEqual(record["exploration"]["cells"], "replicate.gpkg")
        exploration.point_analysis_at_confirmation(self.ctx, record)
        self.assertEqual(self.ctx["artifacts"]["data_registry"][0]["path"], self.path)


def _plan(*steps, hypothesis="as stated"):
    return {"objectives": [{"number": 1, "title": "t", "hypothesis": hypothesis,
                            "hypothesis_ids": ["H1"], "steps": [
        {"step_number": i, "operation_type": "statistical", "operation": op,
         "input_nodes": ins, "output_node": out, "description": op, "expected_output": None}
        for i, (op, ins, out) in enumerate(steps, start=1)]}]}


class RevisionIsMarkedAndConnected(unittest.TestCase):

    def test_changed_steps_are_marked_exploratory(self):
        before = _plan(("OLS", ["cells"], "ols.csv"))
        after = _plan(("OLS", ["cells"], "ols.csv"), ("SEM", ["cells"], "sem.csv"))
        changed = revision.mark_exploratory(before, after)
        self.assertEqual([c["operation"] for c in changed], ["SEM"])
        self.assertNotIn("exploratory", after["objectives"][0]["steps"][0])

    def test_a_revision_may_not_break_step_connectivity(self):
        broken = _plan(("SEM", ["missing.csv"], "sem.csv"))
        with self.assertRaises(InvalidResponse):
            revision.validate_plan(broken, ["cells"])
        # ...but problems the plan already had do not block it.
        tolerated = set(revision.check_connectivity(broken, ["cells"]))
        self.assertIs(revision.validate_plan(broken, ["cells"], tolerated), broken)


class DiagnosticsUseTheFullModel(unittest.TestCase):
    """Pilot T07/T08: single-predictor hypotheses had clustered residuals only
    because the other, spatially smooth predictor was left out."""

    def test_hypotheses_sharing_an_outcome_are_diagnosed_together(self):
        from agents.spatial_reasoning.pipeline import full_models
        hyps = [dict(HYPOTHESIS, id="H1", predictors=["x1"]),
                dict(HYPOTHESIS, id="H2", predictors=["x2"]),
                dict(HYPOTHESIS, id="H3", predictors=["x1", "x2"]),
                dict(HYPOTHESIS, id="H4", outcome="z", predictors=["x1"])]
        models = full_models(hyps)
        self.assertEqual([(m["id"], m["outcome"], m["predictors"], m["hypotheses"]) for m in models],
                         [("M1", "y", ["x1", "x2"], ["H1", "H2", "H3"]), ("M2", "z", ["x1"], ["H4"])])

    def test_omitting_a_smooth_covariate_creates_the_false_alarm_the_full_model_avoids(self):
        from libpysal.weights import lat2W
        rng = np.random.default_rng(1003)     # independent of _grid's own seed
        g = _grid(14, seed=3)
        w = lat2W(14, 14, rook=False)
        w.transform = "r"
        W, _ = w.full()
        smooth = np.linalg.solve(np.eye(196) - 0.9 * W, rng.standard_normal(196))
        g["x2"] = (smooth - smooth.mean()) / smooth.std()
        g["y"] = 1 + 2 * g["x1"] + 3 * g["x2"] + rng.standard_normal(196)
        reduced = diagnostics.residual_dependence(g, "y", ["x1"])
        full = diagnostics.residual_dependence(g, "y", ["x1", "x2"])
        self.assertTrue(reduced["dependence_detected"])
        self.assertFalse(full["dependence_detected"])

    def test_each_hypothesis_is_judged_on_its_full_model(self):
        from agents.spatial_reasoning.pipeline import statuses
        evidence = {"M1": {"hypotheses": ["H1"], "results": {"residual_dependence": {
            "dependence_detected": False, "by_weights": [{"moran_I": 0.0, "moran_p": 0.4, "weights": "queen"}]}}}}
        st = statuses(evidence, [HYPOTHESIS])
        self.assertEqual(st["H1"]["independence"]["status"], "supported")

    def test_diagnostics_already_run_are_not_rerun_on_request(self):
        from agents.spatial_reasoning.pipeline import run_requests
        evidence = {"M1": {"dataset": "cells", "outcome": "y", "predictors": ["x1"], "requested": [],
                           "results": {"residual_dependence": {"by_weights": [{"weights": "queen"}]}}}}
        with mock.patch.object(diagnostics, "residual_dependence",
                               side_effect=AssertionError("must not rerun")):
            run_requests(evidence, {}, [{"diagnostic": "residual_dependence", "model": "M1", "weights": "queen"},
                                        {"diagnostic": "local_variation", "model": "M1"}])
        self.assertTrue(all("already ran" in r["result"]["note"] for r in evidence["M1"]["requested"]))


class TheCriticStaysInItsLane(unittest.TestCase):

    def test_requests_are_refused_when_no_diagnostics_can_run(self):
        answer = {"verdict": "revise",
                  "issues": [{"kind": "missing_diagnostic", "severity": "major", "detail": "x"}],
                  "requested_diagnostics": [{"diagnostic": "local_variation", "model": "H1"}]}
        with self.assertRaises(InvalidResponse):
            critic.validate(answer, ["H1"], diagnostics_available=False)
        self.assertEqual(critic.validate(answer, ["H1"], True)["verdict"], "revise")

    def test_only_major_issues_can_force_a_revision(self):
        """Pilot: the critic never approved, and every round grew the plan."""
        minor = {"kind": "other", "severity": "minor", "detail": "add a LISA map"}
        major = {"kind": "ignored_evidence", "severity": "major", "detail": "residuals are dependent"}
        with self.assertRaises(InvalidResponse):
            critic.validate({"verdict": "revise", "issues": [minor]}, ["M1"], True)
        with self.assertRaises(InvalidResponse):
            critic.validate({"verdict": "approve", "issues": [major]}, ["M1"], True)
        with self.assertRaises(InvalidResponse):
            critic.validate({"verdict": "approve", "issues": [{**minor, "severity": None}]}, ["M1"], True)
        ok = critic.validate({"verdict": "revise", "issues": [minor, major]}, ["M1"], True)
        self.assertEqual(critic.major(ok), [major])
        self.assertEqual(critic.validate({"verdict": "approve", "issues": [minor]}, ["M1"], True)["verdict"],
                         "approve")

    def test_the_critic_is_told_each_remedy_answers_its_own_evidence(self):
        """Pilot T04 (P, after restart): the critic called MGWR unsupported
        because residuals were independent and H1 had assumed a global
        relationship, although local variation was detected."""
        seen = {}

        def answer(messages):
            seen["system"] = messages[0].content
            return json.dumps({"verdict": "approve", "issues": [], "requested_diagnostics": []})
        critic.review({"objectives": []}, [HYPOTHESIS], "", ["M1"], True, answer=answer)
        self.assertIn("supports a local model", seen["system"])
        self.assertIn("even when residuals are independent", seen["system"])
        self.assertIn("contradicts a locked assumption", seen["system"])

    def test_a_follow_up_round_only_judges_earlier_issues_and_the_revision(self):
        """Pilot T05, T09: round 2 raised new major issues about steps it had
        already passed, and its diagnostic requests could never run."""
        earlier = [{"kind": "ignored_evidence", "severity": "major", "detail": "no MGWR"}]
        changed = [{"objective": 1, "step": 8, "status": "added", "operation": "MGWR"}]
        seen = {}

        def answer(messages):
            seen["system"], seen["user"] = messages[0].content, messages[1].content
            return json.dumps({"verdict": "approve", "issues": [], "requested_diagnostics": []})
        critic.review({"objectives": []}, [HYPOTHESIS], "", ["M1"], True, answer=answer,
                      previous=earlier, changes=changed, final=True)
        self.assertIn("follow-up review", seen["system"])
        self.assertIn("final review round", seen["system"])
        self.assertIn("no MGWR", seen["user"])
        self.assertIn("MGWR", seen["user"].split("Steps the revision added or changed")[1])
        new = {"kind": "other", "severity": "major", "detail": "add knn12"}
        with self.assertRaises(InvalidResponse):           # no basis
            critic.validate({"verdict": "revise", "issues": [new]}, ["M1"], True, follow_up=True)
        ok = critic.validate({"verdict": "revise", "issues": [{**new, "basis": "unresolved"}]},
                             ["M1"], True, follow_up=True)
        self.assertEqual(ok["verdict"], "revise")
        req = {"verdict": "approve", "issues": [],
               "requested_diagnostics": [{"diagnostic": "residual_dependence", "model": "M1",
                                          "weights": "knn12"}]}
        replies = iter([json.dumps(req), json.dumps({**req, "requested_diagnostics": []})])
        out = critic.review({"objectives": []}, [HYPOTHESIS], "", ["M1"], True,
                            answer=lambda m: next(replies), previous=[], changes=[], final=True)
        self.assertEqual(out["requested_diagnostics"], [])  # refused, then corrected

    def test_the_critic_is_told_about_the_ecological_fallacy(self):
        seen = {}

        def answer(messages):
            seen["system"] = messages[0].content
            return json.dumps({"verdict": "approve", "issues": [], "requested_diagnostics": []})
        critic.review({"objectives": []}, [HYPOTHESIS], "", ["M1"], True, answer=answer)
        self.assertIn("ecological", seen["system"])
        self.assertIn("Comparing zonings does not address it", seen["system"])
        self.assertNotIn("follow-up review", seen["system"])

    def test_unknown_diagnostics_and_weights_are_refused(self):
        for req in ({"diagnostic": "gwr_magic", "model": "H1"},
                    {"diagnostic": "residual_dependence", "model": "H1", "weights": "rook99"}):
            with self.subTest(req=req), self.assertRaises(InvalidResponse):
                critic.validate({"verdict": "approve", "issues": [],
                                 "requested_diagnostics": [req]}, ["H1"], True)


class TheWholeMethodEndToEnd(_Ctx):
    """P through the real plan graph; only the LLM is scripted."""

    def _scripted(self, condition, critic_replies=None, seen=None):
        from agents.task_manager import research_plan_generator as rpg
        conditions.apply_to_ctx(self.ctx, condition)
        planner = json.dumps(_plan(("OLS", ["cells"], "ols.csv"), hypothesis="planner wording"))
        revised = json.dumps({**_plan(("OLS", ["cells"], "ols.csv"),
                                      ("Spatial error model", ["cells"], "sem.csv"),
                                      hypothesis="the reviser rewrote this"),
                              "revision_log": [{"change": "added SEM", "evidence": "Moran's I"}]})
        calls = []

        def component_llm(messages):
            system = messages[0].content
            calls.append(system.split("\n")[0][:40])
            if "state the\nhypotheses" in system or "State the hypotheses" in messages[1].content:
                return json.dumps({"hypotheses": [HYPOTHESIS]})
            if system.startswith("You revise"):
                return revised
            if system.startswith("You are a spatial-analysis reviewer"):
                if seen is not None:
                    seen.append((system, messages[1].content))
                if critic_replies:
                    return json.dumps(critic_replies.pop(0))
                return json.dumps({"verdict": "approve", "issues": [], "requested_diagnostics": []})
            raise AssertionError(f"unexpected prompt: {system[:80]}")

        fake = mock.MagicMock()
        fake.invoke.side_effect = lambda msgs: mock.MagicMock(content=component_llm(msgs))
        planner_llm = mock.MagicMock()
        planner_llm.bind_tools.side_effect = lambda tools: planner_llm
        with mock.patch.object(llm, "make_llm", return_value=fake), \
             mock.patch.object(rpg, "GIBDChat", return_value=planner_llm), \
             mock.patch.object(rpg, "_build_research_design_tools", return_value=[]), \
             mock.patch.object(rpg, "_run_agent_loop", lambda *a, **k: planner):
            rpg.build_serial_graph(self.ctx)
        return calls

    def test_p_produces_every_record_and_keeps_the_lock(self):
        self._scripted("P")
        for name in ("hypotheses_locked.json", "spatial_diagnostics.json", "plan_revisions.json",
                     "spatial_critic.json", "spatial_assumptions.md", "planner_user_prompt.txt"):
            self.assertTrue(os.path.exists(os.path.join(self.out, name)), name)
        plan = self.ctx["artifacts"]["research_plan_final"]
        steps = plan["objectives"][0]["steps"]
        self.assertEqual(plan["objectives"][0]["hypothesis"], HYPOTHESIS["statement"])
        self.assertTrue(steps[1]["exploratory"])
        self.assertNotIn("exploratory", steps[0])
        # The analysis runs on the confirmation half, not the data the diagnostics saw.
        split = self.ctx["artifacts"]["spatial_split"]
        self.assertEqual(steps[1]["input_node_paths"]["cells"], split["confirmation"]["cells"])
        with open(os.path.join(self.out, "planner_user_prompt.txt"), encoding="utf-8") as f:
            self.assertIn("LOCKED HYPOTHESES", f.read())
        diag = json.load(open(os.path.join(self.out, "spatial_diagnostics.json"), encoding="utf-8"))
        self.assertEqual(diag["assumption_status"]["H1"]["independence"]["status"], "contradicted")
        section = open(os.path.join(self.out, "spatial_assumptions.md"), encoding="utf-8").read()
        self.assertIn("| H1 | Independence of observations |", section)
        self.assertIn("Spatial error model", section)

    def test_the_second_critic_round_is_a_follow_up(self):
        issue = {"kind": "ignored_evidence", "severity": "major", "detail": "dependence ignored"}
        seen = []
        self._scripted("P", critic_replies=[
            {"verdict": "revise", "issues": [issue], "requested_diagnostics": []},
            {"verdict": "approve", "issues": [], "requested_diagnostics": []}], seen=seen)
        self.assertEqual(len(seen), 2)
        self.assertNotIn("follow-up review", seen[0][0])
        self.assertIn("follow-up review", seen[1][0])
        self.assertIn("final review round", seen[1][0])
        self.assertIn("dependence ignored", seen[1][1])
        record = json.load(open(os.path.join(self.out, "spatial_critic.json"), encoding="utf-8"))
        self.assertEqual(record["final_verdict"], "approve")

    def test_no_critic_ablation_never_calls_the_critic(self):
        calls = self._scripted("P-no-critic")
        self.assertFalse(any(c.startswith("You are a spatial-analysis") for c in calls))
        self.assertFalse(os.path.exists(os.path.join(self.out, "spatial_critic.json")))

    def test_baselines_get_none_of_it(self):
        from agents.task_manager import research_plan_generator as rpg
        conditions.apply_to_ctx(self.ctx, "B0")
        planner_llm = mock.MagicMock()
        planner_llm.bind_tools.side_effect = lambda tools: planner_llm
        plan = json.dumps(_plan(("OLS", ["cells"], "ols.csv")))
        with mock.patch.object(llm, "make_llm", side_effect=AssertionError("no component LLM")), \
             mock.patch.object(rpg, "GIBDChat", return_value=planner_llm), \
             mock.patch.object(rpg, "_build_research_design_tools", return_value=[]), \
             mock.patch.object(rpg, "_run_agent_loop", lambda *a, **k: plan):
            rpg.build_serial_graph(self.ctx)
        for name in ("hypotheses_locked.json", "spatial_diagnostics.json", "spatial_assumptions.md"):
            self.assertFalse(os.path.exists(os.path.join(self.out, name)), name)
        self.assertNotIn("hypothesis_lock", self.ctx["artifacts"]["research_plan_final"])


if __name__ == "__main__":
    unittest.main()
