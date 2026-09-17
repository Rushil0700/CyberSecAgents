"""Tests for the scripted opponent, the training loop and the metrics log.

The properties that matter here are procedural rather than numerical: that the three
agent switches produce the behaviour CLAUDE.md 3.7 promises, that SARSA actually receives
the action it needs, that an evaluation cannot train the policy it is measuring, and that
a seeded run replays exactly. Each of those failing produces plausible numbers rather
than an error, which is why they are tested.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from marlsoc.agents.scripted import GreedyDefender, ScriptConfig, ScriptedAttacker
from marlsoc.agents.tabular import Algorithm, LearnerConfig
from marlsoc.config import (
    AgentConfig, Policy, ScenarioConfig, static_firewall,
)
from marlsoc.env import actions as act
from marlsoc.env import detection as det
from marlsoc.env.actions import Verb
from marlsoc.env.minicorp import MiniCorp
from marlsoc.training import loop
from marlsoc.training.loop import (
    LearnedController, NoopController, RandomController, ScriptedController,
    build_controller, evaluate, run_episode, train,
)
from marlsoc.training.metrics import EpisodeRecord, MetricsLog

SCRIPT = AgentConfig(learning=False, policy=Policy.SCRIPTED)
OFF = AgentConfig(enabled=False)


def phase2(max_layer: int = 2, learner: str = "B_dmz", seed: int = 1) -> ScenarioConfig:
    agents = {"R_scout": SCRIPT, "R_breach": SCRIPT}
    for blue in ("B_dmz", "B_corp", "B_secure"):
        if blue != learner:
            agents[blue] = OFF
    return ScenarioConfig(seed=seed, max_layer=max_layer, agents=agents)


class TestScripted:
    def test_the_attacker_only_ever_picks_legal_actions(self) -> None:
        env = MiniCorp(phase2(6))
        env.reset()
        red = {a: ScriptedAttacker(a, rng=np.random.default_rng(0))
               for a in ("R_scout", "R_breach")}
        for _ in range(120):
            joint = {a: p.act(env.state) for a, p in red.items()}
            for agent, action in joint.items():
                assert act.is_legal(env.state, agent, action)
            _, _, done, _ = env.step(joint)
            if done:
                break

    def test_the_attacker_prefers_depth_over_breadth(self) -> None:
        """A script that widened before advancing would spend the episode collecting DMZ
        hosts it has no use for, and Phase 2 would have nothing to defend against."""
        from marlsoc.agents.scripted import BREACH_PRIORITY
        assert BREACH_PRIORITY.index(Verb.LATERAL_MOVE) < BREACH_PRIORITY.index(Verb.EXPLOIT)
        assert BREACH_PRIORITY[0] is Verb.ALTER_CREDENTIALS

    def test_the_attacker_names_no_hosts(self) -> None:
        """It asks the mask which hosts qualify rather than smuggling in the topology,
        so it knows exactly what a learning agent would know."""
        import inspect
        from marlsoc.agents import scripted
        source = inspect.getsource(scripted)
        body = source.split('"""', 2)[-1]   # skip the module docstring
        for host in ("ad-controller", "web-portal", "auth-server", "mfa-service"):
            assert host not in body, host

    def test_script_noise_is_on_by_default(self) -> None:
        """A deterministic opponent lets blue memorise one trajectory: sharp Q-values
        for a single attack, and a policy that collapses when anything varies."""
        assert ScriptConfig().noise > 0.0

    def test_noise_makes_the_attack_vary_between_seeds(self) -> None:
        def trace(seed: int) -> list[str]:
            env = MiniCorp(phase2(6))
            env.reset(seed)
            red = {a: ScriptedAttacker(a, rng=np.random.default_rng(seed))
                   for a in ("R_scout", "R_breach")}
            out = []
            for _ in range(40):
                joint = {a: p.act(env.state) for a, p in red.items()}
                out.append(str(joint["R_breach"]))
                _, _, done, _ = env.step(joint)
                if done:
                    break
            return out
        assert trace(1) != trace(2)

    def test_the_greedy_defender_isolates_what_looks_suspicious(self) -> None:
        # Section 9 baseline 3. It reacts to evidence without weighing what evidence is
        # worth, which is the failure the learned policy has to beat.
        env = MiniCorp(phase2(6))
        env.reset()
        env.state.alerts["dev-box"] = det.DEFAULT.suspicion_threshold + 1.0
        chosen = GreedyDefender("B_corp", rng=np.random.default_rng(0)).act(env.state)
        assert chosen.verb is Verb.ISOLATE and chosen.host == "dev-box"

    def test_the_greedy_defender_does_nothing_without_evidence(self) -> None:
        env = MiniCorp(phase2(6))
        env.reset()
        chosen = GreedyDefender("B_corp", rng=np.random.default_rng(0)).act(env.state)
        assert chosen.verb is Verb.NOOP


class TestControllerSwitches:
    """CLAUDE.md 3.7 -- three switches, and what each one actually does."""

    def test_each_policy_builds_its_own_controller(self) -> None:
        rng = np.random.default_rng(0)
        cases = {
            Policy.LEARNED: LearnedController,
            Policy.GREEDY: LearnedController,
            Policy.RANDOM: RandomController,
            Policy.SCRIPTED: ScriptedController,
            Policy.NOOP: NoopController,
        }
        for policy, expected in cases.items():
            built = build_controller("B_corp", AgentConfig(policy=policy), rng)
            assert isinstance(built, expected), policy

    def test_disabled_outranks_policy(self) -> None:
        """Which is why "all blue agents disabled" *is* the static firewall baseline --
        no separate baseline code exists anywhere in the project."""
        rng = np.random.default_rng(0)
        built = build_controller(
            "B_corp", AgentConfig(enabled=False, policy=Policy.LEARNED), rng
        )
        assert isinstance(built, NoopController)
        assert built.learner is None

    def test_greedy_controllers_do_not_explore(self) -> None:
        rng = np.random.default_rng(0)
        built = build_controller("B_corp", AgentConfig(policy=Policy.GREEDY), rng)
        assert isinstance(built, LearnedController) and built.greedy

    def test_the_static_firewall_scenario_disables_only_blue(self) -> None:
        rng = np.random.default_rng(0)
        cfg = static_firewall()
        for agent in ("B_dmz", "B_corp", "B_secure"):
            assert isinstance(build_controller(agent, cfg.for_agent(agent), rng),
                              NoopController)
        for agent in ("R_scout", "R_breach"):
            assert not isinstance(build_controller(agent, cfg.for_agent(agent), rng),
                                  NoopController)

    def test_a_learner_gets_a_table_sized_from_the_topology(self) -> None:
        rng = np.random.default_rng(0)
        built = build_controller("B_corp", AgentConfig(), rng)
        assert built.learner is not None
        assert built.learner.q.shape == (3072, 13)


class TestEpisodeLoop:
    def test_an_episode_produces_a_complete_record(self) -> None:
        scenario = phase2()
        env = MiniCorp(scenario)
        rng = np.random.default_rng(0)
        controllers = {a: build_controller(a, scenario.for_agent(a), rng)
                       for a in ("R_scout", "R_breach", "B_dmz", "B_corp", "B_secure")}
        record, returns = run_episode(env, controllers, scenario, seed=3)
        assert record.steps > 0
        assert record.outcome in {"red_win", "blue_win", "draw"}
        assert set(returns) == {"R_scout", "R_breach", "B_dmz", "B_corp", "B_secure"}

    def test_the_same_seed_replays_the_same_episode(self) -> None:
        # CLAUDE.md 3.7: a chosen episode must be replayable for the demo.
        def once(seed: int) -> tuple:
            scenario = phase2()
            env = MiniCorp(scenario)
            rng = np.random.default_rng(0)
            controllers = {a: build_controller(a, scenario.for_agent(a), rng)
                           for a in ("R_scout", "R_breach", "B_dmz", "B_corp", "B_secure")}
            record, _ = run_episode(env, controllers, scenario, seed=seed)
            return (record.steps, record.outcome, record.layers_breached)
        assert once(5) == once(5)

    def test_sarsa_actually_receives_the_next_action(self) -> None:
        """If the loop quietly omitted a', SARSA would raise -- which is the point of
        making it raise rather than falling back to a max and silently becoming
        Q-Learning."""
        scenario = phase2()
        cfg = LearnerConfig(algorithm=Algorithm.SARSA)
        env = MiniCorp(scenario)
        rng = np.random.default_rng(0)
        controllers = {a: build_controller(a, scenario.for_agent(a), rng, cfg)
                       for a in ("R_scout", "R_breach", "B_dmz", "B_corp", "B_secure")}
        record, _ = run_episode(env, controllers, scenario, seed=1)   # must not raise
        assert controllers["B_dmz"].learner.updates > 0

    def test_a_frozen_agent_does_not_update_its_table(self) -> None:
        # The `learning` switch, independent of `enabled` and `policy`.
        scenario = ScenarioConfig(
            seed=1, max_layer=2,
            agents={"R_scout": SCRIPT, "R_breach": SCRIPT,
                    "B_dmz": AgentConfig(learning=False),
                    "B_corp": OFF, "B_secure": OFF},
        )
        env = MiniCorp(scenario)
        rng = np.random.default_rng(0)
        controllers = {a: build_controller(a, scenario.for_agent(a), rng)
                       for a in ("R_scout", "R_breach", "B_dmz", "B_corp", "B_secure")}
        run_episode(env, controllers, scenario, seed=1)
        assert np.all(controllers["B_dmz"].learner.q == 0.0)

    def test_disabled_agents_still_occupy_their_slot(self) -> None:
        # They idle rather than vanishing, so the joint action stays well formed.
        scenario = phase2()
        env = MiniCorp(scenario)
        rng = np.random.default_rng(0)
        controllers = {a: build_controller(a, scenario.for_agent(a), rng)
                       for a in ("R_scout", "R_breach", "B_dmz", "B_corp", "B_secure")}
        _, returns = run_episode(env, controllers, scenario, seed=1)
        assert returns["B_corp"] != 0.0   # it still receives the shared team reward


class TestEvaluationDoesNotTrain:
    def test_evaluate_leaves_the_q_table_untouched(self) -> None:
        """The mistake that makes a reported number better than the policy actually is.
        Learning is disabled globally rather than per agent, so an evaluation cannot
        train the thing it is measuring even if a switch is set wrong."""
        scenario = phase2()
        controllers, _ = train(scenario, 60, verbose=False)
        before = controllers["B_dmz"].learner.q.copy()
        evaluate(scenario, controllers, episodes=30)
        assert np.array_equal(before, controllers["B_dmz"].learner.q)

    def test_evaluate_restores_exploration_afterwards(self) -> None:
        # Otherwise training resumed after an evaluation would be silently greedy.
        scenario = phase2()
        controllers, _ = train(scenario, 30, verbose=False)
        evaluate(scenario, controllers, episodes=10)
        assert not controllers["B_dmz"].greedy


class TestMetricsLog:
    def _row(self, i: int, outcome: str = "draw", **kw) -> EpisodeRecord:
        base = dict(
            episode=i, phase="p", learner="B_dmz", red_return=1.0, blue_return=-2.0,
            outcome=outcome, steps=10, hosts_compromised=1, layers_breached=2,
            mttd=3, mttc=None, false_positives=0, honeypot_hits=0, epsilon=0.5,
        )
        base.update(kw)
        return EpisodeRecord(**base)

    def test_rate_counts_outcomes(self) -> None:
        log = MetricsLog.from_rows(
            [self._row(0, "red_win"), self._row(1, "draw"),
             self._row(2, "red_win"), self._row(3, "blue_win")]
        )
        assert log.rate("red_win") == 0.5
        assert log.rate("red_win", last=2) == 0.5

    def test_mean_skips_missing_values(self) -> None:
        """An episode where nothing was compromised has no detection time. Averaging a
        zero in for those would flatter the defender badly."""
        log = MetricsLog.from_rows([self._row(0, mttd=None), self._row(1, mttd=8)])
        assert log.mean("mttd") == 8.0

    def test_rolling_mean_shortens_the_series_by_the_window(self) -> None:
        log = MetricsLog.from_rows([self._row(i, blue_return=float(i)) for i in range(10)])
        curve = log.rolling("blue_return", window=4)
        assert curve.shape == (7,)
        assert curve[0] == pytest.approx(1.5)

    def test_rolling_handles_a_window_larger_than_the_log(self) -> None:
        log = MetricsLog.from_rows([self._row(i) for i in range(3)])
        assert log.rolling("blue_return", window=100).size == 1

    def test_csv_round_trips_every_column(self) -> None:
        log = MetricsLog.from_rows([self._row(0), self._row(1, mttd=None)])
        path = log.to_csv(__import__("pathlib").Path("/tmp/marlsoc_test_metrics.csv"))
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert rows[1]["mttd"] == ""          # None survives as an empty cell
        assert rows[0]["outcome"] == "draw"

    def test_summary_is_printable_even_when_empty(self) -> None:
        assert "no episodes" in MetricsLog().summary()


class TestPlots:
    """Smoke tests only -- a plot's correctness is judged by eye, but it must not crash
    on the edge cases training will actually hand it."""

    def _log(self, n: int) -> MetricsLog:
        return MetricsLog.from_rows([
            EpisodeRecord(
                episode=i, phase="p", learner="B_dmz",
                red_return=float(i), blue_return=-float(i),
                outcome="red_win" if i % 3 else "draw", steps=20,
                hosts_compromised=1, layers_breached=i % 7, mttd=2, mttc=None,
                false_positives=i % 4, honeypot_hits=0, epsilon=0.5,
            )
            for i in range(n)
        ])

    def test_learning_curve_writes_a_file(self, tmp_path) -> None:
        from marlsoc.training import plots
        out = plots.learning_curve(self._log(600), tmp_path / "curve.png", window=100)
        assert out.exists() and out.stat().st_size > 0

    def test_it_survives_a_log_shorter_than_the_window(self, tmp_path) -> None:
        # Happens whenever a smoke run is cut short; it must not take the run down.
        from marlsoc.training import plots
        assert plots.learning_curve(self._log(5), tmp_path / "c.png", window=200).exists()

    def test_comparison_overlays_several_runs(self, tmp_path) -> None:
        from marlsoc.training import plots
        logs = {"q_learning": self._log(400), "sarsa": self._log(400)}
        assert plots.comparison(logs, tmp_path / "cmp.png", window=50).exists()
