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
        controllers, _, _ = train(scenario, 60, verbose=False, eval_every=0)
        before = controllers["B_dmz"].learner.q.copy()
        evaluate(scenario, controllers, episodes=30)
        assert np.array_equal(before, controllers["B_dmz"].learner.q)

    def test_evaluate_restores_exploration_afterwards(self) -> None:
        # Otherwise training resumed after an evaluation would be silently greedy.
        scenario = phase2()
        controllers, _, _ = train(scenario, 30, verbose=False, eval_every=0)
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


class TestCurriculum:
    """Section 7.4's stage progression, and the two rules that are not in the spec."""

    def _config(self, **kw):
        from marlsoc.training.curriculum import CurriculumConfig
        # promote_on_greedy is off by default here: these tests drive Curriculum.record
        # directly and are about the promotion *mechanics* -- the dwell time, the window
        # reset, the forced promotion. Which policy supplies the rate is a separate
        # concern, tested in TestPromotionJudgesTheGreedyPolicy.
        base = dict(promote_window=10, min_episodes_per_stage=10, promote_on_greedy=False,
                    promote_threshold=0.7, max_episodes_per_stage=1000)
        base.update(kw)
        return CurriculumConfig(**base)

    def _curriculum(self, **kw):
        from marlsoc.training.curriculum import Curriculum
        return Curriculum(self._config(**kw))

    def test_it_starts_at_stage_one(self) -> None:
        c = self._curriculum()
        assert c.stage_number == 1 and c.max_layer == 2 and not c.finished

    def test_clearing_a_stage_deepens_the_stack(self) -> None:
        c = self._curriculum()
        transition = None
        for i in range(10):
            transition = c.record(won=True, episode=i) or transition
        assert transition is not None
        assert transition.from_stage == 1 and transition.to_stage == 2
        assert c.max_layer == 3

    def test_a_lucky_streak_cannot_promote_before_the_dwell_time(self) -> None:
        """Early in a stage epsilon is high, and a run of fortunate episodes can clear a
        window before red has learned anything stable. It would then arrive at the next
        stage with a Q-table full of noise and fail there, which reads as 'the curriculum
        stalled' rather than 'it was promoted too early'."""
        c = self._curriculum(min_episodes_per_stage=50)
        for i in range(40):
            assert c.record(won=True, episode=i) is None
        assert c.stage_number == 1

    def test_failing_keeps_red_on_the_stage(self) -> None:
        c = self._curriculum()
        for i in range(60):
            assert c.record(won=False, episode=i) is None
        assert c.stage_number == 1

    def test_the_threshold_is_actually_applied(self) -> None:
        # 50% success against a 70% threshold must not promote.
        c = self._curriculum()
        for i in range(60):
            assert c.record(won=i % 2 == 0, episode=i) is None
        assert c.stage_number == 1

    def test_the_window_resets_after_a_promotion(self) -> None:
        """Otherwise the cleared stage's wins would carry into the next stage's window
        and promote red again immediately, skipping stages it never played."""
        c = self._curriculum()
        for i in range(10):
            c.record(won=True, episode=i)
        assert c.stage_number == 2
        assert c.success_rate == 0.0

    def test_a_stuck_stage_is_force_promoted_and_flagged(self) -> None:
        """A stage red cannot clear would otherwise consume the whole budget and the run
        would say nothing about the later stages. A forced promotion is recorded so it
        can never be mistaken for a real one."""
        c = self._curriculum(max_episodes_per_stage=30)
        transition = None
        for i in range(30):
            transition = c.record(won=False, episode=i) or transition
        assert transition is not None and transition.forced
        assert "FORCED" in c.summary()

    def test_the_final_stage_never_promotes(self) -> None:
        from marlsoc.training.curriculum import STAGES
        c = self._curriculum()
        for stage in range(len(STAGES) - 1):
            for i in range(10):
                c.record(won=True, episode=i)
        assert c.finished and c.max_layer == 6
        for i in range(50):
            assert c.record(won=True, episode=i) is None


class TestExplorationBoost:
    """Not in the spec, and the curriculum does not work without it."""

    def _learner(self, **kw):
        from marlsoc.agents.tabular import LearnerConfig, TabularLearner
        cfg = LearnerConfig(epsilon_start=1.0, epsilon_end=0.05,
                            epsilon_decay_episodes=1000, **kw)
        return TabularLearner("t", 4, 3, cfg, np.random.default_rng(0))

    def test_boosting_restores_the_requested_rate(self) -> None:
        L = self._learner()
        for _ in range(1000):
            L.end_episode()
        assert L.epsilon == pytest.approx(0.05, abs=1e-6)
        L.boost_exploration(0.40)
        assert L.epsilon == pytest.approx(0.40, abs=0.01)

    def test_decay_continues_normally_afterwards(self) -> None:
        """Implemented by inverting the schedule rather than storing an offset, so the
        agent stays on one continuous schedule."""
        L = self._learner()
        for _ in range(1000):
            L.end_episode()
        L.boost_exploration(0.40)
        before = L.epsilon
        for _ in range(300):
            L.end_episode()
        assert L.epsilon < before

    def test_a_request_outside_the_schedule_is_clamped(self) -> None:
        # Asking for more exploration than the schedule ever had just means starting over.
        L = self._learner()
        L.boost_exploration(5.0)
        assert L.epsilon <= 1.0
        L.boost_exploration(0.0)
        assert L.epsilon >= 0.05


class TestCurriculumTraining:
    def test_a_run_progresses_through_stages_and_logs_them(self) -> None:
        from marlsoc.agents.tabular import Algorithm, LearnerConfig
        from marlsoc.config import static_firewall
        from marlsoc.training.curriculum import CurriculumConfig
        from marlsoc.training.loop import train_curriculum

        sc = ScenarioConfig(seed=1, agents=static_firewall().agents)
        cfg = LearnerConfig(algorithm=Algorithm.SARSA, epsilon_decay_episodes=200)
        run = train_curriculum(
            sc, 700, cfg,
            CurriculumConfig(min_episodes_per_stage=100, promote_window=50),
            verbose=False,
        )
        assert run.curriculum.stage_number > 1, "red never cleared stage 1 unopposed"
        assert run.curriculum.transitions
        # The stage is recorded per episode, so the sawtooth comes out of the CSV.
        assert {r.phase for r in run.log.rows} >= {"stage1", "stage2"}

    def test_the_q_table_is_carried_across_stages(self) -> None:
        """The transfer section 7.4 calls the whole point: a table that has learned to
        get a DMZ foothold already knows how when Layer 3 switches on."""
        from marlsoc.agents.tabular import Algorithm, LearnerConfig
        from marlsoc.config import static_firewall
        from marlsoc.training.curriculum import CurriculumConfig
        from marlsoc.training.loop import train_curriculum

        sc = ScenarioConfig(seed=1, agents=static_firewall().agents)
        run = train_curriculum(
            sc, 600, LearnerConfig(algorithm=Algorithm.SARSA, epsilon_decay_episodes=200),
            CurriculumConfig(min_episodes_per_stage=100, promote_window=50),
            verbose=False,
        )
        learner = run.controllers["R_breach"].learner
        assert learner is not None and learner.updates > 0
        assert run.curriculum.transitions          # it did move stages
        assert learner.coverage > 0.0              # ...on one continuous table


class TestTheCurriculumAlwaysTraverses:
    """A stage red cannot clear must not consume the run (CLAUDE.md 3.21).

    The failure this guards is silent and expensive: red reached stage 3, spent 4,885 of
    its 9,000 episodes there winning 2.3%, and was then *evaluated at stage 5* -- a depth
    it had never once trained at. Nothing errored; the agent simply looked incapable.
    """

    def test_an_agent_that_never_wins_still_reaches_the_last_stage(self) -> None:
        from marlsoc.training.curriculum import Curriculum, CurriculumConfig, STAGES
        budget = 9_000
        c = Curriculum(CurriculumConfig(), total_episodes=budget)
        for episode in range(budget):
            c.record(won=False, episode=episode)
        assert c.max_layer == STAGES[-1], "the curriculum never reached the full stack"
        assert all(t.forced for t in c.transitions), "nothing was actually cleared"

    def test_a_losing_run_spreads_itself_across_every_stage(self) -> None:
        from marlsoc.training.curriculum import Curriculum, CurriculumConfig, STAGES
        budget = 9_000
        c = Curriculum(CurriculumConfig(), total_episodes=budget)
        spent, last = [], 0
        for episode in range(budget):
            if c.record(won=False, episode=episode) is not None:
                spent.append(episode - last)
                last = episode
        spent.append(budget - last)
        assert len(spent) == len(STAGES), spent
        # No stage may take more than half the run: that is the pathology itself.
        assert max(spent) <= budget // 2, spent

    def test_a_fixed_cap_is_still_honoured_when_asked_for(self) -> None:
        from marlsoc.training.curriculum import Curriculum, CurriculumConfig
        c = Curriculum(CurriculumConfig(max_episodes_per_stage=700),
                       total_episodes=100_000)
        for episode in range(700):
            t = c.record(won=False, episode=episode)
        assert t is not None and t.forced

    def test_the_dwell_time_still_wins_over_a_tight_budget(self) -> None:
        # A budget so small the even share is below min_episodes_per_stage must not
        # promote on episode two -- an untrained stage promoted early arrives at the next
        # one with a Q-table of noise.
        from marlsoc.training.curriculum import Curriculum, CurriculumConfig
        c = Curriculum(CurriculumConfig(), total_episodes=100)
        for episode in range(100):
            if c.record(won=False, episode=episode) is not None:
                raise AssertionError(f"promoted at episode {episode}, before the dwell")


class TestEvaluationRestoresWhatItFound:
    """An evaluation must not change the agents it measured.

    ``evaluate`` forces every learned controller greedy and used to reset them all to
    *not* greedy afterwards. A defender configured as ``Policy.GREEDY`` -- frozen and
    deployed, per CLAUDE.md 3.7 -- is greedy by design, so its first evaluation silently
    converted it into an epsilon-greedy agent for every run that followed.
    """

    def test_a_frozen_greedy_defender_is_still_greedy_afterwards(self) -> None:
        from marlsoc.config import ALL_AGENTS
        from marlsoc.training.loop import build_controller, evaluate

        sc = ScenarioConfig(seed=1, agents={
            "B_dmz": AgentConfig(learning=False, policy=Policy.GREEDY)})
        rng = np.random.default_rng(1)
        ctrl = {a: build_controller(a, sc.for_agent(a), rng) for a in ALL_AGENTS}
        assert ctrl["B_dmz"].greedy, "a GREEDY-policy controller should start greedy"
        evaluate(sc, ctrl, 3)
        assert ctrl["B_dmz"].greedy, "evaluation turned the frozen defender loose"

    def test_a_training_agent_is_still_exploring_afterwards(self) -> None:
        from marlsoc.config import ALL_AGENTS
        from marlsoc.training.loop import build_controller, evaluate

        sc = ScenarioConfig(seed=1)
        rng = np.random.default_rng(1)
        ctrl = {a: build_controller(a, sc.for_agent(a), rng) for a in ALL_AGENTS}
        assert not ctrl["B_dmz"].greedy
        evaluate(sc, ctrl, 3)
        assert not ctrl["B_dmz"].greedy


class TestPromotionJudgesTheGreedyPolicy:
    """A curriculum must promote on the policy you intend to keep (CLAUDE.md 3.23).

    Training episodes are epsilon-greedy, so their win rate is the *exploring* policy's.
    When most greedy actions have never been updated, the exploring policy finishes the
    chain while the deployable one cannot -- measured, every stage transition reported
    100% success while a greedy evaluation of the same agent against the same static
    defence scored 0.0%.
    """

    def _curriculum(self, **kw):
        from marlsoc.training.curriculum import Curriculum, CurriculumConfig
        base = dict(promote_window=10, min_episodes_per_stage=10,
                    promote_threshold=0.7, max_episodes_per_stage=10_000)
        base.update(kw)
        return Curriculum(CurriculumConfig(**base))

    def test_winning_while_exploring_is_not_enough_to_promote(self) -> None:
        c = self._curriculum()
        for i in range(200):
            assert c.record(won=True, episode=i) is None, "promoted on the training rate"
        assert c.stage_number == 1

    def test_a_good_greedy_evaluation_promotes(self) -> None:
        c = self._curriculum()
        for i in range(20):
            c.record(won=True, episode=i)
        c.record_evaluation(0.9)
        assert c.record(won=True, episode=20) is not None

    def test_a_poor_greedy_evaluation_does_not(self) -> None:
        c = self._curriculum()
        for i in range(20):
            c.record(won=True, episode=i)
        c.record_evaluation(0.1)          # exploring wins every episode, greedy does not
        assert c.record(won=True, episode=20) is None

    def test_the_recorded_rate_is_the_one_promotion_was_judged_on(self) -> None:
        """Otherwise the sawtooth plot is annotated with a number that decided nothing."""
        c = self._curriculum()
        for i in range(20):
            c.record(won=True, episode=i)
        c.record_evaluation(0.85)
        transition = c.record(won=True, episode=20)
        assert transition is not None and transition.success_rate == 0.85

    def test_a_stale_evaluation_does_not_carry_into_the_next_stage(self) -> None:
        c = self._curriculum()
        for i in range(20):
            c.record(won=True, episode=i)
        c.record_evaluation(0.9)
        assert c.record(won=True, episode=20) is not None
        assert c.greedy_rate is None, "the new stage inherited the old stage's evidence"
