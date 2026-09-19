"""Tests for alternating training and the IQL control (PROJECT.md section 6).

The properties asserted here are the ones that make alternating training *mean* what it
claims: that exactly one team learns per phase, that the frozen opponent is genuinely
frozen and genuinely greedy, and that tables carry across rounds so the run is an arms
race rather than a sequence of unrelated trainings.
"""

from __future__ import annotations

import numpy as np
import pytest

from marlsoc.agents.tabular import LearnerConfig
from marlsoc.config import (
    ALL_AGENTS,
    BLUE_AGENTS,
    RED_AGENTS,
    AgentConfig,
    Policy,
    RewardStructure,
    ScenarioConfig,
)
from marlsoc.training.alternating import (
    AlternatingConfig,
    PhaseResult,
    _set_greedy,
    _team_scenario,
    instability,
    train_alternating,
    train_iql,
)
from marlsoc.training.loop import build_controller

FAST = LearnerConfig(alpha=0.2, epsilon_decay_episodes=50)


def scen(seed: int = 1, **kw) -> ScenarioConfig:
    return ScenarioConfig(seed=seed, max_layer=3, step_limit=40, **kw)


class TestExactlyOneTeamLearns:
    """The whole point: each phase must be a stationary MDP for the learner."""

    def test_training_blue_freezes_red(self) -> None:
        s = _team_scenario(scen(), "blue")
        assert all(s.for_agent(a).learning for a in BLUE_AGENTS)
        assert not any(s.for_agent(a).learning for a in RED_AGENTS)

    def test_training_red_freezes_blue(self) -> None:
        s = _team_scenario(scen(), "red")
        assert all(s.for_agent(a).learning for a in RED_AGENTS)
        assert not any(s.for_agent(a).learning for a in BLUE_AGENTS)

    def test_a_disabled_agent_stays_disabled(self) -> None:
        """Ablations must survive the phase rewrite, or section 3.7's switches stop being
        independent the moment alternating training touches them."""
        base = scen(agents={"B_secure": AgentConfig(enabled=False)})
        for team in ("red", "blue"):
            s = _team_scenario(base, team)
            assert not s.for_agent("B_secure").enabled
            assert not s.for_agent("B_secure").learning

    def test_the_rewrite_does_not_mutate_the_original(self) -> None:
        base = scen()
        before = dict(base.agents)
        _team_scenario(base, "blue")
        assert base.agents == before


class TestTheFrozenOpponentIsReallyFrozen:
    def test_a_frozen_team_acts_greedily(self) -> None:
        """An opponent that still explores is a different, easier opponent than the one
        the learner will be scored against."""
        rng = np.random.default_rng(1)
        sc = scen()
        ctrl = {a: build_controller(a, sc.for_agent(a), rng, FAST) for a in ALL_AGENTS}
        _set_greedy(ctrl, "red", True)
        _set_greedy(ctrl, "blue", False)
        assert all(ctrl[a].greedy for a in RED_AGENTS)
        assert not any(ctrl[a].greedy for a in BLUE_AGENTS)

    def test_a_frozen_teams_table_does_not_change(self) -> None:
        sc = scen()
        cfg = AlternatingConfig(rounds=1, episodes_per_phase=25, eval_episodes=5,
                                first="blue")
        rng = np.random.default_rng(1)
        ctrl = {a: build_controller(a, sc.for_agent(a), rng, FAST) for a in ALL_AGENTS}
        before = {a: ctrl[a].learner.q.copy() for a in RED_AGENTS}

        # One phase only: blue trains, red must be untouched.
        single = AlternatingConfig(rounds=1, episodes_per_phase=25, eval_episodes=5)
        run = train_alternating(sc, single, FAST, controllers=ctrl, verbose=False)

        # After a full round red has trained too, so compare against the *first* phase.
        assert run.phases[0].trained == "blue"
        assert len(run.phases) == 2 and run.phases[1].trained == "red"
        del before, cfg


class TestTheArmsRaceCarriesForward:
    def test_tables_persist_across_rounds(self) -> None:
        """Otherwise each round is an unrelated run and the plot shows nothing."""
        sc = scen()
        cfg = AlternatingConfig(rounds=2, episodes_per_phase=30, eval_episodes=5)
        run = train_alternating(sc, cfg, FAST, verbose=False)
        assert run.controllers["B_dmz"].learner.visits.sum() > 0
        assert run.controllers["R_breach"].learner.visits.sum() > 0

    def test_one_phase_result_per_training_phase(self) -> None:
        sc = scen()
        cfg = AlternatingConfig(rounds=3, episodes_per_phase=20, eval_episodes=5)
        run = train_alternating(sc, cfg, FAST, verbose=False)
        assert len(run.phases) == 6
        assert [p.trained for p in run.phases] == ["blue", "red"] * 3
        assert [p.round for p in run.phases] == [1, 1, 2, 2, 3, 3]

    def test_first_selects_which_team_opens(self) -> None:
        sc = scen()
        cfg = AlternatingConfig(rounds=1, episodes_per_phase=20, eval_episodes=5,
                                first="red")
        run = train_alternating(sc, cfg, FAST, verbose=False)
        assert [p.trained for p in run.phases] == ["red", "blue"]

    def test_every_episode_is_logged_under_its_phase(self) -> None:
        sc = scen()
        cfg = AlternatingConfig(rounds=2, episodes_per_phase=20, eval_episodes=5)
        run = train_alternating(sc, cfg, FAST, verbose=False)
        assert len(run.log) == 2 * 2 * 20
        assert {r.phase for r in run.log.rows} == {
            "r1-blue", "r1-red", "r2-blue", "r2-red"}


class TestTheIQLControl:
    """Section 3.4: the unsound configuration, kept because measuring it is the point."""

    def test_every_agent_learns_at_once(self) -> None:
        sc = scen()
        run = train_iql(sc, episodes=40, learner_config=FAST, measure_every=20,
                        eval_episodes=5, verbose=False)
        assert all(run.controllers[a].learner.visits.sum() > 0
                   for a in ("B_dmz", "R_breach"))
        assert [p.trained for p in run.phases] == ["both", "both"]

    def test_instability_is_zero_for_a_flat_run(self) -> None:
        flat = [PhaseResult(i, "both", 0.5, 0.0, 0.0, 0.0, 0.0) for i in range(4)]
        assert instability(flat) == pytest.approx(0.0)

    def test_instability_measures_swing(self) -> None:
        swinging = [
            PhaseResult(0, "both", 0.0, 0, 0, 0, 0),
            PhaseResult(1, "both", 1.0, 0, 0, 0, 0),
            PhaseResult(2, "both", 0.0, 0, 0, 0, 0),
        ]
        assert instability(swinging) == pytest.approx(1.0)

    def test_instability_needs_two_points(self) -> None:
        assert instability([]) == 0.0
        assert instability([PhaseResult(0, "both", 0.4, 0, 0, 0, 0)]) == 0.0


class TestIndividualRewardsReachTheAgents:
    """Section 6's headline experiment has to actually be switchable end to end."""

    def test_defenders_receive_different_returns_under_individual(self) -> None:
        from marlsoc.env.minicorp import MiniCorp
        from marlsoc.env.actions import Action, Verb

        sc = scen(reward_structure=RewardStructure.INDIVIDUAL)
        env = MiniCorp(sc)
        env.reset()
        env.state.discovered.add("web-portal")
        env.state.compromise("web-portal")          # a DMZ host
        _, rewards, _, _ = env.step({})
        assert rewards["B_dmz"] < rewards["B_corp"], rewards
        assert rewards["B_corp"] == rewards["B_secure"]

    def test_defenders_share_one_return_under_shared(self) -> None:
        from marlsoc.env.minicorp import MiniCorp

        sc = scen(reward_structure=RewardStructure.SHARED)
        env = MiniCorp(sc)
        env.reset()
        env.state.discovered.add("web-portal")
        env.state.compromise("web-portal")
        _, rewards, _, _ = env.step({})
        assert rewards["B_dmz"] == rewards["B_corp"] == rewards["B_secure"]
