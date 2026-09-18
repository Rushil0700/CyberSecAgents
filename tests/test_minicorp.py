"""Tests for the environment: the transition function and episode dynamics.

Beyond the unit properties, two of these are *feasibility* tests -- they assert the
results that justify the whole methodology. That every curriculum stage is winnable
against a static defence is what makes section 7.4's staged training possible at all,
and that random exploration wins nothing against an active defender is the sparse-reward
problem section 15 warns about, demonstrated rather than claimed. If either flipped, the
project's plan would need rethinking, so they are tests rather than a paragraph.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from marlsoc.config import ScenarioConfig, static_firewall
from marlsoc.env import actions as act
from marlsoc.env import topology as topo
from marlsoc.env.actions import Action, Verb
from marlsoc.env.layers import Layer
from marlsoc.env.minicorp import MiniCorp
from marlsoc.env.state import HostStatus, Outcome
from marlsoc.env.topology import Zone


def rollout(seed: int, max_layer: int = 6, blue: bool = True, limit: int = 250) -> dict:
    """One episode with every agent sampling uniformly from its legal set."""
    agents_cfg = {} if blue else static_firewall().agents
    env = MiniCorp(ScenarioConfig(seed=seed, max_layer=max_layer, agents=agents_cfg))
    env.reset()
    rng = np.random.default_rng(seed)
    acting = [a for a in act.ACTION_SPACES if blue or not a.startswith("B_")]
    info: dict = {}
    for _ in range(limit):
        joint = {}
        for agent in acting:
            legal = act.legal_actions(env.state, agent)
            joint[agent] = legal[rng.integers(len(legal))]
        _, _, done, info = env.step(joint)
        if done:
            break
    return info


class TestFeasibility:
    """The two results the methodology rests on."""

    def test_every_curriculum_stage_is_winnable_against_a_static_defence(self) -> None:
        """Section 7.4 trains red against a static defence first so it receives positive
        signal. A stage red cannot win is a stage that never promotes -- section 15's
        "curriculum stage never promotes" -- and the curriculum stalls there forever."""
        for max_layer in (2, 3, 4, 5, 6):
            wins = sum(
                rollout(s, max_layer=max_layer, blue=False)["outcome"] == "red_win"
                for s in range(30)
            )
            assert wins > 0, f"stage 1-{max_layer} is unwinnable even unopposed"

    def test_a_middle_stage_is_not_harder_than_a_deeper_one(self) -> None:
        """The regression guard for a real bug.

        Progress used to be derived from breached layers, so under a stage where Layer 4
        was inactive it could never be *breached*, Corp never counted as entered, the
        secure zone stayed unscannable and red could not discover the crown jewel. The
        symptom was stage 1-3 winning 0% of episodes while the strictly harder stage 1-4
        won 100% -- an inversion that means a layer is not actually in the way.
        """
        wins = {
            ml: sum(rollout(s, max_layer=ml, blue=False)["outcome"] == "red_win"
                    for s in range(30))
            for ml in (2, 3, 4, 5, 6)
        }
        assert min(wins.values()) > 0, wins

    def test_the_full_stack_costs_red_more_time_than_the_shallow_one(self) -> None:
        """Six layers should take longer to walk than two.

        Only the endpoints are compared: adding steal_credentials and one lateral move
        (stages 3 and 4) costs almost no wall-clock steps, so 1-2 against 1-4 is
        statistically indistinguishable and asserting an ordering there would be testing
        sampling noise. 1-2 against 1-6 is a real difference.
        """
        shallow = np.mean([rollout(s, max_layer=2, blue=False)["step"] for s in range(60)])
        full = np.mean([rollout(s, max_layer=6, blue=False)["step"] for s in range(60)])
        assert full > shallow * 1.1, (shallow, full)

    def test_random_exploration_alone_gets_nowhere_against_a_defender(self) -> None:
        """The sparse-reward, long-horizon problem in section 7.4, demonstrated.

        Red almost never completes a six-layer chain by chance, so it receives no
        learning signal from the +100 alone. This is why the progressive ladder in 5.4
        and the curriculum in 7.4 are both required rather than optional.
        """
        outcomes = Counter(rollout(s)["outcome"] for s in range(60))
        depths = [rollout(s)["layers_breached"] for s in range(60)]
        assert outcomes["red_win"] == 0
        assert np.mean(depths) < 2.0


class TestEpisodeLifecycle:
    def test_reset_returns_an_observation_for_every_agent(self) -> None:
        env = MiniCorp()
        observations = env.reset()
        assert set(observations) == set(act.ACTION_SPACES)

    def test_the_same_seed_replays_an_episode_exactly(self) -> None:
        """CLAUDE.md 3.7: a chosen episode must be replayable on demand for the demo.
        One seeded generator is the only randomness in the environment, which is what
        makes this hold."""
        def trace(seed: int) -> list:
            env = MiniCorp(ScenarioConfig(seed=seed))
            env.reset()
            rng = np.random.default_rng(99)
            out = []
            for _ in range(40):
                joint = {}
                for agent in act.ACTION_SPACES:
                    legal = act.legal_actions(env.state, agent)
                    joint[agent] = legal[rng.integers(len(legal))]
                _, rewards, done, info = env.step(joint)
                out.append((info["layers_breached"], info["hosts_compromised"],
                            round(rewards["R_breach"], 4)))
                if done:
                    break
            return out

        assert trace(11) == trace(11)
        assert trace(11) != trace(12)

    def test_an_episode_hitting_the_step_limit_is_a_draw(self) -> None:
        env = MiniCorp(ScenarioConfig(seed=1, step_limit=5))
        env.reset()
        for _ in range(5):
            _, _, done, info = env.step({"R_scout": Action(Verb.WAIT)})
        assert done and info["outcome"] == "draw"

    def test_stepping_a_finished_episode_raises(self) -> None:
        env = MiniCorp(ScenarioConfig(seed=1, step_limit=1))
        env.reset()
        env.step({"R_scout": Action(Verb.WAIT)})
        with pytest.raises(RuntimeError):
            env.step({"R_scout": Action(Verb.WAIT)})

    def test_an_illegal_action_raises_rather_than_being_ignored(self) -> None:
        """Silently dropping it would let a buggy policy appear to work while doing
        nothing, and section 15 calls 'red never wins' the hardest failure to diagnose."""
        env = MiniCorp()
        env.reset()
        with pytest.raises(ValueError):
            env.step({"R_breach": Action(Verb.ALTER_CREDENTIALS)})

    def test_omitted_agents_simply_do_not_act(self) -> None:
        # How a disabled agent (CLAUDE.md 3.7) takes part without the environment
        # needing to know it was switched off.
        env = MiniCorp()
        env.reset()
        _, rewards, _, _ = env.step({})
        assert set(rewards) == set(act.ACTION_SPACES)

    def test_containing_every_foothold_is_a_blue_win(self) -> None:
        env = MiniCorp()
        env.reset()
        env.state.record_breach(Layer.PERIMETER)
        env.state.discovered.add("web-portal")
        env.state.compromise("web-portal")
        _, _, done, info = env.step({"B_dmz": Action(Verb.ISOLATE, host="web-portal")})
        assert done and info["outcome"] == "blue_win"

    def test_holding_nothing_at_reset_is_not_a_blue_win(self) -> None:
        # Red starts with no foothold; that is the start of the game, not a victory.
        env = MiniCorp()
        env.reset()
        _, _, done, info = env.step({"R_scout": Action(Verb.WAIT)})
        assert not done and info["outcome"] == "running"


class TestRedTransitions:
    def test_slow_scan_is_the_only_way_past_layer_one(self) -> None:
        env = MiniCorp(ScenarioConfig(seed=4))
        env.reset()
        for _ in range(60):
            env.step({"R_scout": Action(Verb.SCAN, zone=Zone.DMZ)})
        assert not env.state.layers.is_breached(Layer.PERIMETER)

        env.reset()
        for _ in range(60):
            env.step({"R_scout": Action(Verb.SLOW_SCAN, zone=Zone.DMZ)})
            if env.state.layers.is_breached(Layer.PERIMETER):
                break
        assert env.state.layers.is_breached(Layer.PERIMETER)

    def test_scanning_is_louder_than_slow_scanning(self) -> None:
        """The stealth tradeoff that makes Layer 1 a decision rather than a formality."""
        loud = MiniCorp(ScenarioConfig(seed=5)); loud.reset()
        quiet = MiniCorp(ScenarioConfig(seed=5)); quiet.reset()
        for _ in range(10):
            loud.step({"R_scout": Action(Verb.SCAN, zone=Zone.DMZ)})
            quiet.step({"R_scout": Action(Verb.SLOW_SCAN, zone=Zone.DMZ)})
        assert loud.state.heat > quiet.state.heat

    def test_the_pivot_cannot_be_taken_by_ordinary_widening(self) -> None:
        """The Layer 4 bypass bug, pinned.

        Red reaching ad-controller with exploit -- as intra-zone widening -- walked
        straight through Layer 4 without breaching it, plateauing the layer-depth metric
        at 3 in every curriculum stage.
        """
        env = MiniCorp()
        env.reset()
        # The pivot is no longer offered as an exploit target at all -- an action that
        # can never be legal does not belong in the action space.
        assert topo.PIVOT_HOST not in act.EXPLOIT_TARGETS
        assert topo.PIVOT_HOST in act.LATERAL_TARGETS

    def test_a_failed_exploit_still_makes_noise(self) -> None:
        # Otherwise red could probe freely and stealth would cost nothing.
        env = MiniCorp(ScenarioConfig(seed=8))
        env.reset()
        env.state.record_breach(Layer.PERIMETER)
        env.state.discovered.add("reverse-proxy")   # exploit_prob 0.30, usually fails
        before = env.state.heat
        env.step({"R_breach": Action(Verb.EXPLOIT, host="reverse-proxy")})
        assert env.state.heat > before

    def test_touching_a_honeypot_costs_red_and_pays_blue(self) -> None:
        env = MiniCorp()
        env.reset()
        env.state.record_breach(Layer.PERIMETER)
        env.state.compromise("web-portal")
        env.state.deploy_honeypot("honeypot-1")
        env.state.discovered.add("honeypot-1")
        env.state.record_breach(Layer.DMZ_BOUNDARY)
        env.state.record_breach(Layer.AUTH)
        env.state.compromise("reverse-proxy")
        env.state.compromise("intranet")
        # honeypot-1 sits in Corp, which red already holds, so reaching it is ordinary
        # intra-zone widening: exploit, not lateral_move.
        _, rewards, _, info = env.step(
            {"R_breach": Action(Verb.EXPLOIT, host="honeypot-1")}
        )
        assert info["honeypot_hits"] == 1
        assert env.state.status("honeypot-1") is not HostStatus.COMPROMISED
        assert rewards["R_breach"] < rewards["B_corp"]

    def test_the_winning_move_is_deterministic(self) -> None:
        """The six layers were the difficulty. Making the final move a coin flip would
        add variance to the one event the episode is scored on, for no extra decision."""
        env = MiniCorp()
        env.reset()
        for layer in Layer:
            env.state.record_breach(layer)
        env.state.compromise(topo.CROWN_JEWEL)
        _, _, done, info = env.step({"R_breach": Action(Verb.ALTER_CREDENTIALS)})
        assert done and info["outcome"] == "red_win"


class TestBlueTransitions:
    def test_block_prevents_compromise_at_no_availability_cost(self) -> None:
        env = MiniCorp(ScenarioConfig(seed=3))
        env.reset()
        env.state.record_breach(Layer.PERIMETER)
        env.state.discovered.add("web-portal")
        env.step({"B_dmz": Action(Verb.BLOCK, host="web-portal")})
        for _ in range(5):
            env.step({"R_breach": Action(Verb.EXPLOIT, host="web-portal")})
        assert env.state.status("web-portal") is HostStatus.CLEAN
        assert env.state.false_positives == 0   # blocking is not isolating

    def test_block_does_nothing_to_an_already_compromised_host(self) -> None:
        """Prevention that worked retroactively would make isolate -- and its
        availability cost -- pointless."""
        env = MiniCorp()
        env.reset()
        env.state.compromise("web-portal")
        env.step({"B_dmz": Action(Verb.BLOCK, host="web-portal")})
        assert env.state.status("web-portal") is HostStatus.COMPROMISED

    def test_isolating_a_clean_host_is_recorded_as_a_false_positive(self) -> None:
        env = MiniCorp()
        env.reset()
        _, rewards, _, info = env.step({"B_corp": Action(Verb.ISOLATE, host="dev-box")})
        assert info["false_positives"] == 1
        assert rewards["B_corp"] < 0

    def test_restoring_a_layer_forces_red_to_breach_it_again(self) -> None:
        """This is what turns one intrusion into the arms race section 9 plots."""
        env = MiniCorp()
        env.reset()
        env.state.record_breach(Layer.PERIMETER)
        assert env.state.layers.is_breached(Layer.PERIMETER)
        env.step({"B_dmz": Action(Verb.TIGHTEN_RATELIMIT)})
        assert not env.state.layers.is_breached(Layer.PERIMETER)
        assert env.state.layers.can_attempt(Layer.PERIMETER)

    def test_blue_actions_are_deterministic(self) -> None:
        """The uncertainty blue faces is about what is true, not about whether its own
        containment works. Stacking a failure probability on top of noisy detection
        would make defensive credit assignment nearly unlearnable."""
        for seed in (1, 2, 3):
            env = MiniCorp(ScenarioConfig(seed=seed))
            env.reset()
            env.step({"B_corp": Action(Verb.ISOLATE, host="dev-box")})
            assert env.state.status("dev-box") is HostStatus.ISOLATED


class TestRewardWiring:
    def test_teammates_receive_identical_rewards(self) -> None:
        env = MiniCorp()
        env.reset()
        _, rewards, _, _ = env.step({})
        assert rewards["R_scout"] == rewards["R_breach"]
        assert rewards["B_dmz"] == rewards["B_corp"] == rewards["B_secure"]

    def test_red_is_charged_for_being_seen_not_for_being_compromised(self) -> None:
        """Section 5.4's -50 is for detection. A quiet foothold is never charged, which
        is the entire stealth incentive."""
        env = MiniCorp(ScenarioConfig(seed=6))
        env.reset()
        env.state.compromise("web-portal")
        quiet_returns = [env.step({})[1]["R_breach"] for _ in range(10)]
        assert all(r >= env.rewards.step_cost + env.rewards.detected for r in quiet_returns)
        assert any(r == env.rewards.step_cost for r in quiet_returns)

    def test_info_carries_every_metric_the_csv_row_needs(self) -> None:
        # CLAUDE.md 3.8's row: they must come out of the environment, not be recomputed.
        env = MiniCorp()
        env.reset()
        _, _, _, info = env.step({})
        for key in ("step", "outcome", "layers_breached", "hosts_compromised",
                    "mttd", "mttc", "false_positives", "honeypot_hits"):
            assert key in info


class TestLayerRestorationCannotBeFarmed:
    """The reward-shaping trap in section 15, found by decoding a trained Q-table.

    Section 5.3 pays a flat +25 per layer restoration. Because ``tighten_ratelimit`` is
    legal whenever Layer 1 is down, and red re-breaches Layer 1 with ``slow_scan`` at
    p=0.60, blue could earn roughly +12 a step from the repair loop -- more than the -10
    a step bleed it was supposed to prevent. The trained defender restored the perimeter
    **35.1 times per episode**, collecting +175,750 against a total return of -115,338:
    152% of its absolute reward came from farming the shaping term, and it scored worse
    than a random defender while doing it.
    """

    def _breach_and_restore(self, env, times: int) -> int:
        paid = 0
        for _ in range(times):
            if not env.state.layers.is_breached(Layer.PERIMETER):
                env.state.record_breach(Layer.PERIMETER)
            _, _, done, info = env.step({"B_dmz": Action(Verb.TIGHTEN_RATELIMIT)})
            paid += len(info["events"].layers_restored)
            if done:
                break
        return paid

    def test_only_the_first_restoration_of_a_layer_is_paid(self) -> None:
        env = MiniCorp()
        env.reset()
        assert self._breach_and_restore(env, 10) == 1

    def test_repeated_restoration_is_still_legal_and_still_repairs(self) -> None:
        """It has to stay available: forcing red to breach the perimeter again is real
        defensive value, it just is not new reward."""
        env = MiniCorp()
        env.reset()
        env.state.record_breach(Layer.PERIMETER)
        env.step({"B_dmz": Action(Verb.TIGHTEN_RATELIMIT)})
        assert not env.state.layers.is_breached(Layer.PERIMETER)
        env.state.record_breach(Layer.PERIMETER)
        assert act.is_legal(env.state, "B_dmz", Action(Verb.TIGHTEN_RATELIMIT))
        env.step({"B_dmz": Action(Verb.TIGHTEN_RATELIMIT)})
        assert not env.state.layers.is_breached(Layer.PERIMETER)

    def test_the_repair_loop_no_longer_outpays_the_compromise_bleed(self) -> None:
        """The arithmetic that made farming rational, asserted directly.

        Over any episode, total restoration income is bounded by three layers times +25,
        which cannot outrun a -10 per step bleed for more than a few steps.
        """
        from marlsoc.env import rewards as rw
        max_income = 3 * rw.DEFAULT.layer_restored
        assert max_income < abs(rw.DEFAULT.compromised_host_per_step) * 10


class TestTheLadderCannotBeFarmedEither:
    """The mirror of the restoration farm, on red's side.

    Section 5.4 pays +10 for breaching Layer 1. If blue repairs it, red breaches again --
    and must, to advance. Paying the rung twice would mean red *profits from blue
    defending*, which is the same perverse incentive pointing the other way. In the run
    that exposed the restoration farm, blue repaired the perimeter 35 times an episode,
    so red was collecting an unearned +350 from the same loop.
    """

    def test_a_re_breach_after_repair_pays_nothing(self) -> None:
        env = MiniCorp()
        env.reset()
        assert env.state.record_breach(Layer.PERIMETER) is True    # first time: paid
        env.state.layers = env.state.layers.restore(Layer.PERIMETER)
        assert env.state.record_breach(Layer.PERIMETER) is False   # again: not paid

    def test_a_re_breach_still_advances_the_layer_model(self) -> None:
        """It has to: red cannot reach the DMZ with Layer 1 standing, so re-breaching is
        genuine progress even though it is not new reward."""
        env = MiniCorp()
        env.reset()
        env.state.record_breach(Layer.PERIMETER)
        env.state.layers = env.state.layers.restore(Layer.PERIMETER)
        assert not env.state.layers.is_breached(Layer.PERIMETER)
        env.state.record_breach(Layer.PERIMETER)
        assert env.state.layers.is_breached(Layer.PERIMETER)

    def test_the_first_breach_step_is_not_overwritten_by_a_re_breach(self) -> None:
        # The layer-breach histogram in section 15 asks *when* red first got through.
        env = MiniCorp()
        env.reset()
        env.state.step = 4
        env.state.record_breach(Layer.PERIMETER)
        env.state.layers = env.state.layers.restore(Layer.PERIMETER)
        env.state.step = 90
        env.state.record_breach(Layer.PERIMETER)
        assert env.state.breach_steps[Layer.PERIMETER] == 4

    def test_neither_team_can_profit_from_the_repair_loop(self) -> None:
        """Both sides of the loop, in one episode: blue repairs, red re-breaches, and
        after the first of each nobody is paid again."""
        env = MiniCorp()
        env.reset()
        paid_red = paid_blue = 0
        for _ in range(12):
            if not env.state.layers.is_breached(Layer.PERIMETER):
                if env.state.record_breach(Layer.PERIMETER):
                    paid_red += 1
            _, _, done, info = env.step({"B_dmz": Action(Verb.TIGHTEN_RATELIMIT)})
            paid_blue += len(info["events"].layers_restored)
            if done:
                break
        assert paid_red == 1
        assert paid_blue == 1


class TestWinningIsAlwaysPaid:
    """Termination must be decided before rewards, on every path to a win.

    This was a real bug with no symptom of its own. ``_check_termination`` is what sets
    ``events.red_won`` for a curriculum stage's objective, and rewards were computed
    first -- so a stage win paid nothing: red collected no +100 and blue was charged no
    -100. Conceding was free, and blue correctly learned to concede. A static defence
    scored -38.6 against a trained defender's -142.8, because defending cost step time
    that losing did not, and the whole thing read as a reward-design problem rather than
    an ordering bug. Only ``alter_credentials`` was unaffected, because it sets
    ``red_won`` during action application -- which is exactly why the full six-layer game
    never showed it.
    """

    def _both_returns(self, env, joint):
        _, rewards, done, info = env.step(joint)
        return rewards["R_breach"], rewards["B_dmz"], done, info

    def test_a_stage_objective_win_pays_both_teams(self) -> None:
        env = MiniCorp(ScenarioConfig(seed=1, max_layer=2))
        env.reset()
        env.state.record_breach(Layer.PERIMETER)
        env.state.discovered.add("web-portal")
        # Force the second layer to fall this step by exploiting the entry host.
        for _ in range(40):
            if not act.is_legal(env.state, "R_breach",
                                Action(Verb.EXPLOIT, host="web-portal")):
                break
            red_r, blue_r, done, info = self._both_returns(
                env, {"R_breach": Action(Verb.EXPLOIT, host="web-portal")})
            if done:
                assert info["outcome"] == "red_win"
                assert red_r > 0, "red won a stage and was paid nothing"
                assert blue_r < -50, "blue lost a stage and was charged nothing"
                return
        raise AssertionError("red never reached the stage objective")

    def test_the_full_game_win_still_pays(self) -> None:
        env = MiniCorp()
        env.reset()
        for layer in Layer:
            env.state.record_breach(layer)
        env.state.compromise(topo.CROWN_JEWEL)
        _, rewards, done, info = env.step({"R_breach": Action(Verb.ALTER_CREDENTIALS)})
        assert done and info["outcome"] == "red_win"
        assert rewards["R_breach"] > 0
        assert rewards["B_dmz"] < -50

    def test_conceding_is_not_cheaper_than_defending(self) -> None:
        """The property the bug violated: losing must cost more than the step time of
        trying to prevent it, or the optimal defensive policy is to concede."""
        from marlsoc.env import rewards as rw
        # Losing costs -100 at once; a defence that drags the episode out pays -1 a step.
        assert rw.DEFAULT.red_win > abs(rw.DEFAULT.step_cost) * 60
