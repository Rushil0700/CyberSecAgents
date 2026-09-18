"""Tests for the reward functions.

Reward bugs do not crash; they produce a well-trained agent solving the wrong problem.
So the properties asserted here are the *incentives* the report claims, not just the
arithmetic: that the degenerate policy loses, that the game is general-sum, that stealth
pays, and that no action can be farmed for free reward.
"""

from __future__ import annotations

import pytest

from marlsoc.config import AvailabilityCost, RewardShaping
from marlsoc.env import layers as lyr
from marlsoc.env import rewards as rw
from marlsoc.env.layers import Layer, LayerStatus
from marlsoc.env.rewards import RewardConfig, StepEvents
from marlsoc.env.state import EpisodeState

ONE_SHOT = RewardConfig(availability_cost=AvailabilityCost.ONE_SHOT)
PER_STEP = RewardConfig(availability_cost=AvailabilityCost.PER_STEP)
RAW = RewardConfig(shaping=RewardShaping.RAW_LADDER)


def fresh() -> EpisodeState:
    return EpisodeState.initial(LayerStatus.full())


class TestTheDegeneratePolicyTrap:
    """CLAUDE.md amendment 3.2 -- the project's best analysis paragraph, made executable."""

    def test_one_shot_makes_locking_everything_down_optimal(self) -> None:
        """Which is why section 5.3 as written does not prevent it.

        The benefit of over-isolating is per-step; the penalty is one-shot. Twelve hosts
        cost -240 once and then nothing can ever be compromised. Passivity over the same
        horizon costs -10100. Blue is not malfunctioning when it learns this -- it is
        correct, and the reward function is wrong.
        """
        lockdown = rw.degenerate_policy_cost(12, 250, ONE_SHOT)
        passive = -10.0 * 5 * 200 - 100.0
        assert lockdown == -240.0
        assert lockdown > passive * 10   # optimal by more than an order of magnitude

    def test_per_step_makes_targeted_defence_the_best_policy(self) -> None:
        """The ordering blue must face: targeted >> lockdown >> passive.

        Targeted here is "isolate the two genuinely compromised hosts at step 20 and
        carry their availability cost for the rest of the episode".
        """
        lockdown = rw.degenerate_policy_cost(12, 250, PER_STEP)
        passive = -10.0 * 5 * 200 - 100.0
        targeted = (
            2 * PER_STEP.correct_isolation                      # +100 for the catches
            + 2 * PER_STEP.isolated_host_per_step * 230         # availability while down
            + 2 * PER_STEP.compromised_host_per_step * 20       # bleed before containing
        )
        assert targeted > lockdown > passive
        assert lockdown == -6240.0

    def test_containment_is_worth_taking_but_never_free(self) -> None:
        """A correctly isolated host still costs -2 a step. Net against leaving it
        compromised is +8 a step, so blue should contain -- but it cannot treat isolate
        as a costless filler action."""
        assert PER_STEP.isolated_host_per_step < 0
        net = PER_STEP.compromised_host_per_step - PER_STEP.isolated_host_per_step
        assert net == -8.0   # i.e. containing is 8 better per step than not

    def test_a_false_positive_costs_more_than_a_single_step_of_bleed(self) -> None:
        # Otherwise guessing is cheaper than observing and blue learns to shoot first.
        assert abs(PER_STEP.false_positive) > abs(PER_STEP.compromised_host_per_step)


class TestGeneralSum:
    """Section 6: payoffs do not cancel. This is the content of 'general-sum'."""

    def test_a_honeypot_engagement_is_zero_sum(self) -> None:
        events = StepEvents(honeypot_hits=1)
        blue = rw.blue_reward(fresh(), events) - rw.blue_reward(fresh(), StepEvents())
        red = rw.red_reward(fresh(), events) - rw.red_reward(fresh(), StepEvents())
        assert blue == 30.0 and red == -30.0
        assert blue + red == 0.0

    def test_a_false_positive_is_not_zero_sum(self) -> None:
        """Blue pays; red gains nothing. If every event cancelled, the game would be
        zero-sum and section 6's formulation would be wrong."""
        events = StepEvents(false_positives=["mail"])
        blue = rw.blue_reward(fresh(), events) - rw.blue_reward(fresh(), StepEvents())
        red = rw.red_reward(fresh(), events) - rw.red_reward(fresh(), StepEvents())
        assert blue == -20.0 and red == 0.0
        assert blue + red != 0.0

    def test_the_win_transfers_the_same_magnitude_both_ways(self) -> None:
        events = StepEvents(red_won=True)
        assert rw.blue_reward(fresh(), events) == -100.0 + rw.DEFAULT.step_cost
        assert rw.red_reward(fresh(), events) == 100.0 + rw.DEFAULT.step_cost


class TestRedLadder:
    """Section 5.4's ladder, under both shaping modes (CLAUDE.md 3.18)."""

    def test_each_layer_pays_its_own_rung_under_the_raw_ladder(self) -> None:
        """PROJECT.md 5.4's numbers as written, preserved under RAW_LADDER."""
        base = rw.red_reward(fresh(), StepEvents(), RAW)
        for layer, expected in [(Layer.PERIMETER, 10.0), (Layer.DMZ_BOUNDARY, 20.0),
                                (Layer.AUTH, 30.0), (Layer.SEGMENTATION, 40.0),
                                (Layer.PRIVILEGE, 50.0), (Layer.APPROVAL, 60.0)]:
            got = rw.red_reward(fresh(), StepEvents(layers_breached=[layer]), RAW)
            assert got - base == expected

    def test_a_rung_pays_gamma_times_its_value_under_potential_shaping(self) -> None:
        """F = gamma * Phi(s') - Phi(s). Breaching Layer 1 from nothing moves the
        potential 0 -> 10, so the step is credited 0.95 * 10."""
        state = fresh()
        state.record_breach(Layer.PERIMETER)
        events = StepEvents(layers_breached=[Layer.PERIMETER], potential_before=0.0)
        got = rw.red_reward(state, events) - rw.DEFAULT.step_cost
        assert got == pytest.approx(0.95 * 10.0)

    def test_potential_shaping_sums_to_zero_over_a_trajectory(self) -> None:
        """The Ng, Harada & Russell (1999) property, and the whole reason for the change.

            sum_t gamma^t * [gamma * Phi(s_{t+1}) - Phi(s_t)]  =  gamma^T Phi(s_T) - Phi(s_0)

        Both ends are zero -- nothing is breached at reset, and Phi(terminal) = 0 by
        construction -- so shaping contributes exactly nothing to the discounted return
        and cannot change which policy is optimal. It only moves value around *inside*
        the episode, which is what makes it steer exploration for free.
        """
        gamma = rw.DEFAULT.shaping_gamma
        state = fresh()
        total, phi_before = 0.0, 0.0
        for t, layer in enumerate(Layer):
            state.record_breach(layer)
            terminal = layer is Layer.APPROVAL
            events = StepEvents(layers_breached=[layer], potential_before=phi_before,
                                terminal=terminal)
            shaped = rw.red_reward(state, events) - rw.DEFAULT.step_cost
            total += (gamma ** t) * shaped
            phi_before = 0.0 if terminal else lyr.cumulative_breach_reward(
                state.paid_breaches)
        assert total == pytest.approx(0.0, abs=1e-9)

    def test_blue_repairing_a_layer_cannot_lower_reds_potential(self) -> None:
        """Otherwise repair would hand red a fresh rung to re-climb -- CLAUDE.md 3.13's
        farmable loop with an extra step in it. Phi reads the *paid* breach set."""
        state = fresh()
        state.record_breach(Layer.PERIMETER)
        before = lyr.cumulative_breach_reward(state.paid_breaches)
        state.layers.restore(Layer.PERIMETER)
        assert lyr.cumulative_breach_reward(state.paid_breaches) == before

    def test_winning_must_beat_being_contained(self) -> None:
        """Section 5.4's argument, restated for potential-based shaping.

        Under RAW_LADDER the claim was "the ladder outweighs the step costs of walking
        it". That claim is void once shaping telescopes to zero: the *only* thing left
        pulling red towards the crown jewel is the win itself, net of the time and the
        detections it takes to get there. Measured against a trained defender a winning
        run takes about 35 steps and eats two detections, so the win has to clear that.
        """
        cost_of_a_deep_run = 35 * abs(rw.DEFAULT.step_cost) + 2 * abs(rw.DEFAULT.detected)
        assert rw.DEFAULT.red_win + lyr.ALTER_CREDENTIALS_REWARD > cost_of_a_deep_run

    def test_detection_is_the_cliff(self) -> None:
        """Section 7.2: the attacker is punished for exploring. -50 is larger than every
        ladder rung but the last two, so getting caught while probing genuinely undoes
        progress -- which is what makes SARSA's safer path worth learning."""
        caught = abs(rw.DEFAULT.detected)
        assert caught > 40.0     # bigger than the Layer 4 rung
        assert caught >= max(10.0, 20.0, 30.0, 40.0)

    def test_honeypots_punish_red_more_than_a_wasted_step(self) -> None:
        assert abs(rw.DEFAULT.honeypot_cost) > abs(rw.DEFAULT.step_cost) * 10


class TestPerStepTerms:
    def test_the_bleed_scales_with_hosts_held(self) -> None:
        state = fresh()
        base = rw.blue_reward(state, StepEvents())
        state.compromise("web-portal")
        one = rw.blue_reward(state, StepEvents())
        state.compromise("mail")
        two = rw.blue_reward(state, StepEvents())
        assert one - base == -10.0
        assert two - one == -10.0

    def test_isolation_cost_only_applies_in_per_step_mode(self) -> None:
        state = fresh()
        state.isolate("mail")
        assert rw.blue_reward(state, StepEvents(), ONE_SHOT) == ONE_SHOT.step_cost
        assert rw.blue_reward(state, StepEvents(), PER_STEP) == PER_STEP.step_cost - 2.0

    def test_both_teams_pay_for_time(self) -> None:
        # Makes dithering expensive: a policy that wins in 40 steps beats one taking 200.
        assert rw.blue_reward(fresh(), StepEvents()) == rw.DEFAULT.step_cost
        assert rw.red_reward(fresh(), StepEvents()) == rw.DEFAULT.step_cost


class TestSharedRewards:
    def test_blue_agents_share_one_return(self) -> None:
        """Section 6: nobody is told to cooperate. Coordination is simply how you
        maximise a return you both receive."""
        state = fresh()
        state.compromise("ad-controller")   # a Corp host
        # There is one blue_reward function, not three: the DMZ defender pays for a
        # corporate breach, which is what makes hand-off worth learning.
        assert rw.blue_reward(state, StepEvents()) < rw.DEFAULT.step_cost

    def test_red_agents_share_one_return(self) -> None:
        # So R_scout earns credit for discoveries R_breach converts much later --
        # section 4.1's credit-assignment problem.
        state = fresh()
        state.record_breach(Layer.PERIMETER)
        events = StepEvents(layers_breached=[Layer.PERIMETER], potential_before=0.0)
        # One red_reward function, not two: whichever attacker moved the ladder, both
        # are paid for it.
        assert rw.red_reward(state, events) > rw.DEFAULT.step_cost


class TestConfigBaselines:
    def test_the_static_firewall_baseline_is_just_switches(self) -> None:
        """CLAUDE.md 3.7 predicted this: "all blue agents disabled" *is* section 9's
        static-firewall baseline. No separate baseline implementation exists."""
        from marlsoc.config import BLUE_AGENTS, RED_AGENTS, static_firewall
        cfg = static_firewall()
        assert all(not cfg.for_agent(a).enabled for a in BLUE_AGENTS)
        assert all(cfg.for_agent(a).enabled for a in RED_AGENTS)

    def test_the_three_switches_are_independent(self) -> None:
        from marlsoc.config import AgentConfig, Policy
        frozen = AgentConfig(enabled=True, learning=False, policy=Policy.GREEDY)
        assert frozen.enabled and not frozen.learning
        assert frozen.policy is Policy.GREEDY

    def test_per_step_is_the_default_availability_mode(self) -> None:
        from marlsoc.config import ScenarioConfig
        assert ScenarioConfig().availability_cost is AvailabilityCost.PER_STEP
        assert rw.DEFAULT.availability_cost is AvailabilityCost.PER_STEP


class TestPreventionIsCheapButNotFree:
    """A free, effective preventive control would strictly dominate everything else.

    ``block`` stops a clean host being compromised and, unlike ``isolate``, leaves it on
    the network. If it also cost nothing, blue's best policy would be to blanket-block
    its whole zone every few steps -- perfect prevention at zero price -- and section
    5.3's tradeoff would have nothing left to trade.
    """

    def test_blocking_costs_something(self) -> None:
        assert PER_STEP.blocked_host_per_step < 0

    def test_blocking_is_cheaper_than_isolating(self) -> None:
        # It restricts a host rather than removing it, so it should not cost the same.
        assert PER_STEP.blocked_host_per_step > PER_STEP.isolated_host_per_step

    def test_blanket_prevention_costs_about_as_much_as_one_containment(self) -> None:
        """The calibration that forces a choice: covering all four Edge/DMZ hosts is
        roughly the price of one isolation, so blue must decide *where* to spend
        prevention instead of spreading it everywhere."""
        four_blocked = 4 * PER_STEP.blocked_host_per_step
        one_isolated = PER_STEP.isolated_host_per_step
        assert four_blocked == pytest.approx(one_isolated)

    def test_the_cost_scales_with_how_much_is_blocked(self) -> None:
        state = fresh()
        base = rw.blue_reward(state, StepEvents(), PER_STEP)
        state.block("mail", duration=5)
        one = rw.blue_reward(state, StepEvents(), PER_STEP)
        state.block("web-portal", duration=5)
        two = rw.blue_reward(state, StepEvents(), PER_STEP)
        assert one - base == pytest.approx(-0.5)
        assert two - one == pytest.approx(-0.5)

    def test_an_expired_block_stops_costing(self) -> None:
        state = fresh()
        state.block("mail", duration=3)
        assert state.blocked_count == 1
        state.step = 99
        assert state.blocked_count == 0
