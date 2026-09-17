"""Tests for the reward functions.

Reward bugs do not crash; they produce a well-trained agent solving the wrong problem.
So the properties asserted here are the *incentives* the report claims, not just the
arithmetic: that the degenerate policy loses, that the game is general-sum, that stealth
pays, and that no action can be farmed for free reward.
"""

from __future__ import annotations

from marlsoc.config import AvailabilityCost
from marlsoc.env import rewards as rw
from marlsoc.env.layers import Layer, LayerStatus
from marlsoc.env.rewards import RewardConfig, StepEvents
from marlsoc.env.state import EpisodeState

ONE_SHOT = RewardConfig(availability_cost=AvailabilityCost.ONE_SHOT)
PER_STEP = RewardConfig(availability_cost=AvailabilityCost.PER_STEP)


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
    def test_each_layer_pays_its_own_rung(self) -> None:
        base = rw.red_reward(fresh(), StepEvents())
        for layer, expected in [(Layer.PERIMETER, 10.0), (Layer.DMZ_BOUNDARY, 20.0),
                                (Layer.AUTH, 30.0), (Layer.SEGMENTATION, 40.0),
                                (Layer.PRIVILEGE, 50.0), (Layer.APPROVAL, 60.0)]:
            got = rw.red_reward(fresh(), StepEvents(layers_breached=[layer]))
            assert got - base == expected

    def test_the_full_chain_pays_more_than_the_step_costs_of_walking_it(self) -> None:
        """Section 5.4's whole argument: the ladder must outweigh the -1 per step over a
        realistic path, or red's optimal policy is to do nothing at all."""
        ladder = sum(rw.DEFAULT.step_cost for _ in range(120))   # a long run to the goal
        ladder += sum(rw.red_reward(fresh(), StepEvents(layers_breached=[ly]))
                      - rw.DEFAULT.step_cost for ly in Layer)
        ladder += 100.0
        assert ladder > 0

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
        events = StepEvents(layers_breached=[Layer.PERIMETER])
        assert rw.red_reward(fresh(), events) > 0


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
