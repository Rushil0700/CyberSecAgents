"""Run configuration: the switches an experiment turns, in one serialisable place.

Everything here is a frozen dataclass so a run's exact configuration can be written into
``artifacts/runs/<id>/`` beside its curves. Six weeks later, when two curves disagree,
the question "what was different?" has a file that answers it rather than a memory that
does not.

CLAUDE.md amendment 3.7 is the design here: an agent has **three independent switches**,
because "turn this agent off" means three different things and collapsing them would
force a rewrite of the training loop for every experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final


class AvailabilityCost(str, Enum):
    """How the cost of isolating a host is charged. CLAUDE.md amendment 3.2.

    PROJECT.md section 5.3 charges a false positive **once**, at -20. That does not
    prevent the degenerate "isolate everything" policy, because the *benefit* of
    over-isolating is per-step while the penalty is one-shot. Work it out for the twelve
    defended hosts: isolating all of them costs 12 * -20 = -240 once, and then avoids the
    -10 per step per compromised host forever. One compromised host over T=250 steps is
    -2500. The degenerate policy wins by an order of magnitude, blue converges on perfect
    security with zero availability, and the tradeoff the project is about disappears.

    PER_STEP charges an additional per-step cost for every isolated host, so the same
    twelve-host lockdown costs 12 * 250 * -2 = -6000 and loses badly.

    ONE_SHOT is kept deliberately: reproducing the degenerate policy on demand is a
    before/after result worth showing, not a bug to be hidden.
    """

    ONE_SHOT = "one_shot"
    PER_STEP = "per_step"


class RewardShaping(str, Enum):
    """How red's progress along the section 5.4 ladder is credited. CLAUDE.md 3.18.

    The ladder (+10 ... +60, one rung per layer) exists to solve the sparse-reward
    problem section 7.4 demonstrates: red only ever sees +100 at the crown jewel, and
    Phase 1 measured that random exploration reaches mean depth 0.78 of 6 and wins 0% of
    the time. Dense intermediate signal is not optional.

    RAW_LADDER pays each rung as a flat bonus that red **keeps**. That is not
    potential-based shaping, and Ng, Harada & Russell (1999) prove that only
    potential-based terms leave the optimal policy invariant -- any other shaping term
    can change *which* policy is optimal rather than merely how fast it is found. The
    risk here is concrete: a rung red keeps is a reward it can collect and then lose on
    purpose, since nothing claws it back at termination.

    POTENTIAL_BASED credits ``gamma * Phi(s') - Phi(s)`` instead, where ``Phi`` is the
    cumulative ladder value of the layers red has breached and ``Phi(terminal) = 0``. The
    discounted sum over any trajectory telescopes to ``gamma^T Phi(s_T) - Phi(s_0)``,
    which is zero at both ends -- so the shaping shapes the *value function* and steers
    exploration without being a destination red can settle for.

    **RAW_LADDER is the default, and that is a measured decision, not an oversight.**
    Potential-based shaping contributes *exactly zero* to the discounted return -- that is
    the theorem, not a side effect -- so red's only remaining incentive is the terminal
    +100. Discounted over the ~29 steps to the crown jewel at gamma = 0.95 that is worth
    about +22.6 against step costs of about -15.5, and a single -50 detection (which the
    alert process charges **even against a static defence**, since detection is
    environmental rather than defender-driven) turns winning into roughly -23 against
    about -20 for idling. Idling wins. Measured over three seeds against a static defence,
    4,000 episodes:

        raw_ladder        attacker 66.7%   mean depth 5.00 of 6
        potential_based   attacker  0.0%   mean depth 0.00 of 6

    The ladder was never there to preserve optimality; it was there to make a problem
    Phase 1 measured as unlearnable (0% wins at depth 0.78 of 6) learnable. Guaranteeing
    that shaping changes nothing guarantees it does not help.

    POTENTIAL_BASED is kept because the comparison above is a genuine result: it is the
    limit of a famous theorem, demonstrated rather than cited. Use it with a discount
    factor and terminal reward large enough that the true objective dominates.
    """

    RAW_LADDER = "raw_ladder"
    POTENTIAL_BASED = "potential_based"


class Policy(str, Enum):
    """How an agent chooses actions.

    LEARNED reads a Q-table. SCRIPTED follows a fixed heuristic (the Phase 2 attacker,
    and the "greedy heuristic" baseline in section 9). RANDOM samples uniformly from the
    legal set -- the floor every result is measured against. GREEDY is LEARNED with
    epsilon forced to zero, for evaluation and deployment. NOOP always waits.
    """

    LEARNED = "learned"
    SCRIPTED = "scripted"
    RANDOM = "random"
    GREEDY = "greedy"
    NOOP = "noop"


@dataclass(frozen=True)
class AgentConfig:
    """The three orthogonal switches from CLAUDE.md amendment 3.7.

    Attributes:
        enabled: Does this agent act at all? Disabling every blue agent *is* the static
            firewall baseline in section 9 -- no separate baseline code needed.
        learning: Does it update its Q-table? False freezes it, which is how alternating
            training (section 6) keeps each phase a stationary MDP.
        policy: How it picks actions.

    The three are independent on purpose. An agent can be enabled but frozen (deployed),
    enabled and learning (training), or disabled entirely (ablation). Folding these into
    one flag would mean a different training loop for each experiment.
    """

    enabled: bool = True
    learning: bool = True
    policy: Policy = Policy.LEARNED


@dataclass(frozen=True)
class ScenarioConfig:
    """A complete, reproducible experiment: per-agent switches plus a seed.

    Episodes must be **exactly reproducible from a seed** (CLAUDE.md 3.7) so a chosen
    episode can be replayed for a demo. Selecting a representative seeded episode to show
    is legitimate; changing exploit probabilities or detection rates to manufacture an
    outcome and reporting it as a result is not. A demo pairs two runs on the *same seed*
    with one variable changed, and any single episode shown is accompanied by the
    aggregate over many runs.
    """

    seed: int = 0
    agents: dict[str, AgentConfig] = field(default_factory=dict)
    availability_cost: AvailabilityCost = AvailabilityCost.PER_STEP
    shaping: RewardShaping = RewardShaping.RAW_LADDER
    max_layer: int = 6
    step_limit: int = 250

    def for_agent(self, name: str) -> AgentConfig:
        """This agent's switches, defaulting to enabled-and-learning."""
        return self.agents.get(name, AgentConfig())


ALL_AGENTS: Final[tuple[str, ...]] = (
    "R_scout", "R_breach", "B_dmz", "B_corp", "B_secure",
)

RED_AGENTS: Final[tuple[str, ...]] = ("R_scout", "R_breach")
BLUE_AGENTS: Final[tuple[str, ...]] = ("B_dmz", "B_corp", "B_secure")


def static_firewall() -> ScenarioConfig:
    """Section 9 baseline 2: the layered firewall with no defenders acting.

    "What a normal hardened network does", and the headline comparison. It is not
    separate code -- it is every blue agent disabled, which is exactly what amendment
    3.7 predicted would fall out of the three-switch design.
    """
    return ScenarioConfig(
        agents={a: AgentConfig(enabled=False, learning=False, policy=Policy.NOOP)
                for a in BLUE_AGENTS}
    )


def random_defender() -> ScenarioConfig:
    """Section 9 baseline 1: the floor."""
    return ScenarioConfig(
        agents={a: AgentConfig(learning=False, policy=Policy.RANDOM)
                for a in BLUE_AGENTS}
    )
