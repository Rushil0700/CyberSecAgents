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
