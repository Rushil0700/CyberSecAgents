"""Alternating training and the IQL control -- PROJECT.md section 6, CLAUDE.md 3.4.

Why this module exists
----------------------
With all five agents learning at once, each agent's environment *contains the others*,
whose policies keep changing. The environment is therefore **non-stationary**, and
Q-Learning's convergence proof assumes a stationary MDP -- so Independent Q-Learning has
**no convergence guarantee** in a multi-agent setting. This is not a detail: it is the
single piece of multi-agent theory PROJECT.md section 6 says to be able to state out loud.

Alternating training is the answer. Freeze one team, train the other to convergence, swap,
repeat. During each phase exactly one team's policy is changing, so the learner faces a
**fixed** opponent and its phase is a genuine stationary MDP where the proof applies
again.

``train_iql`` is kept beside it deliberately. Section 3.4 calls for one simultaneous
configuration as a *control*, because its failure to settle is the empirical
demonstration that IQL is unsound here -- a measured result rather than a cited one. It
is not a worse implementation of the same thing; it is the thing the main method avoids.

Two design decisions that are not in the spec
---------------------------------------------
**Exploration is rewound at every swap.** An agent arriving from the previous round is
nearly greedy, and the opponent it is about to face has just changed underneath it. A
nearly greedy agent cannot discover the response, because every action it would need to
try is ranked below something that used to work -- the same argument that makes
``epsilon_on_promote`` necessary for the curriculum (see ``curriculum.py``). Without it
the arms race stalls after the first round: each team keeps replaying the counter it
already had.

**Q-tables carry across rounds.** That carry-forward is what makes this an arms race
rather than a sequence of unrelated runs. Round 3's blue inherits everything rounds 1 and
2 taught it, so what the plot shows is two policies co-adapting rather than two fresh
agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np

from marlsoc.agents.tabular import LearnerConfig
from marlsoc.config import (
    ALL_AGENTS,
    BLUE_AGENTS,
    RED_AGENTS,
    AgentConfig,
    Policy,
    ScenarioConfig,
)
from marlsoc.training.loop import (
    Controller,
    LearnedController,
    build_controller,
    evaluate,
    run_episode,
)
from marlsoc.env.minicorp import MiniCorp
from marlsoc.training.metrics import EpisodeRecord, MetricsLog

TEAMS: dict[str, tuple[str, ...]] = {"red": RED_AGENTS, "blue": BLUE_AGENTS}


@dataclass(frozen=True)
class AlternatingConfig:
    """How the arms race is run.

    Attributes:
        rounds: Full red-and-blue cycles. Each round trains both teams once, so the
            number of training phases is ``2 * rounds``.
        episodes_per_phase: Episodes one team trains for while the other is frozen.
        first: Which team trains first. Blue by default, because red starts from a
            curriculum warm-up and blue starts from nothing -- letting blue move first
            means red's first opponent is a defender that has at least seen the game.
        epsilon_on_swap: Exploration to rewind to when a team starts a training phase.
            None disables the rewind; see the module docstring for why that stalls.
        eval_episodes: Episodes in the greedy evaluation taken after every phase. This
            is what the arms-race plot is drawn from, and it is greedy for the reason
            CLAUDE.md 3.23 gives -- a training win rate is the exploring policy's.
    """

    rounds: int = 5
    episodes_per_phase: int = 2_000
    first: str = "blue"
    epsilon_on_swap: float | None = 0.40
    eval_episodes: int = 200


@dataclass(frozen=True)
class PhaseResult:
    """One training phase, measured greedily after it finished."""

    round: int
    trained: str
    attacker_success: float
    red_return: float
    blue_return: float
    layers_breached: float
    false_positives: float

    @property
    def label(self) -> str:
        return f"r{self.round}-{self.trained}"


class AlternatingRun(NamedTuple):
    controllers: dict[str, Controller]
    log: MetricsLog
    phases: list[PhaseResult]


def _team_scenario(scenario: ScenarioConfig, learning_team: str) -> ScenarioConfig:
    """``scenario`` with exactly one team's ``learning`` switch on.

    Rebuilt rather than mutated because ``ScenarioConfig`` is frozen -- which is the point
    of it being frozen. A run's configuration is written beside its curves, and a config
    that changed underneath the run would make that file a lie.
    """
    agents = dict(scenario.agents)
    for team, members in TEAMS.items():
        for agent in members:
            current = scenario.for_agent(agent)
            if not current.enabled:
                # "Disabled but learning" is incoherent -- a NoopController has no learner
                # to update. Harmless in effect, but a scenario is written beside its
                # curves as the record of what ran, so it must not describe an agent as
                # learning when nothing could have updated it.
                agents[agent] = replace(current, learning=False)
                continue
            agents[agent] = replace(
                current,
                learning=(team == learning_team),
                policy=Policy.LEARNED,
            )
    return replace(scenario, agents=agents)


def _set_greedy(controllers: dict[str, Controller], team: str, greedy: bool) -> None:
    """Freeze or unfreeze a team's action selection.

    A frozen opponent must act **greedily**: it is standing in for a deployed policy, and
    an opponent that still explores 5% of the time is a different, easier opponent than
    the one the learner will actually be scored against.
    """
    for agent in TEAMS[team]:
        controller = controllers.get(agent)
        if isinstance(controller, LearnedController):
            controller.greedy = greedy


def _measure(
    scenario: ScenarioConfig,
    controllers: dict[str, Controller],
    episodes: int,
    round_no: int,
    trained: str,
) -> PhaseResult:
    log = evaluate(scenario, controllers, episodes, phase=f"r{round_no}-{trained}")
    return PhaseResult(
        round=round_no,
        trained=trained,
        attacker_success=log.rate("red_win"),
        red_return=log.mean("red_return"),
        blue_return=log.mean("blue_return"),
        layers_breached=log.mean("layers_breached"),
        false_positives=log.mean("false_positives"),
    )


def train_alternating(
    scenario: ScenarioConfig,
    config: AlternatingConfig | None = None,
    learner_config: LearnerConfig | None = None,
    *,
    controllers: dict[str, Controller] | None = None,
    verbose: bool = True,
) -> AlternatingRun:
    """Freeze one team, train the other, swap, repeat. CLAUDE.md 3.4's main method.

    Args:
        scenario: Agent switches and seed. Per-agent ``learning`` flags are overridden
            each phase; everything else is respected.
        config: Rounds, phase length and the swap rewind.
        learner_config: Hyperparameters for every learning agent.
        controllers: Pre-built controllers, so a warm-started team (a curriculum-trained
            red, say) can be carried in rather than relearned.
        verbose: Print a line per phase.

    Returns:
        The controllers, the per-episode log, and one ``PhaseResult`` per training phase.
    """
    cfg = config or AlternatingConfig()
    rng = np.random.default_rng(scenario.seed)
    controllers = controllers or {
        agent: build_controller(agent, scenario.for_agent(agent), rng, learner_config)
        for agent in ALL_AGENTS
    }

    order = (cfg.first, "red" if cfg.first == "blue" else "blue")
    log = MetricsLog()
    phases: list[PhaseResult] = []

    if verbose:
        print(f"Alternating: {cfg.rounds} rounds x {cfg.episodes_per_phase} episodes "
              f"per phase, {order[0]} first")

    for round_no in range(1, cfg.rounds + 1):
        for team in order:
            opponent = "red" if team == "blue" else "blue"
            phase_scenario = _team_scenario(scenario, team)
            # One environment per phase. Built here rather than cached by scenario
            # identity: ScenarioConfig holds a dict so it is unhashable, and keying a
            # cache on id() is unsound because ids are reused after collection -- a
            # stale entry would silently run a phase under the wrong reward structure.
            env = MiniCorp(phase_scenario)

            # The learner explores; the opponent is a deployed policy and acts greedily.
            _set_greedy(controllers, team, False)
            _set_greedy(controllers, opponent, True)

            if cfg.epsilon_on_swap is not None:
                for agent in TEAMS[team]:
                    controller = controllers.get(agent)
                    if controller is not None and controller.learner is not None:
                        controller.learner.boost_exploration(cfg.epsilon_on_swap)

            for episode in range(cfg.episodes_per_phase):
                record, _ = run_episode(
                    env=env,
                    controllers=controllers,
                    scenario=phase_scenario,
                    seed=scenario.seed * 7_919 + round_no * 100_003 + episode,
                )
                log.append(EpisodeRecord(**{
                    **record.__dict__,
                    "episode": len(log),
                    "phase": f"r{round_no}-{team}",
                    "learner": team,
                }))

            result = _measure(phase_scenario, controllers, cfg.eval_episodes,
                              round_no, team)
            phases.append(result)
            if verbose:
                print(f"  round {round_no}  trained {team:>4}  "
                      f"attacker {result.attacker_success:6.1%}  "
                      f"blue_R {result.blue_return:8.1f}  "
                      f"depth {result.layers_breached:4.2f}")

    return AlternatingRun(controllers, log, phases)


def train_iql(
    scenario: ScenarioConfig,
    episodes: int,
    learner_config: LearnerConfig | None = None,
    *,
    controllers: dict[str, Controller] | None = None,
    measure_every: int = 1_000,
    eval_episodes: int = 200,
    verbose: bool = True,
) -> AlternatingRun:
    """The control: every agent learning at once. CLAUDE.md 3.4.

    Deliberately the unsound configuration. Each agent treats the others as part of its
    environment, and that environment keeps changing, so no agent is solving a stationary
    MDP and none of them has a convergence guarantee. What this run is *for* is measuring
    that -- whether the arms race settles or keeps oscillating -- so it is evaluated on
    the same schedule as ``train_alternating`` and its phases carry the same shape.

    ``PhaseResult.trained`` reads ``"both"`` here, which is exactly the difference being
    tested.
    """
    rng = np.random.default_rng(scenario.seed)
    every_agent_learns = replace(scenario, agents={
        agent: replace(scenario.for_agent(agent), learning=True, policy=Policy.LEARNED)
        for agent in ALL_AGENTS
        if scenario.for_agent(agent).enabled
    })
    controllers = controllers or {
        agent: build_controller(agent, every_agent_learns.for_agent(agent), rng,
                                learner_config)
        for agent in ALL_AGENTS
    }

    log = MetricsLog()
    phases: list[PhaseResult] = []
    env = MiniCorp(every_agent_learns)

    if verbose:
        print(f"IQL control: {episodes} episodes, all agents learning simultaneously")

    for episode in range(episodes):
        record, _ = run_episode(
            env=env, controllers=controllers, scenario=every_agent_learns,
            seed=scenario.seed * 31 + episode,
        )
        log.append(EpisodeRecord(**{
            **record.__dict__, "episode": episode, "phase": "iql", "learner": "both",
        }))

        if measure_every and (episode + 1) % measure_every == 0:
            checkpoint = (episode + 1) // measure_every
            result = _measure(every_agent_learns, controllers, eval_episodes,
                              checkpoint, "both")
            phases.append(result)
            if verbose:
                print(f"  ep {episode + 1:>6}  attacker {result.attacker_success:6.1%}  "
                      f"blue_R {result.blue_return:8.1f}  "
                      f"depth {result.layers_breached:4.2f}")

    return AlternatingRun(controllers, log, phases)


def instability(phases: list[PhaseResult]) -> float:
    """Mean absolute change in attacker success between consecutive measurements.

    The number that makes "IQL does not settle" a measurement rather than an impression.
    A converging run drives this towards zero as the policies stop moving; a
    non-stationary one does not, because every improvement by one side invalidates what
    the other had learned.

    Compared like for like, it is only meaningful against a run measured on the same
    schedule -- which is why ``train_iql`` evaluates the same way ``train_alternating``
    does.
    """
    if len(phases) < 2:
        return 0.0
    deltas = [abs(b.attacker_success - a.attacker_success)
              for a, b in zip(phases, phases[1:])]
    return float(np.mean(deltas))
