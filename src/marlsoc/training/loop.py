"""The training loop: where the three agent switches become code.

CLAUDE.md amendment 3.7 gives every agent three independent switches -- ``enabled``,
``learning`` and ``policy``. This module is the one place they are interpreted. Each
agent becomes a **controller** that answers one question, "what action index now?", and
the loop never asks where the answer came from. That is what lets one loop serve
training, evaluation, all three section 9 baselines and the demo without branching.

Design note -- **the loop is written for SARSA, which makes it work for all three.**
SARSA's target needs ``a'``, the action actually taken in ``s'``, so the update for step
``t`` cannot happen until step ``t+1``'s action has been chosen::

    a  <- select(s)
    loop:
        s', r, done <- step(a)
        a'          <- select(s')          # needed before the update
        update(s, a, r, s', a', done)
        s, a <- s', a'

Q-Learning and Expected SARSA ignore ``a'``, so one loop covers all three. Writing the
simpler off-policy loop first and bolting SARSA on later is how the two quietly end up
using different transition sequences and the section 7.2 comparison becomes meaningless.

Design note -- **the training curve and the policy-quality curve are different curves.**
The obvious learning curve plots the return collected *while training*. For this project
that curve is misleading, and measurably so. Eight of ``B_dmz``'s ten actions are
``block`` or ``isolate``, so even at epsilon = 0.05 the agent takes roughly a dozen
expensive random containments per 250-step episode. Measured over 3,000 episodes, online
return got *worse* (-365 to -622) while the greedy policy it had learned was far better
(-159): the decline was the price of exploring, not a policy getting worse.

So ``train`` takes periodic **evaluation snapshots** -- a short greedy, non-learning run
every ``eval_every`` episodes -- and those are what the learning curve plots. The online
series is still recorded, because the gap between the two *is* the cost of exploration
and is worth a paragraph in the report. This is the defender's version of the
cliff-walking problem in section 7.2: an agent punished for exploring looks worse online
than it actually is.

Design note -- **rewards are accumulated undiscounted for reporting.**
The learner discounts by ``gamma`` internally, but the number plotted is the plain sum of
rewards over the episode. Discounted returns are not comparable across episodes of
different lengths, and every curve in section 9 compares across exactly that.
"""

from __future__ import annotations

from dataclasses import replace

from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

from marlsoc.agents.scripted import GreedyDefender, ScriptedAttacker
from marlsoc.agents.tabular import LearnerConfig, TabularLearner
from marlsoc.config import ALL_AGENTS, AgentConfig, Policy, ScenarioConfig
from marlsoc.env import actions as act
from marlsoc.env import observations as obs
from marlsoc.env import topology as topo
from marlsoc.env.actions import Verb
from marlsoc.env.minicorp import MiniCorp
from marlsoc.env.state import EpisodeState
from marlsoc.training.curriculum import Curriculum, CurriculumConfig
from marlsoc.training.metrics import EpisodeRecord, MetricsLog


def dims_for(agent: str) -> tuple[int, ...]:
    """The observation radix per agent -- the shape of its Q-table's state axis."""
    if agent == "R_breach":
        return obs.RED_BREACH_DIMS
    if agent == "R_scout":
        return obs.RED_SCOUT_DIMS
    return obs.blue_dims(agent)


def observation_for(state: EpisodeState, agent: str) -> tuple[int, ...]:
    if agent == "R_breach":
        return obs.red_breach_observation(state)
    if agent == "R_scout":
        return obs.red_scout_observation(state)
    return obs.blue_observation(state, agent)


def state_index(state: EpisodeState, agent: str) -> int:
    return obs.encode(observation_for(state, agent), dims_for(agent))


# --------------------------------------------------------------------------------------
# Controllers
# --------------------------------------------------------------------------------------
class Controller(Protocol):
    """Anything that can choose an action index for an agent."""

    learner: TabularLearner | None

    def select(self, state: EpisodeState, index: int, mask: np.ndarray) -> int: ...


@dataclass
class LearnedController:
    """Epsilon-greedy over a Q-table. ``greedy=True`` is ``Policy.GREEDY``."""

    learner: TabularLearner | None
    greedy: bool = False

    def select(self, state: EpisodeState, index: int, mask: np.ndarray) -> int:
        assert self.learner is not None
        return self.learner.act(index, mask, greedy=self.greedy)


@dataclass
class RandomController:
    """Uniform over the legal set -- section 9's floor."""

    rng: np.random.Generator
    learner: TabularLearner | None = None

    def select(self, state: EpisodeState, index: int, mask: np.ndarray) -> int:
        return int(self.rng.choice(np.flatnonzero(mask)))


@dataclass
class ScriptedController:
    """Wraps a scripted policy that speaks in ``Action`` objects."""

    agent: str
    policy: object
    learner: TabularLearner | None = None

    def select(self, state: EpisodeState, index: int, mask: np.ndarray) -> int:
        space = act.ACTION_SPACES[self.agent]
        legal = tuple(a for a, ok in zip(space, mask, strict=True) if ok)
        action = self.policy.act(state, legal)   # type: ignore[attr-defined]
        return act.action_index(self.agent, action)


@dataclass
class NoopController:
    """Always waits. A disabled agent still occupies its slot in the joint action."""

    agent: str
    learner: TabularLearner | None = None

    def select(self, state: EpisodeState, index: int, mask: np.ndarray) -> int:
        verb = Verb.WAIT if self.agent.startswith("R_") else Verb.NOOP
        for i, action in enumerate(act.ACTION_SPACES[self.agent]):
            if action.verb is verb:
                return i
        raise ValueError(f"{self.agent} has no idle action")


def build_controller(
    agent: str,
    cfg: AgentConfig,
    rng: np.random.Generator,
    learner_config: LearnerConfig | None = None,
) -> Controller:
    """Turn CLAUDE.md 3.7's three switches into something that can act.

    ``enabled=False`` outranks ``policy``: a disabled agent idles regardless of what its
    policy says, which is how "all blue agents disabled" becomes section 9's static
    firewall baseline without any separate baseline code.
    """
    if not cfg.enabled:
        return NoopController(agent)

    if cfg.policy is Policy.NOOP:
        return NoopController(agent)

    if cfg.policy is Policy.RANDOM:
        return RandomController(rng)

    if cfg.policy is Policy.SCRIPTED:
        policy = (
            ScriptedAttacker(agent, rng=rng) if agent.startswith("R_")
            else GreedyDefender(agent, rng=rng)
        )
        return ScriptedController(agent, policy)

    dims = dims_for(agent)
    learner = TabularLearner(
        agent, obs.n_states(dims), len(act.ACTION_SPACES[agent]),
        learner_config, rng,
    )
    return LearnedController(learner, greedy=cfg.policy is Policy.GREEDY)


# --------------------------------------------------------------------------------------
# One episode
# --------------------------------------------------------------------------------------
def run_episode(
    env: MiniCorp,
    controllers: dict[str, Controller],
    scenario: ScenarioConfig,
    seed: int | None = None,
    *,
    learn: bool = True,
    max_layer: int | None = None,
) -> tuple[EpisodeRecord, dict[str, float]]:
    """Run one episode to termination and return its metrics row.

    Args:
        env: The environment. Reset here, so the caller can reuse one instance.
        controllers: One per agent.
        scenario: Supplies each agent's ``learning`` switch.
        seed: Per-episode seed. Varying it across training is what stops every episode
            being the same episode; fixing it is what replays one for a demo.
        learn: Global override. False for evaluation runs, so a measurement never
            changes the thing being measured.
        max_layer: Curriculum stage for this episode; see ``MiniCorp.reset``.

    Returns:
        The episode record and the per-agent undiscounted returns.
    """
    env.reset(seed, max_layer)
    returns = {agent: 0.0 for agent in ALL_AGENTS}

    # A NoopController ignores both its observation and its mask, so computing either
    # for a disabled agent is pure waste -- and mask construction is half the twin's
    # runtime. Restricting the set here is what keeps a five-agent environment cheap to
    # run with two agents switched off, which is exactly the section 9 baselines.
    active = tuple(a for a, c in controllers.items() if not isinstance(c, NoopController))

    indices = {a: state_index(env.state, a) for a in active}
    masks = env.legal_masks(active)
    chosen = {
        a: controllers[a].select(env.state, indices[a], masks[a]) for a in active
    }

    steps = 0
    idle = {
        a: c.select(env.state, 0, np.empty(0))
        for a, c in controllers.items() if isinstance(c, NoopController)
    }
    while True:
        joint = {a: act.ACTION_SPACES[a][i] for a, i in chosen.items()}
        joint.update({a: act.ACTION_SPACES[a][i] for a, i in idle.items()})
        _, rewards, done, info = env.step(joint)
        steps += 1

        for agent, reward in rewards.items():
            returns[agent] += reward

        next_indices = {a: state_index(env.state, a) for a in active}
        next_masks = env.legal_masks(active)
        # SARSA needs a' before the update can happen; see the module docstring.
        next_chosen = {
            a: controllers[a].select(env.state, next_indices[a], next_masks[a])
            for a in active
        }

        for agent in active:
            controller = controllers[agent]
            learner = controller.learner
            if learner is None or not learn:
                continue
            if not scenario.for_agent(agent).learning:
                continue
            learner.update(
                indices[agent], chosen[agent], rewards[agent],
                next_indices[agent], next_masks[agent], done,
                next_action=next_chosen[agent],
            )

        if done:
            break
        indices, masks, chosen = next_indices, next_masks, next_chosen

    for controller in controllers.values():
        if controller.learner is not None and learn:
            controller.learner.end_episode()

    epsilon = next(
        (c.learner.epsilon for c in controllers.values() if c.learner is not None), 0.0
    )
    record = EpisodeRecord(
        episode=0, phase="", learner="",
        red_return=returns["R_breach"],
        blue_return=returns["B_corp"],
        outcome=info["outcome"],
        steps=steps,
        hosts_compromised=info["hosts_compromised"],
        layers_breached=info["layers_breached"],
        mttd=info["mttd"],
        mttc=info["mttc"],
        false_positives=info["false_positives"],
        honeypot_hits=info["honeypot_hits"],
        epsilon=epsilon,
    )
    return record, returns


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
class TrainingRun(NamedTuple):
    """Everything a run produces.

    Attributes:
        controllers: The trained agents.
        log: Per-episode online metrics -- what was collected *while exploring*.
        evals: Periodic greedy snapshots. **This is the learning curve.** See the design
            note on why the two differ.
    """

    controllers: dict[str, Controller]
    log: MetricsLog
    evals: MetricsLog


def train(
    scenario: ScenarioConfig,
    episodes: int,
    learner_config: LearnerConfig | None = None,
    *,
    controllers: dict[str, Controller] | None = None,
    phase: str = "phase2",
    learner_name: str = "B_corp",
    report_every: int = 500,
    eval_every: int = 250,
    eval_episodes: int = 60,
    verbose: bool = True,
) -> TrainingRun:
    """Run ``episodes`` episodes and return the controllers and both metric series.

    Each episode gets its own seed, derived from the scenario seed, so a run is
    reproducible end to end while no two episodes are identical.

    Args:
        eval_every: Episodes between greedy evaluation snapshots. Zero disables them.
        eval_episodes: Episodes per snapshot. Small, because this runs many times; the
            noise it adds is averaged out by the curve rather than by each point.
    """
    rng = np.random.default_rng(scenario.seed)
    env = MiniCorp(scenario)
    controllers = controllers or {
        agent: build_controller(agent, scenario.for_agent(agent), rng, learner_config)
        for agent in ALL_AGENTS
    }

    log = MetricsLog()
    evals = MetricsLog()
    for episode in range(episodes):
        record, _ = run_episode(
            env, controllers, scenario, seed=scenario.seed * 1_000_003 + episode
        )
        log.append(
            EpisodeRecord(**{**record.__dict__, "episode": episode,
                             "phase": phase, "learner": learner_name})
        )

        if eval_every and (episode + 1) % eval_every == 0:
            snapshot = evaluate(scenario, controllers, eval_episodes, phase="snapshot")
            # One row per snapshot, holding that snapshot's means. The curve is then a
            # series of measurements of the policy rather than of the exploration.
            evals.append(EpisodeRecord(
                episode=episode, phase=phase, learner=learner_name,
                red_return=snapshot.mean("red_return"),
                blue_return=snapshot.mean("blue_return"),
                outcome=f"red_win_rate={snapshot.rate('red_win'):.4f}",
                steps=round(snapshot.mean("steps")),
                hosts_compromised=round(snapshot.mean("hosts_compromised")),
                layers_breached=round(snapshot.mean("layers_breached")),
                mttd=None, mttc=None,
                false_positives=round(snapshot.mean("false_positives")),
                honeypot_hits=round(snapshot.mean("honeypot_hits")),
                epsilon=record.epsilon,
            ))

        if verbose and report_every and (episode + 1) % report_every == 0:
            print(log.summary(report_every))

    return TrainingRun(controllers, log, evals)


def evaluate(
    scenario: ScenarioConfig,
    controllers: dict[str, Controller],
    episodes: int = 500,
    *,
    phase: str = "eval",
) -> MetricsLog:
    """Measure a frozen policy: greedy, no learning, fresh seeds.

    Learning is disabled globally rather than by flipping each agent's switch, so an
    evaluation can never accidentally train the thing it is measuring -- the mistake that
    makes a reported number better than the policy actually is.
    """
    env = MiniCorp(scenario)
    # Remember what each controller *was*, rather than assuming it was exploring. A
    # controller configured as Policy.GREEDY -- a frozen, deployed defender -- is greedy
    # by design, and restoring it to False would quietly turn it into an epsilon-greedy
    # agent for every run after its first evaluation.
    was_greedy = {
        name: controller.greedy
        for name, controller in controllers.items()
        if isinstance(controller, LearnedController)
    }
    for controller in controllers.values():
        if isinstance(controller, LearnedController):
            controller.greedy = True

    log = MetricsLog()
    for episode in range(episodes):
        record, _ = run_episode(
            env, controllers, scenario,
            seed=scenario.seed * 7_919 + 500_000 + episode, learn=False,
        )
        log.append(EpisodeRecord(**{**record.__dict__, "episode": episode,
                                    "phase": phase, "learner": "frozen"}))

    for name, previous in was_greedy.items():
        controllers[name].greedy = previous
    return log


# --------------------------------------------------------------------------------------
# Curriculum training (PROJECT.md section 7.4)
# --------------------------------------------------------------------------------------
class CurriculumRun(NamedTuple):
    """A curriculum run: the trained agents, the episode log, and the stage history."""

    controllers: dict[str, Controller]
    log: MetricsLog
    curriculum: Curriculum


def train_curriculum(
    scenario: ScenarioConfig,
    episodes: int,
    learner_config: LearnerConfig | None = None,
    curriculum_config: CurriculumConfig | None = None,
    *,
    controllers: dict[str, Controller] | None = None,
    report_every: int = 1_000,
    verbose: bool = True,
) -> CurriculumRun:
    """Train against a progressively deeper layer stack.

    The Q-table is never reset between stages -- that carry-forward *is* the transfer
    section 7.4 calls the whole point. A table that has learned "get a DMZ foothold"
    already knows how when Layer 3 switches on, so stage 2 only has to learn the new
    layer rather than the whole chain again.

    Each episode's ``phase`` records the stage it was played at, so the sawtooth plot and
    any per-stage analysis come straight out of the CSV without a second bookkeeping
    path.

    Args:
        scenario: Agent switches and seed. Its ``max_layer`` is ignored -- the curriculum
            supplies the stage per episode.
        episodes: Total training episodes across all stages.
        learner_config: Hyperparameters for every learning agent.
        curriculum_config: Promotion rules.
        controllers: Pre-built controllers to train *instead of* fresh ones. This is how
            CLAUDE.md 3.4's alternating training keeps each phase a stationary MDP: hand
            in the opposing team already loaded and frozen, so the learner faces the
            policy it will actually be evaluated against. Training against one opponent
            and evaluating against another measures distribution shift, not learning.
        report_every: Console summary interval.
        verbose: Print progress and transitions.

    Returns:
        The controllers, the per-episode log, and the curriculum with its transitions.
    """
    rng = np.random.default_rng(scenario.seed)
    env = MiniCorp(scenario)
    controllers = controllers or {
        agent: build_controller(agent, scenario.for_agent(agent), rng, learner_config)
        for agent in ALL_AGENTS
    }
    # The budget is handed to the curriculum so its stage cap can be a share of what
    # is left rather than a constant guessed against an unknown run length.
    curriculum = Curriculum(curriculum_config or CurriculumConfig(),
                            total_episodes=episodes)
    log = MetricsLog()

    for episode in range(episodes):
        stage = curriculum.max_layer
        record, _ = run_episode(
            env, controllers, scenario,
            seed=scenario.seed * 1_000_003 + episode,
            max_layer=stage,
        )
        log.append(EpisodeRecord(**{
            **record.__dict__,
            "episode": episode,
            "phase": f"stage{curriculum.stage_number}",
            "learner": "red",
        }))

        # Promotion is judged on the *greedy* policy, not on the exploring one that
        # generated the episode above. See CurriculumConfig.promote_on_greedy: every
        # transition once reported 100% success while a greedy evaluation of the same
        # agent scored 0.0%, because epsilon-greedy explores past its own bad estimates
        # and the deployable policy cannot.
        if curriculum.due_for_evaluation():
            probe = replace(scenario, max_layer=curriculum.max_layer)
            snapshot = evaluate(probe, controllers,
                                curriculum.config.greedy_eval_episodes,
                                phase=f"probe{curriculum.stage_number}")
            curriculum.record_evaluation(snapshot.rate("red_win"))

        transition = curriculum.record(
            won=record.outcome == "red_win", episode=episode
        )
        if transition is not None:
            # A new stage contains a behaviour red has never performed; a nearly greedy
            # policy cannot find it. See curriculum.py's design note.
            boost = curriculum.config.epsilon_on_promote
            if boost is not None:
                for controller in controllers.values():
                    if controller.learner is not None:
                        controller.learner.boost_exploration(boost)
            if verbose:
                tag = "  (FORCED)" if transition.forced else ""
                print(f"  ep {episode:>6}  stage {transition.from_stage} -> "
                      f"{transition.to_stage}  (success {transition.success_rate:.1%},"
                      f" now layers 1-{curriculum.max_layer}){tag}")

        if verbose and report_every and (episode + 1) % report_every == 0:
            print(f"{log.summary(report_every)}  | stage {curriculum.stage_number} "
                  f"(layers 1-{curriculum.max_layer}), "
                  f"recent success {curriculum.success_rate:.1%}")

    return CurriculumRun(controllers, log, curriculum)
