"""Tabular TD control: Q-Learning, SARSA and Expected SARSA, written from scratch.

NumPy only. CLAUDE.md section 2 forbids stable-baselines, RLlib and Tianshou here,
because the rubric penalises calling an RL library without understanding it and every
update rule has to be defensible as our own work.

The three algorithms are **one update** differing in a single term::

    Q(s,a) <- Q(s,a) + alpha * [ TARGET - Q(s,a) ]

    Q-Learning      TARGET = r + gamma * max_{a' in legal} Q(s',a')       off-policy
    SARSA           TARGET = r + gamma * Q(s',a')   a' actually taken     on-policy
    Expected SARSA  TARGET = r + gamma * sum_{a'} pi(a'|s') * Q(s',a')    on-policy

So they share a class and differ in ``_td_target``. That is not a shortcut: it makes the
algorithm sweep in PROJECT.md section 9 a one-line configuration change, and it puts the
three equations next to each other where the difference is actually visible.

Which algorithm goes where (PROJECT.md sections 7.1 and 7.2)
------------------------------------------------------------
Blue gets **Q-Learning**. A defender can explore freely -- a wrong action during training
costs nothing real -- so learning the optimal policy regardless of the exploratory
behaviour is exactly right.

Red gets **SARSA**. The attacker is *punished for exploring*: -50 when detected. This is
the Cliff Walking situation from Sutton & Barto section 6.5. Q-Learning learns the
optimal path along the cliff edge and keeps falling off while it explores; SARSA learns a
safer path that accounts for its own exploration risk. An attacker that gets caught
probing should learn the stealthy route, and SARSA gives that for free.

Design note -- **the max must be taken over legal actions only.**
If Q-Learning bootstraps off an illegal action's value it is chasing a return it can
never collect, and the error compounds backwards through every earlier state. This is the
easiest way to get a subtly wrong tabular agent: nothing crashes, the values are merely
too optimistic in a way that is invisible on a reward curve. Every target here is masked.

Design note -- **argmax breaks ties at random.**
At initialisation every Q-value is 0.0, so ``np.argmax`` returns index 0 every single
time. The agent would greedily take the same action forever and its only exploration
would come from epsilon. Random tie-breaking removes that positional bias, and it matters
again later wherever a row is genuinely flat.

Design note -- **initialisation is a hyperparameter, not an accident of ``np.zeros``.**
Almost every return in this environment is large and negative -- a defender's episode
return is around -150, and a bad one is -2500. Initialising at 0.0 therefore gives every
untried action an optimism bonus of roughly +150 over the true value of the best known
one. During training that is *optimistic initialisation*: useful, systematic exploration
on top of epsilon-greedy, for free.

At **evaluation** it is a disaster, and it is measurable. After 4,000 episodes at
curriculum stage 1-3, state coverage was 14.6%, and in **91% of visited states the greedy
action was an entry that had never once been updated** -- 68% of all decisions weighted by
how often the states occur, against a learned ``Q(noop)`` of -149.7 in the most-visited
state. The greedy policy was, in effect, "always try something you have never tried". That
is why the trained defender scored *worse than always choosing noop*, and why it did
better at epsilon = 0.05 than greedy: exploration occasionally picked an action whose
value it actually knew.

So ``q_init`` is explicit. Zero keeps the textbook optimistic behaviour. Setting it near
the return of doing nothing makes an untried action look about as good as conceding, which
is honest, and stops the greedy policy preferring ignorance. The right value depends on
the reward scale, which is exactly why it should not be hidden inside a call to
``np.zeros``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np


class Algorithm(str, Enum):
    """Which TD target to use. The only difference between the three."""

    Q_LEARNING = "q_learning"
    SARSA = "sarsa"
    EXPECTED_SARSA = "expected_sarsa"

    @property
    def is_on_policy(self) -> bool:
        """SARSA and Expected SARSA value the policy actually being followed."""
        return self is not Algorithm.Q_LEARNING


@dataclass(frozen=True)
class LearnerConfig:
    """Hyperparameters. Frozen so a run's settings serialise beside its curves.

    Attributes:
        algorithm: Which TD target.
        alpha: Learning rate. How far each update moves toward the target. Section 9
            sweeps 0.01 / 0.1 / 0.3.
        gamma: Discount factor -- how much a reward one step later is worth now. Section
            5.5 sets 0.95 and argues it must be high, because with a six-layer path the
            payoff for an early ``steal_credentials`` arrives many steps afterwards.
        epsilon_start: Initial exploration rate.
        epsilon_end: Floor. Never zero during training: a policy that stops exploring
            cannot notice the opponent has changed, which matters enormously in the
            alternating training of section 6.
        epsilon_decay_episodes: Episodes over which epsilon falls from start to end.
            Geometric, so most of the exploration happens early.
        q_init: Value every Q-entry starts at. 0.0 is optimistic in this reward regime --
            see the module docstring for why that helps during training and hurts at
            evaluation. Set it near the return of doing nothing to stop the greedy policy
            preferring actions it has never tried.
    """

    algorithm: Algorithm = Algorithm.Q_LEARNING
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 3_000
    q_init: float = 0.0


class TabularLearner:
    """One agent's Q-table and the update rule that fills it.

    The table is ``(n_states, n_actions)`` float64. Dense rather than sparse because
    CLAUDE.md caps every agent at ~10,000 states and the widest action set is 40, so the
    largest table in the project is 3,072 x 13 -- about 320 KB. A dict-of-dicts would be
    slower and would hide the budget rather than making it obvious.

    Example:
        >>> learner = TabularLearner("B_corp", n_states=3072, n_actions=13)
        >>> a = learner.act(state_index, mask)
        >>> learner.update(state_index, a, reward, next_index, next_mask, done=False)
    """

    def __init__(
        self,
        name: str,
        n_states: int,
        n_actions: int,
        config: LearnerConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.name = name
        self.config = config or LearnerConfig()
        self.rng = rng if rng is not None else np.random.default_rng()

        # See the module docstring: this value is load-bearing, not a default.
        self.q = np.full((n_states, n_actions), self.config.q_init, dtype=np.float64)

        self.episode = 0
        self.updates = 0
        # Per-state visit counts. Not used by the algorithm; it is how we answer "did the
        # agent actually see enough of its state space to have learned anything?", which
        # is the first question to ask of a flat learning curve.
        self.visits = np.zeros(n_states, dtype=np.int64)

    # ----------------------------------------------------------------------------------
    # Exploration schedule
    # ----------------------------------------------------------------------------------
    @property
    def epsilon(self) -> float:
        """Current exploration rate, decaying geometrically ``start -> end``.

        Geometric rather than linear so most exploration happens early, while the
        Q-values are worthless, and the agent spends the long tail of training refining a
        policy rather than randomly walking away from it.
        """
        cfg = self.config
        # A geometric schedule cannot start from zero, and a zero start is exactly what
        # GREEDY_EVAL asks for. Both degenerate cases collapse to a constant epsilon.
        if cfg.epsilon_decay_episodes <= 0 or cfg.epsilon_start <= 0.0:
            return cfg.epsilon_end
        fraction = min(1.0, self.episode / cfg.epsilon_decay_episodes)
        return cfg.epsilon_start * (cfg.epsilon_end / cfg.epsilon_start) ** fraction

    def end_episode(self) -> None:
        """Advance the exploration schedule. Called once per episode, not per step."""
        self.episode += 1

    # ----------------------------------------------------------------------------------
    # Policy
    # ----------------------------------------------------------------------------------
    def act(self, state: int, mask: np.ndarray, *, greedy: bool = False) -> int:
        """Choose an action index by epsilon-greedy over the **legal** actions.

        Args:
            state: Encoded observation index.
            mask: Boolean array over the action space; True where legal.
            greedy: Force exploitation. Used for evaluation and for deployment, which is
                ``Policy.GREEDY`` in ``config.py``.

        Returns:
            An index into the agent's action tuple, always a legal one.

        Raises:
            ValueError: if no action is legal. The environment guarantees this cannot
                happen -- WAIT and NOOP are unconditionally legal -- so if it fires,
                the mask is wrong rather than the situation being hopeless.
        """
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise ValueError(f"{self.name} has no legal action in state {state}")

        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))

        return self._greedy_action(state, legal)

    def _greedy_action(self, state: int, legal: np.ndarray) -> int:
        """``argmax_a Q(s,a)`` over legal actions, breaking ties uniformly at random.

        The tie-break is not cosmetic. At initialisation every value is 0.0, so a plain
        ``argmax`` would return the lowest-indexed legal action in every state forever --
        the agent's greedy behaviour would be a constant, and all of its exploration
        would have to come from epsilon.
        """
        values = self.q[state, legal]
        best = legal[values == values.max()]
        return int(best[0]) if best.size == 1 else int(self.rng.choice(best))

    def action_probabilities(self, state: int, mask: np.ndarray) -> np.ndarray:
        """``pi(a|s)`` for the current epsilon-greedy policy over the legal set.

        Needed by Expected SARSA, which averages over the policy instead of sampling one
        action from it. Returns a full-width array with zeros on illegal actions, so it
        can be dotted straight against a Q-row.

        With ``n`` legal actions and ``k`` tied for the maximum::

            pi(a) = epsilon/n                   for a legal, not best
            pi(a) = epsilon/n + (1-epsilon)/k   for a legal and best
            pi(a) = 0                           for a illegal
        """
        probs = np.zeros(self.q.shape[1], dtype=np.float64)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return probs

        eps = self.epsilon
        probs[legal] = eps / legal.size

        values = self.q[state, legal]
        best = legal[values == values.max()]
        probs[best] += (1.0 - eps) / best.size
        return probs

    # ----------------------------------------------------------------------------------
    # Learning
    # ----------------------------------------------------------------------------------
    def _td_target(
        self,
        reward: float,
        next_state: int,
        next_mask: np.ndarray,
        next_action: int | None,
        done: bool,
    ) -> float:
        """The bootstrapped target. The one line where the three algorithms differ.

        A terminal state has no future, so the target is just ``r``. Bootstrapping past
        termination is a classic tabular bug: the agent learns that dying is followed by
        whatever garbage sits in the terminal row of the table.
        """
        if done:
            return reward

        gamma = self.config.gamma
        legal = np.flatnonzero(next_mask)
        if legal.size == 0:
            return reward

        algo = self.config.algorithm

        if algo is Algorithm.Q_LEARNING:
            # Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
            # The max is over LEGAL actions only; see the module docstring.
            return reward + gamma * float(self.q[next_state, legal].max())

        if algo is Algorithm.SARSA:
            # Q(s,a) <- Q(s,a) + alpha [ r + gamma * Q(s',a') - Q(s,a) ]   a' taken
            if next_action is None:
                raise ValueError("SARSA needs the action actually taken in s'")
            return reward + gamma * float(self.q[next_state, next_action])

        # Q(s,a) <- Q(s,a) + alpha [ r + gamma * sum_a' pi(a'|s') Q(s',a') - Q(s,a) ]
        # Lower variance than SARSA: it averages over the policy instead of sampling one
        # action from it, which helps here because our layer breaches succeed
        # probabilistically and a single sample is noisy.
        probs = self.action_probabilities(next_state, next_mask)
        return reward + gamma * float(probs @ self.q[next_state])

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_mask: np.ndarray,
        done: bool,
        next_action: int | None = None,
    ) -> float:
        """Apply one TD update and return the TD error.

        Args:
            state: Encoded observation the action was chosen in.
            action: Action index taken.
            reward: Reward received.
            next_state: Encoded observation that followed.
            next_mask: Legality mask in ``next_state``.
            done: Whether the episode terminated on this transition.
            next_action: The action taken in ``next_state``. Required for SARSA, ignored
                by the other two.

        Returns:
            The TD error ``delta``. Returned rather than discarded because its magnitude
            over time is the most direct evidence that learning is converging -- a curve
            that has flattened while ``|delta|`` is still large means the policy is
            oscillating, not settled.
        """
        target = self._td_target(reward, next_state, next_mask, next_action, done)
        delta = target - self.q[state, action]
        self.q[state, action] += self.config.alpha * delta

        self.visits[state] += 1
        self.updates += 1
        return float(delta)

    # ----------------------------------------------------------------------------------
    # Persistence and inspection
    # ----------------------------------------------------------------------------------
    @property
    def untried_greedy_fraction(self) -> float:
        """Share of visited states whose greedy action has never been updated.

        The diagnostic that exposed the initialisation trap. A high value means the
        greedy policy is largely choosing by ignorance rather than by value, and any
        evaluation of it is measuring the initialisation rather than the learning.
        """
        seen = np.flatnonzero(self.visits)
        if seen.size == 0:
            return 0.0
        init = self.config.q_init
        untried = sum(
            1 for s in seen
            if self.q[s].max() == init and self.q[s].min() < init
        )
        return untried / seen.size

    @property
    def coverage(self) -> float:
        """Fraction of the state space visited at least once.

        The first diagnostic to check against a flat learning curve: an agent that has
        seen 3% of its states has not failed to learn, it has failed to *explore*, and
        those are different problems with different fixes.
        """
        return float((self.visits > 0).mean())

    def save(self, path) -> None:
        """Persist the table for deployment to the Docker lab (PROJECT.md section 8)."""
        np.savez_compressed(
            path, q=self.q, visits=self.visits,
            episode=self.episode, algorithm=self.config.algorithm.value,
        )

    def load(self, path) -> None:
        """Restore a table. Shape mismatches raise rather than broadcasting."""
        data = np.load(path, allow_pickle=False)
        if data["q"].shape != self.q.shape:
            raise ValueError(
                f"table shape {data['q'].shape} does not match {self.q.shape}"
            )
        self.q = data["q"]
        self.visits = data["visits"]
        self.episode = int(data["episode"])


GREEDY_EVAL: Final[LearnerConfig] = LearnerConfig(
    epsilon_start=0.0, epsilon_end=0.0, epsilon_decay_episodes=0
)
