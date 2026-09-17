"""Tests for the from-scratch TD learners.

The most important test in this file is the cliff-walking one. It is simultaneously a
correctness check on three hand-written algorithms and the empirical justification for
PROJECT.md section 7.2's decision to give the defender Q-Learning and the attacker SARSA.
If it stopped reproducing Sutton & Barto's result, either the implementations are wrong
or the argument in the report is.
"""

from __future__ import annotations

import numpy as np
import pytest

from marlsoc.agents.tabular import Algorithm, LearnerConfig, TabularLearner


def learner(algo=Algorithm.Q_LEARNING, n_states=1, n_actions=3, seed=0, **kw):
    cfg = LearnerConfig(algorithm=algo, **kw)
    return TabularLearner("t", n_states, n_actions, cfg, np.random.default_rng(seed))


ALL = np.ones(3, dtype=bool)


class TestUpdateRule:
    def test_one_update_moves_alpha_of_the_way_to_the_target(self) -> None:
        """Q(s,a) <- Q(s,a) + alpha [ target - Q(s,a) ]. With gamma=0 and a terminal
        transition the target is just r, so the arithmetic is checkable by hand."""
        L = learner(alpha=0.25, gamma=0.0)
        delta = L.update(0, 1, reward=8.0, next_state=0, next_mask=ALL, done=True)
        assert delta == 8.0
        assert L.q[0, 1] == pytest.approx(2.0)      # 0 + 0.25 * (8 - 0)
        L.update(0, 1, 8.0, 0, ALL, done=True)
        assert L.q[0, 1] == pytest.approx(3.5)      # 2 + 0.25 * (8 - 2)

    def test_repeated_updates_converge_to_the_true_value(self) -> None:
        L = learner(alpha=0.1, gamma=0.0)
        for _ in range(500):
            L.update(0, 0, 4.0, 0, ALL, done=True)
        assert L.q[0, 0] == pytest.approx(4.0, abs=1e-3)

    def test_a_terminal_transition_does_not_bootstrap(self) -> None:
        """Bootstrapping past termination teaches the agent that dying is followed by
        whatever garbage happens to sit in the terminal row."""
        L = learner(n_states=2, alpha=1.0, gamma=0.9)
        L.q[1, :] = 100.0
        L.update(0, 0, reward=-1.0, next_state=1, next_mask=ALL, done=True)
        assert L.q[0, 0] == pytest.approx(-1.0)

    def test_gamma_discounts_the_future(self) -> None:
        L = learner(n_states=2, alpha=1.0, gamma=0.5)
        L.q[1, :] = 10.0
        L.update(0, 0, reward=0.0, next_state=1, next_mask=ALL, done=False)
        assert L.q[0, 0] == pytest.approx(5.0)   # 0 + 1.0 * (0 + 0.5 * 10 - 0)

    def test_all_three_algorithms_learn_the_same_bandit(self) -> None:
        # Different targets, same answer when there is no future to disagree about.
        for algo in Algorithm:
            L = learner(algo, alpha=0.1, gamma=0.0,
                        epsilon_start=0.3, epsilon_end=0.3)
            for _ in range(3000):
                a = L.act(0, ALL)
                L.update(0, a, [1.0, 5.0, -2.0][a], 0, ALL, False,
                         next_action=L.act(0, ALL))
            assert L.q[0].argmax() == 1
            assert L.q[0] == pytest.approx([1.0, 5.0, -2.0], abs=0.3)


class TestTheThreeTargetsDiffer:
    """Each algorithm must actually compute its own target, not a shared one."""

    def _setup(self, algo):
        L = learner(algo, n_states=2, alpha=1.0, gamma=1.0,
                    epsilon_start=0.0, epsilon_end=0.0)
        L.q[1] = np.array([0.0, 10.0, -10.0])   # best=10, worst=-10
        return L

    def test_q_learning_bootstraps_off_the_best_next_action(self) -> None:
        L = self._setup(Algorithm.Q_LEARNING)
        L.update(0, 0, 0.0, 1, ALL, False, next_action=2)   # took the worst action
        assert L.q[0, 0] == pytest.approx(10.0)             # ...valued the best anyway

    def test_sarsa_bootstraps_off_the_action_actually_taken(self) -> None:
        L = self._setup(Algorithm.SARSA)
        L.update(0, 0, 0.0, 1, ALL, False, next_action=2)
        assert L.q[0, 0] == pytest.approx(-10.0)            # it valued what it did

    def test_expected_sarsa_averages_over_the_policy(self) -> None:
        """With epsilon=0 the expectation collapses onto the greedy action, so it agrees
        with Q-Learning; with epsilon>0 it must sit strictly between the best and the
        mean, which is what gives it lower variance than SARSA."""
        L = self._setup(Algorithm.EXPECTED_SARSA)
        L.update(0, 0, 0.0, 1, ALL, False)
        assert L.q[0, 0] == pytest.approx(10.0)

        L2 = learner(Algorithm.EXPECTED_SARSA, n_states=2, alpha=1.0, gamma=1.0,
                     epsilon_start=0.5, epsilon_end=0.5)
        L2.q[1] = np.array([0.0, 10.0, -10.0])
        L2.update(0, 0, 0.0, 1, ALL, False)
        assert 0.0 < L2.q[0, 0] < 10.0

    def test_sarsa_refuses_to_update_without_the_next_action(self) -> None:
        # Silently falling back to a max would make SARSA into Q-Learning and quietly
        # invalidate the entire section 7.2 comparison.
        L = self._setup(Algorithm.SARSA)
        with pytest.raises(ValueError):
            L.update(0, 0, 0.0, 1, ALL, done=False)


class TestMasking:
    def test_only_legal_actions_are_ever_chosen(self) -> None:
        L = learner(epsilon_start=1.0, epsilon_end=1.0)   # pure exploration
        mask = np.array([False, True, False])
        assert {L.act(0, mask) for _ in range(200)} == {1}

    def test_the_greedy_choice_ignores_a_better_illegal_action(self) -> None:
        L = learner(epsilon_start=0.0, epsilon_end=0.0)
        L.q[0] = np.array([100.0, 1.0, 2.0])
        assert L.act(0, np.array([False, True, True])) == 2

    def test_bootstrapping_never_uses_an_illegal_action_s_value(self) -> None:
        """The subtlest way to get a wrong tabular agent: nothing crashes, the values
        are merely too optimistic because the agent is chasing a return it can never
        collect, and the error propagates backwards through every earlier state."""
        L = learner(Algorithm.Q_LEARNING, n_states=2, alpha=1.0, gamma=1.0)
        L.q[1] = np.array([999.0, 5.0, 3.0])
        L.update(0, 0, 0.0, 1, np.array([False, True, True]), done=False)
        assert L.q[0, 0] == pytest.approx(5.0)

    def test_expected_sarsa_puts_no_probability_on_illegal_actions(self) -> None:
        L = learner(Algorithm.EXPECTED_SARSA, epsilon_start=1.0, epsilon_end=1.0)
        probs = L.action_probabilities(0, np.array([True, False, True]))
        assert probs[1] == 0.0
        assert probs.sum() == pytest.approx(1.0)

    def test_an_empty_mask_raises_rather_than_guessing(self) -> None:
        # The environment guarantees WAIT/NOOP are always legal, so an empty mask means
        # the mask is wrong, not that the situation is hopeless.
        L = learner()
        with pytest.raises(ValueError):
            L.act(0, np.zeros(3, dtype=bool))


class TestTieBreaking:
    def test_ties_are_broken_uniformly_rather_than_by_index(self) -> None:
        """At initialisation every value is 0.0. A plain argmax would return action 0 in
        every state forever, so the agent's greedy behaviour would be a constant and all
        exploration would have to come from epsilon."""
        L = learner(epsilon_start=0.0, epsilon_end=0.0, seed=3)
        chosen = {L.act(0, ALL) for _ in range(300)}
        assert chosen == {0, 1, 2}

    def test_a_clear_winner_is_always_taken(self) -> None:
        L = learner(epsilon_start=0.0, epsilon_end=0.0)
        L.q[0] = np.array([1.0, 7.0, 1.0])
        assert {L.act(0, ALL) for _ in range(50)} == {1}


class TestEpsilonSchedule:
    def test_epsilon_decays_from_start_to_end(self) -> None:
        L = learner(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=100)
        assert L.epsilon == pytest.approx(1.0)
        for _ in range(100):
            L.end_episode()
        assert L.epsilon == pytest.approx(0.05)

    def test_epsilon_is_monotone_and_never_leaves_the_bounds(self) -> None:
        L = learner(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=50)
        seen = []
        for _ in range(200):
            seen.append(L.epsilon)
            L.end_episode()
        assert seen == sorted(seen, reverse=True)
        assert min(seen) >= 0.05 and max(seen) <= 1.0

    def test_exploration_never_reaches_zero_during_training(self) -> None:
        """A policy that stops exploring cannot notice that the opponent has changed,
        which matters enormously under the alternating training in section 6."""
        L = learner(epsilon_decay_episodes=10)
        for _ in range(10_000):
            L.end_episode()
        assert L.epsilon > 0.0


class TestCliffWalking:
    """Sutton & Barto Example 6.6, reproduced with our own learners.

    This is why the attacker gets SARSA and the defender gets Q-Learning (section 7.2).
    The cliff is our -50 detection penalty: an agent punished *for exploring* should
    learn a route that accounts for its own exploration, not the optimal route it keeps
    falling off.
    """

    ROWS, COLS = 4, 12
    START, GOAL = (3, 0), (3, 11)
    MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def _step(self, pos, a):
        r = min(self.ROWS - 1, max(0, pos[0] + self.MOVES[a][0]))
        c = min(self.COLS - 1, max(0, pos[1] + self.MOVES[a][1]))
        if r == 3 and 1 <= c <= 10:
            return self.START, -100.0, False     # off the cliff, back to the start
        return (r, c), -1.0, (r, c) == self.GOAL

    def _train(self, algo, episodes=600, seed=0):
        L = TabularLearner(
            "cliff", self.ROWS * self.COLS, 4,
            LearnerConfig(algorithm=algo, alpha=0.5, gamma=1.0,
                          epsilon_start=0.1, epsilon_end=0.1, epsilon_decay_episodes=0),
            np.random.default_rng(seed),
        )
        mask = np.ones(4, dtype=bool)
        returns = []
        for _ in range(episodes):
            pos, total = self.START, 0.0
            a = L.act(pos[0] * self.COLS + pos[1], mask)
            for _ in range(500):
                s = pos[0] * self.COLS + pos[1]
                nxt, r, done = self._step(pos, a)
                ns = nxt[0] * self.COLS + nxt[1]
                na = L.act(ns, mask)
                L.update(s, a, r, ns, mask, done, next_action=na)
                total += r
                pos, a = nxt, na
                if done:
                    break
            returns.append(total)
            L.end_episode()
        return L, returns

    def _greedy_rows(self, L):
        """Rows visited by the greedy policy. Row 2 hugs the cliff; row 0 is safest."""
        pos, rows = self.START, []
        for _ in range(60):
            a = L._greedy_action(pos[0] * self.COLS + pos[1], np.arange(4))
            pos, _, done = self._step(pos, a)
            rows.append(pos[0])
            if done or pos == self.START:
                break
        return rows

    def test_sarsa_beats_q_learning_on_online_performance(self) -> None:
        """The headline of Figure 6.4. Q-Learning learns the optimal path and keeps
        falling off it while exploring; SARSA learns a path that survives its own
        epsilon-greedy behaviour, so it collects more reward *while learning*."""
        _, q_returns = self._train(Algorithm.Q_LEARNING)
        _, s_returns = self._train(Algorithm.SARSA)
        assert np.mean(s_returns[-200:]) > np.mean(q_returns[-200:])

    def test_expected_sarsa_beats_both_by_averaging_out_the_noise(self) -> None:
        # Section 7.3: lower variance, because it averages over the policy rather than
        # sampling one action from it.
        _, q_returns = self._train(Algorithm.Q_LEARNING)
        _, e_returns = self._train(Algorithm.EXPECTED_SARSA)
        assert np.mean(e_returns[-200:]) > np.mean(q_returns[-200:])

    def test_q_learning_walks_closer_to_the_cliff_than_sarsa(self) -> None:
        """The behavioural difference behind the numbers, not just the score.

        Q-Learning's greedy policy is the *optimal* one -- along the cliff edge. SARSA
        detours away from it, because its values include the cost of the exploratory
        steps it will actually take.
        """
        q_learner, _ = self._train(Algorithm.Q_LEARNING)
        s_learner, _ = self._train(Algorithm.SARSA)
        q_rows = self._greedy_rows(q_learner)
        s_rows = self._greedy_rows(s_learner)
        assert np.mean(q_rows) > np.mean(s_rows)   # higher row index = nearer the cliff


class TestDiagnostics:
    def test_coverage_reports_how_much_of_the_space_was_seen(self) -> None:
        """The first thing to check against a flat learning curve: an agent that has
        seen 3% of its states has failed to explore, not failed to learn, and those have
        different fixes."""
        L = learner(n_states=10)
        assert L.coverage == 0.0
        for s in range(5):
            L.update(s, 0, 1.0, s, ALL, done=True)
        assert L.coverage == 0.5

    def test_saving_and_loading_round_trips(self, tmp_path) -> None:
        L = learner(n_states=8)
        L.q[3, 1] = 4.2
        L.end_episode()
        L.save(tmp_path / "q.npz")
        other = learner(n_states=8)
        other.load(tmp_path / "q.npz")
        assert other.q[3, 1] == 4.2
        assert other.episode == 1

    def test_loading_a_mismatched_table_raises(self, tmp_path) -> None:
        # Deploying a Q-table onto the wrong agent must fail loudly, not broadcast.
        learner(n_states=8).save(tmp_path / "q.npz")
        with pytest.raises(ValueError):
            learner(n_states=9).load(tmp_path / "q.npz")
