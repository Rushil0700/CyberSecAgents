"""Curriculum learning: train red against a progressively deeper layer stack.

PROJECT.md section 7.4 is the specification. The problem it solves is the one Phase 1
demonstrated empirically: with all six layers active from episode one, random exploration
essentially never completes the chain, red receives no learning signal, and the curve
flatlines. Measured then: red wins 0% of episodes against even a random defender, at a
mean depth of 0.78 of 6.

The fix is to make the early stages winnable, so red gets positive signal, and to carry
the Q-table forward as the stack deepens. That carry-forward is the *transfer* the spec
calls "the whole point" -- a Q-table trained on "get a DMZ foothold" already knows how to
get a DMZ foothold when Layer 3 switches on, so stage 2 only has to learn the new layer.

    stage 1   layers 1-2   get a foothold in the DMZ
    stage 2   layers 1-3   + steal credentials
    stage 3   layers 1-4   + pivot to the ad-controller
    stage 4   layers 1-5   + escalate privilege
    stage 5   layers 1-6   + degrade MFA, then alter the admin credentials

Design note -- **promotion needs a dwell time, not just a success rate.**
Section 7.4 promotes when success exceeds a threshold over a recent window. On its own
that is exploitable by luck: early in a stage epsilon is high, and a run of fortunate
episodes can clear a 200-episode window before red has learned anything stable. It then
arrives at the next stage with a Q-table full of noise and fails there, which reads as
"the curriculum stalled" rather than "it was promoted too early". So a stage also has a
minimum number of episodes.

Design note -- **exploration is boosted on promotion, and it has to be.**
This is not in the spec and the curriculum does not work without it. Epsilon decays over a
run; by the end of stage 2 it is near its floor. Red then arrives at stage 3 with a nearly
greedy policy and **cannot explore the new layer it has never seen** -- every action it
would need to try is one its current values rank below what already works. The curriculum
stalls at the first stage that needs a genuinely new behaviour, which is exactly section
15's "curriculum stage never promotes". Promotion therefore rewinds exploration partway
up. Note the cost: each boost temporarily *lowers* measured success, which is part of what
produces the sawtooth in the plot.

Design note -- **the curriculum belongs to the training loop, not the environment.**
``MiniCorp`` takes the stage per ``reset`` rather than owning a mutable one. That keeps
``ScenarioConfig`` frozen and serialisable, keeps the environment ignorant of training
schedules, and means an evaluation at a fixed stage is a normal reset rather than a
special mode.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Final

# Deepest active layer per stage -- PROJECT.md section 7.4's table.
STAGES: Final[tuple[int, ...]] = (2, 3, 4, 5, 6)


@dataclass(frozen=True)
class CurriculumConfig:
    """When to move red to the next stage.

    Attributes:
        stages: Deepest active layer at each stage.
        promote_threshold: Success rate over the recent window required to promote.
            Section 7.4 suggests 0.70.
        promote_window: Episodes the success rate is measured over.
        min_episodes_per_stage: Dwell time. Prevents a lucky early streak promoting red
            before it has learned anything stable; see the design note.
        epsilon_on_promote: Exploration rate to rewind to when a stage is cleared. The
            new stage contains a behaviour red has never performed, and a nearly greedy
            policy cannot find it. ``None`` disables the boost, which is worth running
            once as an ablation because the curriculum visibly stalls without it.
        max_episodes_per_stage: Give up and promote anyway. Without this a stage red
            cannot clear consumes the entire training budget and the run produces no
            information about the later stages at all -- better to reach stage 5 badly
            than to spend 50,000 episodes failing stage 3. A forced promotion is recorded
            so it can never be mistaken for a real one.
    """

    stages: tuple[int, ...] = STAGES
    promote_threshold: float = 0.70
    promote_window: int = 200
    min_episodes_per_stage: int = 600
    epsilon_on_promote: float | None = 0.40
    max_episodes_per_stage: int = 8_000


@dataclass
class StageTransition:
    """One promotion, for the sawtooth plot and the run report."""

    episode: int
    from_stage: int
    to_stage: int
    success_rate: float
    forced: bool = False


@dataclass
class Curriculum:
    """Tracks red's progress and decides when to deepen the stack.

    Example:
        >>> curriculum = Curriculum()
        >>> curriculum.max_layer          # deepest active layer right now
        2
        >>> curriculum.record(won=True, episode=0)
    """

    config: CurriculumConfig = field(default_factory=CurriculumConfig)
    index: int = 0
    episodes_at_stage: int = 0
    transitions: list[StageTransition] = field(default_factory=list)
    _recent: deque[bool] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        self._recent = deque(maxlen=self.config.promote_window)

    # ----------------------------------------------------------------------------------
    @property
    def max_layer(self) -> int:
        """Deepest layer active at the current stage -- passed to ``MiniCorp.reset``."""
        return self.config.stages[self.index]

    @property
    def stage_number(self) -> int:
        """1-based stage, as section 7.4 numbers them."""
        return self.index + 1

    @property
    def finished(self) -> bool:
        """Red is on the final stage; there is nothing left to promote to."""
        return self.index >= len(self.config.stages) - 1

    @property
    def success_rate(self) -> float:
        """Red's win rate over the recent window."""
        return sum(self._recent) / len(self._recent) if self._recent else 0.0

    # ----------------------------------------------------------------------------------
    def record(self, *, won: bool, episode: int) -> StageTransition | None:
        """Log an episode outcome and promote if the stage is cleared.

        Args:
            won: Whether red achieved the stage objective.
            episode: Global episode index, recorded on any transition.

        Returns:
            The transition if red was promoted this episode, otherwise None. The caller
            uses a non-None return to rewind exploration and to mark the plot.
        """
        self._recent.append(won)
        self.episodes_at_stage += 1

        if self.finished:
            return None

        forced = self.episodes_at_stage >= self.config.max_episodes_per_stage
        ready = (
            self.episodes_at_stage >= self.config.min_episodes_per_stage
            and len(self._recent) >= self.config.promote_window
            and self.success_rate >= self.config.promote_threshold
        )
        if not (ready or forced):
            return None

        transition = StageTransition(
            episode=episode,
            from_stage=self.stage_number,
            to_stage=self.stage_number + 1,
            success_rate=self.success_rate,
            forced=forced and not ready,
        )
        self.index += 1
        self.episodes_at_stage = 0
        self._recent.clear()
        self.transitions.append(transition)
        return transition

    # ----------------------------------------------------------------------------------
    def summary(self) -> str:
        """One line per transition, for the run report."""
        if not self.transitions:
            return "no stage transitions -- red never cleared stage 1"
        lines = []
        for t in self.transitions:
            tag = "  (FORCED -- stage not actually cleared)" if t.forced else ""
            lines.append(
                f"  ep {t.episode:>6}: stage {t.from_stage} -> {t.to_stage}  "
                f"success {t.success_rate:5.1%}{tag}"
            )
        return "\n".join(lines)
