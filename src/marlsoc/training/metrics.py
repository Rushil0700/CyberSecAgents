"""Per-episode metrics: the row CLAUDE.md 3.8 requires, and the curves it feeds.

Amendment 3.8 is explicit that metrics cannot be retrofitted -- the environment records
detection and containment times as they happen, and this module only collects them. A
row is written for **every** episode, not sampled, because the interesting questions are
about variance and about where a curve changes slope, and a sampled log answers neither.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class EpisodeRecord:
    """One episode. The columns are CLAUDE.md 3.8's row, in its order.

    ``mttd`` and ``mttc`` are optional because an episode where nothing was ever
    compromised has no meaningful detection time, and averaging a zero in for those would
    flatter the defender badly. ``None`` survives to the CSV as an empty cell.
    """

    episode: int
    phase: str
    learner: str
    red_return: float
    blue_return: float
    outcome: str
    steps: int
    hosts_compromised: int
    layers_breached: int
    mttd: int | None
    mttc: int | None
    false_positives: int
    honeypot_hits: int
    epsilon: float

    @property
    def red_won(self) -> bool:
        return self.outcome == "red_win"

    @property
    def blue_won(self) -> bool:
        return self.outcome == "blue_win"


COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(EpisodeRecord))


class MetricsLog:
    """An append-only list of episode records, plus the summaries training prints."""

    def __init__(self) -> None:
        self.rows: list[EpisodeRecord] = []

    def append(self, record: EpisodeRecord) -> None:
        self.rows.append(record)

    def __len__(self) -> int:
        return len(self.rows)

    # ----------------------------------------------------------------------------------
    def column(self, name: str, last: int | None = None) -> list[Any]:
        rows = self.rows[-last:] if last else self.rows
        return [getattr(r, name) for r in rows]

    def rolling(self, name: str, window: int = 100) -> np.ndarray:
        """Moving average of a column.

        Raw per-episode returns in an adversarial, stochastic game are far too noisy to
        read. Every learning curve in the report is a rolling mean, and saying which
        window was used is part of reporting it honestly.
        """
        values = np.array([float(v) for v in self.column(name)], dtype=np.float64)
        if values.size == 0:
            return values
        window = min(window, values.size)
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="valid")

    def rate(self, outcome: str, last: int | None = None) -> float:
        """Fraction of episodes ending in ``outcome``. The headline number in section 9."""
        outcomes = self.column("outcome", last)
        return outcomes.count(outcome) / len(outcomes) if outcomes else 0.0

    def mean(self, name: str, last: int | None = None) -> float:
        """Mean of a column, skipping ``None`` -- see ``EpisodeRecord``."""
        values = [v for v in self.column(name, last) if v is not None]
        return fmean(values) if values else float("nan")

    # ----------------------------------------------------------------------------------
    def summary(self, last: int = 500) -> str:
        """The live console line, printed every N episodes during training."""
        n = min(last, len(self.rows))
        if n == 0:
            return "no episodes yet"
        return (
            f"ep {self.rows[-1].episode:>6}  "
            f"eps {self.rows[-1].epsilon:.3f}  "
            f"| last {n}: "
            f"red_win {self.rate('red_win', n):5.1%}  "
            f"blue_win {self.rate('blue_win', n):5.1%}  "
            f"draw {self.rate('draw', n):5.1%}  "
            f"| layers {self.mean('layers_breached', n):4.2f}  "
            f"blue_R {self.mean('blue_return', n):8.1f}  "
            f"FP {self.mean('false_positives', n):4.1f}  "
            f"MTTD {self.mean('mttd', n):5.1f}"
        )

    def to_csv(self, path: Path) -> Path:
        """Write every row. One file per run, beside its curves."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(asdict(row))
        return path

    @classmethod
    def from_rows(cls, rows: Iterable[EpisodeRecord]) -> MetricsLog:
        log = cls()
        for row in rows:
            log.append(row)
        return log
