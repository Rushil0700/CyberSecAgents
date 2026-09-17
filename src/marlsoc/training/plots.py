"""Learning curves. CLAUDE.md 3.8: per-run CSV and curve PNGs under artifacts/runs/<id>/.

Design note -- **every curve is a rolling mean, and the window is stated on the axis.**
Per-episode returns in a stochastic adversarial game are far too noisy to read: the
underlying trend is invisible under the variance. Smoothing is therefore not cosmetic, it
is what makes the plot legible -- but a smoothed curve that does not say how much it was
smoothed is not an honest plot, so the window goes in the axis label rather than being
mentioned in a caption somewhere else.

Design note -- **matplotlib only, no seaborn.**
One fewer dependency, and the default style is adequate for a report. Nothing here is
imported by the environment or the agents, so a machine without matplotlib can still run
training and write the CSV.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # no display in CI; must be set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402

from marlsoc.training.metrics import MetricsLog  # noqa: E402

WINDOW = 200


def _panel(ax, log: MetricsLog, column: str, label: str, window: int, color: str):
    curve = log.rolling(column, window)
    if curve.size == 0:
        return
    ax.plot(range(window - 1, window - 1 + curve.size), curve, color=color, lw=1.4)
    ax.set_ylabel(label)
    ax.grid(alpha=0.25, lw=0.5)


def learning_curve(
    log: MetricsLog,
    path: Path,
    *,
    title: str = "Phase 2 — single defender vs scripted attacker",
    window: int = WINDOW,
) -> Path:
    """The four-panel figure: blue return, attacker success, layers breached, false positives.

    These four together answer the question a single reward curve cannot: *how* did blue
    improve? A return that rises while false positives also rise means blue is buying
    security with availability, which is section 5.3's whole argument and would be
    invisible on a reward plot alone.
    """
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

    _panel(axes[0], log, "blue_return", "blue return", window, "#1f77b4")
    axes[0].set_title(title)

    # Attacker success needs converting to a 0/1 series before smoothing.
    wins = MetricsLog.from_rows(log.rows)
    success = [1.0 if r.outcome == "red_win" else 0.0 for r in log.rows]
    if success:
        import numpy as np
        w = min(window, len(success))
        curve = np.convolve(np.array(success), np.ones(w) / w, mode="valid")
        axes[1].plot(range(w - 1, w - 1 + curve.size), curve, color="#d62728", lw=1.4)
    axes[1].set_ylabel("attacker success")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.25, lw=0.5)

    _panel(axes[2], log, "layers_breached", "layers breached", window, "#9467bd")
    axes[2].set_ylim(0, 6.2)

    _panel(axes[3], log, "false_positives", "false positives", window, "#ff7f0e")
    axes[3].set_xlabel(f"episode   (rolling mean, window = {window})")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def comparison(
    logs: dict[str, MetricsLog],
    path: Path,
    *,
    column: str = "blue_return",
    title: str = "",
    window: int = WINDOW,
) -> Path:
    """Several runs on one axis -- the shape every section 9 sweep takes.

    Used for algorithm comparisons, the curriculum on/off experiment, and the
    availability-cost before/after.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, log in logs.items():
        curve = log.rolling(column, window)
        if curve.size:
            ax.plot(range(window - 1, window - 1 + curve.size), curve, lw=1.4, label=label)
    ax.set_xlabel(f"episode   (rolling mean, window = {window})")
    ax.set_ylabel(column.replace("_", " "))
    ax.set_title(title or column.replace("_", " "))
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
