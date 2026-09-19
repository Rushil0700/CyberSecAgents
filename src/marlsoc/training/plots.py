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


def curriculum_curve(
    log: MetricsLog,
    transitions,
    path: Path,
    *,
    title: str = "Curriculum learning — red vs a static defence",
    window: int = 200,
) -> Path:
    """The sawtooth. PROJECT.md section 7.4 calls this the report's best figure.

    Success rate climbs, a layer is switched on, it drops, it climbs again. Two things
    make the drop legible rather than confusing, and both are marked on the plot: the
    stage transition itself, and the exploration boost that accompanies it -- part of
    each drop is red exploring again rather than red having got worse.

    The layer-depth panel underneath is section 9's "money graph" -- it degrades
    gracefully where a binary win/lose curve does not, and it shows red pushing deeper
    stage by stage.
    """
    import numpy as np

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    success = np.array([1.0 if r.outcome == "red_win" else 0.0 for r in log.rows])
    if success.size:
        w = min(window, success.size)
        curve = np.convolve(success, np.ones(w) / w, mode="valid")
        axes[0].plot(range(w - 1, w - 1 + curve.size), curve, color="#d62728", lw=1.5)
    axes[0].set_ylabel("attacker success")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25, lw=0.5)

    _panel(axes[1], log, "layers_breached", "layers breached", window, "#9467bd")
    axes[1].set_ylim(0, 6.4)
    axes[1].set_xlabel(f"episode   (rolling mean, window = {window})")

    for transition in transitions:
        for ax in axes:
            ax.axvline(transition.episode, color="#444", ls="--", lw=0.9, alpha=0.7)
        axes[0].annotate(
            f"stage {transition.to_stage}\nlayers 1-{transition.to_stage + 1}",
            xy=(transition.episode, 1.0), xytext=(4, -12),
            textcoords="offset points", fontsize=8,
            color="#b00" if transition.forced else "#444",
        )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def arms_race(
    phases,
    path: Path,
    *,
    title: str = "Alternating training — the arms race",
    iql_phases=None,
) -> Path:
    """CLAUDE.md 3.8's cross-phase plot: both teams' scores across training phases.

    Each point is a **greedy** evaluation taken after a training phase, for the reason
    3.23 gives -- a training win rate measures the exploring policy, not the one being
    deployed. Phases where blue trained are shaded, so the expected sawtooth is readable
    as cause and effect: attacker success should fall while blue trains and rise while
    red trains, with both teams improving in absolute terms round on round.

    Passing ``iql_phases`` overlays the simultaneous-learning control (3.4). The claim
    that plot is making is not "IQL is worse" but "IQL does not settle" -- so what to
    look at is the *amplitude* of the oscillation rather than its level. The number
    underneath it is ``alternating.instability``.
    """
    import numpy as np

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = np.arange(1, len(phases) + 1)

    axes[0].plot(x, [p.attacker_success for p in phases], "o-",
                 color="#d62728", lw=1.8, label="alternating")
    axes[0].set_ylabel("attacker success")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title(title)

    axes[1].plot(x, [p.blue_return for p in phases], "o-",
                 color="#1f77b4", lw=1.8, label="alternating")
    axes[1].set_ylabel("blue return")
    axes[1].set_xlabel("training phase")

    if iql_phases:
        xi = np.linspace(1, len(phases), num=len(iql_phases))
        axes[0].plot(xi, [p.attacker_success for p in iql_phases], "s--",
                     color="#7f7f7f", lw=1.4, alpha=0.9, label="IQL (simultaneous)")
        axes[1].plot(xi, [p.blue_return for p in iql_phases], "s--",
                     color="#7f7f7f", lw=1.4, alpha=0.9, label="IQL (simultaneous)")

    # Shade the phases in which blue was the learner, so the sawtooth reads as cause
    # and effect rather than as noise.
    for i, phase in enumerate(phases, start=1):
        if phase.trained == "blue":
            for ax in axes:
                ax.axvspan(i - 0.5, i + 0.5, color="#1f77b4", alpha=0.07, lw=0)

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc="best", fontsize=8)

    axes[0].text(0.01, 0.97, "shaded = blue trained this phase", transform=axes[0].transAxes,
                 fontsize=8, va="top", color="#1f77b4")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
