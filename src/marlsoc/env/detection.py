"""The noisy alert process: how blue finds out anything at all.

CLAUDE.md amendment 3.1 splits ground truth from what a defender perceives. This module
is that split. It reads ``EpisodeState`` -- which is ground truth -- and produces alert
evidence; ``env/observations.py`` then turns evidence into the status a blue agent
actually sees. The two files are kept separate on purpose: observations must never be
able to reach past the evidence to the truth behind it, and the easiest way to guarantee
that is for the module that builds observations to receive alerts rather than state.

The model
---------
Every step, every host on the network draws an alert:

    P(alert | compromised) = clamp(base + gain * action_noise, .., max_detect_rate)
    P(alert | clean)       = false_alert_rate
    P(alert | honeypot touched) = honeypot_detect_rate   (near-certain, by design)
    P(alert | isolated)    = 0                            (off the network)

Alerts accumulate per host and decay geometrically. ``observations.py`` thresholds the
accumulation into SUSPICIOUS.

Design note -- **the base rate is what makes blue's job real.**
A 2% per-step false-alert rate looks negligible until you multiply it by twelve clean
hosts: roughly 0.24 spurious alerts per step against roughly 0.4 genuine ones from a
single compromised host. Those are the same order of magnitude. Blue therefore cannot
learn "isolate whatever alerted" -- it has to learn that a *sustained* signal on one host
means something different from a scattered one across many. This is the base rate
fallacy, and encoding it is what turns the section 5.3 availability tradeoff from a
sentence in the report into a property of the environment. Set ``false_alert_rate`` to
zero and blue converges on the degenerate isolate-everything policy within a few hundred
episodes; that ablation is worth running once as a result.

Design note -- **detection is never certain.**
``max_detect_rate`` is 0.85, not 1.0. If the loudest action were always caught, red's
stealth choice would have a known answer and there would be no cliff for section 7.2's
cliff-walking argument to work with -- SARSA only beats Q-Learning for the attacker
because exploring near detection is genuinely risky rather than fatal.

Design note -- **alerts decay, and they have to.**
Without decay, at a 2% base rate over T=250 steps every host crosses any fixed threshold
eventually, and late-episode blue observes a uniformly suspicious network carrying no
information. Geometric decay gives blue a fading memory, and it is also what makes "go
quiet and let the heat die down" a strategy red can discover rather than one we script.

Design note -- **this module never mutates the layer model or statuses.**
It writes ``state.alerts`` and stamps ``mark_detected``. Detection observes; it does not
contain. Containment is a blue *action*, which is the entire point of the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from marlsoc.env import topology as topo
from marlsoc.env.state import EpisodeState, HostStatus
from marlsoc.env.topology import Zone


@dataclass(frozen=True)
class DetectionConfig:
    """Tunable parameters of the alert process.

    These are the numbers an experiment sweeps, so they live in one frozen object rather
    than as module constants -- a run's config can then be serialised into the artifacts
    directory beside its curves, which is what makes a result reproducible six weeks
    later when someone asks why this curve looks different from that one.

    Attributes:
        false_alert_rate: P(alert) for a clean host, per step. The base rate.
        base_detect_rate: P(alert) for a compromised host red did nothing loud on.
            Non-zero because a foothold generates background activity, which is why
            red cannot simply sit still forever once inside.
        noise_gain: Additional detection probability per alert point of red's action
            noise. This is the coefficient that prices stealth.
        max_detect_rate: Ceiling. Detection is never certain; see the design note.
        honeypot_detect_rate: P(alert) when red touches a live honeypot. A decoy that
            might fail to notice being touched is not a decoy.
        decay: Multiplier applied to accumulated alerts each step, giving blue a fading
            memory.
        suspicion_threshold: Accumulated alert points at which a host reads SUSPICIOUS.
            Deliberately above 1.0. At exactly 1.0 a single alert trips it, "suspicious"
            collapses into "alerted this step", and the sustained-versus-scattered
            distinction the base rate note above is entirely about stops existing. At
            1.5 one isolated alert is not enough but two within a couple of steps are
            (1.0, then 0.85 + 1.0 = 1.85).
        zone_alert_cuts: Accumulated zone-wide points separating low / medium / high
            alert levels -- the third component of every blue agent's state vector.
    """

    false_alert_rate: float = 0.02
    base_detect_rate: float = 0.15
    noise_gain: float = 0.08
    max_detect_rate: float = 0.85
    honeypot_detect_rate: float = 0.95
    decay: float = 0.85
    suspicion_threshold: float = 1.5
    zone_alert_cuts: tuple[float, float] = (1.0, 3.0)


DEFAULT: Final[DetectionConfig] = DetectionConfig()

# Alert level codes; blue's state vector carries this, not the raw float.
ALERT_LOW: Final[int] = 0
ALERT_MEDIUM: Final[int] = 1
ALERT_HIGH: Final[int] = 2
N_ALERT_LEVELS: Final[int] = 3


def detection_prob(
    state: EpisodeState,
    host: str,
    action_noise: float,
    cfg: DetectionConfig = DEFAULT,
) -> float:
    """P(this host emits an alert this step).

    Args:
        state: Ground truth. Read only -- this function decides nothing about it.
        host: Host name.
        action_noise: Alert points generated by red's action against this host this
            step, from ``Host.noise`` and ``LayerSpec.noise``. Zero if red did not act
            here, which is the difference between a noisy breach and a quiet one.
        cfg: Detection parameters.

    Returns:
        A probability in [0, 1].
    """
    status = state.status(host)

    if status is HostStatus.ISOLATED:
        return 0.0  # off the network; nothing to observe

    if host in state.honeypots_live and action_noise > 0.0:
        return cfg.honeypot_detect_rate

    if status is HostStatus.COMPROMISED:
        return min(cfg.base_detect_rate + cfg.noise_gain * action_noise,
                   cfg.max_detect_rate)

    # Clean. The base rate -- small per host, substantial across the network.
    return cfg.false_alert_rate


def step_alerts(
    state: EpisodeState,
    rng: np.random.Generator,
    action_noise: dict[str, float] | None = None,
    cfg: DetectionConfig = DEFAULT,
) -> frozenset[str]:
    """Advance the alert process one step, in place.

    Decays every host's accumulated alerts, draws a fresh alert per host, and stamps
    mean-time-to-detection the first time a *genuinely* compromised host crosses the
    suspicion threshold. A clean host crossing the threshold is a false positive and
    deliberately does **not** stamp detection -- MTTD must measure blue finding the
    attacker, not blue getting startled.

    Args:
        state: Mutated: ``state.alerts`` and possibly ``state.detected_step``.
        rng: Seeded NumPy generator. Passed in rather than created here so an episode
            is exactly reproducible from a seed (CLAUDE.md 3.7).
        action_noise: Per-host alert points from red's action this step.
        cfg: Detection parameters.

    Returns:
        The hosts that alerted this step. Returned rather than only recorded because
        the logger and the LLM copilot both want the per-step event, not the running
        total.
    """
    noise = action_noise or {}
    fired: set[str] = set()

    for host in list(state.alerts):
        state.alerts[host] *= cfg.decay

        p = detection_prob(state, host, noise.get(host, 0.0), cfg)
        if p > 0.0 and rng.random() < p:
            state.alerts[host] += 1.0
            fired.add(host)

    # MTTD stamps only on a true positive; see the docstring.
    for host in fired:
        if (state.status(host) is HostStatus.COMPROMISED
                and state.alerts[host] >= cfg.suspicion_threshold):
            state.mark_detected()
            break

    return frozenset(fired)


def is_suspicious(state: EpisodeState, host: str, cfg: DetectionConfig = DEFAULT) -> bool:
    """Whether accumulated evidence makes this host read as SUSPICIOUS to its defender.

    This is a statement about *evidence*, not about truth: a clean host can be
    suspicious (a false positive waiting to happen) and a compromised host can be
    perfectly quiet (the stealth red is trying to achieve).
    """
    return state.alerts.get(host, 0.0) >= cfg.suspicion_threshold


def zone_alert_level(
    state: EpisodeState,
    zone: Zone,
    cfg: DetectionConfig = DEFAULT,
) -> int:
    """Low / medium / high alert for a zone -- the last element of blue's state vector.

    Summing the zone's accumulated alerts rather than counting flagged hosts is
    deliberate: it lets one screaming host and three murmuring ones produce different
    levels, which is exactly the distinction blue has to learn to make under the base
    rate fallacy.
    """
    total = sum(
        state.alerts.get(h.name, 0.0)
        for h in topo.hosts_in(zone)
        if state.status(h.name) is not HostStatus.ISOLATED
    )
    low, high = cfg.zone_alert_cuts
    if total >= high:
        return ALERT_HIGH
    if total >= low:
        return ALERT_MEDIUM
    return ALERT_LOW


def heat_level(state: EpisodeState, cuts: tuple[float, float] = (3.0, 8.0)) -> int:
    """Red's own view of how much attention it has drawn: the ``heat_level`` feature
    in PROJECT.md section 5.2, bucketed to three values.

    Red observes its accumulated action noise, not blue's alert totals. Letting red read
    blue's alerts would hand it perfect information about the defender's beliefs and
    collapse the partial observability the Markov Game formulation in section 6 rests on.
    """
    low, high = cuts
    if state.heat >= high:
        return ALERT_HIGH
    if state.heat >= low:
        return ALERT_MEDIUM
    return ALERT_LOW


def decay_heat(state: EpisodeState, cfg: DetectionConfig = DEFAULT) -> None:
    """Red's heat fades at the same rate blue's memory does.

    Same rate on purpose: it means "go quiet and wait" is worth the same to both sides,
    so ``wait`` is a genuine strategic option for red rather than a dominated one.
    """
    state.heat *= cfg.decay
