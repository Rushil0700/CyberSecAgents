"""``O_i(s)`` -- what each agent actually sees, and how that becomes a Q-table index.

PROJECT.md section 6 says the true global state ``S`` is observed by nobody; each agent
receives ``o_i = O_i(s)``. This module is every ``O_i``. It is the last file between
ground truth and the agents, so it is the file where partial observability is either
honoured or quietly lost.

What blue is allowed to read
----------------------------
Exactly two things: the accumulated alert evidence for hosts in its own zones, and which
of those hosts it has itself isolated. Nothing else. In particular it never reads
``state.true_status`` for CLEAN versus COMPROMISED -- that distinction is inferred from
evidence, which is the entire point of CLAUDE.md amendment 3.1. ISOLATED is the one
ground-truth fact blue may have, and only because blue caused it: an agent knows what it
switched off.

Design note -- **the fourth status is a confidence level, not a fact.**
Section 5.1 wants four observed statuses and its 4**5 * 3 = 3072 arithmetic depends on
it, but truth only has three (``state.py``). So the fourth is manufactured as a graded
belief over the same evidence::

    ISOLATED     true status is isolated        (blue did it, so blue knows)
    COMPROMISED  accumulated alerts >= 2.5      (confirmed)
    SUSPICIOUS   accumulated alerts >= 1.5      (something is off)
    CLEAN        otherwise

Read that against the steady-state accumulation in ``detection.py``. An idle compromised
host settles at 1.00 -- it never even reaches SUSPICIOUS. One that acts reaches 1.53 to
4.20 depending on how loud the action was. So a stealthy attacker shows up as SUSPICIOUS
at worst and **never** as COMPROMISED, and blue has to choose between isolating on
suspicion alone -- paying the availability cost in section 5.3 -- or waiting for a
confirmation a careful red will never provide. Meanwhile a clean host can cross 2.5 on a
run of bad luck in roughly one episode in forty, so "confirmed" is not a synonym for
"true" either. That is the security/availability tradeoff at its sharpest, and it falls
out of the detection calibration rather than being bolted on.

Design note -- **observations are tuples, and encoding is separate.**
Each ``O_i`` returns a tuple of small integers, and ``encode`` turns a tuple into a flat
index for a NumPy Q-table. Keeping them apart means the observation is readable in a
debugger and in a log line, while the Q-table still gets the dense integer it needs. It
also means the encoding is testable as a bijection, which matters: a non-injective
encoding silently merges two different situations into one Q-table row, and the only
symptom is a policy that looks confused in exactly one part of the state space.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

from marlsoc.env import detection as det
from marlsoc.env import topology as topo
from marlsoc.env.detection import DetectionConfig
from marlsoc.env.state import EpisodeState, HostStatus
from marlsoc.env.topology import Zone


class ObservedStatus(IntEnum):
    """What a defender believes about a host. Four values, per PROJECT.md section 5.1.

    ``IntEnum`` because these are digits of a mixed-radix index; see ``encode``.
    """

    CLEAN = 0
    SUSPICIOUS = 1
    COMPROMISED = 2
    ISOLATED = 3


N_STATUSES: Final[int] = len(ObservedStatus)

# Evidence needed before a defender reads a host as confirmed-compromised rather than
# merely suspicious. Above detection's suspicion_threshold of 1.5; see the design note.
CONFIRM_THRESHOLD: Final[float] = 2.5

# Buckets for red's foothold count (section 5.2): 0, 1, 2-3, 4+. Bucketed rather than
# counted because the strategic difference between four footholds and five is nil, while
# the difference between zero and one is everything -- and an unbucketed count would
# multiply red's state space by the host count for no decision-making benefit.
FOOTHOLD_BUCKETS: Final[tuple[int, ...]] = (0, 1, 2, 4)
N_FOOTHOLD_BUCKETS: Final[int] = len(FOOTHOLD_BUCKETS)

# Same idea for R_scout's discovered-host count.
DISCOVERY_BUCKETS: Final[tuple[int, ...]] = (0, 1, 3, 6)
N_DISCOVERY_BUCKETS: Final[int] = len(DISCOVERY_BUCKETS)


def _bucket(value: int, edges: tuple[int, ...]) -> int:
    """Index of the highest bucket edge at or below ``value``."""
    idx = 0
    for i, edge in enumerate(edges):
        if value >= edge:
            idx = i
    return idx


# --------------------------------------------------------------------------------------
# Blue
# --------------------------------------------------------------------------------------
def observed_status(
    state: EpisodeState,
    host: str,
    cfg: DetectionConfig = det.DEFAULT,
) -> ObservedStatus:
    """What the zone's defender believes about ``host``.

    Note what is *not* consulted: ``state.status(host)`` is read only to recognise
    ISOLATED. Whether a live host is clean or compromised is inferred from alerts alone,
    so this function can be -- and regularly is -- wrong in both directions.
    """
    if state.status(host) is HostStatus.ISOLATED:
        return ObservedStatus.ISOLATED

    evidence = state.alerts.get(host, 0.0)
    if evidence >= CONFIRM_THRESHOLD:
        return ObservedStatus.COMPROMISED
    if evidence >= cfg.suspicion_threshold:
        return ObservedStatus.SUSPICIOUS
    return ObservedStatus.CLEAN


def blue_observation(
    state: EpisodeState,
    agent: str,
    cfg: DetectionConfig = det.DEFAULT,
) -> tuple[int, ...]:
    """``O_i(s)`` for a blue agent: a believed status per defended host, then the zone
    alert level.

    The host ordering comes from ``topology.defended_hosts`` and is therefore stable
    across reset, training and evaluation. If it were not, every Q-table lookup after a
    reordering would be reading a different host's status and the learned policy would
    be silently scrambled.

    For an agent spanning two zones (``B_dmz`` holds Edge and DMZ) the alert level is
    the maximum over its zones -- an agent responsible for two segments should react to
    the worse one, and taking a maximum keeps the state vector at one alert digit rather
    than one per zone.
    """
    statuses = tuple(
        int(observed_status(state, h.name, cfg))
        for h in topo.defended_hosts(agent)
    )
    alert = max(
        det.zone_alert_level(state, zone, cfg)
        for zone in topo.DEFENDER_ZONES[agent]
    )
    return statuses + (alert,)


def blue_dims(agent: str) -> tuple[int, ...]:
    """Radix of each digit of ``blue_observation`` -- the shape of the agent's Q-table."""
    n_hosts = len(topo.defended_hosts(agent))
    return (N_STATUSES,) * n_hosts + (det.N_ALERT_LEVELS,)


# --------------------------------------------------------------------------------------
# Red
# --------------------------------------------------------------------------------------
def red_breach_observation(state: EpisodeState) -> tuple[int, ...]:
    """``O_i(s)`` for ``R_breach``, exactly the vector in PROJECT.md section 5.2::

        (current_zone, footholds_bucket, creds_held, privilege_escalated,
         mfa_degraded, heat_level)

    |S| = 4 * 4 * 2 * 2 * 2 * 3 = 768.

    The three flags are the layer model's layers 3, 5 and 6, read through ``state``'s
    derived properties rather than stored separately -- which is why the observation and
    the precondition model cannot disagree about whether red holds credentials.

    This is also the answer to "how did you keep tabular RL feasible with six layers?":
    each layer contributes **one binary digit**, not a new dimension of hosts. Going from
    two layers to six multiplied red's state space by 8 and blue's by nothing.
    """
    return (
        topo.ZONE_DEPTH[state.current_zone()],
        _bucket(len(state.footholds), FOOTHOLD_BUCKETS),
        int(state.creds_held),
        int(state.privilege_escalated),
        int(state.mfa_degraded),
        det.heat_level(state),
    )


RED_BREACH_DIMS: Final[tuple[int, ...]] = (4, N_FOOTHOLD_BUCKETS, 2, 2, 2,
                                           det.N_ALERT_LEVELS)


def red_scout_observation(state: EpisodeState) -> tuple[int, ...]:
    """``O_i(s)`` for ``R_scout`` (section 4.1): reachability, discoveries, heat.

        (current_zone, discovered_bucket, perimeter_breached, heat_level)

    |S| = 4 * 4 * 2 * 3 = 96.

    The scout carries ``perimeter_breached`` rather than the full layer flags because
    Layer 1 is the only layer it can act on -- a scout that observed the credential and
    privilege flags would be carrying state it has no action to influence, which inflates
    the Q-table without changing any decision.
    """
    from marlsoc.env.layers import Layer

    return (
        topo.ZONE_DEPTH[state.current_zone()],
        _bucket(len(state.discovered), DISCOVERY_BUCKETS),
        int(state.layers.is_breached(Layer.PERIMETER)),
        det.heat_level(state),
    )


RED_SCOUT_DIMS: Final[tuple[int, ...]] = (4, N_DISCOVERY_BUCKETS, 2, det.N_ALERT_LEVELS)


# --------------------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------------------
def encode(obs: tuple[int, ...], dims: tuple[int, ...]) -> int:
    """Flatten an observation tuple to a single Q-table row index (mixed radix).

    This is ordinary positional numbering with a different base per digit::

        index = sum(obs[i] * prod(dims[i+1:]))

    Args:
        obs: One observation tuple.
        dims: Radix of each digit; ``blue_dims(agent)`` or ``RED_*_DIMS``.

    Returns:
        An integer in ``[0, prod(dims))``.

    Raises:
        ValueError: on a digit out of range. This is worth failing loudly on: an
            out-of-range digit that silently wrapped would merge two distinct situations
            into one Q-table row, and the only symptom would be a policy that looks
            confused in exactly one region of the state space.
    """
    if len(obs) != len(dims):
        raise ValueError(f"observation has {len(obs)} digits, dims has {len(dims)}")
    index = 0
    for value, radix in zip(obs, dims, strict=True):
        if not 0 <= value < radix:
            raise ValueError(f"digit {value} out of range for radix {radix}")
        index = index * radix + value
    return index


def decode(index: int, dims: tuple[int, ...]) -> tuple[int, ...]:
    """Inverse of ``encode``. Used by tests and by the dashboard to name a Q-table row."""
    digits: list[int] = []
    for radix in reversed(dims):
        digits.append(index % radix)
        index //= radix
    return tuple(reversed(digits))


def n_states(dims: tuple[int, ...]) -> int:
    """Number of distinct observations -- the Q-table row count, for budget checks."""
    total = 1
    for radix in dims:
        total *= radix
    return total
