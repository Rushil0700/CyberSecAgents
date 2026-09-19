"""``A_i`` -- per-agent action sets and the legality mask over them.

PROJECT.md section 3.4: "actions whose preconditions are unmet are masked out of the
agent's choice set. This keeps the effective branching factor small and stops the agent
wasting exploration on impossible moves." This module is both halves -- what the actions
are, and which of them are legal right now.

Masking is split across two files by design. ``layers.py`` answers "is this layer
attemptable?" using nothing but layer flags, so it is testable on plain booleans. Here we
add the half that needs episode state: do I hold a host with network reach to the target,
is the target still on the network, is it already mine. A legal action is one where both
halves say yes.

Design note -- **the action space is built per agent from the topology.**
CLAUDE.md amendment 3.5: section 5.1's flat "|A| = 12" assumed every zone had five hosts.
They do not, so each agent's action set is generated from the hosts it actually defends.
Never assume a shared action space; an agent indexing another agent's action list is a
silent mis-mapping that produces a policy doing the wrong thing with total confidence.

Design note -- **R_scout acts on zones, not on hosts.**
Section 4.1 lists ``probe(host)``. But red's state in section 5.2 carries only a
*bucketed count* of discoveries -- it cannot tell one undiscovered host from another. An
action space finer than the state space is capacity the Q-table can never use: the agent
would be choosing between targets it has no way to distinguish, and those Q-values would
average into noise. So the scout chooses a zone, and which host it finds there is the
environment's business.

Design note -- **exploit and lateral_move are given a real difference.**
Section 4.1 lists both and maps them to Layers 2 and 4, but as specified they are the
same mechanic under two names. Here they differ in a way that matters strategically and
that the mask can express:

    exploit(h)       compromise an ordinary host in a zone red is already in -- or its
                     very first foothold, from outside. *Widen.*
    lateral_move(h)  compromise a host sitting behind a segmentation boundary: one in a
                     deeper zone, **or the pivot**. *Advance.*

The pivot is included even though it shares the corporate zone with hosts red may
already hold, because Layer 4 is micro-segmentation *inside* Corp -- reaching
``ad-controller`` is a lateral move whether or not the zone changes. Leaving it out was
a real bug caught by a stage-by-stage win-rate check: red took the pivot with
``exploit`` as ordinary intra-zone widening and walked straight through Layer 4 without
ever breaching it, plateauing at depth 3 in every curriculum stage. ``exploit`` is
therefore barred from the pivot outright.

That keeps the one-to-one action-to-layer mapping in section 4.1 honest rather than
having two names for one move, and it makes "spread out in the DMZ before pushing in" a
strategy red can express.

Design note -- **blue's reinforce action differs per agent.**
Section 4.2 gives each defender the one action that restores *its own* layers:
``tighten_ratelimit`` for L1, ``rotate_credentials`` for L3, ``harden_mfa`` for L6. That
is what differentiates the defenders by the layers they hold rather than only by
geography, and it is why their action sets are not interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np

from marlsoc.env import topology as topo
from marlsoc.env.layers import Layer
from marlsoc.env.state import EpisodeState, HostStatus
from marlsoc.env.topology import Role, Zone


class Verb(str, Enum):
    """Every action verb in the game, both teams."""

    # --- red: recon (R_scout) ---
    SCAN = "scan"                  # fast, loud, finds more
    SLOW_SCAN = "slow_scan"        # quiet; the only way past Layer 1

    # --- red: intrusion (R_breach) ---
    EXPLOIT = "exploit"                      # Layer 2; widen within a zone
    LATERAL_MOVE = "lateral_move"            # Layer 4; advance into a deeper zone
    STEAL_CREDENTIALS = "steal_credentials"  # Layer 3
    ESCALATE_PRIVILEGE = "escalate_privilege"  # Layer 5
    DEGRADE_MFA = "degrade_mfa"              # Layer 6
    ALTER_CREDENTIALS = "alter_credentials"  # the win

    # --- blue ---
    BLOCK = "block"                          # cut one inbound path, cheap, reversible
    ISOLATE = "isolate"                      # remove the host entirely, costs availability
    TIGHTEN_RATELIMIT = "tighten_ratelimit"  # restore Layer 1 (B_dmz)
    ROTATE_CREDENTIALS = "rotate_credentials"  # restore Layer 3 (B_corp)
    HARDEN_MFA = "harden_mfa"                # restore Layer 6 (B_secure)
    DEPLOY_HONEYPOT = "deploy_honeypot"
    NOOP = "noop"

    # --- both ---
    WAIT = "wait"


@dataclass(frozen=True)
class Action:
    """One concrete choice: a verb, optionally aimed at a host or a zone.

    Frozen and hashable so an action can key a dictionary and be compared by value. The
    Q-table indexes actions by position in the agent's action tuple; this object exists
    so logs, tests and the LLM advisor can talk about actions by name rather than by
    integer, which is the difference between a readable episode trace and a column of
    numbers.
    """

    verb: Verb
    host: str | None = None
    zone: Zone | None = None

    def __str__(self) -> str:
        target = self.host or (self.zone.value if self.zone else None)
        return f"{self.verb.value}({target})" if target else self.verb.value


# --------------------------------------------------------------------------------------
# Which hosts red can aim at
# --------------------------------------------------------------------------------------
# The two gate hosts are excluded: they have exploit_prob 0.0 and are beaten by their own
# verbs (slow_scan, degrade_mfa), so offering exploit(edge-gateway) would be offering an
# action that can never succeed -- the exact waste masking exists to remove. The logger
# is excluded because it is a receive-only sink and never a target.
ATTACKABLE: Final[tuple[str, ...]] = tuple(
    h.name for h in topo.HOSTS
    if not h.is_gate and h.role is not Role.LOG_SINK
)

# Each targeted verb gets its own target list rather than reusing ATTACKABLE, because a
# blanket list produces actions that can never be legal in any reachable state. Those are
# not harmless: an action that is never legal is never taken, so it is never updated, so
# it keeps its optimistic 0.0 initialisation forever, and it costs a column in every
# Q-table. This was found by decoding a trained table -- the "best" action in most states
# had Q exactly 0.00 beside a spread of 120, because it was one the agent had never been
# allowed to try.
#
# exploit widens within a held zone, and is barred from the pivot (Layer 4 guards it), so
# the pivot can never be an exploit target.
EXPLOIT_TARGETS: Final[tuple[str, ...]] = tuple(
    name for name in ATTACKABLE if topo.BY_NAME[name].role is not Role.PIVOT
)

# lateral_move advances into a strictly deeper zone, or onto the pivot. That makes the
# valid targets exactly the **destinations of cross-zone edges**, plus the pivot -- not
# every deep host. ``fileserver`` is in Corp but is reachable only from inside Corp, so
# red must already hold a Corp foothold to touch it, and at that point nothing in Corp is
# "deeper" than what it holds: the action could never be legal. The single DMZ-to-Corp
# edge is ``reverse-proxy -> intranet``, which is why ``intranet`` qualifies and the rest
# of the corporate zone does not. Both honeypots are likewise reachable only from within
# their own zone.
LATERAL_TARGETS: Final[tuple[str, ...]] = tuple(
    name for name in ATTACKABLE
    if name == topo.PIVOT_HOST
    or any(
        dst == name and topo.ZONE_DEPTH[topo.BY_NAME[dst].zone] > topo.ZONE_DEPTH[Zone.DMZ]
        for _, dst in topo.CROSS_ZONE_EDGES
    )
)

# Credentials are read from a host red already **holds**, and red can never hold a
# honeypot: touching one is recorded as an engagement and returns without compromising
# anything. So a honeypot can never be a steal target.
STEAL_TARGETS: Final[tuple[str, ...]] = tuple(
    name for name in ATTACKABLE if not topo.BY_NAME[name].is_honeypot_slot
)

RECON_ZONES: Final[tuple[Zone, ...]] = (Zone.EDGE, Zone.DMZ, Zone.CORP, Zone.SECURE)

# The layer each blue reinforcement action restores (section 5.3, "+25 restoring a
# breached layer"). Declared as data so rewards.py does not repeat the mapping.
REINFORCE_LAYER: Final[dict[Verb, Layer]] = {
    Verb.TIGHTEN_RATELIMIT: Layer.PERIMETER,
    Verb.ROTATE_CREDENTIALS: Layer.AUTH,
    Verb.HARDEN_MFA: Layer.APPROVAL,
}

AGENT_REINFORCE: Final[dict[str, Verb]] = {
    "B_dmz": Verb.TIGHTEN_RATELIMIT,
    "B_corp": Verb.ROTATE_CREDENTIALS,
    "B_secure": Verb.HARDEN_MFA,
}


# --------------------------------------------------------------------------------------
# Action space construction
# --------------------------------------------------------------------------------------
def _honeypot_slots(agent: str) -> tuple[topo.Host, ...]:
    """Honeypot slots inside an agent's zones. B_dmz has none -- honeypots live in Corp
    and Secure, where there is something worth faking."""
    return tuple(
        h for zone in topo.DEFENDER_ZONES[agent]
        for h in topo.hosts_in(zone)
        if h.is_honeypot_slot
    )


def blue_actions(agent: str) -> tuple[Action, ...]:
    """``A_i`` for a defender: block and isolate each defended host, reinforce its own
    layer, deploy a honeypot where there is a slot for one, or do nothing.

    Size is ``2n + 2``, plus one if the agent's zones contain a honeypot slot: 10 for
    B_dmz, 13 for B_corp, 9 for B_secure -- different per agent, per CLAUDE.md
    amendment 3.5.

    ``deploy_honeypot`` is omitted for ``B_dmz`` because PROJECT.md section 3.3 places
    both honeypots in Corp and Secure, so the Edge and DMZ zones have no slot to deploy
    into and the action could never be legal. This was found by decoding a trained
    Q-table: a permanently illegal action is never taken, so it is never updated, so it
    keeps its optimistic 0.0 initialisation forever and dominates any unmasked argmax
    over the row. It costs a column in the table and makes the action-space size a
    misstatement. An action that can never be legal does not belong in the space.
    """
    hosts = topo.defended_hosts(agent)
    actions = [Action(Verb.BLOCK, host=h.name) for h in hosts]
    actions += [Action(Verb.ISOLATE, host=h.name) for h in hosts]
    actions.append(Action(AGENT_REINFORCE[agent]))
    if _honeypot_slots(agent):
        actions.append(Action(Verb.DEPLOY_HONEYPOT))
    actions.append(Action(Verb.NOOP))
    return tuple(actions)


def red_scout_actions() -> tuple[Action, ...]:
    """``A_i`` for ``R_scout``: scan or slow-scan a zone, or wait. Nine actions.

    The scan/slow_scan pair *is* the Layer 1 decision: ``scan`` finds more per step but
    generates far more noise, ``slow_scan`` is the only action that gets past the rate
    limiter. Red is never told which to prefer; that is the first thing the curriculum's
    stage 1 has to teach it.
    """
    actions = [Action(Verb.SCAN, zone=z) for z in RECON_ZONES]
    actions += [Action(Verb.SLOW_SCAN, zone=z) for z in RECON_ZONES]
    actions.append(Action(Verb.WAIT))
    return tuple(actions)


def red_breach_actions() -> tuple[Action, ...]:
    """``A_i`` for ``R_breach``: the targeted verbs over every attackable host, plus the
    three untargeted layer moves, the win, and wait.

    Size is 29 -- eleven exploit targets, four lateral-move targets, ten
    steal-credentials targets and four untargeted actions. It started at 40; the eleven
    removed were all provably unsatisfiable, and a test constructs states directly to
    keep it that way. Larger than any blue action
    set, which is correct: the attacker has the initiative and therefore the wider
    choice. Each verb uses its own target list so that no column of the Q-table holds an
    action that can never be legal; see the target-list comments above.
    """
    actions: list[Action] = []
    for verb, targets in (
        (Verb.EXPLOIT, EXPLOIT_TARGETS),
        (Verb.LATERAL_MOVE, LATERAL_TARGETS),
        (Verb.STEAL_CREDENTIALS, STEAL_TARGETS),
    ):
        actions += [Action(verb, host=h) for h in targets]
    actions.append(Action(Verb.ESCALATE_PRIVILEGE))
    actions.append(Action(Verb.DEGRADE_MFA))
    actions.append(Action(Verb.ALTER_CREDENTIALS))
    actions.append(Action(Verb.WAIT))
    return tuple(actions)


ACTION_SPACES: Final[dict[str, tuple[Action, ...]]] = {
    "R_scout": red_scout_actions(),
    "R_breach": red_breach_actions(),
    **{agent: blue_actions(agent) for agent in topo.DEFENDER_ZONES},
}

# Action -> column index, per agent. Built once, because the obvious alternatives are
# both linear scans over a tuple of dataclasses: ``action in ACTION_SPACES[agent]`` and
# ``tuple.index(action)``. Called once per action per mask, that is quadratic in the
# action-space size on every single step -- 1,600 dataclass comparisons per step for
# R_breach alone, and measurably the twin's bottleneck. Section 8 budgets ~10,000
# episodes a minute; this is part of paying for it.
ACTION_INDEX: Final[dict[str, dict[Action, int]]] = {
    agent: {action: i for i, action in enumerate(space)}
    for agent, space in ACTION_SPACES.items()
}


# --------------------------------------------------------------------------------------
# Legality
# --------------------------------------------------------------------------------------
def _red_can_reach(state: EpisodeState, target: str) -> bool:
    """Red holds some live foothold with a network path to ``target``.

    Before red holds anything, only hosts the WAF fronts are reachable -- red is outside,
    at the edge, and the gateway is what it is talking to.
    """
    if state.status(target) is HostStatus.ISOLATED:
        return False
    if not state.footholds:
        return topo.can_reach(topo.WAF_HOST, target)
    return any(topo.can_reach(src, target) for src in state.footholds)


def is_legal(state: EpisodeState, agent: str, action: Action) -> bool:
    """Whether ``agent`` may take ``action`` in ``state``.

    The layer half of each answer is delegated to ``state.layers``; the reachability and
    ownership half is here. Both must agree.

    Raises:
        ValueError: if ``action`` is not in ``agent``'s action space at all. This is the
            mis-mapping the module docstring warns about -- an agent indexing another
            agent's action list -- and it must fail loudly, because the alternative is a
            defender confidently taking a move it does not possess.
    """
    if action not in ACTION_INDEX[agent]:
        raise ValueError(f"{action} is not in {agent}'s action space")

    verb = action.verb

    # ---- always available ------------------------------------------------------------
    if verb in (Verb.WAIT, Verb.NOOP):
        return True

    # ---- red: recon ------------------------------------------------------------------
    if verb in (Verb.SCAN, Verb.SLOW_SCAN):
        # Scanning a zone red has no path into tells it nothing. Depth-1 beyond the
        # deepest zone reached keeps the frontier visible without letting the scout see
        # the secure zone from the DMZ.
        assert action.zone is not None
        return topo.ZONE_DEPTH[action.zone] <= state.deepest_zone_depth() + 1

    # ---- red: targeted intrusion -----------------------------------------------------
    if verb is Verb.EXPLOIT:
        assert action.host is not None
        host = topo.BY_NAME[action.host]
        if host.role is Role.PIVOT:
            # Layer 4 guards the pivot. Allowing ordinary intra-zone widening onto it
            # would let red past that layer without breaching it -- see the docstring.
            return False
        if state.status(action.host) is not HostStatus.CLEAN:
            return False              # already owned, or off the network
        if action.host not in state.discovered:
            return False              # cannot attack what has not been found
        if not _red_can_reach(state, action.host):
            return False
        # Widen: same zone as an existing foothold, or the first foothold from outside.
        same_zone = any(topo.BY_NAME[f].zone is host.zone for f in state.footholds)
        first_foothold = not state.footholds and host.zone is Zone.DMZ
        if not (same_zone or first_foothold):
            return False
        # The first DMZ foothold *is* Layer 2, so Layer 1 must already be down.
        if first_foothold:
            return state.layers.is_satisfied(Layer.PERIMETER)
        return True

    if verb is Verb.LATERAL_MOVE:
        assert action.host is not None
        host = topo.BY_NAME[action.host]
        if state.status(action.host) is not HostStatus.CLEAN:
            return False
        if action.host not in state.discovered:
            return False
        if not _red_can_reach(state, action.host):
            return False
        # Advance: behind a segmentation boundary -- a deeper zone, or the pivot.
        if not state.footholds:
            return False
        deepest_held = max(topo.ZONE_DEPTH[topo.BY_NAME[f].zone] for f in state.footholds)
        deeper = topo.ZONE_DEPTH[host.zone] > deepest_held
        if not (deeper or host.role is Role.PIVOT):
            return False
        # Entering Corp needs credentials (L3); entering Secure needs privilege (L5).
        gate = {Zone.CORP: Layer.AUTH, Zone.SECURE: Layer.PRIVILEGE}.get(host.zone)
        return gate is None or state.layers.is_satisfied(gate)

    if verb is Verb.STEAL_CREDENTIALS:
        assert action.host is not None
        # Only from a host red already owns: this is reading a credential cache, not an
        # attack. Layer 3 must still be standing, or there is nothing left to steal.
        return (action.host in state.footholds
                and state.layers.can_attempt(Layer.AUTH))

    # ---- red: untargeted layer moves -------------------------------------------------
    if verb is Verb.ESCALATE_PRIVILEGE:
        # Needs the pivot: the ACL being beaten is reachable only from there.
        return (state.layers.can_attempt(Layer.PRIVILEGE)
                and topo.PIVOT_HOST in state.footholds)

    if verb is Verb.DEGRADE_MFA:
        # Needs network reach to the MFA service, which the pivot provides. Deliberately
        # *not* gated on privilege: that is the partial order in layers.py.
        return (state.layers.can_attempt(Layer.APPROVAL)
                and _red_can_reach(state, topo.MFA_HOST))

    if verb is Verb.ALTER_CREDENTIALS:
        # Section 3.4's winning move: hold the crown jewel, with L5 and L6 satisfied.
        return topo.CROWN_JEWEL in state.footholds and state.layers.can_win()

    # ---- blue ------------------------------------------------------------------------
    if verb in (Verb.BLOCK, Verb.ISOLATE):
        assert action.host is not None
        # Isolating an already-isolated host is a wasted turn, not an error. Masking it
        # stops blue from learning that repeating a containment is free.
        return state.status(action.host) is not HostStatus.ISOLATED

    if verb in REINFORCE_LAYER:
        # Restoring a layer only makes sense if it is currently breached. Otherwise blue
        # could farm the +25 in section 5.3 by hardening an intact layer forever -- the
        # reward-shaping trap section 15 warns about.
        return state.layers.is_breached(REINFORCE_LAYER[verb])

    if verb is Verb.DEPLOY_HONEYPOT:
        return any(h.name not in state.honeypots_live
                   for h in _honeypot_slots(agent))

    raise ValueError(f"unhandled verb {verb}")


def legal_mask(state: EpisodeState, agent: str) -> np.ndarray:
    """Boolean mask over ``ACTION_SPACES[agent]``, True where the action is legal.

    Returned as a NumPy array because the policy multiplies it against a row of
    Q-values. Epsilon-greedy then samples uniformly from the True entries, so illegal
    actions are never explored -- section 3.4's "keeps the effective branching factor
    small".

    The mask is never empty: WAIT and NOOP are unconditionally legal, which guarantees
    every agent always has at least one move and the episode can never deadlock.
    """
    space = ACTION_SPACES[agent]
    return np.fromiter(
        (is_legal(state, agent, a) for a in space), dtype=bool, count=len(space)
    )


def legal_actions(state: EpisodeState, agent: str) -> tuple[Action, ...]:
    """The legal subset, for logging and for validating an LLM suggestion.

    PROJECT.md section 10.2: an LLM-proposed action is checked against this before being
    taken, and a proposal outside it falls back to a random legal action. The advisor
    can suggest; it cannot make an illegal move legal.
    """
    space = ACTION_SPACES[agent]
    return tuple(a for a, ok in zip(space, legal_mask(state, agent), strict=True) if ok)


def action_index(agent: str, action: Action) -> int:
    """Position of ``action`` in its agent's action tuple -- the Q-table column."""
    return ACTION_INDEX[agent][action]
