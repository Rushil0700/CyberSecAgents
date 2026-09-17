"""MiniCorp network topology: the map of the world.

This module is **pure data plus the rules that derive more data**. It contains no
simulation logic, no randomness and no agent code. It is the single source of truth
for two consumers that must never disagree:

  * the simulated twin (``env/minicorp.py``), which uses the reachability matrix to
    decide whether an attack action is even legal, and
  * the real Docker lab (``docker-compose.yml``), which is *generated* from this file.

If those two ever drift apart, the sim-to-real gap measurement in PROJECT.md section 8
stops measuring transfer and starts measuring our own typos. Hence: one file, both
consumers.

Design note -- the firewall matrix is **derived, not typed out**. Writing 15x15 = 225
reachability cells by hand invites a single-character mistake that silently opens a
path from the DMZ to the secure zone. That failure is invisible in testing: the attacker
simply looks unusually good. Instead we declare the *policy* (full mesh inside a zone,
plus an explicit allow-list of cross-zone edges) and compute the matrix from it, so
every cross-zone path is one reviewable line.

Design note -- reachability and credentials are kept separate. PROJECT.md section 3.3
says ``dev-box`` holds "a reused credential from web-portal". That is not a network
path; it is a reason an attack on ``dev-box`` *succeeds* more often once reachable.
So credential weakness lives in ``exploit_prob`` and the firewall stays honest.

Design note -- **some hosts are gates, not targets.** PROJECT.md section 3.1 makes each
of the six layers beatable by a *different class of action*: Layer 1 falls to
``slow_scan``, Layer 6 to ``degrade_mfa``. If ``edge-gateway`` and ``mfa-service`` had
an ordinary ``exploit_prob``, red would learn the far cheaper policy of simply owning
them, and the six heterogeneous layers would collapse into "exploit six hosts in a
row" -- the exact thing section 3.1 argues against. So the two gate hosts have
``exploit_prob = 0.0`` and a separate ``bypass_prob``. Owning them is impossible;
getting past them is a different move. That one data decision is what makes the layer
taxonomy real rather than decorative.

Design note -- host *roles* are data, not conditionals. CLAUDE.md section 2 forbids
hard-coded tactics such as ``if host == "ad-controller"``. Declaring
``role=Role.PIVOT`` here is the legitimate alternative: the layer model and the reward
function ask the topology "which host is the pivot?" instead of naming it inline, so
the map stays the only place a host name appears.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


class Zone(str, Enum):
    """The four network segments an attacker must traverse in order.

    Inheriting from ``str`` means a Zone is usable directly as a dict key, a YAML
    scalar and a Docker network name without conversion, while still being a proper
    enum for exhaustiveness checks.
    """

    EDGE = "edge"      # the WAF sits here; Layer 1
    DMZ = "dmz"        # Layers 2 and 3
    CORP = "corp"      # Layers 4 and 5
    SECURE = "secure"  # Layer 6 and the crown jewel
    INFRA = "infra"    # the logger; not a defended zone, no blue agent owns it


# How deep into the network a zone is. Red's state (PROJECT.md section 5.2) tracks
# ``deepest_zone_reached``, which needs an ordering; putting it here keeps the ordering
# next to the zones themselves rather than duplicated in the agent code.
ZONE_DEPTH: Final[dict[Zone, int]] = {
    Zone.EDGE: 0,
    Zone.DMZ: 1,
    Zone.CORP: 2,
    Zone.SECURE: 3,
    Zone.INFRA: -1,  # off the path entirely
}


class Role(str, Enum):
    """What a host *is for*, structurally.

    Every one of the six layers anchors on a host with a particular role. The layer
    model (``env/layers.py``) and the reward table look roles up here rather than
    naming hosts inline, which is what keeps CLAUDE.md section 2's "never hard-code
    tactics" rule enforceable: grep the codebase for ``"ad-controller"`` and it should
    appear in this file and nowhere else.
    """

    NORMAL = "normal"
    WAF = "waf"                  # Layer 1 -- rate limiting; evaded, never owned
    ENTRY = "entry"              # red's first foothold candidate
    BRIDGE = "bridge"            # Layer 2 -- the one DMZ host with inward reach
    PIVOT = "pivot"              # Layer 4 -- the one Corp host with secure-zone reach
    CROWN_JEWEL = "crown_jewel"  # holds the admin login; altering it is red's win
    MFA_GATE = "mfa_gate"        # Layer 6 -- approval check; degraded, never owned
    HONEYPOT = "honeypot"        # decoy, deployed by blue mid-episode
    LOG_SINK = "log_sink"        # receive-only


@dataclass(frozen=True)
class Host:
    """A single machine in MiniCorp.

    Frozen because topology is configuration, not state. Anything that changes during
    an episode (compromised, isolated, alert counts) lives in ``env/state.py``. Keeping
    the two apart is what stops episode state from leaking into the map.

    Attributes:
        name: Container name; also the Docker service name in the generated compose file.
        zone: Which segment the host sits in.
        role: Structural purpose; see ``Role``. Drives the layer model.
        exploit_prob: ``p_h`` from PROJECT.md section 3.4 -- probability that an exploit
            action against this host succeeds, *given* the attacker already holds a host
            with network reach to it and every layer guarding it is already satisfied.
            Higher means a weaker host. This is where planted credential weakness is
            expressed. **Zero means the host cannot be owned at all** -- it is a gate.
        bypass_prob: Probability that the layer-specific move which *gets past* this
            host succeeds (``slow_scan`` against the WAF, ``degrade_mfa`` against the
            MFA gate). Zero for ordinary hosts, which have nothing to bypass.
        noise: How many alert points an action against this host generates before
            detection noise is applied. Feeds ``env/detection.py``. Loud hosts are
            easier to defend; this is the attacker's stealth tradeoff.
        defended: Whether this host appears in its zone defender's state vector.
            Honeypot slots and the logger do not, which is what keeps the blue state
            spaces small.
        weakness: Human-readable note, for the report and the LLM copilot's grounding.
            Never read by the RL code.
    """

    name: str
    zone: Zone
    role: Role = Role.NORMAL
    exploit_prob: float = 0.0
    bypass_prob: float = 0.0
    noise: int = 1
    defended: bool = True
    weakness: str = "none"

    @property
    def is_entry(self) -> bool:
        return self.role is Role.ENTRY

    @property
    def is_crown_jewel(self) -> bool:
        return self.role is Role.CROWN_JEWEL

    @property
    def is_honeypot_slot(self) -> bool:
        return self.role is Role.HONEYPOT

    @property
    def is_gate(self) -> bool:
        """A host that is passed rather than owned (Layer 1 WAF, Layer 6 MFA)."""
        return self.role in (Role.WAF, Role.MFA_GATE)


# --------------------------------------------------------------------------------------
# The hosts (PROJECT.md section 3.3)
# --------------------------------------------------------------------------------------
# Exploit probabilities encode the intended difficulty gradient: the entry point is easy
# (0.9) so red gets early positive signal and does not stall at episode one, and the
# pivot to the secure zone is hard (0.4) so the chokepoint is genuinely a chokepoint.
# PROJECT.md section 3.4 fixes those two anchors; the rest interpolate.

HOSTS: Final[tuple[Host, ...]] = (
    # ---- Zone 0: Edge ------------------------------------------------------------
    # LAYER 1. exploit_prob is deliberately 0.0: the WAF is evaded with slow_scan, not
    # compromised. See the module docstring's "gates, not targets" note.
    Host("edge-gateway", Zone.EDGE, Role.WAF, exploit_prob=0.0, bypass_prob=0.60,
         noise=3, weakness="rate threshold evadable by scanning below it"),

    # ---- Zone 1: DMZ -------------------------------------------------------------
    # LAYER 2 is the DMZ boundary; reverse-proxy is the only host with inward reach.
    Host("reverse-proxy", Zone.DMZ, Role.BRIDGE, exploit_prob=0.30, noise=3,
         weakness="none; routing only, but it is the only DMZ host with a path to Corp"),
    Host("web-portal", Zone.DMZ, Role.ENTRY, exploit_prob=0.90, noise=2,
         weakness="weak admin credential"),
    Host("mail", Zone.DMZ, exploit_prob=0.70, noise=2,
         weakness="exposed config endpoint"),

    # ---- Zone 2: Corporate LAN ---------------------------------------------------
    # LAYER 3 (credentials) gates entry to this zone; LAYER 4 gates reaching the pivot.
    Host("intranet", Zone.CORP, exploit_prob=0.65, noise=2,
         weakness="trusts DMZ session token"),
    Host("fileserver", Zone.CORP, exploit_prob=0.75, noise=1,
         weakness="world-readable share"),
    Host("dev-box", Zone.CORP, exploit_prob=0.80, noise=1,
         weakness="credential reused from web-portal"),
    Host("ci-runner", Zone.CORP, exploit_prob=0.60, noise=3,
         weakness="over-privileged service account"),
    Host("ad-controller", Zone.CORP, Role.PIVOT, exploit_prob=0.40, noise=4,
         weakness="credential cache; the only host with a path to the secure zone"),

    # ---- Zone 3: Secure ----------------------------------------------------------
    Host("auth-server", Zone.SECURE, Role.CROWN_JEWEL, exploit_prob=0.50, noise=5,
         weakness="stores the admin login; reachable only via ad-controller"),
    # LAYER 6. Also a gate: degraded under load, never owned.
    Host("mfa-service", Zone.SECURE, Role.MFA_GATE, exploit_prob=0.0, bypass_prob=0.45,
         noise=4, weakness="approval check on credential change; degradable under load"),
    Host("backup", Zone.SECURE, exploit_prob=0.55, noise=3,
         weakness="weak backup credential"),

    # ---- Honeypots: deployed by blue, absent at reset, excluded from state --------
    Host("honeypot-1", Zone.CORP, Role.HONEYPOT, exploit_prob=0.95, noise=5,
         defended=False, weakness="decoy; looks valuable, alerts loudly on any touch"),
    Host("honeypot-2", Zone.SECURE, Role.HONEYPOT, exploit_prob=0.95, noise=5,
         defended=False, weakness="decoy; looks valuable, alerts loudly on any touch"),

    # ---- Infrastructure ----------------------------------------------------------
    Host("logger", Zone.INFRA, Role.LOG_SINK, exploit_prob=0.0, noise=0, defended=False,
         weakness="n/a; receives logs from all zones, never a target"),
)

BY_NAME: Final[dict[str, Host]] = {h.name: h for h in HOSTS}


def host_with_role(role: Role) -> Host:
    """The unique host holding ``role``.

    Raises if the role is absent or ambiguous, which turns "someone deleted the pivot"
    into an immediate, loud failure rather than a subtly easier game.
    """
    matches = [h for h in HOSTS if h.role is role]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one host with role {role.value}, got {len(matches)}")
    return matches[0]


# --------------------------------------------------------------------------------------
# The firewall policy (PROJECT.md section 3.2)
# --------------------------------------------------------------------------------------
# Rule 1: inside a zone, every host reaches every other host (flat segments -- realistic,
#         and it is what makes intra-zone lateral movement cheap).
# Rule 2: across zones, ONLY the edges listed below. Each one is a deliberate hole.
# Rule 3: every host may ship logs to the logger, one-way. Not an attack path.
#
# These cross-zone edges are the entire reason this project has a decision problem.
# Delete them and the network is unwinnable; add one carelessly and red learns to skip a
# layer. This list is the thing to review hardest.

CROSS_ZONE_EDGES: Final[tuple[tuple[str, str], ...]] = (
    # Edge -> DMZ: the WAF fronts all three DMZ services. Passing it is Layer 1;
    # these edges are network paths only and say nothing about whether Layer 1 is open.
    ("edge-gateway", "reverse-proxy"),
    ("edge-gateway", "web-portal"),
    ("edge-gateway", "mail"),

    # DMZ -> Corp: the only way inward, and only from the bridge host. Layer 2.
    ("reverse-proxy", "intranet"),

    # Corp -> Secure: the pivot. Layer 4 is what makes reaching ad-controller itself hard.
    ("ad-controller", "auth-server"),
    ("ad-controller", "mfa-service"),
    ("ad-controller", "backup"),
)

LOG_SINK: Final[str] = "logger"


def build_reachability() -> dict[str, frozenset[str]]:
    """Derive the full reachability matrix from the firewall policy above.

    Returns:
        Mapping from source host name to the set of host names it can open a
        connection to. Used by the twin to decide whether an attack action is legal,
        and by the compose generator to decide which Docker networks each service
        attaches to.

    The matrix is derived rather than stored so that the policy -- seven cross-zone
    edges and a full-mesh rule -- stays the reviewable artifact.

    Note this is *network* reachability only. Whether an action is actually legal also
    depends on the six layer preconditions in ``env/layers.py``; reachability is a
    necessary condition, never a sufficient one.
    """
    reach: dict[str, set[str]] = {h.name: set() for h in HOSTS}

    for src in HOSTS:
        if src.zone is Zone.INFRA:
            continue  # the logger initiates nothing; it only receives

        # Rule 1 -- full mesh within the zone, excluding self.
        for dst in HOSTS:
            if dst.zone is src.zone and dst.name != src.name:
                reach[src.name].add(dst.name)

        # Rule 3 -- everyone ships logs one way.
        reach[src.name].add(LOG_SINK)

    # Rule 2 -- the explicit cross-zone allow-list.
    for src_name, dst_name in CROSS_ZONE_EDGES:
        reach[src_name].add(dst_name)

    return {name: frozenset(dsts) for name, dsts in reach.items()}


REACHABILITY: Final[dict[str, frozenset[str]]] = build_reachability()


def can_reach(src: str, dst: str) -> bool:
    """Whether ``src`` may open a connection to ``dst`` under the firewall policy."""
    return dst in REACHABILITY[src]


def hosts_in(zone: Zone, *, defended_only: bool = False) -> tuple[Host, ...]:
    """Hosts belonging to ``zone``.

    Args:
        zone: The segment to list.
        defended_only: If True, exclude honeypot slots and infrastructure. This is the
            set that appears in a blue agent's state vector, so it is what determines
            that agent's state-space size.
    """
    return tuple(
        h for h in HOSTS
        if h.zone is zone and (h.defended or not defended_only)
    )


# Which zones a blue agent owns. B_dmz defends Edge *and* DMZ (PROJECT.md section 4.2):
# the WAF is not worth a whole agent, and the agent that holds Layer 1 should be the one
# that holds Layer 2, since tightening the rate limit and isolating a DMZ host are two
# responses to the same intrusion.
DEFENDER_ZONES: Final[dict[str, tuple[Zone, ...]]] = {
    "B_dmz": (Zone.EDGE, Zone.DMZ),
    "B_corp": (Zone.CORP,),
    "B_secure": (Zone.SECURE,),
}

DEFENDED_ZONES: Final[tuple[Zone, ...]] = (Zone.EDGE, Zone.DMZ, Zone.CORP, Zone.SECURE)

ENTRY_HOST: Final[str] = host_with_role(Role.ENTRY).name
CROWN_JEWEL: Final[str] = host_with_role(Role.CROWN_JEWEL).name
PIVOT_HOST: Final[str] = host_with_role(Role.PIVOT).name
BRIDGE_HOST: Final[str] = host_with_role(Role.BRIDGE).name
WAF_HOST: Final[str] = host_with_role(Role.WAF).name
MFA_HOST: Final[str] = host_with_role(Role.MFA_GATE).name


def defended_hosts(agent: str) -> tuple[Host, ...]:
    """The hosts appearing in ``agent``'s state vector, in a stable order.

    Stable ordering matters: the state vector is indexed positionally, so if this
    ordering changed between training and evaluation every Q-table lookup would be
    reading a different host's status. Tuple order in ``HOSTS`` is the ordering.
    """
    return tuple(
        h
        for zone in DEFENDER_ZONES[agent]
        for h in hosts_in(zone, defended_only=True)
    )


def state_space_size(agent: str, *, n_statuses: int = 4, n_alert_levels: int = 3) -> int:
    """Size of a defender's state space: ``n_statuses ** |hosts| * n_alert_levels``.

    PROJECT.md section 5.1 works this out for ``B_corp`` as 4**5 * 3 = 3072. This
    function exists so the arithmetic is executable rather than asserted -- the 10,000
    state budget in CLAUDE.md section 2 is enforced by a test, not by good intentions.
    """
    return n_statuses ** len(defended_hosts(agent)) * n_alert_levels


# Each blue agent gets one layer-reinforcing action for the layers it holds
# (PROJECT.md section 4.2): tighten_ratelimit, rotate_credentials, harden_mfa.
def action_space_size(agent: str) -> int:
    """Size of a defender's action set: ``block(h) + isolate(h) + reinforce + noop``,
    plus ``deploy_honeypot`` where the agent's zones actually contain a honeypot slot.

    Note this differs per agent (10 / 13 / 9), which is CLAUDE.md amendment 3.5 -- the
    spec's flat "12 actions" assumed every zone had the same number of hosts. B_dmz is
    also the one agent with no honeypot slot to deploy into; see ``actions.blue_actions``.
    """
    has_slot = any(
        h.is_honeypot_slot
        for zone in DEFENDER_ZONES[agent]
        for h in hosts_in(zone)
    )
    return 2 * len(defended_hosts(agent)) + 2 + int(has_slot)
