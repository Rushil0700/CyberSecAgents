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

Design note -- the firewall matrix is **derived, not typed out**. Writing 13x13 = 169
reachability cells by hand invites a single-character mistake that silently opens a
path from the DMZ to the database. That failure is invisible in testing: the attacker
simply looks unusually good. Instead we declare the *policy* (full mesh inside a zone,
plus an explicit allow-list of cross-zone edges) and compute the matrix from it, so
every cross-zone path is one reviewable line.

Design note -- reachability and credentials are kept separate. PROJECT.md section 3.2
says ``dev-box`` holds "a reused credential from web-portal". That is not a network
path; it is a reason an attack on ``dev-box`` *succeeds* more often once reachable.
So credential weakness lives in ``exploit_prob`` and the firewall stays honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final


class Zone(str, Enum):
    """The three network segments an attacker must traverse in order.

    Inheriting from ``str`` means a Zone is usable directly as a dict key, a YAML
    scalar and a Docker network name without conversion, while still being a proper
    enum for exhaustiveness checks.
    """

    DMZ = "dmz"
    CORP = "corp"
    SECURE = "secure"
    INFRA = "infra"  # the logger; not a defended zone, no blue agent owns it


@dataclass(frozen=True)
class Host:
    """A single machine in MiniCorp.

    Frozen because topology is configuration, not state. Anything that changes during
    an episode (compromised, isolated, alert counts) lives in ``env/state.py``. Keeping
    the two apart is what stops episode state from leaking into the map.

    Attributes:
        name: Container name; also the Docker service name in the generated compose file.
        zone: Which segment the host sits in.
        exploit_prob: ``p_h`` from PROJECT.md section 3.3 -- probability that an exploit
            action against this host succeeds, *given* the attacker already holds a host
            with network reach to it. Higher means a weaker host. This is where planted
            credential weakness is expressed.
        noise: How many alert points a successful or attempted exploit on this host
            generates, before detection noise is applied. Feeds ``env/detection.py``.
            Loud hosts are easier to defend; this is the attacker's stealth tradeoff.
        is_entry: The attacker's starting foothold candidate.
        is_crown_jewel: Reaching this host terminates the episode as a red win.
        is_honeypot_slot: Not present at reset. Blue's ``deploy_honeypot`` action
            activates the slot in its own zone.
        defended: Whether this host appears in its zone defender's state vector. Honeypot
            slots and the logger do not, which is what keeps the state spaces small.
        weakness: Human-readable note, for the report and the LLM copilot's grounding.
            Never read by the RL code.
    """

    name: str
    zone: Zone
    exploit_prob: float = 0.0
    noise: int = 1
    is_entry: bool = False
    is_crown_jewel: bool = False
    is_honeypot_slot: bool = False
    defended: bool = True
    weakness: str = "none"


# --------------------------------------------------------------------------------------
# The hosts (PROJECT.md section 3.2)
# --------------------------------------------------------------------------------------
# Exploit probabilities encode the intended difficulty gradient: the entry point is easy
# (0.9) so red gets early positive signal and does not stall at episode one, and the
# pivot to the secure zone is hard (0.4) so the chokepoint is genuinely a chokepoint.
# PROJECT.md section 3.3 fixes those two anchors; the rest interpolate.

HOSTS: Final[tuple[Host, ...]] = (
    # ---- Zone 1: DMZ -------------------------------------------------------------
    Host("reverse-proxy", Zone.DMZ, exploit_prob=0.30, noise=3,
         weakness="none; routing only, but it is the only DMZ host with a path to Corp"),
    Host("web-portal", Zone.DMZ, exploit_prob=0.90, noise=2, is_entry=True,
         weakness="weak admin credential"),
    Host("mail", Zone.DMZ, exploit_prob=0.70, noise=2,
         weakness="exposed config endpoint"),

    # ---- Zone 2: Corporate LAN ---------------------------------------------------
    Host("intranet", Zone.CORP, exploit_prob=0.65, noise=2,
         weakness="trusts DMZ session token"),
    Host("fileserver", Zone.CORP, exploit_prob=0.75, noise=1,
         weakness="world-readable share"),
    Host("dev-box", Zone.CORP, exploit_prob=0.80, noise=1,
         weakness="credential reused from web-portal"),
    Host("ci-runner", Zone.CORP, exploit_prob=0.60, noise=3,
         weakness="over-privileged service account"),
    Host("ad-controller", Zone.CORP, exploit_prob=0.40, noise=4,
         weakness="credential cache; the only host with a path to the secure zone"),

    # ---- Zone 3: Secure ----------------------------------------------------------
    Host("db-primary", Zone.SECURE, exploit_prob=0.50, noise=5, is_crown_jewel=True,
         weakness="reachable only from ad-controller"),
    Host("backup", Zone.SECURE, exploit_prob=0.55, noise=3,
         weakness="weak backup credential"),

    # ---- Honeypots: deployed by blue, absent at reset, excluded from state --------
    Host("honeypot-corp", Zone.CORP, exploit_prob=0.95, noise=5,
         is_honeypot_slot=True, defended=False,
         weakness="decoy; looks valuable, alerts loudly on any touch"),
    Host("honeypot-secure", Zone.SECURE, exploit_prob=0.95, noise=5,
         is_honeypot_slot=True, defended=False,
         weakness="decoy; looks valuable, alerts loudly on any touch"),

    # ---- Infrastructure ----------------------------------------------------------
    Host("logger", Zone.INFRA, exploit_prob=0.0, noise=0, defended=False,
         weakness="n/a; receives logs from all zones, never a target"),
)

BY_NAME: Final[dict[str, Host]] = {h.name: h for h in HOSTS}


# --------------------------------------------------------------------------------------
# The firewall policy (PROJECT.md section 3.1)
# --------------------------------------------------------------------------------------
# Rule 1: inside a zone, every host reaches every other host (flat segments -- realistic,
#         and it is what makes intra-zone lateral movement cheap).
# Rule 2: across zones, ONLY the edges listed below. Each one is a deliberate hole.
# Rule 3: every host may ship logs to the logger, one-way. Not an attack path.
#
# These three cross-zone edges are the entire reason this project has a decision problem.
# Delete them and the network is flat; add a fourth carelessly and red learns to skip the
# chokepoint. This list is the thing to review hardest.

CROSS_ZONE_EDGES: Final[tuple[tuple[str, str], ...]] = (
    ("reverse-proxy", "intranet"),      # DMZ -> Corp: the only way in
    ("ad-controller", "db-primary"),    # Corp -> Secure: the pivot
    ("ad-controller", "backup"),        # Corp -> Secure: the pivot, secondary target
)

LOG_SINK: Final[str] = "logger"


def build_reachability() -> dict[str, frozenset[str]]:
    """Derive the full reachability matrix from the firewall policy above.

    Returns:
        Mapping from source host name to the set of host names it can open a
        connection to. Used by the twin to decide whether an attack action is legal,
        and by the compose generator to decide which Docker networks each service
        attaches to.

    The matrix is derived rather than stored so that the policy -- three cross-zone
    edges and a full-mesh rule -- stays the reviewable artifact.
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


DEFENDED_ZONES: Final[tuple[Zone, ...]] = (Zone.DMZ, Zone.CORP, Zone.SECURE)

ENTRY_HOST: Final[str] = next(h.name for h in HOSTS if h.is_entry)
CROWN_JEWEL: Final[str] = next(h.name for h in HOSTS if h.is_crown_jewel)


def state_space_size(zone: Zone, *, n_statuses: int = 4, n_alert_levels: int = 3) -> int:
    """Size of the zone defender's state space: ``n_statuses ** |hosts| * n_alert_levels``.

    PROJECT.md section 5 works this out for the Corp defender as 4**5 * 3 = 3072. This
    function exists so the arithmetic is executable rather than asserted -- the 10,000
    state budget in CLAUDE.md section 2 is enforced by a test, not by good intentions.
    """
    return n_statuses ** len(hosts_in(zone, defended_only=True)) * n_alert_levels


def action_space_size(zone: Zone) -> int:
    """Size of the zone defender's action set: ``block(h) + isolate(h) + honeypot + noop``.

    Note this differs per zone (8 / 12 / 10), which is CLAUDE.md amendment 3.5 -- the
    spec's flat "12 actions" assumed every zone had five hosts.
    """
    n = len(hosts_in(zone, defended_only=True))
    return 2 * n + 2
