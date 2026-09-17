"""Generate ``docker-compose.yml`` from the topology.

Run with ``python -m marlsoc.env.compose_gen``.

Why generate rather than hand-write: the compose file and the simulated twin must agree
about who can reach whom, or the sim-to-real gap measured in PROJECT.md section 8 is
measuring transcription errors rather than transfer. Generating it makes disagreement
structurally impossible -- both read ``topology.REACHABILITY``.

How zone segmentation is expressed in Docker
--------------------------------------------
Docker's unit of connectivity is the network: two containers on a shared network can
reach each other, and there is no per-host rule inside one. So a naive "one network per
zone, multi-home the bridging hosts" would be **over-permissive**: putting
``reverse-proxy`` on the corporate network to reach ``intranet`` would also hand it
``fileserver``, ``dev-box``, ``ci-runner`` and ``ad-controller`` -- flattening the very
chokepoint the project is about.

Instead we emit one network per zone *plus a dedicated two-or-three-host bridge network
per cross-zone edge group*. ``reverse-proxy`` and ``intranet`` share a private network
that nothing else joins, so the DMZ-to-Corp hole is exactly one host wide, matching the
twin.

The logger is attached to all three zone networks rather than having its own. Giving it
a shared network that every host also joined would create a flat any-to-any path through
the log sink -- a real and commonly-missed misconfiguration. Multi-homing the sink keeps
each zone's log path inside that zone.

Known sim-to-real gaps (write these up; they are findings, not bugs)
-------------------------------------------------------------------
1. **Direction.** ``topology.CROSS_ZONE_EDGES`` is directed; Docker networks are not.
   ``db-primary`` can technically open a connection back to ``ad-controller`` in the
   lab but not in the twin. Benign here because the attacker only moves forward, but it
   is a genuine difference and belongs in the results section.
2. **Probabilities.** The twin's ``exploit_prob`` is a coin flip. In the lab the same
   action is an HTTP request to our own Flask route, which either works or does not.
   Phase 4 reproduces the probability by having the service fail deliberately at rate
   ``1 - p_h``, so the two remain comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from marlsoc.env import topology as topo
from marlsoc.env.topology import Host, Zone

ZONE_NETWORK = {
    Zone.DMZ: "net-dmz",
    Zone.CORP: "net-corp",
    Zone.SECURE: "net-secure",
}

# One private network per cross-zone hole, named for the edge it implements. Keeping
# these separate from the zone networks is what makes the hole one host wide.
BRIDGE_NETWORKS: dict[str, tuple[str, ...]] = {
    "net-bridge-dmz-corp": ("reverse-proxy", "intranet"),
    "net-bridge-corp-secure": ("ad-controller", "db-primary", "backup"),
}


def networks_for(host: Host) -> list[str]:
    """Which Docker networks a host attaches to, derived from the topology."""
    nets: list[str] = []

    if host.name == topo.LOG_SINK:
        # Multi-homed into every zone so logs never need a flat shared network.
        return list(ZONE_NETWORK.values())

    if host.zone in ZONE_NETWORK:
        nets.append(ZONE_NETWORK[host.zone])

    for bridge, members in BRIDGE_NETWORKS.items():
        if host.name in members:
            nets.append(bridge)

    return nets


def service_for(host: Host) -> dict[str, Any]:
    """One compose service definition for a host.

    Every service is the same tiny Flask image parameterised by environment variables.
    Thirteen near-identical containers is the point: the interesting variation is in the
    topology and the agents, not in the services.
    """
    service: dict[str, Any] = {
        "build": {"context": "./services", "dockerfile": "Dockerfile"},
        "container_name": host.name,
        "hostname": host.name,
        "environment": {
            "HOST_NAME": host.name,
            "ZONE": host.zone.value,
            "EXPLOIT_PROB": str(host.exploit_prob),
            "NOISE": str(host.noise),
            "WEAKNESS": host.weakness,
            "IS_HONEYPOT": str(host.is_honeypot_slot).lower(),
            "IS_CROWN_JEWEL": str(host.is_crown_jewel).lower(),
            "LOG_SINK": f"http://{topo.LOG_SINK}:8000/ingest",
        },
        "networks": networks_for(host),
        "restart": "unless-stopped",
    }

    if host.is_honeypot_slot:
        # Honeypots exist but stay stopped until blue's deploy_honeypot action starts
        # them, mirroring the twin where the slot is absent at reset.
        service["profiles"] = ["honeypot"]

    return service


def build_compose() -> dict[str, Any]:
    """Assemble the full compose document."""
    all_networks = list(ZONE_NETWORK.values()) + list(BRIDGE_NETWORKS)

    return {
        # SAFETY: internal: true means Docker creates no gateway to the host or the
        # internet. There is physically nowhere for a packet to leave to. This is the
        # airgap in CLAUDE.md section 2, enforced by the platform rather than by policy.
        "networks": {name: {"internal": True} for name in all_networks},
        "services": {h.name: service_for(h) for h in topo.HOSTS},
    }


def write(path: Path) -> dict[str, Any]:
    """Write the compose document and return it."""
    doc = build_compose()
    header = (
        "# GENERATED FILE -- do not edit by hand.\n"
        "# Regenerate with:  python -m marlsoc.env.compose_gen\n"
        "# Source of truth:  src/marlsoc/env/topology.py\n"
        "#\n"
        "# Every network is internal: true -- no gateway, no route off the host.\n\n"
    )
    path.write_text(header + yaml.safe_dump(doc, sort_keys=False, width=100))
    return doc


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    out = root / "docker-compose.yml"
    doc = write(out)
    print(f"wrote {out.relative_to(root)}")
    print(f"  {len(doc['services'])} services, {len(doc['networks'])} internal networks")
