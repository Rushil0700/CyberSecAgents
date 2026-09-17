"""Tests that the generated Docker lab matches the simulated twin.

This is the test that protects the sim-to-real claim. If Docker connectivity and the
twin's firewall matrix diverge, every number in the transfer analysis is meaningless.
"""

from __future__ import annotations

import itertools

from marlsoc.env import compose_gen, topology as topo


def _docker_connected(doc: dict, a: str, b: str) -> bool:
    """Two containers can talk iff they share at least one network."""
    nets_a = set(doc["services"][a]["networks"])
    nets_b = set(doc["services"][b]["networks"])
    return bool(nets_a & nets_b)


class TestAirgap:
    def test_every_network_is_internal(self) -> None:
        # CLAUDE.md section 2, the hard constraint. internal: true removes the gateway,
        # so there is no route off the host for any container.
        doc = compose_gen.build_compose()
        for name, net in doc["networks"].items():
            assert net["internal"] is True, f"{name} is not airgapped"

    def test_no_service_publishes_a_port(self) -> None:
        # A published port would bridge the lab to the host's network stack and defeat
        # internal: true for that service.
        doc = compose_gen.build_compose()
        for name, svc in doc["services"].items():
            assert "ports" not in svc, f"{name} publishes a port"


class TestConnectivityMatchesTopology:
    def test_docker_connectivity_equals_firewall_policy(self) -> None:
        """For every host pair, sharing a network must mean the firewall allows it.

        Docker networks are undirected, so we compare against the undirected closure of
        the twin's directed matrix. The remaining direction difference is the documented
        sim-to-real gap 1 in compose_gen's docstring.
        """
        doc = compose_gen.build_compose()
        for a, b in itertools.combinations([h.name for h in topo.HOSTS], 2):
            allowed = topo.can_reach(a, b) or topo.can_reach(b, a)
            assert _docker_connected(doc, a, b) == allowed, (
                f"{a} <-> {b}: docker={_docker_connected(doc, a, b)} policy={allowed}"
            )

    def test_dmz_cannot_touch_the_secure_zone(self) -> None:
        doc = compose_gen.build_compose()
        for src in topo.hosts_in(topo.Zone.DMZ):
            for dst in topo.hosts_in(topo.Zone.SECURE):
                assert not _docker_connected(doc, src.name, dst.name)

    def test_reverse_proxy_reaches_only_intranet_in_corp(self) -> None:
        """The DMZ-to-Corp hole must be exactly one host wide.

        The naive "multi-home the proxy onto net-corp" approach would fail this, which
        is why bridge networks exist.
        """
        doc = compose_gen.build_compose()
        reachable_corp = [
            h.name for h in topo.hosts_in(topo.Zone.CORP)
            if _docker_connected(doc, "reverse-proxy", h.name)
        ]
        assert reachable_corp == ["intranet"]

    def test_logger_does_not_create_a_flat_network(self) -> None:
        """The log sink must not become an any-to-any path between zones."""
        doc = compose_gen.build_compose()
        assert not _docker_connected(doc, "web-portal", "db-primary")
        assert not _docker_connected(doc, "mail", "ci-runner")
        # ...while still receiving from every zone.
        for zone in topo.DEFENDED_ZONES:
            for host in topo.hosts_in(zone):
                assert _docker_connected(doc, host.name, topo.LOG_SINK), host.name


class TestServices:
    def test_every_host_becomes_a_service(self) -> None:
        doc = compose_gen.build_compose()
        assert set(doc["services"]) == {h.name for h in topo.HOSTS}

    def test_honeypots_are_behind_a_profile(self) -> None:
        # Absent at reset in the twin; not started by default in the lab.
        doc = compose_gen.build_compose()
        assert doc["services"]["honeypot-corp"]["profiles"] == ["honeypot"]
        assert "profiles" not in doc["services"]["db-primary"]
