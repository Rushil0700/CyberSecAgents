"""Tests for the network map.

These are the most important tests in the project and they are worth understanding.
Everything downstream -- the attacker's decision problem, the state-space budget, the
generated compose file -- assumes properties of this map. If the firewall is wrong, the
learning curves still look fine; they are just curves for a different problem than the
one we claim to be solving. That is a silent failure, so we test it loudly.
"""

from __future__ import annotations

from marlsoc.env import topology as topo
from marlsoc.env.topology import Zone


def _reachable_closure(start: str, *, blocked: frozenset[str] = frozenset()) -> set[str]:
    """Every host reachable from ``start`` by chaining allowed connections.

    This is a breadth-first search over the reachability graph. It models the attacker's
    true capability: not "who can I talk to right now" but "where could I eventually get
    to, hopping host by host". ``blocked`` lets us delete a host from the graph and ask
    what becomes unreachable, which is how we prove a chokepoint is really a chokepoint.
    """
    seen = {start}
    frontier = [start]
    while frontier:
        current = frontier.pop()
        for nxt in topo.REACHABILITY[current]:
            if nxt not in seen and nxt not in blocked:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


class TestForcedPath:
    """The DMZ -> Corp -> Secure traversal must be genuinely forced."""

    def test_attacker_entry_can_eventually_reach_crown_jewel(self) -> None:
        # If this fails the game is unwinnable for red and no amount of training helps.
        assert topo.CROWN_JEWEL in _reachable_closure(topo.ENTRY_HOST)

    def test_no_direct_path_from_dmz_to_secure(self) -> None:
        # One hop only: no DMZ host may open a connection straight into the secure zone.
        for src in topo.hosts_in(Zone.DMZ):
            for dst in topo.hosts_in(Zone.SECURE):
                assert not topo.can_reach(src.name, dst.name), (
                    f"{src.name} -> {dst.name} skips the corporate zone entirely"
                )

    def test_ad_controller_is_the_only_pivot_into_secure(self) -> None:
        """Removing ad-controller must cut the secure zone off from the attacker.

        This is the test that makes "chokepoint defence" a real emergent finding. If
        blue learns to concentrate on ad-controller, that behaviour is only meaningful
        because ad-controller genuinely is the single point of failure -- which this
        asserts rather than assumes.
        """
        without_pivot = _reachable_closure(
            topo.ENTRY_HOST, blocked=frozenset({"ad-controller"})
        )
        assert topo.CROWN_JEWEL not in without_pivot
        assert "backup" not in without_pivot

    def test_reverse_proxy_is_the_only_bridge_into_corp(self) -> None:
        without_bridge = _reachable_closure(
            topo.ENTRY_HOST, blocked=frozenset({"reverse-proxy"})
        )
        for host in topo.hosts_in(Zone.CORP):
            assert host.name not in without_bridge


class TestFirewallPolicy:
    def test_cross_zone_edges_are_the_only_cross_zone_paths(self) -> None:
        """Every cross-zone connection must appear in the explicit allow-list.

        Guards against the failure mode the module docstring warns about: a stray edge
        that quietly flattens the network. The logger is exempt -- it is a one-way sink,
        not an attack path.
        """
        allowed = set(topo.CROSS_ZONE_EDGES)
        for src_name, dsts in topo.REACHABILITY.items():
            src = topo.BY_NAME[src_name]
            for dst_name in dsts:
                dst = topo.BY_NAME[dst_name]
                if dst.zone is src.zone or dst_name == topo.LOG_SINK:
                    continue
                assert (src_name, dst_name) in allowed, (
                    f"undeclared cross-zone edge {src_name} -> {dst_name}"
                )

    def test_logger_initiates_nothing(self) -> None:
        # A log sink that can open connections is a lateral-movement path through the
        # one host attached to all three zones. It must be receive-only.
        assert topo.REACHABILITY[topo.LOG_SINK] == frozenset()

    def test_no_host_reaches_itself(self) -> None:
        for name, dsts in topo.REACHABILITY.items():
            assert name not in dsts


class TestHostDefinitions:
    def test_exactly_one_entry_and_one_crown_jewel(self) -> None:
        assert sum(h.is_entry for h in topo.HOSTS) == 1
        assert sum(h.is_crown_jewel for h in topo.HOSTS) == 1

    def test_thirteen_hosts(self) -> None:
        assert len(topo.HOSTS) == 13

    def test_exploit_probabilities_are_valid(self) -> None:
        for host in topo.HOSTS:
            assert 0.0 <= host.exploit_prob <= 1.0, host.name

    def test_entry_is_easy_and_pivot_is_hard(self) -> None:
        """The difficulty gradient from PROJECT.md section 3.3.

        Red needs early positive signal or it never gets off the ground (section 15,
        "red never succeeding"), and the pivot needs to be expensive or the chokepoint
        is not worth defending.
        """
        assert topo.BY_NAME["web-portal"].exploit_prob == 0.90
        assert topo.BY_NAME["ad-controller"].exploit_prob == 0.40
        assert topo.BY_NAME["ad-controller"].exploit_prob < topo.BY_NAME["dev-box"].exploit_prob

    def test_host_names_are_valid_docker_service_names(self) -> None:
        # These names are transcribed directly into docker-compose.yml.
        for host in topo.HOSTS:
            assert host.name.replace("-", "").isalnum()
            assert host.name.islower()


class TestBudgets:
    """CLAUDE.md section 2: at most ~10,000 states per agent, enforced not promised."""

    def test_every_defender_is_within_the_state_budget(self) -> None:
        for zone in topo.DEFENDED_ZONES:
            size = topo.state_space_size(zone)
            assert size <= 10_000, f"{zone.value} needs {size} states, over budget"

    def test_corp_matches_the_worked_example_in_the_spec(self) -> None:
        # PROJECT.md section 5 derives 4**5 * 3 = 3072 for the Corp defender.
        assert topo.state_space_size(Zone.CORP) == 3072

    def test_action_spaces_differ_per_zone(self) -> None:
        # CLAUDE.md amendment 3.5 -- the spec's flat "12 actions" is wrong for DMZ/Secure.
        assert topo.action_space_size(Zone.CORP) == 12
        assert topo.action_space_size(Zone.DMZ) == 8
        assert len({topo.action_space_size(z) for z in topo.DEFENDED_ZONES}) > 1
