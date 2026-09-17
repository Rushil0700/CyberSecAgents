"""Tests for the action spaces and the legality mask.

Action masking is where a bug is most expensive and least visible. Mask something red
needs and its curve flatlines with no error anywhere (section 15 lists exactly this as a
thing to check). Fail to mask something it should not have and red learns a policy that
exploits a hole in our own rules, which looks like a brilliant emergent strategy right
up until someone asks how it worked.
"""

from __future__ import annotations

import numpy as np
import pytest

from marlsoc.env import actions as act
from marlsoc.env import topology as topo
from marlsoc.env.actions import Action, Verb
from marlsoc.env.layers import Layer, LayerStatus
from marlsoc.env.state import EpisodeState
from marlsoc.env.topology import Zone


def fresh() -> EpisodeState:
    return EpisodeState.initial(LayerStatus.full())


def into_dmz() -> EpisodeState:
    """Red past Layer 1 with a DMZ foothold and the DMZ discovered."""
    state = fresh()
    state.record_breach(Layer.PERIMETER)
    state.discovered.update(h.name for h in topo.hosts_in(Zone.DMZ))
    state.compromise("web-portal")
    state.record_breach(Layer.DMZ_BOUNDARY)
    return state


def at_the_pivot() -> EpisodeState:
    """Red holding ad-controller with credentials -- Layers 1-4 down."""
    state = into_dmz()
    state.record_breach(Layer.AUTH)
    state.compromise("reverse-proxy")
    state.discovered.update(h.name for h in topo.HOSTS)
    state.compromise("intranet")
    state.compromise("ad-controller")
    state.record_breach(Layer.SEGMENTATION)
    return state


class TestActionSpaces:
    def test_sizes_are_per_agent_not_shared(self) -> None:
        # CLAUDE.md amendment 3.5. An agent indexing another agent's action list is a
        # silent mis-mapping producing a confident, wrong policy.
        sizes = {a: len(s) for a, s in act.ACTION_SPACES.items()}
        assert sizes == {"R_scout": 9, "R_breach": 29,
                         "B_dmz": 10, "B_corp": 13, "B_secure": 9}

    def test_blue_sizes_match_the_topology_formula(self) -> None:
        for agent in topo.DEFENDER_ZONES:
            assert len(act.ACTION_SPACES[agent]) == topo.action_space_size(agent)

    def test_every_action_is_unique_within_its_space(self) -> None:
        # A duplicate would give one move two Q-values that never reconcile.
        for agent, space in act.ACTION_SPACES.items():
            assert len(set(space)) == len(space), agent

    def test_each_defender_reinforces_its_own_layer(self) -> None:
        """Section 4.2: defenders are differentiated by the layers they hold, not only
        by geography. Identical action sets would make the zone split cosmetic."""
        verbs = {
            agent: {a.verb for a in space} & set(act.REINFORCE_LAYER)
            for agent, space in act.ACTION_SPACES.items() if agent.startswith("B_")
        }
        assert verbs == {"B_dmz": {Verb.TIGHTEN_RATELIMIT},
                         "B_corp": {Verb.ROTATE_CREDENTIALS},
                         "B_secure": {Verb.HARDEN_MFA}}

    def test_gate_hosts_are_not_offered_as_exploit_targets(self) -> None:
        """edge-gateway and mfa-service have exploit_prob 0.0. Offering exploit() on
        them would be offering an action that can never succeed -- precisely the wasted
        exploration masking exists to remove."""
        assert topo.WAF_HOST not in act.ATTACKABLE
        assert topo.MFA_HOST not in act.ATTACKABLE
        assert topo.LOG_SINK not in act.ATTACKABLE
        assert topo.CROWN_JEWEL in act.ATTACKABLE

    def test_each_verb_has_its_own_target_list(self) -> None:
        """A blanket target list produces actions that can never be legal anywhere.
        The pivot is never an exploit target (Layer 4 guards it), and no DMZ host is ever
        a lateral-move target, because red's first foothold is always in the DMZ so no
        DMZ host is ever deeper than what it already holds."""
        assert topo.PIVOT_HOST not in act.EXPLOIT_TARGETS
        assert topo.PIVOT_HOST in act.LATERAL_TARGETS
        for host in topo.hosts_in(Zone.DMZ):
            assert host.name not in act.LATERAL_TARGETS
            assert host.name in act.EXPLOIT_TARGETS

        # Only cross-zone-edge destinations qualify for lateral_move. fileserver is deep
        # but reachable only from inside Corp, so red must already hold Corp to touch it
        # and nothing in Corp is then "deeper" than what it holds.
        assert "fileserver" not in act.LATERAL_TARGETS
        assert "intranet" in act.LATERAL_TARGETS

        # Red can never hold a honeypot -- touching one is an engagement, not a
        # compromise -- so it can never steal credentials from one.
        for slot in ("honeypot-1", "honeypot-2"):
            assert slot not in act.STEAL_TARGETS
            assert slot in act.EXPLOIT_TARGETS

    def test_the_scout_chooses_zones_because_its_state_only_tracks_zones(self) -> None:
        """An action space finer than the state space is capacity the Q-table can never
        use: the agent would choose between targets it cannot distinguish."""
        for action in act.ACTION_SPACES["R_scout"]:
            assert action.host is None


class TestMaskIsNeverEmpty:
    def test_every_agent_always_has_a_move(self) -> None:
        # Otherwise epsilon-greedy samples from an empty set and the episode deadlocks.
        for state in (fresh(), into_dmz(), at_the_pivot()):
            for agent in act.ACTION_SPACES:
                assert act.legal_mask(state, agent).any(), agent

    def test_wait_and_noop_are_unconditionally_legal(self) -> None:
        state = at_the_pivot()
        assert act.is_legal(state, "R_breach", Action(Verb.WAIT))
        assert act.is_legal(state, "B_corp", Action(Verb.NOOP))


class TestRedMasking:
    def test_red_can_only_wait_before_it_has_scanned_anything(self) -> None:
        """Nothing discovered and Layer 1 still up: the branching factor is 1 out of 40.
        This is what forces R_scout to matter and creates the credit-assignment problem
        in section 4.1."""
        assert act.legal_actions(fresh(), "R_breach") == (Action(Verb.WAIT),)

    def test_the_first_foothold_requires_layer_one_to_be_down(self) -> None:
        state = fresh()
        state.discovered.add("web-portal")
        assert not act.is_legal(state, "R_breach", Action(Verb.EXPLOIT, host="web-portal"))
        state.record_breach(Layer.PERIMETER)
        assert act.is_legal(state, "R_breach", Action(Verb.EXPLOIT, host="web-portal"))

    def test_undiscovered_hosts_cannot_be_attacked(self) -> None:
        state = fresh()
        state.record_breach(Layer.PERIMETER)
        assert not act.is_legal(state, "R_breach", Action(Verb.EXPLOIT, host="mail"))

    def test_an_owned_host_is_not_a_target_twice(self) -> None:
        state = into_dmz()
        assert not act.is_legal(state, "R_breach", Action(Verb.EXPLOIT, host="web-portal"))

    def test_exploit_widens_and_lateral_move_advances(self) -> None:
        """The distinction this module introduces. exploit stays in the zone red holds;
        lateral_move crosses into a deeper one."""
        state = into_dmz()
        state.discovered.update(h.name for h in topo.hosts_in(Zone.CORP))
        # Same zone -> exploit. lateral_move onto a DMZ host is not merely illegal, it
        # is not in the action space at all, which is the stronger guarantee.
        assert act.is_legal(state, "R_breach", Action(Verb.EXPLOIT, host="mail"))
        assert "mail" not in act.LATERAL_TARGETS
        # Deeper zone -> lateral_move is the verb, but Layer 3 is still standing.
        assert not act.is_legal(state, "R_breach", Action(Verb.EXPLOIT, host="intranet"))
        assert not act.is_legal(state, "R_breach",
                                Action(Verb.LATERAL_MOVE, host="intranet"))

    def test_entering_corp_needs_credentials(self) -> None:
        state = into_dmz()
        state.discovered.update(h.name for h in topo.hosts_in(Zone.CORP))
        state.compromise("reverse-proxy")  # the only host with a path inward
        move = Action(Verb.LATERAL_MOVE, host="intranet")
        assert not act.is_legal(state, "R_breach", move)
        state.record_breach(Layer.AUTH)
        assert act.is_legal(state, "R_breach", move)

    def test_network_reach_is_required_not_just_credentials(self) -> None:
        """Holding mail (no inward path) must not open Corp even with credentials --
        otherwise Layer 2's 'a DMZ host with inward reach' means nothing."""
        state = into_dmz()
        state.record_breach(Layer.AUTH)
        state.discovered.update(h.name for h in topo.hosts_in(Zone.CORP))
        assert not act.is_legal(state, "R_breach",
                                Action(Verb.LATERAL_MOVE, host="intranet"))

    def test_credentials_are_stolen_only_from_a_held_host(self) -> None:
        state = into_dmz()
        assert act.is_legal(state, "R_breach",
                            Action(Verb.STEAL_CREDENTIALS, host="web-portal"))
        assert not act.is_legal(state, "R_breach",
                                Action(Verb.STEAL_CREDENTIALS, host="mail"))

    def test_credentials_cannot_be_stolen_twice(self) -> None:
        # Otherwise red farms the +30 rung of the 5.4 ladder instead of advancing.
        state = into_dmz()
        state.record_breach(Layer.AUTH)
        assert not act.is_legal(state, "R_breach",
                                Action(Verb.STEAL_CREDENTIALS, host="web-portal"))

    def test_escalation_requires_holding_the_pivot(self) -> None:
        state = into_dmz()
        state.record_breach(Layer.AUTH)
        assert not act.is_legal(state, "R_breach", Action(Verb.ESCALATE_PRIVILEGE))
        assert act.is_legal(at_the_pivot(), "R_breach", Action(Verb.ESCALATE_PRIVILEGE))

    def test_degrading_mfa_does_not_require_escalated_privilege(self) -> None:
        """The partial order from layers.py, enforced at the action level.

        If this required Layer 5 the six-layer environment would be a corridor and red
        would have no ordering left to learn.
        """
        state = at_the_pivot()
        assert not state.privilege_escalated
        assert act.is_legal(state, "R_breach", Action(Verb.DEGRADE_MFA))
        assert act.is_legal(state, "R_breach", Action(Verb.ESCALATE_PRIVILEGE))

    def test_the_win_needs_the_crown_jewel_and_both_deep_layers(self) -> None:
        state = at_the_pivot()
        win = Action(Verb.ALTER_CREDENTIALS)
        assert not act.is_legal(state, "R_breach", win)
        state.record_breach(Layer.PRIVILEGE)
        state.compromise(topo.CROWN_JEWEL)
        assert not act.is_legal(state, "R_breach", win)   # MFA still standing
        state.record_breach(Layer.APPROVAL)
        assert act.is_legal(state, "R_breach", win)

    def test_the_win_is_impossible_without_holding_the_crown_jewel(self) -> None:
        state = at_the_pivot()
        state.record_breach(Layer.PRIVILEGE)
        state.record_breach(Layer.APPROVAL)
        assert state.layers.can_win()
        assert not act.is_legal(state, "R_breach", Action(Verb.ALTER_CREDENTIALS))

    def test_scanning_cannot_see_arbitrarily_deep(self) -> None:
        # One zone beyond the frontier, so the scout can find what is next without
        # reading the secure zone from the DMZ.
        state = fresh()
        assert act.is_legal(state, "R_scout", Action(Verb.SLOW_SCAN, zone=Zone.DMZ))
        assert not act.is_legal(state, "R_scout", Action(Verb.SCAN, zone=Zone.SECURE))

    def test_isolated_hosts_drop_out_of_red_s_choice_set(self) -> None:
        state = into_dmz()
        assert act.is_legal(state, "R_breach", Action(Verb.EXPLOIT, host="mail"))
        state.isolate("mail")
        assert not act.is_legal(state, "R_breach", Action(Verb.EXPLOIT, host="mail"))


class TestBlueMasking:
    def test_isolating_an_isolated_host_is_masked(self) -> None:
        """Repeating a containment must not be free, or blue learns that isolate is a
        safe filler action and the availability cost stops biting."""
        state = fresh()
        assert act.is_legal(state, "B_corp", Action(Verb.ISOLATE, host="dev-box"))
        state.isolate("dev-box")
        assert not act.is_legal(state, "B_corp", Action(Verb.ISOLATE, host="dev-box"))

    def test_reinforcing_an_intact_layer_is_masked(self) -> None:
        """Otherwise blue farms the +25 in section 5.3 by hardening a layer that was
        never breached -- the reward-shaping trap in section 15."""
        state = fresh()
        assert not act.is_legal(state, "B_dmz", Action(Verb.TIGHTEN_RATELIMIT))
        state.record_breach(Layer.PERIMETER)
        assert act.is_legal(state, "B_dmz", Action(Verb.TIGHTEN_RATELIMIT))

    def test_a_defender_cannot_reinforce_another_zone_s_layer(self) -> None:
        state = at_the_pivot()
        assert Action(Verb.ROTATE_CREDENTIALS) not in act.ACTION_SPACES["B_secure"]
        assert Action(Verb.HARDEN_MFA) not in act.ACTION_SPACES["B_corp"]

    def test_b_dmz_has_no_honeypot_action_because_it_has_no_slot(self) -> None:
        """Both honeypots live in Corp and Secure (section 3.3), so the action could
        never be legal for B_dmz. Found by decoding a trained Q-table: an action that is
        never legal is never taken, so it is never updated, so it keeps its optimistic
        0.0 forever and dominates any unmasked argmax over the row."""
        verbs = {a.verb for a in act.ACTION_SPACES["B_dmz"]}
        assert Verb.DEPLOY_HONEYPOT not in verbs
        for agent in ("B_corp", "B_secure"):
            assert Verb.DEPLOY_HONEYPOT in {a.verb for a in act.ACTION_SPACES[agent]}

    def test_no_action_is_unsatisfiable(self) -> None:
        """Every action must be legal in *some* coherent state.

        Constructing states directly rather than sampling trajectories, because the two
        are different claims. A trajectory sample only shows what a particular opponent
        happens to reach -- the scripted attacker beelines for the pivot and never widens
        inside Corp, so its rollouts make perfectly satisfiable actions look dead. What
        matters here is whether an action's legality predicate can be satisfied at all,
        and that is answered by building the state, not by waiting for one.

        The states are coherent by construction: layers are breached along a valid prefix
        of the partial order, so no state has Layer 4 down while Layer 3 still stands.
        """
        import numpy as np
        from marlsoc.env.layers import Layer

        rng = np.random.default_rng(0)
        ever = {a: np.zeros(len(act.ACTION_SPACES[a]), dtype=bool)
                for a in act.ACTION_SPACES}

        orderings = (
            (Layer.PERIMETER, Layer.DMZ_BOUNDARY, Layer.AUTH, Layer.SEGMENTATION,
             Layer.PRIVILEGE, Layer.APPROVAL),
            (Layer.PERIMETER, Layer.DMZ_BOUNDARY, Layer.AUTH, Layer.SEGMENTATION,
             Layer.APPROVAL, Layer.PRIVILEGE),   # the L5/L6 branch, taken the other way
        )
        hosts = [h.name for h in topo.HOSTS if not h.is_honeypot_slot
                 and h.role is not topo.Role.LOG_SINK]

        for trial in range(1_500):
            state = fresh()
            order = orderings[trial % 2]
            for layer in order[: int(rng.integers(0, len(order) + 1))]:
                state.layers = state.layers.breach(layer)

            state.discovered.update(h.name for h in topo.HOSTS)
            for name in hosts:
                roll = rng.random()
                if roll < 0.35:
                    state.compromise(name)
                elif roll < 0.45:
                    state.isolate(name)
            if rng.random() < 0.5:
                state.deploy_honeypot("honeypot-1")
            if rng.random() < 0.5:
                state.deploy_honeypot("honeypot-2")

            for agent in act.ACTION_SPACES:
                ever[agent] |= act.legal_mask(state, agent)

        for agent, seen in ever.items():
            dead = [str(act.ACTION_SPACES[agent][i]) for i in np.flatnonzero(~seen)]
            assert not dead, f"{agent} can never legally: {dead}"

    def test_honeypots_cannot_be_deployed_twice(self) -> None:
        state = fresh()
        assert act.is_legal(state, "B_corp", Action(Verb.DEPLOY_HONEYPOT))
        state.deploy_honeypot("honeypot-1")
        assert not act.is_legal(state, "B_corp", Action(Verb.DEPLOY_HONEYPOT))

    def test_blue_can_only_act_on_its_own_hosts(self) -> None:
        # Section 3.4: "each defender acts only within its own zone". Enforced by
        # construction -- another zone's host is not in the action space at all.
        corp_hosts = {a.host for a in act.ACTION_SPACES["B_corp"] if a.host}
        assert corp_hosts == {h.name for h in topo.defended_hosts("B_corp")}
        assert "web-portal" not in corp_hosts


class TestMaskMechanics:
    def test_mask_aligns_with_the_action_space(self) -> None:
        state = at_the_pivot()
        for agent, space in act.ACTION_SPACES.items():
            mask = act.legal_mask(state, agent)
            assert mask.dtype == np.bool_
            assert mask.shape == (len(space),)
            assert list(act.legal_actions(state, agent)) == [
                a for a, ok in zip(space, mask, strict=True) if ok
            ]

    def test_action_index_round_trips(self) -> None:
        for agent, space in act.ACTION_SPACES.items():
            for i, action in enumerate(space):
                assert act.action_index(agent, action) == i

    def test_masking_cuts_the_branching_factor_hard(self) -> None:
        # Section 3.4's stated purpose, measured: 40 actions down to a handful.
        legal = act.legal_mask(into_dmz(), "R_breach").sum()
        assert legal < 0.3 * len(act.ACTION_SPACES["R_breach"])

    def test_an_unknown_verb_raises_rather_than_defaulting_to_legal(self) -> None:
        # Defaulting to legal would let a new action slip past every precondition.
        state = fresh()
        with pytest.raises(ValueError):
            act.is_legal(state, "B_corp", Action(Verb.SCAN, zone=Zone.CORP))
