"""Tests for the observation functions.

Two jobs. First, prove that partial observability is real -- that a blue agent cannot
recover ground truth from what it is handed, in either direction. Second, prove the
encoding is a bijection, because a non-injective encoding merges two distinct situations
into one Q-table row and the only symptom is a policy that looks confused in exactly one
region of the state space.
"""

from __future__ import annotations

import itertools

from marlsoc.env import detection as det
from marlsoc.env import observations as obs
from marlsoc.env import topology as topo
from marlsoc.env.layers import Layer, LayerStatus
from marlsoc.env.observations import ObservedStatus
from marlsoc.env.state import EpisodeState
from marlsoc.env.topology import Zone


def fresh() -> EpisodeState:
    return EpisodeState.initial(LayerStatus.full())


class TestPartialObservabilityIsReal:
    """CLAUDE.md 3.1: blue infers, it does not know."""

    def test_a_silent_compromised_host_looks_clean(self) -> None:
        """A false negative. If this were impossible, detection would be free and
        mean time to detection would be 1 step in every episode."""
        state = fresh()
        state.compromise("ad-controller")
        assert obs.observed_status(state, "ad-controller") is ObservedStatus.CLEAN

    def test_an_unlucky_clean_host_looks_compromised(self) -> None:
        """A false positive at maximum confidence. 'Confirmed' is not a synonym for
        'true', which is what keeps the availability cost in section 5.3 payable even
        by a careful defender."""
        state = fresh()
        state.alerts["mail"] = obs.CONFIRM_THRESHOLD
        assert obs.observed_status(state, "mail") is ObservedStatus.COMPROMISED

    def test_a_stealthy_attacker_never_gets_past_suspicious(self) -> None:
        """The core tradeoff, asserted.

        An idle foothold's steady-state accumulation is base_detect_rate / (1 - decay)
        = 1.00, which clears neither threshold; even sustained quiet activity reaches
        only 1.53, over SUSPICIOUS but under CONFIRM. So blue must either isolate on
        suspicion alone and pay the availability cost, or wait for a confirmation a
        careful red never provides.
        """
        cfg = det.DEFAULT
        idle = cfg.base_detect_rate / (1 - cfg.decay)
        quiet_activity = (cfg.base_detect_rate + cfg.noise_gain * 1) / (1 - cfg.decay)
        assert idle < cfg.suspicion_threshold
        assert cfg.suspicion_threshold < quiet_activity < obs.CONFIRM_THRESHOLD

    def test_confirmation_needs_more_evidence_than_suspicion(self) -> None:
        assert obs.CONFIRM_THRESHOLD > det.DEFAULT.suspicion_threshold

    def test_blue_knows_what_it_isolated(self) -> None:
        # The one ground-truth fact blue is allowed, and only because blue caused it.
        state = fresh()
        state.compromise("mail")
        state.isolate("mail")
        assert obs.observed_status(state, "mail") is ObservedStatus.ISOLATED

    def test_the_observation_never_reads_true_status_for_a_live_host(self) -> None:
        """Two states identical in evidence must be indistinguishable to blue, however
        different the truth behind them is."""
        clean, dirty = fresh(), fresh()
        dirty.compromise("dev-box")
        for state in (clean, dirty):
            state.alerts["dev-box"] = 2.0
        assert (obs.blue_observation(clean, "B_corp")
                == obs.blue_observation(dirty, "B_corp"))


class TestBlueObservationShape:
    def test_shape_is_one_digit_per_defended_host_plus_an_alert_level(self) -> None:
        for agent in topo.DEFENDER_ZONES:
            o = obs.blue_observation(fresh(), agent)
            assert len(o) == len(topo.defended_hosts(agent)) + 1
            assert len(o) == len(obs.blue_dims(agent))

    def test_host_ordering_is_stable(self) -> None:
        """If the ordering changed between training and evaluation, every Q-table lookup
        would be reading a different host's status and the policy would be scrambled
        with no error raised anywhere."""
        first = [h.name for h in topo.defended_hosts("B_corp")]
        second = [h.name for h in topo.defended_hosts("B_corp")]
        assert first == second
        assert len(set(first)) == len(first)

    def test_a_two_zone_agent_reports_its_worse_alert_level(self) -> None:
        # B_dmz holds Edge and DMZ; it should react to whichever segment is worse.
        state = fresh()
        for host in topo.hosts_in(Zone.DMZ):
            state.alerts[host.name] = 2.0
        assert obs.blue_observation(state, "B_dmz")[-1] == det.ALERT_HIGH

    def test_each_agent_sees_only_its_own_zones(self) -> None:
        """Section 4.2's "genuine partial observability". A corporate breach must be
        invisible to the DMZ defender, or the zone split buys nothing."""
        state = fresh()
        before = obs.blue_observation(state, "B_dmz")
        for host in topo.hosts_in(Zone.CORP):
            state.compromise(host.name)
            state.alerts[host.name] = 9.0
        assert obs.blue_observation(state, "B_dmz") == before


class TestRedObservation:
    def test_breach_observation_matches_the_spec_vector(self) -> None:
        state = fresh()
        o = obs.red_breach_observation(state)
        assert len(o) == 6  # zone, footholds, creds, privilege, mfa, heat
        assert o == (0, 0, 0, 0, 0, 0)

    def test_the_layer_flags_are_digits_three_four_and_five(self) -> None:
        state = fresh()
        for layer in (Layer.PERIMETER, Layer.DMZ_BOUNDARY, Layer.AUTH):
            state.record_breach(layer)
        assert obs.red_breach_observation(state)[2] == 1   # creds_held
        assert obs.red_breach_observation(state)[3] == 0   # privilege_escalated
        assert obs.red_breach_observation(state)[4] == 0   # mfa_degraded

    def test_red_breach_state_space_is_384_not_the_spec_s_768(self) -> None:
        """PROJECT.md section 5.2 writes 4 * 4 * 2 * 2 * 2 * 3 = 768. That product is
        384; the spec doubled once too often.

        The *argument* built on the number survives intact, which is the part that
        matters for the report: section 5.2 claims six layers cost 8x red's state space
        against two layers. Two layers means no creds/privilege/mfa flags, so
        4 * 4 * 3 = 48, and 48 * 8 = 384. Three binary flags is 2**3 = 8 exactly. Only
        the absolute figure was wrong, and the corrected one is further inside budget.
        """
        assert obs.n_states(obs.RED_BREACH_DIMS) == 384
        two_layer_dims = (4, obs.N_FOOTHOLD_BUCKETS, det.N_ALERT_LEVELS)
        assert obs.n_states(two_layer_dims) == 48
        assert obs.n_states(obs.RED_BREACH_DIMS) == 8 * obs.n_states(two_layer_dims)

    def test_each_layer_costs_one_binary_digit_not_a_dimension_of_hosts(self) -> None:
        # The answer to "how did you keep tabular RL feasible with six layers?"
        assert obs.RED_BREACH_DIMS[2:5] == (2, 2, 2)

    def test_the_scout_carries_only_state_it_can_act_on(self) -> None:
        """A scout observing the credential and privilege flags would be carrying state
        it has no action to influence: a bigger Q-table and not one different decision."""
        state = fresh()
        o = obs.red_scout_observation(state)
        assert len(o) == 4
        assert obs.n_states(obs.RED_SCOUT_DIMS) == 96

    def test_footholds_are_bucketed_not_counted(self) -> None:
        # The strategic difference between four footholds and five is nil; between zero
        # and one it is everything. Counting would multiply the state space for nothing.
        state = fresh()
        assert obs.red_breach_observation(state)[1] == 0
        state.compromise("web-portal")
        assert obs.red_breach_observation(state)[1] == 1
        for host in ("mail", "reverse-proxy"):
            state.compromise(host)
        assert obs.red_breach_observation(state)[1] == 2
        for host in ("intranet", "fileserver", "dev-box"):
            state.compromise(host)
        assert obs.red_breach_observation(state)[1] == 3  # saturates at the top bucket


class TestEncoding:
    def test_encode_is_a_bijection_over_every_agent_s_state_space(self) -> None:
        """Exhaustive, because these spaces are small enough to enumerate -- which is
        itself the argument for tabular RL here."""
        spaces = [obs.blue_dims(a) for a in topo.DEFENDER_ZONES]
        spaces += [obs.RED_BREACH_DIMS, obs.RED_SCOUT_DIMS]
        for dims in spaces:
            seen = set()
            for combo in itertools.product(*(range(d) for d in dims)):
                idx = obs.encode(combo, dims)
                assert 0 <= idx < obs.n_states(dims)
                assert idx not in seen, f"collision at {combo}"
                seen.add(idx)
                assert obs.decode(idx, dims) == combo
            assert len(seen) == obs.n_states(dims)

    def test_out_of_range_digits_are_refused(self) -> None:
        # A digit that wrapped silently would merge two situations into one Q-table row.
        import pytest
        with pytest.raises(ValueError):
            obs.encode((4, 0, 0), (4, 4, 3))
        with pytest.raises(ValueError):
            obs.encode((0, 0), (4, 4, 3))

    def test_real_observations_encode_inside_the_table(self) -> None:
        state = fresh()
        state.compromise("ad-controller")
        state.alerts["ad-controller"] = 3.0
        for agent in topo.DEFENDER_ZONES:
            dims = obs.blue_dims(agent)
            assert 0 <= obs.encode(obs.blue_observation(state, agent), dims) < obs.n_states(dims)
        assert 0 <= obs.encode(obs.red_breach_observation(state),
                               obs.RED_BREACH_DIMS) < 384


class TestBudget:
    def test_every_agent_is_inside_the_ten_thousand_state_budget(self) -> None:
        # CLAUDE.md section 2, now covering red as well as blue.
        sizes = {a: obs.n_states(obs.blue_dims(a)) for a in topo.DEFENDER_ZONES}
        sizes["R_breach"] = obs.n_states(obs.RED_BREACH_DIMS)
        sizes["R_scout"] = obs.n_states(obs.RED_SCOUT_DIMS)
        for agent, size in sizes.items():
            assert size <= 10_000, f"{agent} needs {size} states"
        assert max(sizes.values()) == 3072  # B_corp, the worked example in section 5.1
