"""Tests for the true episode state.

Two things are being protected here. First, that ground truth stays three-valued, so the
detection problem is real rather than free (CLAUDE.md 3.1). Second, that every section 9
metric is stamped at the instant its event happens -- these are the numbers the whole
evaluation rests on, and a metric that is subtly wrong produces a plausible-looking
report rather than a crash.
"""

from __future__ import annotations

from marlsoc.env import topology as topo
from marlsoc.env.layers import Layer, LayerStatus
from marlsoc.env.state import EpisodeState, HostStatus, Outcome
from marlsoc.env.topology import Zone


def fresh() -> EpisodeState:
    return EpisodeState.initial(LayerStatus.full())


class TestGroundTruthIsThreeValued:
    """CLAUDE.md amendment 3.1 -- the amendment that makes detection non-trivial."""

    def test_there_is_no_suspicious_true_status(self) -> None:
        """If ``suspicious`` were ground truth, detection would be free: every attacked
        host would flag immediately, MTTD would be 1 in every episode and a false
        positive would be impossible. It is a *belief*, so it belongs to observations."""
        assert {s.value for s in HostStatus} == {"clean", "compromised", "isolated"}
        assert not hasattr(HostStatus, "SUSPICIOUS")

    def test_every_non_honeypot_host_starts_clean(self) -> None:
        state = fresh()
        for host in topo.HOSTS:
            if host.is_honeypot_slot:
                continue
            assert state.status(host.name) is HostStatus.CLEAN

    def test_undeployed_honeypots_are_off_the_network(self) -> None:
        # Absent at reset in the twin, behind a compose profile in the lab. Reading as
        # ISOLATED means nothing can touch them until blue deploys them.
        state = fresh()
        for host in topo.HOSTS:
            if host.is_honeypot_slot:
                assert state.status(host.name) is HostStatus.ISOLATED


class TestRedStartsOutside:
    def test_red_holds_nothing_at_reset(self) -> None:
        """A free foothold on the entry host would pre-breach Layer 2 and make the
        first rung of the 5.4 ladder unearnable -- the rung a cold-start red needs
        most."""
        state = fresh()
        assert state.footholds == frozenset()
        assert not state.is_alive
        assert state.current_zone() is Zone.EDGE

    def test_current_zone_is_the_deepest_foothold(self) -> None:
        state = fresh()
        state.compromise("web-portal")
        assert state.current_zone() is Zone.DMZ
        state.compromise("ad-controller")
        assert state.current_zone() is Zone.CORP
        # Losing the deep foothold walks the zone back...
        state.isolate("ad-controller")
        assert state.current_zone() is Zone.DMZ

    def test_deepest_zone_reached_does_not_walk_back(self) -> None:
        """...but *progress* must not un-learn when blue pushes red out.

        If it did, red's state would flip between two Q-table rows for the same
        strategic situation and the values learned in each would keep contradicting the
        other.
        """
        state = fresh()
        state.record_breach(Layer.PERIMETER)
        state.compromise("web-portal")
        before = state.deepest_zone_depth()
        state.isolate("web-portal")
        assert state.deepest_zone_depth() == before


class TestLayerFlagsAreDerived:
    """Section 5.2's three flags must be views onto the layer model, never copies."""

    def test_flags_follow_the_layer_model(self) -> None:
        state = fresh()
        assert not state.creds_held
        for layer in (Layer.PERIMETER, Layer.DMZ_BOUNDARY, Layer.AUTH):
            state.record_breach(layer)
        assert state.creds_held
        assert not state.privilege_escalated
        assert not state.mfa_degraded

    def test_flags_cannot_be_set_independently_of_the_layers(self) -> None:
        # They are read-only properties; there is no setter to drift out of sync with.
        state = fresh()
        for name in ("creds_held", "privilege_escalated", "mfa_degraded"):
            assert isinstance(getattr(type(state), name), property)
            assert getattr(type(state), name).fset is None


class TestMetricsAreStampedWhenTheyHappen:
    """CLAUDE.md 3.8 -- none of these can be reconstructed from a terminal state."""

    def test_mttd_measures_from_first_compromise_not_from_reset(self) -> None:
        state = fresh()
        state.step = 10
        state.compromise("web-portal")
        state.step = 14
        state.mark_detected()
        assert state.mttd == 4

    def test_detection_is_stamped_once(self) -> None:
        # Otherwise MTTD reports the *latest* detection and blue looks slower the better
        # it is at finding things.
        state = fresh()
        state.step = 2
        state.compromise("web-portal")
        state.mark_detected()
        state.step = 40
        state.mark_detected()
        assert state.detected_step == 2

    def test_mttd_is_none_when_nothing_was_ever_compromised(self) -> None:
        # Distinguishable from "detected instantly", which a stored difference would not
        # be. Averaging a 0 in for every clean episode would flatter blue badly.
        assert fresh().mttd is None

    def test_false_positive_is_counted_before_the_status_is_overwritten(self) -> None:
        """The retrofit problem in miniature: after the write there is no record of what
        the host used to be."""
        state = fresh()
        state.isolate("mail")
        assert state.false_positives == 1
        state.compromise("web-portal")
        state.isolate("web-portal")
        assert state.false_positives == 1  # a true positive, not counted

    def test_containment_stamps_when_the_last_foothold_falls(self) -> None:
        state = fresh()
        state.step = 5
        state.compromise("web-portal")
        state.compromise("mail")
        state.step = 9
        state.isolate("web-portal")
        assert state.contained_step is None  # one foothold still standing
        state.step = 12
        state.isolate("mail")
        assert state.contained_step == 12
        assert state.mttc == 7

    def test_layer_breach_steps_are_recorded_for_the_histogram(self) -> None:
        # Section 15 diagnoses a flatlining red by asking *where* it stalls, which needs
        # per-layer timing rather than a final count.
        state = fresh()
        state.step = 3
        state.record_breach(Layer.PERIMETER)
        state.step = 11
        state.record_breach(Layer.DMZ_BOUNDARY)
        assert state.breach_steps == {Layer.PERIMETER: 3, Layer.DMZ_BOUNDARY: 11}
        assert state.layers_breached == 2


class TestHoneypots:
    def test_deploying_brings_the_slot_onto_the_network(self) -> None:
        state = fresh()
        state.deploy_honeypot("honeypot-1")
        assert state.status("honeypot-1") is HostStatus.CLEAN
        assert "honeypot-1" in state.honeypots_live

    def test_redeploying_is_a_wasted_action_not_a_reset(self) -> None:
        """Blue must not be able to spam deploy_honeypot for free -- an action that
        always 'succeeds' with no cost distorts every comparison against noop."""
        state = fresh()
        state.deploy_honeypot("honeypot-1")
        state.compromise("honeypot-1")
        state.deploy_honeypot("honeypot-1")
        assert state.status("honeypot-1") is HostStatus.COMPROMISED


class TestOutcomes:
    def test_episodes_start_running(self) -> None:
        assert fresh().outcome is Outcome.RUNNING

    def test_the_three_terminal_outcomes_exist(self) -> None:
        # Section 3.4: red win, blue win, draw at T.
        assert {o.value for o in Outcome} == {"running", "red_win", "blue_win", "draw"}
