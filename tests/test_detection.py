"""Tests for the noisy alert process.

These are statistical tests, so they run many steps with a fixed seed and assert on
aggregates rather than on single draws. The properties being protected are the ones that
make blue's problem non-trivial: the base rate is comparable to the signal, detection is
never certain, and a quiet attacker is genuinely hard to see.
"""

from __future__ import annotations

import numpy as np

from marlsoc.env import detection as det
from marlsoc.env import topology as topo
from marlsoc.env.layers import LayerStatus
from marlsoc.env.state import EpisodeState, HostStatus
from marlsoc.env.topology import Zone


def fresh() -> EpisodeState:
    return EpisodeState.initial(LayerStatus.full())


class TestDetectionProbability:
    def test_isolated_hosts_are_silent(self) -> None:
        # Off the network: there is nothing left to observe, and an isolated host that
        # kept alerting would teach blue to isolate hosts it had already isolated.
        state = fresh()
        state.isolate("mail")
        assert det.detection_prob(state, "mail", action_noise=6.0) == 0.0

    def test_a_clean_host_alerts_at_the_base_rate(self) -> None:
        state = fresh()
        assert det.detection_prob(state, "mail", 0.0) == det.DEFAULT.false_alert_rate

    def test_noisier_actions_are_likelier_to_be_caught(self) -> None:
        """The stealth gradient. If this were flat, slow_scan would be pointless and
        Layer 1 would have nothing to beat it with."""
        state = fresh()
        state.compromise("ad-controller")
        quiet = det.detection_prob(state, "ad-controller", 1.0)   # slow_scan
        loud = det.detection_prob(state, "ad-controller", 6.0)    # escalate_privilege
        assert loud > quiet > det.DEFAULT.false_alert_rate

    def test_detection_is_never_certain(self) -> None:
        """Section 7.2's cliff-walking argument needs exploring near detection to be
        risky, not fatal. A certainly-caught action has a known answer and SARSA's
        advantage over Q-Learning for the attacker disappears."""
        state = fresh()
        state.compromise("ad-controller")
        assert det.detection_prob(state, "ad-controller", 1000.0) == det.DEFAULT.max_detect_rate
        assert det.DEFAULT.max_detect_rate < 1.0

    def test_touching_a_honeypot_is_near_certainly_noticed(self) -> None:
        state = fresh()
        state.deploy_honeypot("honeypot-1")
        p = det.detection_prob(state, "honeypot-1", action_noise=5.0)
        assert p == det.DEFAULT.honeypot_detect_rate
        assert p > det.DEFAULT.max_detect_rate  # honeypots beat the normal ceiling

    def test_an_untouched_honeypot_behaves_like_any_clean_host(self) -> None:
        # Otherwise blue could identify its own honeypots by their alert rate, which is
        # harmless, but red could too -- and a honeypot red can recognise is useless.
        state = fresh()
        state.deploy_honeypot("honeypot-1")
        assert det.detection_prob(state, "honeypot-1", 0.0) == det.DEFAULT.false_alert_rate


class TestStealthIsPossible:
    """The property the whole attacker side of the project depends on."""

    def test_an_idle_compromised_host_stays_below_the_suspicion_threshold(self) -> None:
        """Steady-state accumulation is p / (1 - decay). For an idle foothold that is
        0.15 / 0.15 = 1.0, under the 1.5 threshold -- so red can hide by doing nothing
        and ``wait`` is a genuine strategic option rather than a wasted turn."""
        cfg = det.DEFAULT
        idle = cfg.base_detect_rate / (1 - cfg.decay)
        assert idle < cfg.suspicion_threshold

    def test_acting_pushes_a_foothold_over_the_threshold(self) -> None:
        cfg = det.DEFAULT
        acting = (cfg.base_detect_rate + cfg.noise_gain * 1) / (1 - cfg.decay)
        assert acting > cfg.suspicion_threshold

    def test_a_single_alert_is_not_enough_to_look_suspicious(self) -> None:
        """At a threshold of exactly 1.0 one alert would trip it, 'suspicious' would
        collapse into 'alerted this step', and the sustained-versus-scattered
        distinction blue has to learn would not exist."""
        assert det.DEFAULT.suspicion_threshold > 1.0

    def test_alerts_decay_so_a_quiet_host_clears(self) -> None:
        """Without decay, every host crosses any threshold over T=250 steps at a 2%
        base rate, and late-episode blue observes a uniformly suspicious network
        carrying no information."""
        state = fresh()
        state.alerts["mail"] = 5.0
        rng = np.random.default_rng(1)
        for i in range(40):
            state.step = i
            det.step_alerts(state, rng, {})
        assert state.alerts["mail"] < det.DEFAULT.suspicion_threshold


class TestBaseRate:
    def test_false_alerts_are_comparable_in_volume_to_true_ones(self) -> None:
        """The base rate fallacy, asserted numerically.

        Twelve clean hosts at 2% each produce alerts at a rate within a factor of two of
        one compromised host acting periodically. If this ratio collapsed, blue could
        learn 'isolate whatever alerted' and section 5.3's tradeoff would vanish.
        """
        state = fresh()
        state.compromise("ad-controller")
        rng = np.random.default_rng(0)
        true_alerts = false_alerts = 0
        for i in range(400):
            state.step = i
            fired = det.step_alerts(state, rng, {"ad-controller": 6.0} if i % 5 == 0 else {})
            true_alerts += "ad-controller" in fired
            false_alerts += len(fired - {"ad-controller"})
        assert true_alerts > 0 and false_alerts > 0
        assert 0.4 < false_alerts / true_alerts < 2.5, (true_alerts, false_alerts)

    def test_turning_the_base_rate_off_makes_detection_trivial(self) -> None:
        # Kept as an executable statement of the ablation worth running once: with no
        # false alerts, any alert is proof, and blue converges on isolate-everything.
        cfg = det.DetectionConfig(false_alert_rate=0.0)
        state = fresh()
        rng = np.random.default_rng(3)
        for i in range(200):
            state.step = i
            det.step_alerts(state, rng, {}, cfg)
        assert all(v == 0.0 for v in state.alerts.values())


class TestMttdStampsOnlyOnTruePositives:
    def test_a_false_positive_does_not_stamp_detection(self) -> None:
        """MTTD must measure blue finding the attacker, not blue getting startled.

        Stamping on a false alert would make MTTD *improve* as the false-positive rate
        rises, which is exactly backwards.
        """
        state = fresh()
        rng = np.random.default_rng(7)
        for i in range(300):
            state.step = i
            det.step_alerts(state, rng, {})
        assert state.detected_step is None

    def test_a_loud_compromised_host_is_eventually_detected(self) -> None:
        state = fresh()
        state.compromise("ad-controller")
        rng = np.random.default_rng(2)
        for i in range(60):
            state.step = i
            det.step_alerts(state, rng, {"ad-controller": 6.0})
        assert state.detected_step is not None
        assert state.mttd is not None and state.mttd >= 0


class TestReproducibility:
    def test_the_same_seed_gives_the_same_alerts(self) -> None:
        # CLAUDE.md 3.7: an episode must be exactly replayable for the demo. The rng is
        # passed in rather than created here precisely so this holds.
        def run(seed: int) -> list[float]:
            state = fresh()
            state.compromise("web-portal")
            rng = np.random.default_rng(seed)
            for i in range(50):
                state.step = i
                det.step_alerts(state, rng, {"web-portal": 2.0})
            return sorted(state.alerts.values())

        assert run(42) == run(42)
        assert run(42) != run(43)


class TestZoneAlertLevel:
    def test_levels_are_ordered_and_bounded(self) -> None:
        state = fresh()
        assert det.zone_alert_level(state, Zone.CORP) == det.ALERT_LOW
        for host in topo.hosts_in(Zone.CORP):
            state.alerts[host.name] = 1.0
        assert det.zone_alert_level(state, Zone.CORP) == det.ALERT_HIGH

    def test_isolated_hosts_stop_contributing_to_the_level(self) -> None:
        """Otherwise isolating a noisy host would leave its alarm ringing forever and
        blue would keep reacting to a host it has already dealt with."""
        state = fresh()
        state.alerts["ad-controller"] = 5.0
        assert det.zone_alert_level(state, Zone.CORP) == det.ALERT_HIGH
        state.isolate("ad-controller")
        assert det.zone_alert_level(state, Zone.CORP) == det.ALERT_LOW

    def test_there_are_exactly_three_levels(self) -> None:
        # Blue's state size is 4**n * 3; a fourth level would multiply every Q-table.
        assert det.N_ALERT_LEVELS == 3


class TestHeat:
    def test_red_observes_its_own_noise_not_blues_alerts(self) -> None:
        """Letting red read blue's alert totals would hand it perfect information about
        the defender's beliefs and collapse the partial observability section 6 rests
        on."""
        state = fresh()
        for host in state.alerts:
            state.alerts[host] = 10.0
        assert det.heat_level(state) == det.ALERT_LOW  # blue is screaming; red cannot tell

    def test_heat_fades_at_the_same_rate_as_blues_memory(self) -> None:
        # Same rate on purpose: "go quiet and wait" is then worth the same to both
        # sides, so wait is a genuine option for red rather than a dominated one.
        state = fresh()
        state.heat = 10.0
        det.decay_heat(state)
        assert state.heat == 10.0 * det.DEFAULT.decay

    def test_heat_buckets_into_three_levels(self) -> None:
        state = fresh()
        state.heat = 0.0
        assert det.heat_level(state) == det.ALERT_LOW
        state.heat = 5.0
        assert det.heat_level(state) == det.ALERT_MEDIUM
        state.heat = 20.0
        assert det.heat_level(state) == det.ALERT_HIGH
