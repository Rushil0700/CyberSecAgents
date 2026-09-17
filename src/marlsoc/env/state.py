"""The true episode state -- the ground truth no agent is allowed to see.

This is ``S`` from the Markov Game in PROJECT.md section 6: "true global state (host
statuses, footholds, all six layer flags). **No agent observes it.**" Each agent sees
``o_i = O_i(s)``, computed in ``env/observations.py`` from a noisy alert process in
``env/detection.py``. Keeping ground truth in its own module is what makes that leak
structurally visible: if an agent ever imports this file, the partial observability the
project claims is gone.

Design note -- **the true status is three-valued, not four.**
PROJECT.md section 5.1 writes ``status in {clean, suspicious, compromised, isolated}``
and puts it straight into blue's state vector. CLAUDE.md amendment 3.1 overrides that,
and this is where the override lives. ``suspicious`` is not a fact about a host; it is a
*belief a defender holds*. If it were ground truth, a host would turn suspicious the
instant it was attacked -- mean time to detection would be exactly one step in every
episode, false positives would be impossible, and the security/availability tradeoff in
section 5.3 (the project's single best analysis paragraph) would not exist. So truth has
three values and ``SUSPICIOUS`` is manufactured downstream by the detection model. Blue's
Q-table still indexes four statuses; it simply no longer gets the fourth for free.

Design note -- **red's layer flags are views, not fields.**
Section 5.2 lists ``creds_held``, ``privilege_escalated`` and ``mfa_degraded`` among
red's state features. Those are precisely layers 3, 5 and 6, so they are *derived* from
``LayerStatus`` rather than stored beside it. Two copies of one fact is how a state
vector and a precondition model come to disagree, and the symptom -- red acting as if it
holds credentials the layer model says it lacks -- would be almost impossible to spot in
a learning curve.

Design note -- **metrics are stamped as they happen.**
CLAUDE.md 3.8: detection and containment times cannot be retrofitted. There is no way to
recover "when was this host first detected" from a terminal state, so the fields below
are written at the instant the event occurs and never recomputed. They are part of the
state for exactly that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from marlsoc.env import topology as topo
from marlsoc.env.layers import Layer, LayerStatus
from marlsoc.env.topology import Zone


class HostStatus(str, Enum):
    """What is *actually* true of a host. Three values; see the module docstring.

    ISOLATED is terminal for the episode: a host cut off the network cannot be
    re-attacked, and cannot serve the attacker as a foothold either. That is what makes
    ``isolate`` a real decision rather than a free action -- it removes the host from
    both sides of the game, which is why it carries an availability cost.
    """

    CLEAN = "clean"
    COMPROMISED = "compromised"
    ISOLATED = "isolated"


class Outcome(str, Enum):
    """How an episode ended (PROJECT.md section 3.4)."""

    RUNNING = "running"
    RED_WIN = "red_win"      # admin credentials altered
    BLUE_WIN = "blue_win"    # every attacker foothold isolated
    DRAW = "draw"            # step limit T reached


STEP_LIMIT: Final[int] = 250  # T, raised from 100 in section 3.4 for the six-layer path


@dataclass
class EpisodeState:
    """Ground truth for one episode.

    Mutable, unlike ``LayerStatus`` and ``Host``: this is the thing the transition
    function advances, and threading a full copy through every step would make the
    twin's 10,000-episodes-per-minute budget (section 8) much harder to hit. The
    immutable pieces are the ones that get shared by reference and could leak across
    episodes; this object is created fresh by ``reset()`` and never shared.

    Attributes:
        step: Timestep within the episode, 0 at reset.
        layers: Which of the six layers are active and which red has breached.
        true_status: Ground-truth status per host. Honeypot slots start absent from
            play and are added by blue's ``deploy_honeypot``.
        discovered: Hosts red knows exist. Recon (``scan`` / ``slow_scan``) grows this;
            red cannot act on a host it has not discovered, which is what gives
            ``R_scout`` something to contribute and creates the credit-assignment
            problem in section 4.1.
        heat: Red's accumulated noise, decayed each step. Drives detection probability
            and is bucketed into red's ``heat_level`` observation.
        alerts: Per-host accumulated alert points on the *blue* side. Written by
            ``detection.py``; the raw counter behind each zone's alert level.
        honeypots_live: Honeypot slots blue has deployed this episode.
        outcome: Terminal status, or RUNNING.
        detected_step: First step at which any truly compromised host was flagged
            suspicious. Numerator of mean time to detection.
        first_compromise_step: First step at which any host became compromised.
            MTTD is ``detected_step - first_compromise_step``; storing both rather than
            the difference keeps an episode that was never compromised distinguishable
            from one detected instantly.
        contained_step: Step at which every attacker foothold was simultaneously
            isolated. Feeds mean time to containment.
        false_positives: Count of clean hosts isolated. The availability cost in
            section 5.3 and the metric that exposes the degenerate policy.
        honeypot_hits: Times red acted against a live honeypot. Section 9's deception
            metric.
        breach_steps: Step at which each layer fell, for the layer-breach histogram
            section 15 recommends for diagnosing where red stalls.
    """

    layers: LayerStatus
    step: int = 0
    true_status: dict[str, HostStatus] = field(default_factory=dict)
    discovered: set[str] = field(default_factory=set)
    heat: float = 0.0
    alerts: dict[str, float] = field(default_factory=dict)
    honeypots_live: set[str] = field(default_factory=set)
    outcome: Outcome = Outcome.RUNNING

    # ---- metrics, stamped at the moment they occur -----------------------------------
    detected_step: int | None = None
    first_compromise_step: int | None = None
    contained_step: int | None = None
    false_positives: int = 0
    honeypot_hits: int = 0
    breach_steps: dict[Layer, int] = field(default_factory=dict)

    # ----------------------------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------------------------
    @classmethod
    def initial(cls, layers: LayerStatus) -> EpisodeState:
        """A fresh episode: every host clean, nothing discovered, no honeypots.

        Red starts with **no foothold at all**, not a free one on the entry host. Giving
        red ``web-portal`` for free would pre-breach Layer 2 and make the first rung of
        the section 5.4 ladder unearnable, which is exactly the signal a cold-start red
        needs most.
        """
        return cls(
            layers=layers,
            true_status={
                h.name: HostStatus.CLEAN
                for h in topo.HOSTS
                if not h.is_honeypot_slot
            },
            alerts={h.name: 0.0 for h in topo.HOSTS if not h.is_honeypot_slot},
        )

    # ----------------------------------------------------------------------------------
    # Views onto ground truth
    # ----------------------------------------------------------------------------------
    def status(self, host: str) -> HostStatus:
        """Ground-truth status. A honeypot not yet deployed reads as ISOLATED -- it is
        not on the network, so nothing can touch it."""
        return self.true_status.get(host, HostStatus.ISOLATED)

    @property
    def footholds(self) -> frozenset[str]:
        """Hosts red currently owns. Derived, never stored: a foothold *is* a host whose
        true status is COMPROMISED, and keeping a second list would let the two drift."""
        return frozenset(
            name for name, st in self.true_status.items() if st is HostStatus.COMPROMISED
        )

    @property
    def is_alive(self) -> bool:
        """Red still has somewhere to act from."""
        return bool(self.footholds)

    def current_zone(self) -> Zone:
        """The deepest zone red holds a foothold in -- red's ``current_zone`` feature.

        EDGE when red holds nothing, which is where it starts: outside, at the WAF.
        """
        if not self.footholds:
            return Zone.EDGE
        return max(
            (topo.BY_NAME[h].zone for h in self.footholds),
            key=lambda z: topo.ZONE_DEPTH[z],
        )

    def deepest_zone_depth(self) -> int:
        """How far in red has *ever* been, not where it is now.

        Distinct from ``current_zone`` because blue can isolate red back out of a zone;
        the progress feature should not un-learn when that happens, or red's state would
        oscillate between two rows of the Q-table for the same strategic situation.
        Computed from breached layers rather than from footholds for that reason.
        """
        return max(
            (topo.ZONE_DEPTH[z] for z in (Zone.EDGE, Zone.DMZ, Zone.CORP, Zone.SECURE)
             if self._zone_entered(z)),
            default=0,
        )

    def _zone_entered(self, zone: Zone) -> bool:
        gate = {
            Zone.EDGE: None,
            Zone.DMZ: Layer.PERIMETER,
            Zone.CORP: Layer.SEGMENTATION,
            Zone.SECURE: Layer.PRIVILEGE,
        }[zone]
        return gate is None or self.layers.is_breached(gate)

    # ---- section 5.2 flags: derived from the layer model, never stored ---------------
    @property
    def creds_held(self) -> bool:
        """Layer 3 breached."""
        return self.layers.is_breached(Layer.AUTH)

    @property
    def privilege_escalated(self) -> bool:
        """Layer 5 breached."""
        return self.layers.is_breached(Layer.PRIVILEGE)

    @property
    def mfa_degraded(self) -> bool:
        """Layer 6 breached."""
        return self.layers.is_breached(Layer.APPROVAL)

    # ----------------------------------------------------------------------------------
    # Mutations, each of which stamps its own metric
    # ----------------------------------------------------------------------------------
    def compromise(self, host: str) -> None:
        """Red owns ``host`` from now on. Stamps the compromise clock on first use."""
        self.true_status[host] = HostStatus.COMPROMISED
        if self.first_compromise_step is None:
            self.first_compromise_step = self.step

    def isolate(self, host: str) -> None:
        """Blue cuts ``host`` off. Counts a false positive if it was clean.

        The count is taken *before* the status changes, because after the write there is
        no longer any record of what the host was -- this is the retrofit problem in
        miniature.
        """
        if self.status(host) is HostStatus.CLEAN:
            self.false_positives += 1
        self.true_status[host] = HostStatus.ISOLATED
        if not self.footholds and self.contained_step is None:
            self.contained_step = self.step

    def mark_detected(self) -> None:
        """A truly compromised host has been flagged. Stamps MTTD once."""
        if self.detected_step is None:
            self.detected_step = self.step

    def record_breach(self, layer: Layer) -> None:
        """Advance the layer model and stamp when the layer fell."""
        self.layers = self.layers.breach(layer)
        self.breach_steps[layer] = self.step

    def deploy_honeypot(self, host: str) -> None:
        """Bring a honeypot slot onto the network. Idempotent: redeploying an already
        live honeypot wastes the action rather than resetting it, so blue cannot spam
        the action for free."""
        if host in self.honeypots_live:
            return
        self.honeypots_live.add(host)
        self.true_status[host] = HostStatus.CLEAN
        self.alerts.setdefault(host, 0.0)

    # ----------------------------------------------------------------------------------
    # Derived metrics (section 9)
    # ----------------------------------------------------------------------------------
    @property
    def mttd(self) -> int | None:
        """Steps from first compromise to first detection, or None if never detected."""
        if self.detected_step is None or self.first_compromise_step is None:
            return None
        return self.detected_step - self.first_compromise_step

    @property
    def mttc(self) -> int | None:
        """Steps from first compromise to full containment, or None if never contained."""
        if self.contained_step is None or self.first_compromise_step is None:
            return None
        return self.contained_step - self.first_compromise_step

    @property
    def hosts_compromised(self) -> int:
        return len(self.footholds)

    @property
    def layers_breached(self) -> int:
        """Section 9's headline metric, 0-6."""
        return self.layers.depth
