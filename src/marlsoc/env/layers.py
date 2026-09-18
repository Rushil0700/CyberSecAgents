"""The six-layer defence-in-depth model: preconditions, rewards and noise.

PROJECT.md section 3.1 is the specification this file implements. Its central claim is
that real defence in depth is *not* one control repeated -- it is different **kinds** of
control stacked, so no single attacker capability gets you through. Each of the six
layers below is beaten by a different class of action:

    L1 perimeter / WAF        -> slow_scan            (evade a rate limit)
    L2 DMZ network boundary   -> exploit a DMZ host   (own a host with inward reach)
    L3 authentication gate    -> steal_credentials    (identity, not network)
    L4 internal segmentation  -> lateral_move to pivot(a specific host, not any host)
    L5 privilege escalation   -> escalate_privilege   (rights on a host already held)
    L6 approval / MFA         -> degrade_mfa          (availability attack, not access)

This module imports ``topology`` and **nothing else** -- no episode state, no agents, no
randomness. Every predicate here takes plain booleans, so the whole layer model is unit
testable before an environment exists to run it in. ``env/state.py`` owns a
``LayerStatus`` as a field; the dependency runs that way and never back.

Design note -- **the layers are a partial order, not a chain.**
The obvious reading of section 3.1 is a corridor: L1, then L2, ... then L6. We
deliberately do not build that. ``APPROVAL`` (L6) requires only ``SEGMENTATION`` (L4),
not ``PRIVILEGE`` (L5), because once red holds the pivot it has network reach to the MFA
service and degrading it is physically possible right then::

    L1 -> L2 -> L3 -> L4 -+-> L5 -+-> alter_credentials
                          +-> L6 -+

A total order would make the "six-layer decision problem" a single path with no
decisions in it, and red would learn to press six buttons in sequence -- a corridor with
extra steps, which is exactly what section 3.1 argues against. With the branch, red must
*learn* an ordering, and that ordering interacts with detection: ``escalate_privilege``
is the noisiest action in the game, so taking it early means carrying heat for longer.
Whichever order red converges on is a finding rather than a script.

Design note -- **inactive is not the same as breached.**
The curriculum in section 7.4 switches layers on progressively: stage 1 runs with layers
1-2 only. An inactive layer must let actions pass *without* paying its breach reward,
otherwise stage 1 would hand red the +210 ladder for four layers it never fought and the
progressive reward signal that curriculum learning depends on would be meaningless. So
``LayerStatus`` carries two sets -- ``active`` and ``breached`` -- and "satisfied" means
inactive **or** breached.

Design note -- **why breach rewards live here and not in ``rewards.py``.**
The +10/+20/.../+60 ladder in section 5.4 is not a free parameter of the reward function;
it is a property of the layer, and section 5.4's own argument ("a lonely +100 at the end
of a six-layer chain gives random exploration no signal") only holds if the ladder is
monotone in layer depth. Keeping the number next to the layer means a test can assert
that monotonicity directly. ``rewards.py`` reads the ladder from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

from marlsoc.env.topology import Role


class Layer(IntEnum):
    """The six layers, numbered as in PROJECT.md section 3.1.

    ``IntEnum`` rather than ``Enum`` so that ``max(breached)`` and ``layer <= stage``
    work directly -- the curriculum and the "deepest layer reached" metric both want to
    compare layers by depth, and spelling that ``.value`` everywhere would be noise.
    """

    PERIMETER = 1        # rate limiting / IP reputation at the edge
    DMZ_BOUNDARY = 2     # network segmentation into the DMZ
    AUTH = 3             # identity: valid credentials
    SEGMENTATION = 4     # micro-segmentation to the pivot host
    PRIVILEGE = 5        # privilege escalation gate
    APPROVAL = 6         # MFA approval on a credential change


@dataclass(frozen=True)
class LayerSpec:
    """Everything that is true of a layer regardless of episode state.

    Attributes:
        layer: Which layer this is.
        control: The *class* of control, in security terms. Two layers sharing a class
            would be the "just add more zones" design section 3.1 rejects, so a test
            asserts all six are distinct.
        enforced_by: Role of the host that implements the control, or None for layers
            enforced by the firewall matrix rather than by a single host.
        beaten_by: Name of the red action class that breaches it. This is the 1:1
            action-to-layer mapping in section 4.1.
        requires: Layers that must be satisfied before this one can be attempted. The
            partial order described in the module docstring.
        breach_reward: The section 5.4 ladder entry, paid once, the first time.
        success_prob: Probability the breach attempt succeeds, when that probability is
            a property of the *action* rather than of a host. None means "look it up on
            the host" -- L1 and L6 use the gate host's ``bypass_prob``, L2 and L4 use the
            target host's ``exploit_prob``.
        noise: Alert points the attempt generates, win or lose. Feeds ``detection.py``.
            This is the stealth tradeoff: the deepest layers are also the loudest, so
            red's optimal path is not simply the shortest one.
    """

    layer: Layer
    control: str
    enforced_by: Role | None
    beaten_by: str
    requires: tuple[Layer, ...]
    breach_reward: float
    success_prob: float | None
    noise: int


# --------------------------------------------------------------------------------------
# The six layers (PROJECT.md sections 3.1 and 5.4)
# --------------------------------------------------------------------------------------
# success_prob is None wherever the probability belongs to a host rather than to the
# action, so that topology.py stays the only place a per-host number is written down.

LAYERS: Final[dict[Layer, LayerSpec]] = {
    Layer.PERIMETER: LayerSpec(
        layer=Layer.PERIMETER,
        control="rate limiting / IP reputation",
        enforced_by=Role.WAF,
        beaten_by="slow_scan",
        requires=(),
        breach_reward=10.0,
        success_prob=None,   # edge-gateway.bypass_prob
        noise=1,             # the whole point of slow_scan: quiet
    ),
    Layer.DMZ_BOUNDARY: LayerSpec(
        layer=Layer.DMZ_BOUNDARY,
        control="network segmentation",
        enforced_by=None,    # the firewall matrix, not a host
        beaten_by="exploit",
        requires=(Layer.PERIMETER,),
        breach_reward=20.0,
        success_prob=None,   # target host's exploit_prob
        noise=3,
    ),
    Layer.AUTH: LayerSpec(
        layer=Layer.AUTH,
        control="identity",
        enforced_by=None,    # every internal service checks the token
        beaten_by="steal_credentials",
        requires=(Layer.DMZ_BOUNDARY,),
        breach_reward=30.0,
        success_prob=0.55,   # a property of the action: read a credential cache
        noise=2,
    ),
    Layer.SEGMENTATION: LayerSpec(
        layer=Layer.SEGMENTATION,
        control="network micro-segmentation",
        enforced_by=Role.PIVOT,
        beaten_by="lateral_move",
        requires=(Layer.AUTH,),
        breach_reward=40.0,
        success_prob=None,   # ad-controller.exploit_prob, the hard one at 0.40
        noise=4,
    ),
    Layer.PRIVILEGE: LayerSpec(
        layer=Layer.PRIVILEGE,
        control="privilege",
        enforced_by=Role.CROWN_JEWEL,   # the auth-server ACL is what is being beaten
        beaten_by="escalate_privilege",
        requires=(Layer.SEGMENTATION,),
        breach_reward=50.0,
        success_prob=0.35,   # section 3.1: "noisy, high detection risk"
        noise=6,             # the loudest action in the game
    ),
    Layer.APPROVAL: LayerSpec(
        layer=Layer.APPROVAL,
        control="application control",
        enforced_by=Role.MFA_GATE,
        # NOTE the requires: L4, *not* L5. See the partial-order note in the docstring.
        beaten_by="degrade_mfa",
        requires=(Layer.SEGMENTATION,),
        breach_reward=60.0,
        success_prob=None,   # mfa-service.bypass_prob
        noise=5,
    ),
}

ALL_LAYERS: Final[tuple[Layer, ...]] = tuple(Layer)

# The reward for the terminal move itself (section 5.4). Kept here beside the ladder it
# tops so the "does the ladder actually reach the goal" argument is checkable in one file.
ALTER_CREDENTIALS_REWARD: Final[float] = 100.0

# Layers that must be breached before ``alter_credentials`` is legal at all: hold the
# crown jewel with escalated privilege (L5) *and* with MFA degraded (L6). Section 3.4,
# "the winning move".
WIN_REQUIRES: Final[tuple[Layer, ...]] = (Layer.PRIVILEGE, Layer.APPROVAL)


@dataclass(frozen=True)
class LayerStatus:
    """Which layers exist this episode, and which red has already broken.

    Frozen and returned-by-value: ``breach()`` hands back a new status rather than
    mutating. Episode state gets rewound and replayed (CLAUDE.md 3.7 requires exact
    reproducibility from a seed), and a mutable flag set shared by reference is the
    classic way for one episode's progress to leak into the next.

    Attributes:
        active: Layers switched on this episode. The curriculum (section 7.4) narrows
            this; a full run has all six.
        breached: Layers red has actually broken, a subset of ``active``.
    """

    active: frozenset[Layer]
    breached: frozenset[Layer] = frozenset()

    @classmethod
    def for_stage(cls, deepest: Layer | int) -> LayerStatus:
        """Layers 1..``deepest`` active, nothing breached -- one curriculum stage.

        Section 7.4's stages are "layers 1-2", "layers 1-3" and so on, so a stage is
        fully described by its deepest active layer.
        """
        return cls(active=frozenset(ly for ly in Layer if ly <= int(deepest)))

    @classmethod
    def full(cls) -> LayerStatus:
        """All six layers active: the real game, and curriculum stage 5."""
        return cls(active=frozenset(ALL_LAYERS))

    def is_active(self, layer: Layer) -> bool:
        return layer in self.active

    def is_breached(self, layer: Layer) -> bool:
        """Red has actually broken this layer. An inactive layer is never *breached*."""
        return layer in self.breached

    def is_satisfied(self, layer: Layer) -> bool:
        """This layer is not standing in red's way -- switched off, or already broken.

        The distinction from ``is_breached`` is the whole of the second design note:
        preconditions ask "satisfied?", rewards ask "breached?".
        """
        return layer not in self.active or layer in self.breached

    def can_attempt(self, layer: Layer) -> bool:
        """Whether red may even try this layer: active, unbroken, prerequisites met.

        This is the layer half of action masking (PROJECT.md section 3.4). The other
        half -- do I hold a host with network reach to the target -- lives in
        ``actions.py``, because it needs episode state and this module has none.
        """
        if not self.is_active(layer) or self.is_breached(layer):
            return False
        return all(self.is_satisfied(req) for req in LAYERS[layer].requires)

    def attemptable(self) -> frozenset[Layer]:
        """Every layer red could currently attack. The frontier of the partial order."""
        return frozenset(ly for ly in ALL_LAYERS if self.can_attempt(ly))

    def breach(self, layer: Layer) -> LayerStatus:
        """Return a new status with ``layer`` broken.

        Raises:
            ValueError: if the layer was not attemptable. Silently permitting an
                out-of-order breach would mean the six-layer model is not actually
                enforcing anything, and the only symptom would be red looking
                mysteriously good -- so we fail loudly instead.
        """
        if not self.can_attempt(layer):
            raise ValueError(f"{layer.name} is not currently attemptable")
        return LayerStatus(active=self.active, breached=self.breached | {layer})

    def restore(self, layer: Layer) -> LayerStatus:
        """Return a new status with ``layer`` repaired -- blue's reinforcement actions.

        Red must then breach it again, which is what turns a single intrusion into the
        arms race section 9 wants to plot. Restoring an unbreached layer is a no-op
        rather than an error, because the action mask already prevents it and a second
        guard here would only duplicate that rule in two places.
        """
        if layer not in self.breached:
            return self
        return LayerStatus(active=self.active, breached=self.breached - {layer})

    def can_win(self) -> bool:
        """Whether ``alter_credentials`` is unblocked by the layer model.

        Still requires holding the crown jewel, which is episode state; see
        ``actions.py``. Note ``is_satisfied``: in a curriculum stage where L6 is off,
        MFA is simply not in the way, so a stage-4 red can win without degrading it.
        """
        return all(self.is_satisfied(ly) for ly in WIN_REQUIRES)

    def objective_met(self, ever_breached: frozenset[Layer] | set[Layer] | None = None) -> bool:
        """Whether red has achieved **this stage's** objective.

        Args:
            ever_breached: Layers red has broken at any point this episode. Defaults to
                the currently-breached set, but the caller should pass the monotone one
                (``EpisodeState.paid_breaches``). A stage objective is an *achievement*,
                not a state red has to hold: testing the current set would let blue deny
                the win forever by repairing one layer each time red completed the set,
                which is the reward-farm loop again with the reward removed.

        PROJECT.md section 7.4's stage table says stage 1 (layers 1-2) should teach red
        to "get a foothold in the DMZ". Requiring the crown jewel at every stage does not
        do that: switching off layers 3-6 removes the obstacles but leaves the same
        full-length journey, so a "shallow" stage is not shallow -- it is the whole
        network with the defences turned off, which is *easier for red*, the opposite of
        a curriculum.

        So a stage's objective is to breach every **active** layer. At stage 1 that is
        exactly a DMZ foothold, which is what section 7.4 asks for.

        Layer 6 is the exception, and for a real reason rather than a special case:
        ``alter_credentials`` is the move Layer 6 guards, so when Layer 6 is in play
        breaching it is not the objective -- executing the move it was protecting is.
        The environment checks that separately.
        """
        if Layer.APPROVAL in self.active:
            return False   # the full game ends on alter_credentials, not on a layer
        reached = self.breached if ever_breached is None else frozenset(ever_breached)
        return self.active <= reached

    @property
    def depth(self) -> int:
        """How many layers red has broken -- the section 9 "layers breached" metric.

        A count rather than ``max(breached)`` precisely because the order is partial:
        breaching L6 before L5 must not report a depth of 6. Counting degrades
        gracefully, which section 9 says is why this graph beats binary win/lose.
        """
        return len(self.breached)


def breach_reward(layer: Layer) -> float:
    """The section 5.4 progressive reward for breaching ``layer``."""
    return LAYERS[layer].breach_reward


def total_ladder() -> float:
    """Sum of every breach reward plus the win. Sanity figure for the report: 310."""
    return sum(spec.breach_reward for spec in LAYERS.values()) + ALTER_CREDENTIALS_REWARD


def layer_for_action(action_class: str) -> Layer | None:
    """Reverse the 1:1 action-to-layer mapping in PROJECT.md section 4.1.

    Returns None for actions that breach nothing (``scan``, ``wait``,
    ``alter_credentials`` -- the last of which is the win, not a layer).
    """
    for spec in LAYERS.values():
        if spec.beaten_by == action_class:
            return spec.layer
    return None
