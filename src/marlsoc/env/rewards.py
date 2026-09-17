"""``R_i`` -- the reward functions for both teams (PROJECT.md sections 5.3 and 5.4).

Design note -- **rewards price events; they do not inspect history.**
The transition function records what happened during a step into a ``StepEvents``
record, and these functions price that record plus the standing per-step costs. Keeping
it that way means the reward function is nearly pure and can be unit tested by handing
it an event record, with no environment needed. It also makes the reward table in the
report a literal transcription of this file rather than a reconstruction of it.

Design note -- **the general-sum property is a consequence, not a decoration.**
Section 6 claims a general-sum Markov Game: payoffs do not cancel. A honeypot engagement
is +30 blue and -30 red, which does cancel; a false positive is -20 blue and 0 red, which
does not. That asymmetry is the whole content of "general-sum, not zero-sum", and it is
asserted in the tests rather than trusted.

Design note -- **the availability cost is the interesting parameter.**
See ``config.AvailabilityCost``. Section 5.3's one-shot -20 does not prevent the
degenerate isolate-everything policy, because the benefit of over-isolating is per-step
while the penalty is not. The default here charges per step and the degenerate policy
loses by a factor of twenty-five.

Design note -- **red's ladder is read from ``layers.py``.**
The +10/+20/.../+60 progression is a property of the layer, not a free parameter of the
reward function, and section 5.4's sparse-reward argument only holds if the ladder is
monotone in depth. Reading it from one place means a test can assert that monotonicity
where the numbers actually live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from marlsoc.config import AvailabilityCost
from marlsoc.env import layers as lyr
from marlsoc.env.layers import Layer
from marlsoc.env.state import EpisodeState, HostStatus


@dataclass(frozen=True)
class RewardConfig:
    """Every number in PROJECT.md sections 5.3 and 5.4, in one sweepable object.

    Attributes:
        red_win: Paid to red for altering the admin credentials; the negation is charged
            to the whole blue team.
        compromised_host_per_step: The bleed. Per compromised host, per step.
        correct_isolation: Blue isolating a genuinely compromised host.
        false_positive: Blue isolating a clean host. One-shot under either availability
            mode -- this is the "you got it wrong" charge, distinct from the "this host
            is down" charge below.
        isolated_host_per_step: Charged per isolated host per step under PER_STEP. Applies
            to *correctly* isolated hosts too: containment should be worth taking (+50,
            and it stops the -10 bleed, netting +8 per step) but never free.
        honeypot_engagement: Blue's payoff when red touches a decoy.
        honeypot_cost: Red's side of the same event. Equal and opposite, deliberately.
        layer_restored: Blue repairing a breached layer -- **paid once per layer per
            episode**. PROJECT.md section 5.3 specifies a flat +25 per restoration, and
            that is farmable: ``tighten_ratelimit`` is legal whenever Layer 1 is down,
            red re-breaches Layer 1 with ``slow_scan`` at p=0.60 (about two steps), so
            blue earns roughly +12 a step from the repair loop -- more than the -10 a
            step bleed it is meant to be preventing. Blue then *profits from red
            succeeding*. Measured on a trained agent before the fix: 35.1 restorations
            per episode, contributing +175,750 against a total return of -115,338, or
            152% of the absolute reward. The agent had learned to farm the shaping term
            instead of defending, and scored worse than a random defender while doing it.
            Paying once per layer keeps the intent -- repairing a breached defence is
            worth something -- and removes the loop.
        step_cost: Charged to both teams, every step. Makes dithering expensive and is
            why an agent that can win in forty steps beats one that takes two hundred.
        detected: Charged to red when a compromised host of its is confirmed. This is the
            cliff in section 7.2's cliff-walking argument.
    """

    # --- blue (section 5.3) ---
    red_win: float = 100.0
    compromised_host_per_step: float = -10.0
    correct_isolation: float = 50.0
    false_positive: float = -20.0
    isolated_host_per_step: float = -2.0
    honeypot_engagement: float = 30.0
    layer_restored: float = 25.0

    # --- red (section 5.4) ---
    honeypot_cost: float = -30.0
    detected: float = -50.0

    # --- both ---
    step_cost: float = -1.0

    availability_cost: AvailabilityCost = AvailabilityCost.PER_STEP


DEFAULT: Final[RewardConfig] = RewardConfig()


@dataclass
class StepEvents:
    """What happened during one timestep. Filled by the transition function.

    Everything the reward functions need, and nothing else. A field here is a commitment
    that the environment records the event at the moment it occurs, which is CLAUDE.md
    3.8 applied to rewards as well as to metrics.

    Attributes:
        layers_breached: Layers red broke this step **for the first time**. Usually zero
            or one. A re-breach after blue repaired a layer is excluded: it is recovering
            lost ground, not a new rung of the 5.4 ladder, and paying it again would mean
            red profits from blue defending -- the mirror of the restoration farm
            described under ``RewardConfig.layer_restored``.
        layers_restored: Layers blue repaired this step.
        hosts_compromised: Hosts red took this step.
        correct_isolations: Compromised hosts blue isolated this step.
        false_positives: Clean hosts blue isolated this step.
        honeypot_hits: Times red acted against a live honeypot.
        red_detected: Whether a compromised host was confirmed this step.
        red_won: Whether red altered the admin credentials this step.
    """

    layers_breached: list[Layer] = field(default_factory=list)
    layers_restored: list[Layer] = field(default_factory=list)
    hosts_compromised: list[str] = field(default_factory=list)
    correct_isolations: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)
    honeypot_hits: int = 0
    red_detected: bool = False
    red_won: bool = False


def blue_reward(
    state: EpisodeState,
    events: StepEvents,
    cfg: RewardConfig = DEFAULT,
) -> float:
    """``R_blue`` for one step, shared across all three defenders.

    Shared rather than per-agent because section 6 makes cooperation a consequence of a
    shared return rather than an instruction: nobody is told to cooperate, it is simply
    how you maximise a reward you both receive. The individual-reward variant is the
    headline experiment in section 6, and it is a different function, not a flag here.

        R = -100 [red won]
            - 10 * (compromised hosts)          per step
            + 50 * (correct isolations)
            - 20 * (false positives)
            + 30 * (honeypot engagements)
            + 25 * (layers restored)
            -  2 * (isolated hosts)             per step, PER_STEP mode only
            -  1                                per step
    """
    reward = cfg.step_cost

    if events.red_won:
        reward -= cfg.red_win

    # The bleed: every host red currently holds costs blue every step it holds it.
    reward += cfg.compromised_host_per_step * state.hosts_compromised

    reward += cfg.correct_isolation * len(events.correct_isolations)
    reward += cfg.false_positive * len(events.false_positives)
    reward += cfg.honeypot_engagement * events.honeypot_hits
    reward += cfg.layer_restored * len(events.layers_restored)

    if cfg.availability_cost is AvailabilityCost.PER_STEP:
        # The fix for the degenerate policy. See config.AvailabilityCost.
        isolated = sum(
            1 for st in state.true_status.values() if st is HostStatus.ISOLATED
        )
        reward += cfg.isolated_host_per_step * isolated

    return reward


def red_reward(
    state: EpisodeState,
    events: StepEvents,
    cfg: RewardConfig = DEFAULT,
) -> float:
    """``R_red`` for one step, shared across both attackers.

    Shared so that ``R_scout`` earns credit for discoveries ``R_breach`` later converts
    -- section 4.1's credit-assignment discussion. Under individual rewards the scout
    would receive almost nothing, since finding a host pays only when someone else
    exploits it much later.

        R = + the section 5.4 ladder for each layer breached this step
            + 100 [credentials altered]
            -  50 [detected]
            -  30 * (honeypot engagements)
            -   1                              per step
    """
    reward = cfg.step_cost

    for layer in events.layers_breached:
        reward += lyr.breach_reward(layer)

    if events.red_won:
        reward += lyr.ALTER_CREDENTIALS_REWARD

    if events.red_detected:
        reward += cfg.detected

    reward += cfg.honeypot_cost * events.honeypot_hits

    return reward


def degenerate_policy_cost(
    n_hosts: int,
    horizon: int,
    cfg: RewardConfig = DEFAULT,
) -> float:
    """What "isolate every host at step 1" costs blue over a full episode.

    Exists so the argument in ``config.AvailabilityCost`` is executable rather than
    asserted in a docstring -- section 5.3 calls this the project's single best analysis
    paragraph, and a paragraph whose arithmetic nobody can rerun is a weaker one.

    Args:
        n_hosts: Number of hosts isolated.
        horizon: Episode length.
        cfg: Reward parameters.

    Returns:
        Total reward (negative) for the lockdown, excluding the step cost.
    """
    one_shot = cfg.false_positive * n_hosts
    if cfg.availability_cost is AvailabilityCost.ONE_SHOT:
        return one_shot
    return one_shot + cfg.isolated_host_per_step * n_hosts * horizon
