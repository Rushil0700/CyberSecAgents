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

from marlsoc.config import AvailabilityCost, RewardShaping
from marlsoc.env import actions as act
from marlsoc.env import layers as lyr
from marlsoc.env import topology as topo
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
        blocked_host_per_step: Charged per blocked host per step. Smaller than isolation,
            because blocking restricts a host rather than removing it -- but **not zero**.
            A preventive control that is both free and effective strictly dominates
            everything else: blue would simply blanket-block its whole zone and the
            section 5.3 tradeoff would have nothing left to trade. At -0.5, covering all
            four Edge/DMZ hosts costs about as much as one isolation, so blue has to
            choose *where* to spend prevention rather than applying it everywhere.
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
    blocked_host_per_step: float = -0.5
    honeypot_engagement: float = 30.0
    layer_restored: float = 25.0

    # --- red (section 5.4) ---
    honeypot_cost: float = -30.0
    detected: float = -50.0

    # --- both ---
    step_cost: float = -1.0

    availability_cost: AvailabilityCost = AvailabilityCost.PER_STEP
    shaping: RewardShaping = RewardShaping.RAW_LADDER

    # Must match the learner's discount factor. If the two drift apart the Ng et al.
    # invariance guarantee no longer holds -- the shaping stops being free and starts
    # quietly re-weighting depth again, which is the bug this flag exists to fix.
    shaping_gamma: float = 0.95


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
    potential_before: float = 0.0
    terminal: bool = False


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
            + 25 * (layers restored, once each)
            -  2 * (isolated hosts)             per step, PER_STEP mode only
            -0.5 * (blocked hosts)              per step, PER_STEP mode only
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
        reward += cfg.blocked_host_per_step * state.blocked_count

    return reward


def owns(agent: str, host: str) -> bool:
    """Whether ``host`` falls in ``agent``'s zones.

    Attribution for the individual-reward variant is exact rather than heuristic: action
    spaces are built per zone, so the only defender that *can* isolate a host in a zone is
    the one that owns it. Reading ownership off the topology therefore recovers who acted,
    without ``StepEvents`` having to carry an actor field it needs for nothing else.
    """
    return topo.BY_NAME[host].zone in topo.DEFENDER_ZONES[agent]


def blue_reward_individual(
    agent: str,
    state: EpisodeState,
    events: StepEvents,
    cfg: RewardConfig = DEFAULT,
) -> float:
    """``R_blue_i`` -- each defender paid only for its own zone. PROJECT.md section 6.

    The headline emergent-behaviour experiment, and deliberately a **separate function**
    rather than a flag on ``RewardConfig``: it is a different reward *structure*, not a
    different number, and hiding a structural change behind a boolean is how a sweep ends
    up comparing two things nobody can name afterwards.

    Every term is restricted to hosts the agent owns:

    - the **bleed** counts only compromised hosts in its zones, so letting an attacker
      through stops being its problem the moment the attacker leaves;
    - **isolations and false positives** are its own, by the attribution argument in
      ``owns``;
    - the **-100 for losing the crown jewel** lands only on the defender whose zone holds
      it, which is ``B_secure``.

    That last line is what makes section 6's prediction sharp. ``B_dmz`` has **no stake at
    all** in the crown jewel: isolating is cheap for it, and the entire downstream cost of
    a missed intrusion is charged to somebody else. The prediction is that it becomes
    trigger-happy -- more isolations, more false positives -- while shared reward teaches
    restraint and hand-off. Nobody is instructed to cooperate or to defect in either case;
    the behaviour falls out of which return each agent is maximising.
    """
    reward = cfg.step_cost

    if events.red_won and owns(agent, topo.CROWN_JEWEL):
        reward -= cfg.red_win

    reward += cfg.compromised_host_per_step * sum(
        1 for host, st in state.true_status.items()
        if st is HostStatus.COMPROMISED and owns(agent, host)
    )
    reward += cfg.correct_isolation * sum(
        1 for h in events.correct_isolations if owns(agent, h)
    )
    reward += cfg.false_positive * sum(
        1 for h in events.false_positives if owns(agent, h)
    )
    reward += cfg.layer_restored * sum(
        1 for layer in events.layers_restored
        if _restorer(layer) == agent
    )

    # The honeypot payoff belongs to whoever runs the decoy that was touched. Events carry
    # only a count, so it is split evenly rather than misattributed -- honeypots are rare
    # enough that the approximation changes no conclusion, and guessing an owner would.
    if events.honeypot_hits:
        reward += (cfg.honeypot_engagement * events.honeypot_hits
                   / len(topo.DEFENDER_ZONES))

    if cfg.availability_cost is AvailabilityCost.PER_STEP:
        reward += cfg.isolated_host_per_step * sum(
            1 for host, st in state.true_status.items()
            if st is HostStatus.ISOLATED and owns(agent, host)
        )
        reward += cfg.blocked_host_per_step * sum(
            1 for host in state.blocked_hosts() if owns(agent, host)
        )

    return reward


def _restorer(layer: Layer) -> str | None:
    """Which defender repairs ``layer``, or None if no action restores it.

    Read from ``actions.AGENT_REINFORCE``, which is the authority: section 4.2 gives each
    defender exactly one reinforcement action, and that mapping is what the environment
    actually executes. Deriving it from ``LayerSpec.enforced_by`` instead looks equivalent
    and is not -- Layer 3 (AUTH) has ``enforced_by = None`` because no single host
    implements it, yet ``B_corp`` restores it with ``rotate_credentials``. That version
    paid ``B_corp`` nothing for its own repair action.
    """
    for agent, verb in act.AGENT_REINFORCE.items():
        if act.REINFORCE_LAYER[verb] is layer:
            return agent
    return None


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

        R = + gamma * Phi(s') - Phi(s)     potential-based shaping over the 5.4 ladder
            + 100 [credentials altered]
            -  50 [detected]
            -  30 * (honeypot engagements)
            -   1                              per step
    """
    reward = cfg.step_cost

    if cfg.shaping is RewardShaping.RAW_LADDER:
        # PROJECT.md section 5.4 as written: a flat bonus per rung, kept. Retained so the
        # "attacker learns to lose on purpose" result can be reproduced -- see
        # config.RewardShaping.
        for layer in events.layers_breached:
            reward += lyr.breach_reward(layer)
    else:
        #   F(s, s') = gamma * Phi(s') - Phi(s),   Phi(terminal) = 0
        # Ng, Harada & Russell (1999). Phi is the banked ladder value; the terminal zero
        # is the clause that stops red treating shaping as a destination, because the
        # discounted sum over a trajectory then telescopes to zero.
        phi_after = (
            0.0
            if events.terminal
            else lyr.cumulative_breach_reward(state.paid_breaches)
        )
        reward += cfg.shaping_gamma * phi_after - events.potential_before

    if events.red_won:
        reward += lyr.ALTER_CREDENTIALS_REWARD

    if events.red_detected:
        reward += cfg.detected

    reward += cfg.honeypot_cost * events.honeypot_hits

    return reward


def attack_margin(
    steps_to_win: int,
    detections: int,
    gamma: float,
    cfg: RewardConfig = DEFAULT,
) -> float:
    """How much red prefers winning to idling, in *discounted* return.

    Exists for the same reason ``degenerate_policy_cost`` does: CLAUDE.md 3.18's argument
    for keeping the raw ladder is arithmetic, and arithmetic nobody can rerun is a weaker
    argument. Positive means attacking is worth it; negative means red's optimal policy is
    to sit still, and a learner that sits still is correct rather than broken.

    Under ``RewardShaping.POTENTIAL_BASED`` the shaping telescopes to zero over any
    trajectory, so it contributes nothing here and red is left with the terminal reward
    against the time and the detections spent reaching it. Under ``RAW_LADDER`` the rungs
    red keeps are added, which is the entire difference between a red that attacks and a
    red that does not.

    Args:
        steps_to_win: Length of a winning episode.
        detections: Detections suffered on the way, each charged ``cfg.detected``.
        gamma: The learner's discount factor.
        cfg: Reward parameters.

    Returns:
        Discounted value of winning minus the discounted value of idling to the horizon.
    """
    discount = gamma ** steps_to_win
    win = lyr.ALTER_CREDENTIALS_REWARD * discount
    # Detections land somewhere in the middle of the run; the midpoint is the fair
    # summary and the conclusion does not turn on the choice.
    win += cfg.detected * detections * (gamma ** (steps_to_win // 2))
    if cfg.shaping is RewardShaping.RAW_LADDER:
        win += lyr.total_ladder() - lyr.ALTER_CREDENTIALS_REWARD

    def time_cost(n: int) -> float:
        return cfg.step_cost * (1.0 - gamma ** n) / (1.0 - gamma)

    idle = time_cost(250)
    return (win + time_cost(steps_to_win)) - idle


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
