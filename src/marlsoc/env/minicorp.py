"""The simulated twin: ``P``, ``reset()`` and ``step()``.

This is the transition function of the Markov Game in PROJECT.md section 6, and the
only file in ``env/`` that mutates anything. Everything it needs has already been
decided elsewhere and is imported rather than re-derived:

    topology.py      who can reach whom, and how hard each host is
    layers.py        which layers exist and what order they may fall in
    state.py         what is true
    detection.py     what blue gets to find out
    observations.py  what each agent is handed
    actions.py       what each agent may choose
    rewards.py       what each outcome is worth

"Where is your transition function?" is answered by opening this file, which is the
one-file-per-MDP-symbol convention in CLAUDE.md section 4.

Design note -- **simultaneous choice, ordered application.**
Section 6 is a Markov Game: all five agents choose from the *same* state ``s`` and the
joint action ``a = (a_1..a_5)`` is applied to it. So ``step()`` takes every agent's
action at once. Only the *application* needs an order, and it is red, then blue, then
detection. That means blue isolating a host red took during the same step counts as a
correct catch -- but blue chose that action before seeing red's move, so it is a lucky
guess rather than clairvoyance, and it makes red's gains revocable, which is the arms
race section 9 wants to plot.

Design note -- **the environment never looks at whose policy produced an action.**
``step()`` takes actions, not agents. Whether an action came from a Q-table, a script, a
random draw or the LLM advisor is invisible here. That is what lets the same environment
serve training, all three section 9 baselines and the demo without a branch, and it is
why CLAUDE.md 3.7's three switches live in the training loop rather than in the world.

Design note -- **every stochastic draw goes through one seeded generator.**
CLAUDE.md 3.7 requires an episode to be exactly reproducible from a seed so a chosen
episode can be replayed on demand. One ``np.random.Generator``, created in ``reset()``
from the scenario seed, is the only source of randomness in the whole environment.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from marlsoc.config import RewardStructure, ScenarioConfig
from marlsoc.env import actions as act
from marlsoc.env import detection as det
from marlsoc.env import layers as lyr
from marlsoc.env import observations as obs
from marlsoc.env import rewards as rw
from marlsoc.env import topology as topo
from marlsoc.env.actions import Action, Verb
from marlsoc.env.detection import DetectionConfig
from marlsoc.env.layers import Layer, LayerStatus
from marlsoc.env.rewards import RewardConfig, StepEvents
from marlsoc.env.state import EpisodeState, HostStatus, Outcome
from marlsoc.env.topology import Zone

BLOCK_DURATION: Final[int] = 8
SCAN_DISCOVERIES: Final[int] = 2   # fast and loud
SLOW_SCAN_DISCOVERIES: Final[int] = 1
SCAN_NOISE: Final[float] = 3.0
SLOW_SCAN_NOISE: Final[float] = 1.0

# Which zone each blue agent may deploy a honeypot into, and which layer entering a zone
# is gated on. Both derived from the topology so no host name appears below.
_ZONE_ENTRY_LAYER: Final[dict[Zone, Layer]] = {
    Zone.DMZ: Layer.DMZ_BOUNDARY,
    Zone.CORP: Layer.SEGMENTATION,
}


class MiniCorp:
    """The simulated MiniCorp network. A Gymnasium-style API we wrote ourselves.

    Not an actual ``gymnasium.Env`` subclass: Gymnasium's API is single-agent, and
    bending a five-agent general-sum game into it would mean either a fake joint agent
    or a wrapper that lies about the observation space. Section 11 asks for a
    Gymnasium-*style* ``step``/``reset``, which is what this is.

    Example:
        >>> env = MiniCorp(ScenarioConfig(seed=7))
        >>> observations = env.reset()
        >>> obs_, rewards, done, info = env.step({"R_scout": Action(Verb.WAIT)})
    """

    def __init__(
        self,
        scenario: ScenarioConfig | None = None,
        reward_cfg: RewardConfig | None = None,
        detection_cfg: DetectionConfig | None = None,
    ) -> None:
        self.scenario = scenario or ScenarioConfig()
        # Build the reward config *from the scenario* when one is not supplied. It used
        # to be `reward_cfg or rw.DEFAULT`, which silently ignored the scenario -- so
        # ScenarioConfig.availability_cost was dead and CLAUDE.md 3.2's ONE_SHOT
        # before/after could not actually be run through a scenario at all. Harmless only
        # because the default happened to be the wanted one.
        self.rewards = reward_cfg or rw.RewardConfig(
            availability_cost=self.scenario.availability_cost,
            shaping=self.scenario.shaping,
        )
        self.detection = detection_cfg or det.DEFAULT
        self.state: EpisodeState
        self.rng: np.random.Generator
        self.reset()

    # ----------------------------------------------------------------------------------
    # Episode lifecycle
    # ----------------------------------------------------------------------------------
    def reset(
        self, seed: int | None = None, max_layer: int | None = None
    ) -> dict[str, tuple[int, ...]]:
        """Start a fresh episode and return each agent's first observation.

        Args:
            seed: Overrides the scenario seed. Passing the same seed twice replays an
                episode exactly, which is what CLAUDE.md 3.7 requires for the demo.
            max_layer: Deepest active layer for this episode, overriding the scenario.
                This is how the curriculum in section 7.4 deepens the stack: the schedule
                belongs to the training loop, not to the environment, so ScenarioConfig
                stays frozen and serialisable and the environment never has to know what
                a "stage" is.

        Returns:
            ``{agent: observation tuple}`` for all five agents.
        """
        self.rng = np.random.default_rng(
            self.scenario.seed if seed is None else seed
        )
        stage = self.scenario.max_layer if max_layer is None else max_layer
        self.state = EpisodeState.initial(LayerStatus.for_stage(stage))
        return self.observations()

    def observations(self) -> dict[str, tuple[int, ...]]:
        """``O_i(s)`` for every agent. The only channel from the world to the agents."""
        state = self.state
        return {
            "R_scout": obs.red_scout_observation(state),
            "R_breach": obs.red_breach_observation(state),
            **{a: obs.blue_observation(state, a, self.detection)
               for a in topo.DEFENDER_ZONES},
        }

    def legal_masks(self, agents: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        """The legality mask per agent, for epsilon-greedy to sample within.

        Args:
            agents: Restrict to these agents. Building a mask means evaluating every
                precondition for every action, which profiling showed to be half the
                twin's runtime -- so a disabled agent, which ignores its mask entirely,
                should not have one computed.
        """
        names = agents if agents is not None else tuple(act.ACTION_SPACES)
        return {a: act.legal_mask(self.state, a) for a in names}

    # ----------------------------------------------------------------------------------
    # The transition function
    # ----------------------------------------------------------------------------------
    def step(
        self, joint_action: dict[str, Action]
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, float], bool, dict[str, Any]]:
        """Advance one timestep under the joint action.

        Args:
            joint_action: ``{agent: action}``. Agents may be omitted, which is how a
                disabled agent (CLAUDE.md 3.7) participates in an episode without acting
                -- the environment does not need to know it was switched off.

        Returns:
            ``(observations, rewards, done, info)``. ``rewards`` carries one entry per
            agent, with teammates sharing a value: section 6 makes cooperation a
            consequence of a shared return rather than an instruction.

        Raises:
            ValueError: if any action is illegal in the current state. Silently ignoring
                an illegal action would let a buggy policy appear to work while quietly
                doing nothing, and section 15 lists "red never wins" as the hardest
                failure to diagnose.
        """
        if self.state.outcome is not Outcome.RUNNING:
            raise RuntimeError("episode has already terminated; call reset()")

        events = StepEvents()
        # Phi(s), captured before anything moves. The shaping term needs the potential of
        # the state we are leaving, and by the time rewards are computed the state object
        # has already been mutated into s'.
        events.potential_before = lyr.cumulative_breach_reward(self.state.paid_breaches)
        noise: dict[str, float] = {}

        for agent, action in joint_action.items():
            if not act.is_legal(self.state, agent, action):
                raise ValueError(f"{agent} chose illegal action {action}")

        # --- red first: it has the initiative ------------------------------------------
        for agent in ("R_scout", "R_breach"):
            if agent in joint_action:
                self._apply_red(joint_action[agent], events, noise)

        # --- then blue: containment can revoke a gain made this same step ---------------
        for agent in topo.DEFENDER_ZONES:
            if agent in joint_action:
                self._apply_blue(agent, joint_action[agent], events)

        # --- then detection: alerts reflect how loud this step actually was -------------
        det.step_alerts(self.state, self.rng, noise, self.detection)
        det.decay_heat(self.state, self.detection)
        events.red_detected = self._confirmed_compromise()

        # --- termination BEFORE rewards --------------------------------------------------
        # Order matters and getting it wrong is silent. _check_termination is what sets
        # events.red_won for a curriculum stage's objective, so computing rewards first
        # meant a stage win paid nothing at all: red collected no +100 and blue was
        # charged no -100. Conceding was literally free, and blue correctly learned to
        # concede -- a static defence scored -38.6 while a trained one scored -142.8,
        # because defending cost step time that losing did not. The bug looked exactly
        # like a reward-design problem. Only the alter_credentials path was unaffected,
        # since it sets red_won during action application, which is why the full
        # six-layer game hid it.
        self.state.step += 1
        done = self._check_termination(events)
        # Phi(terminal) = 0 is what makes the shaping policy-invariant; the reward
        # function cannot know it is terminal unless we say so, and we can only say so
        # because 3.15 put the termination check ahead of the rewards.
        events.terminal = done

        rewards = {
            "R_scout": rw.red_reward(self.state, events, self.rewards),
            "R_breach": rw.red_reward(self.state, events, self.rewards),
        }
        if self.scenario.reward_structure is RewardStructure.INDIVIDUAL:
            # Section 6's headline experiment. A different reward *structure*, so a
            # different function rather than a parameter -- see rewards.blue_reward_individual.
            rewards.update({
                a: rw.blue_reward_individual(a, self.state, events, self.rewards)
                for a in topo.DEFENDER_ZONES
            })
        else:
            blue = rw.blue_reward(self.state, events, self.rewards)
            rewards.update({a: blue for a in topo.DEFENDER_ZONES})

        return self.observations(), rewards, done, self._info(events)

    # ----------------------------------------------------------------------------------
    # Red
    # ----------------------------------------------------------------------------------
    def _apply_red(
        self, action: Action, events: StepEvents, noise: dict[str, float]
    ) -> None:
        """Resolve one red action. Every branch is stochastic except the win itself."""
        state, verb = self.state, action.verb

        if verb is Verb.WAIT:
            return  # heat still decays at the end of the step: going quiet is the point

        if verb in (Verb.SCAN, Verb.SLOW_SCAN):
            self._apply_recon(action, events, noise)
            return

        if verb is Verb.ALTER_CREDENTIALS:
            # No probability: the six layers were the difficulty. Making the final move a
            # coin flip would add variance to the one event the whole episode is scored
            # on, for no extra decision.
            events.red_won = True
            return

        if verb in (Verb.EXPLOIT, Verb.LATERAL_MOVE):
            self._apply_intrusion(action, events, noise)
            return

        # The three untargeted layer moves: steal_credentials, escalate_privilege,
        # degrade_mfa. Each attempts exactly one layer.
        layer = lyr.layer_for_action(verb.value)
        assert layer is not None
        spec = lyr.LAYERS[layer]
        host = self._layer_host(layer, action)
        self._add_noise(noise, host, spec.noise)

        if self.rng.random() < self._layer_success(layer, host):
            if state.record_breach(layer):
                events.layers_breached.append(layer)

    def _apply_recon(
        self, action: Action, events: StepEvents, noise: dict[str, float]
    ) -> None:
        """Scanning: find hosts, and -- for ``slow_scan`` -- attempt Layer 1.

        ``scan`` finds more per step but generates three times the noise; only
        ``slow_scan`` can get past the rate limiter. Red is never told which to prefer.
        """
        state = self.state
        assert action.zone is not None
        slow = action.verb is Verb.SLOW_SCAN

        self.state.heat += SLOW_SCAN_NOISE if slow else SCAN_NOISE
        self._add_noise(noise, topo.WAF_HOST, SLOW_SCAN_NOISE if slow else SCAN_NOISE)

        undiscovered = [
            h.name for h in topo.hosts_in(action.zone)
            if h.name not in state.discovered
            and state.status(h.name) is not HostStatus.ISOLATED
        ]
        found = SLOW_SCAN_DISCOVERIES if slow else SCAN_DISCOVERIES
        for name in undiscovered[:found]:
            state.discovered.add(name)

        # Layer 1 falls only to a slow scan, and only from outside the DMZ.
        if slow and state.layers.can_attempt(Layer.PERIMETER):
            if self.rng.random() < topo.BY_NAME[topo.WAF_HOST].bypass_prob:
                if state.record_breach(Layer.PERIMETER):
                    events.layers_breached.append(Layer.PERIMETER)

    def _apply_intrusion(
        self, action: Action, events: StepEvents, noise: dict[str, float]
    ) -> None:
        """``exploit`` or ``lateral_move``: try to take a host, and maybe a layer with it."""
        state = self.state
        assert action.host is not None
        host = topo.BY_NAME[action.host]

        self._add_noise(noise, host.name, host.noise)
        state.heat += host.noise

        if host.name in state.honeypots_live:
            # The decoy. Red spends the action, learns nothing, and lights up the zone.
            state.honeypot_hits += 1
            events.honeypot_hits += 1
            return

        if state.is_blocked(host.name):
            return  # blue got there first, at no availability cost

        if self.rng.random() >= host.exploit_prob:
            return  # failed attempt; the noise was still generated

        state.compromise(host.name)
        events.hosts_compromised.append(host.name)

        # Taking a host may breach the layer that guarded its zone.
        layer = self._layer_taken_by(host, action.verb)
        if layer is not None and state.layers.can_attempt(layer):
            if state.record_breach(layer):
                events.layers_breached.append(layer)

    def _layer_taken_by(self, host: topo.Host, verb: Verb) -> Layer | None:
        """Which layer, if any, falls when ``host`` is taken by ``verb``.

        Derived from roles and zones rather than from host names, so this stays inside
        CLAUDE.md section 2's ban on hard-coded tactics.
        """
        if verb is Verb.EXPLOIT and host.zone is Zone.DMZ:
            return Layer.DMZ_BOUNDARY
        if verb is Verb.LATERAL_MOVE and host.role is topo.Role.PIVOT:
            return Layer.SEGMENTATION
        return None

    def _layer_host(self, layer: Layer, action: Action) -> str:
        """The host an untargeted layer move is aimed at, for noise accounting."""
        if layer is Layer.AUTH:
            assert action.host is not None
            return action.host
        if layer is Layer.APPROVAL:
            return topo.MFA_HOST
        return topo.PIVOT_HOST   # privilege escalation happens on the pivot

    def _layer_success(self, layer: Layer, host: str) -> float:
        """Success probability for a layer attempt.

        ``LayerSpec.success_prob`` of None means the probability belongs to a host
        rather than to the action, so topology.py stays the only place a per-host number
        is written down.
        """
        spec = lyr.LAYERS[layer]
        if spec.success_prob is not None:
            return spec.success_prob
        return topo.BY_NAME[host].bypass_prob or topo.BY_NAME[host].exploit_prob

    def _add_noise(self, noise: dict[str, float], host: str, amount: float) -> None:
        noise[host] = noise.get(host, 0.0) + amount

    # ----------------------------------------------------------------------------------
    # Blue
    # ----------------------------------------------------------------------------------
    def _apply_blue(self, agent: str, action: Action, events: StepEvents) -> None:
        """Resolve one blue action. Blue's actions are deterministic.

        Deliberately so: the uncertainty blue faces is about *what is true*, not about
        whether its own containment works. Adding a failure probability on top of noisy
        detection would make the credit assignment for a defensive action nearly
        impossible to learn from, and the interesting difficulty is already in the
        observation.
        """
        state, verb = self.state, action.verb

        if verb is Verb.NOOP:
            return

        if verb is Verb.ISOLATE:
            assert action.host is not None
            was_compromised = state.status(action.host) is HostStatus.COMPROMISED
            state.isolate(action.host)
            if was_compromised:
                events.correct_isolations.append(action.host)
            else:
                events.false_positives.append(action.host)
            return

        if verb is Verb.BLOCK:
            assert action.host is not None
            state.block(action.host, BLOCK_DURATION)
            return

        if verb is Verb.DEPLOY_HONEYPOT:
            for slot in act._honeypot_slots(agent):
                if slot.name not in state.honeypots_live:
                    state.deploy_honeypot(slot.name)
                    return
            return

        if verb in act.REINFORCE_LAYER:
            layer = act.REINFORCE_LAYER[verb]
            state.layers = state.layers.restore(layer)
            # Repairing again is legal and still useful -- red must re-breach it -- but
            # it is only *paid* the first time. See RewardConfig.layer_restored for the
            # measurement that forced this.
            if layer not in state.restored_layers:
                state.restored_layers.add(layer)
                events.layers_restored.append(layer)
            return

        raise ValueError(f"unhandled blue verb {verb}")

    # ----------------------------------------------------------------------------------
    # Detection and termination
    # ----------------------------------------------------------------------------------
    def _confirmed_compromise(self) -> bool:
        """Whether any genuinely compromised host now reads as confirmed to its defender.

        This is what charges red the -50 in section 5.4, and it is deliberately based on
        the *observed* status rather than on truth: red is punished for being seen, not
        for being compromised. A red that stays quiet is never charged, which is the
        whole stealth incentive.
        """
        return any(
            obs.observed_status(self.state, host, self.detection)
            is obs.ObservedStatus.COMPROMISED
            for host in self.state.footholds
        )

    def _check_termination(self, events: StepEvents) -> bool:
        """Section 3.4's three endings, checked in order of decisiveness."""
        state = self.state

        if events.red_won:
            state.outcome = Outcome.RED_WIN
            return True

        # A curriculum stage ends when red has breached every layer that stage activates.
        # See LayerStatus.objective_met: without this, switching layers off does not make
        # a stage shallower, it just removes the obstacles from the same long path.
        if state.layers.objective_met(state.paid_breaches):
            state.outcome = Outcome.RED_WIN
            events.red_won = True
            return True

        # Blue wins by containing every foothold -- but only if there was ever one to
        # contain. At reset red holds nothing, and that is not a defensive victory.
        if state.first_compromise_step is not None and not state.footholds:
            state.outcome = Outcome.BLUE_WIN
            return True

        if state.step >= self.scenario.step_limit:
            state.outcome = Outcome.DRAW
            return True

        return False

    def _info(self, events: StepEvents) -> dict[str, Any]:
        """Per-step diagnostics. These become the CSV row in CLAUDE.md 3.8."""
        state = self.state
        return {
            "step": state.step,
            "outcome": state.outcome.value,
            "layers_breached": state.layers_breached,
            "hosts_compromised": state.hosts_compromised,
            "mttd": state.mttd,
            "mttc": state.mttc,
            "false_positives": state.false_positives,
            "honeypot_hits": state.honeypot_hits,
            "heat": round(state.heat, 3),
            "events": events,
        }
