"""A scripted attacker: the fixed opponent blue learns against in Phase 2.

Why this is not a violation of the no-hard-coded-tactics rule
------------------------------------------------------------
CLAUDE.md section 2 bans hard-coded tactics, and a priority-ordered attacker looks like
exactly that. The distinction that matters:

* The ban applies to **agents whose behaviour we claim is emergent**. If blue's
  chokepoint defence were an ``if host == "ad-controller"`` rule, the finding would be
  fabricated.
* A scripted opponent is the **fixed environment** blue learns against. PROJECT.md
  section 12 specifies Phase 2 as "single blue agent, Q-Learning, vs scripted attacker",
  and section 9 lists a scripted heuristic among the baselines to beat. Nothing about
  blue's learned policy is scripted.
* The thing that *would* cost us is warm-starting **red's own Q-table** from this script.
  CLAUDE.md amendment 3.3 flags that as a contingency that forfeits the emergence claim
  for whatever the script taught. We are not doing it, and if we ever do it must be
  stated explicitly in the notes.

Design note -- **the script is deliberately noisy.**
It takes a random legal action a fraction of the time. A perfectly deterministic opponent
would let blue memorise one exact trajectory: it would learn a lookup table for a single
attack rather than a defence, its Q-values would be sharp and meaningless, and the policy
would collapse as soon as anything varied. Section 9's whole premise is that blue
generalises across intrusions, so the opponent has to vary.

Design note -- **priorities are expressed over verbs, not host names.**
The script asks "is a lateral move available?" and lets the action mask decide which
hosts qualify. No host name appears in this file. That keeps the script honest about
what it knows -- the same information the learning agent would have -- rather than
smuggling in the topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from marlsoc.env import actions as act
from marlsoc.env import topology as topo
from marlsoc.env.actions import Action, Verb
from marlsoc.env.state import EpisodeState

# Priority order for R_breach. Read top to bottom: take the win if it is there, otherwise
# advance, otherwise unlock the next layer, otherwise widen, otherwise wait.
#
# ALTER_CREDENTIALS first because the mask already guarantees every precondition.
# LATERAL_MOVE above EXPLOIT because depth is what wins; a script that widened first
# would spend the episode collecting DMZ hosts it has no use for.
BREACH_PRIORITY: Final[tuple[Verb, ...]] = (
    Verb.ALTER_CREDENTIALS,
    Verb.LATERAL_MOVE,
    Verb.ESCALATE_PRIVILEGE,
    Verb.DEGRADE_MFA,
    Verb.STEAL_CREDENTIALS,
    Verb.EXPLOIT,
    Verb.WAIT,
)

# R_scout prefers the quiet scan. That is not a tactic smuggled in -- slow_scan is the
# only action that can breach Layer 1 at all, so a scout that preferred SCAN would never
# get the attacker started and Phase 2 would have no intrusion to defend against.
SCOUT_PRIORITY: Final[tuple[Verb, ...]] = (Verb.SLOW_SCAN, Verb.SCAN, Verb.WAIT)


@dataclass(frozen=True)
class ScriptConfig:
    """Attributes:
        noise: Probability of taking a uniformly random legal action instead of the
            scripted one. Non-zero on purpose; see the module docstring.
        prefer_weakest: Among equally-ranked targets, attack the host with the highest
            ``exploit_prob``. A greedy-but-reasonable heuristic; it makes the script a
            competent opponent rather than a trivially bad one, so beating it means
            something.
    """

    noise: float = 0.15
    prefer_weakest: bool = True


DEFAULT: Final[ScriptConfig] = ScriptConfig()


class ScriptedAttacker:
    """A fixed, non-learning red policy.

    Stateless apart from its RNG: every decision is a function of the current legal set,
    so replaying an episode with the same seed replays the same attack.
    """

    def __init__(
        self,
        agent: str,
        config: ScriptConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if agent not in ("R_scout", "R_breach"):
            raise ValueError(f"{agent} is not an attacker")
        self.agent = agent
        self.config = config or DEFAULT
        self.rng = rng if rng is not None else np.random.default_rng()
        self.priority = (
            SCOUT_PRIORITY if agent == "R_scout" else BREACH_PRIORITY
        )

    def act(self, state: EpisodeState, legal: tuple[Action, ...] | None = None) -> Action:
        """Choose an action. Always legal, because it only ever picks from the mask.

        Args:
            legal: The already-computed legal set. The caller usually has it, and
                recomputing it here means evaluating every precondition twice per step.
        """
        legal = act.legal_actions(state, self.agent) if legal is None else legal

        if self.rng.random() < self.config.noise:
            return legal[int(self.rng.integers(len(legal)))]

        for verb in self.priority:
            candidates = [a for a in legal if a.verb is verb]
            if candidates:
                return self._choose(candidates)

        # Unreachable in practice: WAIT is unconditionally legal and is in both
        # priority lists. Kept as a safety net rather than an assertion.
        return legal[int(self.rng.integers(len(legal)))]

    def _choose(self, candidates: list[Action]) -> Action:
        """Break ties among equally-ranked actions.

        Targets are ranked by how weak the host is; zone-targeted recon is ranked by
        depth, so the scout pushes at the frontier rather than re-scanning the edge.
        """
        if len(candidates) == 1:
            return candidates[0]

        if candidates[0].host is not None and self.config.prefer_weakest:
            return max(candidates, key=lambda a: topo.BY_NAME[a.host].exploit_prob)

        if candidates[0].zone is not None:
            return max(candidates, key=lambda a: topo.ZONE_DEPTH[a.zone])

        return candidates[int(self.rng.integers(len(candidates)))]


class GreedyDefender:
    """PROJECT.md section 9 baseline 3: "isolate anything suspicious".

    Expected to beat random and to lose badly on false positives, because it reacts to
    evidence without weighing what the evidence is worth. That failure *is* the result:
    it is the concrete thing blue's learned policy has to be better than, and the gap
    between them is the availability tradeoff made visible.
    """

    def __init__(
        self,
        agent: str,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.agent = agent
        self.rng = rng if rng is not None else np.random.default_rng()

    def act(self, state: EpisodeState, legal: tuple[Action, ...] | None = None) -> Action:
        from marlsoc.env import detection as det

        legal = act.legal_actions(state, self.agent) if legal is None else legal
        suspicious = [
            a for a in legal
            if a.verb is Verb.ISOLATE and det.is_suspicious(state, a.host)
        ]
        if suspicious:
            return max(suspicious, key=lambda a: state.alerts.get(a.host, 0.0))

        noops = [a for a in legal if a.verb is Verb.NOOP]
        return noops[0] if noops else legal[0]
