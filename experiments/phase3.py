"""Phase 3: red learns to attack, with and without a curriculum.

Run with::

    python experiments/phase3.py            # full run, ~9000 episodes per arm
    python experiments/phase3.py --quick    # short run, for a smoke check

Outputs land in ``artifacts/runs/phase3/``.

What this experiment is
-----------------------
PROJECT.md section 7.4 claims a curriculum is what makes the six-layer attack learnable.
Phase 1 established the premise by measurement: with all six layers live from episode one,
random exploration reaches mean depth 0.78 of 6 and wins 0% of episodes, so red receives
no signal from the +100 alone.

Three findings shaped the design of this experiment, and each cost a run to discover.

**The opponent has to be chosen deliberately, and there are only bad options.** Against a
*static* defence red already wins ~100% at every stage unopposed, so the curriculum has
nothing to demonstrate. Against a *random* defender red wins 0% regardless, because a
defender picking uniformly at random accidentally plays the degenerate lockdown -- it ends
an episode with 3.46 of its 4 hosts isolated (CLAUDE.md 3.17). The only opponent that
makes the question meaningful is the **trained** Phase 2 defender, which is penalised for
lockdown and therefore has to defend selectively.

**Red must train against the defender it is scored against** (CLAUDE.md 3.19). Loading the
frozen defender after training rather than before cost a full day of wrong conclusions: red
had quietly learned to beat an untrained greedy controller, which ties-breaks at random and
so *is* the random defender, and was then scored against something else entirely.

**A curriculum that cannot promote is worse than no curriculum** (CLAUDE.md 3.21). Section
7.4's 0.70 promotion threshold is an absolute win rate, and it was calibrated against a
static defence where red wins ~100%. Against a defender that fights back even the scripted
expert manages 11%, so the threshold is unreachable by construction: red reached stage 3,
spent 4,885 of its 9,000 episodes there winning 2.3%, and was then evaluated at stage 5 --
a depth it had never once trained at. The stage cap is now a share of the remaining budget.

**Correct shaping is not the same as useful shaping** (CLAUDE.md 3.18). Making §5.4's
ladder potential-based -- `γΦ(s') − Φ(s)` with `Φ(terminal) = 0`, per Ng, Harada & Russell
(1999) -- is right in the sense the theorem means: it cannot change the optimal policy.
Here that is exactly the problem. The shaping then sums to zero over any trajectory, so
red's whole incentive is the terminal +100, and discounted over the ~29 steps to the crown
jewel that is +22.6 against -15.5 of step costs; one -50 detection, which the alert process
charges *even against a static defence*, makes winning about -23 against about -20 for
idling. Red learned to idle -- 250 steps, return exactly -250, never breaching Layer 1 --
and was right to. Three seeds, 4,000 episodes, against a static defence:

    raw_ladder        attacker 66.7%   mean depth 5.00 of 6
    potential_based   attacker  0.0%   mean depth 0.00 of 6

The ladder is the default. `RewardShaping.POTENTIAL_BASED` is kept because that comparison
is a result worth showing.

The four arms
-------------
``scripted``    The fixed reference. A hand-written priority list, not a learner, so it
                does not move when either side is retrained. It is an *oracle* baseline:
                it reads the true state directly, while a learned red sees only its
                384-state observation.
``direct``      Learned red at full depth from episode one. No curriculum.
``curriculum``  Learned red, staged 1-2 through 1-6, against the same defender throughout.
``warmup``      Section 3.3 as written: the curriculum is run against a *static* defence
                first, where red gets positive signal and promotions are genuine, then the
                Q-table is carried over and fine-tuned against the trained defender.

Every arm is evaluated identically: 300 greedy episodes at the full six layers against the
frozen Phase 2 defender.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marlsoc.agents.tabular import Algorithm, LearnerConfig            # noqa: E402
from marlsoc.config import (                                           # noqa: E402
    ALL_AGENTS, AgentConfig, Policy, ScenarioConfig,
)
from marlsoc.env.minicorp import MiniCorp                              # noqa: E402
from marlsoc.training import plots                                     # noqa: E402
from marlsoc.training.curriculum import CurriculumConfig               # noqa: E402
from marlsoc.training.loop import (                                    # noqa: E402
    build_controller, evaluate, run_episode, train, train_curriculum,
)
from marlsoc.training.metrics import MetricsLog                        # noqa: E402

OUT = ROOT / "artifacts" / "phase3"
BLUE_QTABLE = ROOT / "artifacts" / "runs" / "phase2" / "B_dmz_qtable.npz"
LEARNER = "R_breach"

OFF = AgentConfig(enabled=False)
FROZEN_BLUE = AgentConfig(learning=False, policy=Policy.GREEDY)
STATIC_BLUE = AgentConfig(enabled=False, learning=False, policy=Policy.NOOP)
SCRIPTED_RED = AgentConfig(learning=False, policy=Policy.SCRIPTED)


def scenario(seed: int, *, blue: AgentConfig, max_layer: int = 6) -> ScenarioConfig:
    """Red learning, one defender, the other two zones switched off.

    Only ``B_dmz`` is active because it is the agent Phase 2 actually trained; scoring
    red against defenders that were never trained would flatter it.
    """
    return ScenarioConfig(seed=seed, max_layer=max_layer, agents={
        "B_dmz": blue, "B_corp": OFF, "B_secure": OFF,
    })


def controllers_with_trained_blue(
    sc: ScenarioConfig, seed: int, lcfg: LearnerConfig | None,
) -> dict:
    """Controllers with the Phase 2 defender already loaded and frozen.

    Built *before* training rather than after it -- CLAUDE.md 3.19. Handing these to
    ``train`` is what keeps red's training distribution the same as its evaluation one.
    """
    rng = np.random.default_rng(seed)
    ctrl = {a: build_controller(a, sc.for_agent(a), rng, lcfg) for a in ALL_AGENTS}
    ctrl["B_dmz"].learner.load(BLUE_QTABLE)
    ctrl["B_dmz"].greedy = True
    return ctrl


def red_config(episodes: int, q_init: float = 0.0) -> LearnerConfig:
    """SARSA, because section 7.2's cliff is the whole reason red is on-policy.

    Red is punished -50 for being detected while probing, which is Sutton & Barto's cliff
    walk: Q-Learning learns the optimal path along the edge and then falls off it while
    exploring, SARSA learns the safer path that accounts for its own exploration. An
    attacker that must survive its own mistakes wants the safer path.
    """
    return LearnerConfig(
        algorithm=Algorithm.SARSA, alpha=0.1, gamma=0.95,
        epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_episodes=max(1, episodes // 3), q_init=q_init,
    )


def score(controllers: dict, sc: ScenarioConfig, episodes: int = 300) -> dict:
    """Greedy evaluation at the full six layers. Never trains the policy it measures."""
    ev = evaluate(sc, controllers, episodes)
    return {
        "attacker_success": ev.rate("red_win"),
        "layers_breached": ev.mean("layers_breached"),
        "steps": ev.mean("steps"),
        "red_return": ev.mean("red_return"),
        "blue_return": ev.mean("blue_return"),
    }


# --------------------------------------------------------------------------------------
# The arms
# --------------------------------------------------------------------------------------
def arm_scripted(seed: int, episodes: int) -> tuple[dict, None]:
    sc = scenario(seed, blue=FROZEN_BLUE)
    rng = np.random.default_rng(seed)
    ctrl = {a: build_controller(a, SCRIPTED_RED if a.startswith("R_") else sc.for_agent(a),
                                rng)
            for a in ALL_AGENTS}
    ctrl["B_dmz"].learner.load(BLUE_QTABLE)
    ctrl["B_dmz"].greedy = True
    return score(ctrl, sc), None


def arm_direct(seed: int, episodes: int) -> tuple[dict, None]:
    sc = scenario(seed, blue=FROZEN_BLUE)
    lcfg = red_config(episodes)
    ctrl = controllers_with_trained_blue(sc, seed, lcfg)
    run = train(sc, episodes, lcfg, controllers=ctrl, phase="phase3-direct",
                learner_name=LEARNER, verbose=False)
    return score(run.controllers, sc), run.log


# Promotion is judged on a greedy probe (CLAUDE.md 3.23), which costs real episodes:
# at the defaults that is 100 evaluation episodes every 200 training ones, or 50%
# overhead on every curriculum arm. 60 every 250 holds the same signal at 24%. It is
# named here rather than inherited so the run's cost is part of the experiment's record.
CURRICULUM = CurriculumConfig(greedy_eval_episodes=60, greedy_eval_every=250)


def arm_curriculum(seed: int, episodes: int) -> tuple[dict, object]:
    sc = scenario(seed, blue=FROZEN_BLUE)
    lcfg = red_config(episodes)
    ctrl = controllers_with_trained_blue(sc, seed, lcfg)
    run = train_curriculum(sc, episodes, lcfg, CURRICULUM,
                           controllers=ctrl, verbose=False)
    return score(run.controllers, sc), run


def arm_warmup(seed: int, episodes: int) -> tuple[dict, object]:
    """CLAUDE.md 3.3: curriculum against a static defence, then fine-tune against blue.

    The split is half and half. The warm-up is where the curriculum can actually promote
    on merit -- red wins ~100% per stage unopposed -- so this is the only arm in which
    the sawtooth means anything. Exploration is rewound before the fine-tune because red
    arrives from the warm-up nearly greedy and is about to meet an opponent that
    invalidates much of what it learned.
    """
    warm, tune = episodes // 2, episodes - episodes // 2

    static_sc = scenario(seed, blue=STATIC_BLUE)
    lcfg = red_config(warm)
    warm_run = train_curriculum(static_sc, warm, lcfg, CURRICULUM, verbose=False)

    live_sc = scenario(seed, blue=FROZEN_BLUE)
    ctrl = controllers_with_trained_blue(live_sc, seed, red_config(tune))
    for agent in ("R_scout", LEARNER):
        ctrl[agent].learner.q = warm_run.controllers[agent].learner.q.copy()
        ctrl[agent].learner.visits = warm_run.controllers[agent].learner.visits.copy()
        ctrl[agent].learner.boost_exploration(0.40)

    run = train(live_sc, tune, red_config(tune), controllers=ctrl,
                phase="phase3-warmup", learner_name=LEARNER, verbose=False)
    return score(run.controllers, live_sc), warm_run


ARMS = {
    "scripted":   ("scripted red (oracle reference)", arm_scripted),
    "direct":     ("learned red, no curriculum", arm_direct),
    "curriculum": ("learned red, curriculum vs trained blue", arm_curriculum),
    "warmup":     ("learned red, static warm-up then fine-tune", arm_warmup),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=9_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    episodes = 1_200 if args.quick else args.episodes
    seeds = list(range(1, args.seeds + 1))

    if not BLUE_QTABLE.exists():
        raise SystemExit(f"missing {BLUE_QTABLE} -- run experiments/phase2.py first")

    results: dict[str, list[dict]] = {}
    artefacts: dict[str, object] = {}
    print(f"Phase 3: {episodes} episodes per arm, seeds {seeds}\n")
    print(f"{'arm':<44} {'attacker success':>18} {'layers':>8} {'steps':>7}")

    for name, (label, fn) in ARMS.items():
        per_seed = []
        for seed in seeds:
            result, artefact = fn(seed, episodes)
            per_seed.append(result)
            if seed == seeds[0] and artefact is not None:
                artefacts[name] = artefact
            if name == "scripted":
                break                      # deterministic given the frozen defender
        results[name] = per_seed
        rates = [r["attacker_success"] for r in per_seed]
        mean = statistics.fmean(rates)
        sd = statistics.pstdev(rates) if len(rates) > 1 else 0.0
        print(f"{label:<44} {mean:>10.1%} (sd {sd:4.1%}) "
              f"{statistics.fmean(r['layers_breached'] for r in per_seed):>8.2f} "
              f"{statistics.fmean(r['steps'] for r in per_seed):>7.1f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(
        {"episodes": episodes, "seeds": seeds,
         "results": {k: v for k, v in results.items()}}, indent=2))

    for name in ("curriculum", "warmup"):
        run = artefacts.get(name)
        if run is not None and hasattr(run, "curriculum"):
            plots.curriculum_curve(
                run.log, run.curriculum.transitions, OUT / f"sawtooth_{name}.png",
                title=("Curriculum learning — red vs a static defence"
                       if name == "warmup" else
                       "Curriculum learning — red vs the trained defender"),
            )
            print(f"\n{name} stage transitions:")
            print(run.curriculum.summary())

    print(f"\nwrote {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
