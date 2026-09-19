"""Phase 4 -- all five agents, alternating training, IQL control, all six layers.

PROJECT.md section 12 calls this "All 5 agents, IQL, alternating training, all six
layers". Everything before it trained a single defender with the other two switched off;
this is the first time the whole Markov Game runs.

Four questions, one arm each
----------------------------
1. **Does alternating training produce an arms race?** Freeze one team, train the other,
   swap, repeat. Each phase is a stationary MDP, so each team's learning is sound. The
   plot to look at is ``arms_race.png``: attacker success should fall while blue trains
   and rise while red trains, with both improving round on round.

2. **Does simultaneous IQL settle?** Section 3.4 keeps this as a *control*, not as an
   inferior method. With everyone learning at once nobody faces a stationary environment,
   so Q-Learning has no convergence guarantee -- and ``alternating.instability`` measures
   whether that shows up as a run that will not stop moving.

3. **Shared or individual reward?** Section 6's headline emergent-behaviour experiment.
   Under individual reward each defender is charged only for its own zone, and the -100
   for losing the crown jewel lands on ``B_secure`` alone. The prediction is that
   ``B_dmz`` turns trigger-happy -- more isolations, more false positives -- because the
   downstream cost of letting an attacker past is somebody else's. Nobody is told to
   cooperate or defect either way.

4. **Does alternating training fix red's reliability?** Phase 3 left the attacker working
   in roughly one seed in three against a single frozen defender. A team trained against a
   *sequence* of opponents may generalise where one trained against a fixed opponent
   overfits. This arm is why the Phase 3 caveat is worth re-measuring here rather than
   assumed to carry over.

Red is warm-started from the Phase 3 curriculum where a table exists, because section 3.3
trains red against a static defence first so it receives positive signal at all. That is
the curriculum doing its job, not a shortcut -- nothing is hand-written into the table.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marlsoc.agents.tabular import Algorithm, LearnerConfig             # noqa: E402
from marlsoc.config import (                                            # noqa: E402
    ALL_AGENTS,
    AgentConfig,
    Policy,
    RewardStructure,
    ScenarioConfig,
)
from marlsoc.training.alternating import (                              # noqa: E402
    AlternatingConfig,
    instability,
    train_alternating,
    train_iql,
)
from marlsoc.training.curriculum import CurriculumConfig                # noqa: E402
from marlsoc.training.loop import (                                     # noqa: E402
    build_controller, evaluate, train_curriculum,
)
from marlsoc.training.plots import arms_race                            # noqa: E402

OUT = Path("artifacts/phase4")
STATIC_BLUE = AgentConfig(enabled=False, learning=False, policy=Policy.NOOP)
CURRICULUM = CurriculumConfig(greedy_eval_episodes=60, greedy_eval_every=250)


def scenario(seed: int, structure: RewardStructure) -> ScenarioConfig:
    """All five agents live, all six layers. The full game for the first time."""
    return ScenarioConfig(seed=seed, max_layer=6, reward_structure=structure)


def red_config(episodes: int) -> LearnerConfig:
    """SARSA for red -- section 7.2's cliff-walking argument: the attacker is punished
    for exploring, so an on-policy learner that accounts for its own exploration takes
    the safer path."""
    return LearnerConfig(
        algorithm=Algorithm.SARSA, alpha=0.1, gamma=0.95,
        epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_episodes=max(1, episodes // 3), q_init=0.0,
    )


def blue_config(episodes: int) -> LearnerConfig:
    """Q-Learning for blue (section 7.1), with the ``q_init`` Phase 2 settled by
    measurement -- see CLAUDE.md 3.14 and 3.24: initialising at zero when returns are
    large and negative makes the greedy policy prefer whatever it has never tried."""
    return LearnerConfig(
        algorithm=Algorithm.Q_LEARNING, alpha=0.1, gamma=0.95,
        epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_episodes=max(1, episodes // 3), q_init=-150.0,
    )


def warm_started_controllers(sc: ScenarioConfig, seed: int, warm_episodes: int):
    """Controllers with red carrying a curriculum warm-up (CLAUDE.md 3.3).

    Red trains against a static defence first, because against a live one it receives no
    positive signal to learn from -- Phase 1 measured 0% wins at mean depth 0.78 of 6.
    Blue starts from nothing: it has positive signal from episode one, since red attacking
    is exactly what blue is paid to notice.
    """
    rng = np.random.default_rng(seed)
    controllers = {
        agent: build_controller(
            agent, sc.for_agent(agent), rng,
            red_config(warm_episodes) if agent.startswith("R_")
            else blue_config(warm_episodes),
        )
        for agent in ALL_AGENTS
    }
    if warm_episodes <= 0:
        return controllers

    static = replace(sc, agents={a: STATIC_BLUE for a in ("B_dmz", "B_corp", "B_secure")})
    warm = train_curriculum(static, warm_episodes, red_config(warm_episodes),
                            CURRICULUM, verbose=False)
    for agent in ("R_scout", "R_breach"):
        controllers[agent].learner.q = warm.controllers[agent].learner.q.copy()
        controllers[agent].learner.visits = warm.controllers[agent].learner.visits.copy()
    return controllers


def score(sc: ScenarioConfig, controllers, episodes: int = 300) -> dict:
    log = evaluate(sc, controllers, episodes, phase="phase4")
    return {
        "attacker_success": log.rate("red_win"),
        "blue_return": log.mean("blue_return"),
        "red_return": log.mean("red_return"),
        "layers_breached": log.mean("layers_breached"),
        "false_positives": log.mean("false_positives"),
        "mttd": log.mean("mttd"),
    }


def isolation_profile(sc: ScenarioConfig, controllers, episodes: int = 300) -> dict:
    """How trigger-happy each defender is -- section 6's prediction, measured.

    False positives per episode is the cleanest single number: it is availability spent
    on nothing, and under individual reward ``B_dmz`` has no reason to economise on it.
    """
    log = evaluate(sc, controllers, episodes, phase="profile")
    return {
        "false_positives": log.mean("false_positives"),
        "blue_return": log.mean("blue_return"),
        "attacker_success": log.rate("red_win"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--episodes-per-phase", type=int, default=2_000)
    parser.add_argument("--warm", type=int, default=3_000,
                        help="curriculum warm-up episodes for red (0 to disable)")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    alt_cfg = AlternatingConfig(rounds=args.rounds,
                                episodes_per_phase=args.episodes_per_phase)
    total = 2 * args.rounds * args.episodes_per_phase

    print(f"Phase 4: {args.rounds} rounds x {args.episodes_per_phase} episodes per phase "
          f"({total} training episodes per seed), seeds {seeds}")
    print()

    results: dict[str, list[dict]] = {}
    phase_traces: dict[str, list] = {}

    for structure in (RewardStructure.SHARED, RewardStructure.INDIVIDUAL):
        key = f"alternating-{structure.value}"
        results[key] = []
        for seed in seeds:
            sc = scenario(seed, structure)
            ctrl = warm_started_controllers(sc, seed, args.warm)
            run = train_alternating(sc, alt_cfg, blue_config(args.episodes_per_phase),
                                    controllers=ctrl, verbose=False)
            results[key].append({
                **score(sc, run.controllers),
                "instability": instability(run.phases),
            })
            if seed == seeds[0]:
                phase_traces[key] = run.phases

    # The control: everybody learning at once (CLAUDE.md 3.4).
    results["iql"] = []
    for seed in seeds:
        sc = scenario(seed, RewardStructure.SHARED)
        ctrl = warm_started_controllers(sc, seed, args.warm)
        run = train_iql(sc, total, blue_config(total), controllers=ctrl,
                        measure_every=max(1, total // (2 * args.rounds)),
                        verbose=False)
        results["iql"].append({
            **score(sc, run.controllers),
            "instability": instability(run.phases),
        })
        if seed == seeds[0]:
            phase_traces["iql"] = run.phases

    # blue_R is EpisodeRecord.blue_return, which is B_corp's return. Under SHARED that
    # is the whole team's value; under INDIVIDUAL it is one defender's slice of it. The
    # two are different quantities and must not be compared across structures -- the same
    # category error as comparing returns across shaping modes (CLAUDE.md 3.18). The
    # cross-structure comparison is attacker success and false positives, which are
    # properties of the *behaviour* and do not depend on how it was priced.
    print("cross-structure comparison: attacker success and FP/ep only.")
    print("blue_R is within-structure (it is B_corp's return, not the team's).")
    print()
    print(f"{'arm':<34}{'attacker':>10}{'blue_R*':>10}{'FP/ep':>8}{'instability':>13}")
    for key, rows in results.items():
        a = np.mean([r["attacker_success"] for r in rows])
        b = np.mean([r["blue_return"] for r in rows])
        f = np.mean([r["false_positives"] for r in rows])
        i = np.mean([r["instability"] for r in rows])
        print(f"{key:<34}{a:>9.1%}{b:>10.1f}{f:>8.2f}{i:>13.3f}")

    print()
    print("per seed (attacker success) -- report the spread, not just the mean:")
    for key, rows in results.items():
        spread = [f"{r['attacker_success']:.1%}" for r in rows]
        print(f"  {key:<32} {spread}")

    OUT.mkdir(parents=True, exist_ok=True)
    for key, phases in phase_traces.items():
        if key.startswith("alternating"):
            arms_race(phases, OUT / f"arms_race_{key.split('-')[1]}.png",
                      title=f"Alternating training — {key.split('-')[1]} reward",
                      iql_phases=phase_traces.get("iql"))
    (OUT / "summary.json").write_text(json.dumps({
        "rounds": args.rounds,
        "episodes_per_phase": args.episodes_per_phase,
        "warm": args.warm,
        "seeds": seeds,
        "results": results,
        "phases": {k: [asdict(p) for p in v] for k, v in phase_traces.items()},
    }, indent=2))
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
