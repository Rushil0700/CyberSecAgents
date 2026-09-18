"""Phase 2: one learning defender against a scripted attacker. The first learning curve.

Run with::

    python experiments/phase2.py            # full run, ~6000 episodes
    python experiments/phase2.py --quick    # 800 episodes, for a smoke check

Outputs land in ``artifacts/runs/phase2/``: the per-episode CSV, the evaluation-snapshot
CSV, and the curve PNGs.

What this experiment is
-----------------------
PROJECT.md section 12 specifies Phase 2 as "single blue agent, Q-Learning, vs scripted
attacker, layers 1-2 only". Two corrections, both established by measurement rather than
argument (see ``notes/phase2-first-curve.md``).

**A stage must end at its own objective.** Originally, switching layers 3-6 off removed
the obstacles but left red the same full-length journey to the crown jewel, so a "shallow"
stage was the whole network with its defences disabled -- easier for red, not shallower.
A stage's objective is now to breach every *active* layer, which is what section 7.4's
stage table actually describes.

**Stage 1-2 is too shallow to contain a defensive decision.** Once stages were graded
properly, a scripted attacker reaches the stage 1-2 objective in 3.7 steps: blue barely
gets a turn, and a trained defender moves attacker success only from 100% to 95%. So
Phase 2 uses **stage 1-3** -- the shallowest stage in which the defender has time to act.
Measured over four seeds:

    stage 1-2   attacker success 100% -> 95.0%  (sd  3.0)   too fast to defend
    stage 1-3   attacker success 100% -> 73.4%  (sd  8.3)   <- Phase 2
    stage 1-4   attacker success 100% -> 15.0%  (sd 21.8)   bigger effect, one seed in
                                                             four fails outright

That trade is itself a result worth reporting: as the task deepens the defender's effect
grows, and so does the variance.

The learning agent is ``B_dmz``, which holds layers 1 and 2 per section 4.2. The other two
defenders are switched off so any improvement is attributable to this one agent.

The baselines it is measured against are section 9's, and none of them is separate code
-- each is a different setting of CLAUDE.md 3.7's three switches.

Why this reports a mean over seeds
----------------------------------
A single run is not evidence here, and that was established the hard way. With epsilon
decaying over 70% of training, two runs of the *same* configuration scored 6.2% and 76.5%
attacker success. The reason is visible in the evaluation snapshots: the policy is
unstable for as long as epsilon keeps moving, and whether it happens to have converged
when training stops is close to luck.

The instability is not mysterious. Blue's own policy determines the state distribution it
experiences -- isolating a DMZ host *ends the episode*, so a containment-happy defender
only ever sees early-episode states while a passive one sees deep ones. On top of that,
most of blue's return comes from the -10 per step bleed on Corp and Secure hosts that
``B_dmz`` has no action to touch, which is uncontrollable variance in every TD target. So
this script runs several seeds and reports the mean and spread. Quoting the best seed
would be exactly the dishonesty CLAUDE.md 3.7 warns about.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marlsoc.agents.tabular import Algorithm, LearnerConfig
from marlsoc.config import AgentConfig, Policy, ScenarioConfig
from marlsoc.training import plots
from marlsoc.training.loop import evaluate, train
from marlsoc.training.metrics import MetricsLog

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "runs" / "phase2"

LEARNER = "B_dmz"
SCRIPTED_RED = AgentConfig(learning=False, policy=Policy.SCRIPTED)
DISABLED = AgentConfig(enabled=False)


STAGE = 3   # layers 1-3; see the module docstring for why not 1-2


def scenario(blue: AgentConfig, seed: int = 1, max_layer: int = STAGE) -> ScenarioConfig:
    """One learning defender, a scripted attacker, the other two defenders switched off.

    Isolating a single defender is what makes the result attributable: any improvement is
    this agent's, not a team effect.
    """
    agents = {"R_scout": SCRIPTED_RED, "R_breach": SCRIPTED_RED, LEARNER: blue}
    agents.update({a: DISABLED for a in ("B_dmz", "B_corp", "B_secure") if a != LEARNER})
    return ScenarioConfig(seed=seed, max_layer=max_layer, agents=agents)


def baselines(episodes: int, stage: int = STAGE) -> dict[str, MetricsLog]:
    """Section 9's three baselines, each a different setting of the three switches."""
    from marlsoc.env.minicorp import MiniCorp
    from marlsoc.training.loop import build_controller, run_episode
    import numpy as np

    from marlsoc.config import ALL_AGENTS

    configs = {
        "static firewall": DISABLED,
        "random defender": AgentConfig(learning=False, policy=Policy.RANDOM),
        "greedy heuristic": AgentConfig(learning=False, policy=Policy.SCRIPTED),
    }

    results: dict[str, MetricsLog] = {}
    for label, blue_cfg in configs.items():
        sc = scenario(blue_cfg, max_layer=stage)
        env = MiniCorp(sc)
        rng = np.random.default_rng(sc.seed)
        controllers = {a: build_controller(a, sc.for_agent(a), rng) for a in ALL_AGENTS}
        log = MetricsLog()
        for i in range(episodes):
            record, _ = run_episode(env, controllers, sc, seed=90_000 + i, learn=False)
            log.append(record)
        results[label] = log
    return results


def report(label: str, log: MetricsLog) -> dict:
    return {
        "policy": label,
        "attacker_success": round(log.rate("red_win"), 4),
        "defender_success": round(log.rate("blue_win"), 4),
        "draw_rate": round(log.rate("draw"), 4),
        "blue_return": round(log.mean("blue_return"), 1),
        "layers_breached": round(log.mean("layers_breached"), 2),
        "false_positives": round(log.mean("false_positives"), 2),
        "mttd": round(log.mean("mttd"), 2),
        "mttc": round(log.mean("mttc"), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="short smoke run")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=4,
                        help="repeat over this many seeds and report mean +- sd")
    parser.add_argument("--stage", type=int, default=STAGE,
                        help="deepest active layer (curriculum stage)")
    parser.add_argument("--decay", type=float, default=0.25,
                        help="fraction of training over which epsilon decays")
    args = parser.parse_args()

    episodes = args.episodes or (800 if args.quick else 6_000)
    eval_n = 100 if args.quick else 400

    print(f"Phase 2 -- {LEARNER} (Q-Learning) vs scripted attacker, "
          f"curriculum stage layers 1-{args.stage}")
    print(f"{episodes} episodes, epsilon decays over {args.decay:.0%} of training, "
          f"{args.seeds} seed(s) from {args.seed}\n")

    def learner_config() -> LearnerConfig:
        return LearnerConfig(
            algorithm=Algorithm.Q_LEARNING,
            alpha=0.1, gamma=0.95,
            epsilon_start=1.0, epsilon_end=0.05,
            epsilon_decay_episodes=max(1, int(episodes * args.decay)),
        )

    # Extra seeds, reported as a spread. See the module docstring on why one run is not
    # evidence for this configuration.
    repeats: list[float] = []
    for extra in range(1, args.seeds):
        sc_extra = scenario(AgentConfig(), seed=args.seed + extra, max_layer=args.stage)
        run_extra = train(sc_extra, episodes, learner_config(), verbose=False,
                          eval_every=0)
        rate = evaluate(sc_extra, run_extra.controllers, eval_n).rate("red_win")
        repeats.append(rate)
        print(f"  seed {args.seed + extra}: attacker success {rate:.1%}")

    learner_cfg = learner_config()
    sc = scenario(AgentConfig(), seed=args.seed, max_layer=args.stage)
    run = train(
        sc, episodes, learner_cfg,
        phase="phase2", learner_name=LEARNER,
        report_every=max(250, episodes // 12),
        eval_every=max(100, episodes // 30),
        eval_episodes=60,
    )

    print("\nBaselines (section 9):")
    rows = [report(label, log) for label, log in baselines(eval_n, args.stage).items()]
    for row in rows:
        print(f"  {row['policy']:<18} attacker success {row['attacker_success']:6.1%}"
              f"  blue return {row['blue_return']:9.1f}"
              f"  FP {row['false_positives']:5.2f}")

    final = evaluate(sc, run.controllers, eval_n, phase="final")
    trained = report(f"learned {LEARNER}", final)
    rows.append(trained)
    print(f"  {trained['policy']:<18} attacker success {trained['attacker_success']:6.1%}"
          f"  blue return {trained['blue_return']:9.1f}"
          f"  FP {trained['false_positives']:5.2f}")

    coverage = run.controllers[LEARNER].learner.coverage
    print(f"\nstate-space coverage: {coverage:.1%}")

    if repeats:
        import statistics
        all_rates = [final.rate("red_win"), *repeats]
        print(f"attacker success across {len(all_rates)} seeds: "
              f"mean {statistics.fmean(all_rates):.1%}, "
              f"sd {statistics.pstdev(all_rates):.1%}, "
              f"range {min(all_rates):.1%}-{max(all_rates):.1%}")

    OUT.mkdir(parents=True, exist_ok=True)
    run.log.to_csv(OUT / "episodes.csv")
    run.evals.to_csv(OUT / "eval_snapshots.csv")
    final.to_csv(OUT / "final_eval.csv")
    (OUT / "summary.json").write_text(json.dumps(
        {"episodes": episodes, "seed": args.seed, "seeds": args.seeds,
         "epsilon_decay_fraction": args.decay, "coverage": round(coverage, 4),
         "attacker_success_per_seed": [round(final.rate("red_win"), 4), *[round(r, 4) for r in repeats]],
         "results": rows}, indent=2))
    run.controllers[LEARNER].learner.save(OUT / f"{LEARNER}_qtable.npz")

    plots.learning_curve(
        run.log, OUT / "training_online.png", window=max(50, episodes // 40),
        title=f"Phase 2 online return while exploring ({LEARNER}, Q-Learning)",
    )
    plots.learning_curve(
        run.evals, OUT / "learning_curve.png", window=3,
        title=f"Phase 2 learning curve -- greedy policy quality ({LEARNER}, Q-Learning)",
    )
    print(f"\nwrote {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
