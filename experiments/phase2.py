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

**The shallowest stages contain no defensive decision worth making.** Once stages were
graded properly, four seeds per stage at 4,000 episodes gave:

    stage 1-2   attacker success 100.0% -> 99.3%  (sd 1.2)   blue return -113 -> -154
    stage 1-3   attacker success 100.0% -> 87.8%  (sd 4.5)   blue return -138 -> -227
    stage 1-4   attacker success  99.7% ->  6.8%  (sd 7.6)   blue return -1315 -> -224

Stage 1-2 is over in 3.4 steps and blue barely gets a turn. Stage 1-3 gives blue time but
not *value*: red's objective is only credential theft, so the episode ends cheaply either
way and a long defensive stand costs more than the breach it prevents -- which is why
blue's return is worse than doing nothing at both stages.

Stage 1-4 requires red to reach the pivot, about 29 steps, which is long enough for
containment to pay for itself. That is **Phase 2**: the shallowest stage where the
defender both has time to act and is rewarded for acting.

Why q_init is -150 and why the episode budget is 12,000
-------------------------------------------------------
Both were settled by measurement, and the two interact. Initialising at 0.0 is optimistic
in a regime where every return is negative, which helps exploration but means untried
entries stay pinned at 0.0 while learned values go more negative -- so the gap *widens*
with training and the greedy policy increasingly prefers actions it has never tried. Three
seeds at stage 1-4:

     4,000 eps, q_init    0:  attacker 18.4% (sd 15.8)  untried 80%
     4,000 eps, q_init -150:  attacker 24.7% (sd 19.6)  untried 23%
    12,000 eps, q_init    0:  attacker 30.6% (sd 21.6)  untried 77%   <- worse with more
    12,000 eps, q_init -150:  attacker  4.3% (sd  5.7)  untried 29%   <- the configuration

At a short budget the optimistic agent looks better; at an adequate one the ordering
reverses and the optimistic agent actually *degrades* with further training. The
pessimistic agent wins on score, on variance and on having a policy that is genuinely
learned rather than a bias toward the unexplored.

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


STAGE = 4   # layers 1-4; see the module docstring for why not 1-2 or 1-3


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
    parser.add_argument("--q-init", type=float, default=-150.0, dest="q_init",
                        help="initial Q value; see agents/tabular.py on why it matters")
    parser.add_argument("--decay", type=float, default=0.375,
                        help="fraction of training over which epsilon decays")
    args = parser.parse_args()

    episodes = args.episodes or (800 if args.quick else 12_000)
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
            q_init=args.q_init,
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

    learner = run.controllers[LEARNER].learner
    coverage = learner.coverage
    print(f"\nstate-space coverage: {coverage:.1%}"
          f"   greedy picks a never-updated action in "
          f"{learner.untried_greedy_fraction:.0%} of visited states")

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
