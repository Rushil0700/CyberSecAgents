# Phase 3 — the attacker learns, and what "learning" turned out to mean

Revision material for Phase 3. Read `notes/phase2-first-curve.md` first; this assumes the
trained `B_dmz` defender from it.

## What Phase 3 was supposed to show

`PROJECT.md` §7.4 claims that a **curriculum** is what makes the six-layer attack
learnable. Phase 1 established the premise by measurement rather than assertion: with all
six layers live from episode one, random exploration reaches mean depth **0.78 of 6** and
wins **0%** of episodes. Red never completes the chain by accident, so it never sees the
+100, so it has nothing to learn from. That is the *sparse reward, long horizon* problem.

A curriculum attacks it by making the early stages winnable and carrying the Q-table
forward as the stack deepens:

| stage | layers | new behaviour red must learn |
|---|---|---|
| 1 | 1–2 | get a foothold in the DMZ |
| 2 | 1–3 | + steal credentials |
| 3 | 1–4 | + pivot to the ad-controller |
| 4 | 1–5 | + escalate privilege |
| 5 | 1–6 | + degrade MFA, then alter the admin credentials |

The carry-forward *is* the transfer. A table that already knows "get a DMZ foothold" does
not relearn it when Layer 3 switches on — stage 2 only has to learn the new layer.

## Result 1 — the curriculum works, and the sawtooth is real

Against a static defence, red clears **every stage at 100%**, and all four promotions are
earned rather than forced:

```
ep  599: stage 1 -> 2  success 100.0%
ep 1199: stage 2 -> 3  success 100.0%
ep 1799: stage 3 -> 4  success 100.0%
ep 2399: stage 4 -> 5  success 100.0%
```

`artifacts/phase3/sawtooth_warmup.png` is the figure. Success climbs, a layer switches on,
it drops, it climbs again. **Part of each drop is not red getting worse** — it is the
exploration rewind that accompanies promotion, and the plot marks both so the two are not
confused.

### Why exploration has to be rewound on promotion

This is not in the spec and the curriculum does not work without it. ε decays over a run,
so by the end of a stage red is nearly greedy. It then arrives at a stage containing a
behaviour it has **never performed** — and a nearly greedy policy cannot find it, because
every action it would need to try is ranked below something that already works. The
curriculum stalls at the first stage needing genuinely new behaviour. `boost_exploration`
inverts the geometric ε schedule to rewind partway up.

> **Viva question.** *Why does your success-rate curve go down four times?*
> Each drop is a stage promotion: a new defensive layer switches on, so the task got
> harder, and exploration is deliberately rewound because the new stage contains a
> behaviour the agent has never performed and a nearly greedy policy can't discover it.
> The recovery after each drop is the transfer — it climbs back much faster than it did
> the first time, because everything below the new layer is already in the table.

## Result 2 — a promotion threshold is relative to an opponent

§7.4 promotes a stage at a **0.70 absolute win rate**. That number is only meaningful
against a fixed opponent. Against the trained Phase 2 defender even the *scripted expert*
wins 11%, so 0.70 is unreachable by construction.

The failure was silent and expensive. Red reached stage 3, spent **4,885 of its 9,000
episodes** there winning 2.3%, and was then evaluated at stage 5 — **a depth it had never
once trained at**. Nothing errored. The agent looked incapable; it was faithfully
executing the four-layer policy it had actually been taught, which is exactly why its mean
depth sat at ~2.5.

`max_episodes_per_stage` was meant to be the escape hatch, but an absolute cap has to be
guessed against a training budget it does not know — 8,000 against a 9,000-episode run can
fire at most once. It is now **a share of the remaining budget over the remaining stages**,
recomputed every episode, which guarantees traversal and lets an easy stage donate its
leftovers to a hard one. Forced promotions stay labelled forced: reaching stage 5 badly
beats never seeing it, but the two must never be confused in a plot.

## Result 3 — the baselines are not what §9 says they are

Two corrections, both measured.

**The random defender is not a security floor.** `isolate` is deterministic and permanent,
so a defender choosing uniformly at random *accidentally plays the degenerate lockdown*:
it ends an episode with **3.46 of its 4 hosts isolated**, matching the trained defender's
security (9.0% attacker success against 10.5%) at **2.3× the cost** (−980 against −420;
2.67 false positives against 1.42). It is the floor on **availability**. Quote it beside
its false-positive rate or it reads as a far stronger baseline than it is.

**The scripted attacker is an oracle, not a peer.** It reads the true state directly —
`BREACH_PRIORITY` is a list of host names. The learned attacker sees only its 384-state
observation, which contains no host identity at all. Comparing them conflates policy
quality with observability, so it is reported as a *reference*, not a competitor.

## Result 4 — how much the observation actually costs red

Worth measuring rather than assuming, and the first answer was wrong.

A tabular agent maps one observation to one action, so if the scripted policy takes
different actions at the same observation, no tabular red can imitate it. Measured over
9,957 decisions:

```
ceiling ignoring the action mask:   82.6%
ceiling WITH the action mask:       97.1%   (143 distinct (obs, legal-set) pairs)
```

The 82.6% looked damning — and it is the wrong number. A **masked** tabular agent
distinguishes observations whose *legal sets* differ, and once that is accounted for, red
can represent **97.1%** of the expert policy. Red's observation was never the bottleneck.

> **Viva question.** *Is your attacker's state representation good enough?*
> I measured it rather than argued it. I ran the expert policy, recorded every
> (observation, action) pair, and computed the best any single-action-per-state policy
> could match. Naively that's 82.6%, but the agent also gets a legal-action mask, so the
> real ceiling is 97.1%. The representation is adequate; the failures were elsewhere.

## Result 5 — potential-based shaping, and the trap it opened

§5.4's ladder (+10 … +60 per layer) is *reward shaping*: dense intermediate signal, added
because the +100 alone is too sparse. **Ng, Harada & Russell (1999)** prove shaping leaves
the optimal policy unchanged **iff** it is potential-based:

```
F(s, s') = γ·Φ(s') − Φ(s)     with   Φ(terminal) = 0
```

A flat rung red *keeps* is not that. Nothing claws it back at termination, so collecting
shaping and then losing on purpose is a policy the reward function permits. Φ is now the
banked ladder value, so the discounted sum telescopes to `γ^T Φ(s_T) − Φ(s_0)` — zero at
both ends. Shaping steers exploration and cannot be a destination.

Two details that matter:

- Φ reads the **paid** breach set, not the currently-breached one. Otherwise blue
  repairing a layer would lower red's potential and hand it a fresh rung to re-climb —
  the farmable loop of Phase 2, with an extra step in it.
- `shaping_gamma` must equal the learner's γ, or the invariance guarantee is gone.

**And it opened a trap.** Under the raw ladder red's returns were ≈ +9 to +18, so
`q_init = 0.0` was roughly neutral. Potential-based shaping moved them to ≈ −90 to −124 —
making `q_init = 0.0` a standing **+100 optimism bonus on every untried action**. This is
Phase 2's §3.14 trap, reappearing on the other team because a *correct* reward change moved
the scale of returns.

> **Viva question.** *You fixed your reward function and results got worse. Why?*
> The fix was right — it made the shaping potential-based, so it can no longer change
> which policy is optimal. But it moved the scale of returns from about zero to about
> −100, and my Q-tables were initialised at 0.0. That had been neutral and was now a
> +100 optimism bonus on every untried action, so the greedy policy increasingly
> preferred things it had never tried. `q_init` isn't a free hyperparameter; it's a claim
> about the scale of your returns, and any change that moves that scale invalidates it.

## What to be able to explain from this phase

1. Why a curriculum is necessary here, in terms of sparse reward and horizon — with the
   Phase 1 numbers (depth 0.78 of 6, 0% wins) as the evidence.
2. What the sawtooth is, and why part of each drop is the exploration rewind.
3. Why an absolute promotion threshold is wrong against an adversary, and how the failure
   presents (an agent evaluated at a depth it never trained at).
4. The potential-based shaping theorem, the form of F, and why `Φ(terminal) = 0` is the
   load-bearing clause.
5. Why the scripted attacker is an oracle baseline and the random defender is not a
   security floor.
6. Why `q_init` is coupled to the reward scale, and the diagnostic that reveals it
   (`untried_greedy_fraction`).

## The methodological lesson

Three hypotheses for why the learned attacker lost to the scripted one — `q_init`
optimism, reward misalignment, observation aliasing — were each coherent, each had a
mechanism, and each was **wrong**. All three died to measurements that took minutes. The
actual cause was visible in a three-line printout of episodes-per-stage, which should have
been the first thing checked.

**When an agent underperforms, instrument what it did before theorising about why.**
