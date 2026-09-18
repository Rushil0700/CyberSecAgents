# Phase 2 — The first learning curve

**What exists:** Q-Learning, SARSA and Expected SARSA written from scratch and validated
against Sutton & Barto; a scripted opponent; a training loop with evaluation snapshots;
per-episode CSV and curve PNGs; a policy inspector; and `experiments/phase2.py`.

```
agents/tabular.py       the three TD algorithms, one update, three targets
agents/scripted.py      the fixed opponent, and the greedy-heuristic baseline
training/loop.py        controllers, episodes, training, evaluation
training/metrics.py     the CLAUDE.md 3.8 row, rolling means, CSV
training/plots.py       four-panel learning curves
experiments/phase2.py   the experiment
experiments/inspect_policy.py   decode a Q-table into a readable policy
```

---

## 1. The three algorithms are one update

```
Q(s,a) ← Q(s,a) + α·[ TARGET − Q(s,a) ]

Q-Learning      TARGET = r + γ·max_{a'∈legal} Q(s',a')          off-policy
SARSA           TARGET = r + γ·Q(s',a')        a' actually taken  on-policy
Expected SARSA  TARGET = r + γ·Σ_{a'∈legal} π(a'|s')·Q(s',a')    on-policy
```

They differ in exactly one term, so they share a class and one `_td_target` method. That
makes §9's algorithm sweep a one-line change and puts the three equations side by side.

### Cliff Walking — why the attacker gets SARSA

Sutton & Barto Example 6.6, reproduced with our own learners (600 episodes, α=0.5, γ=1.0,
ε=0.1):

| | Online return, last 200 episodes | Greedy path |
|---|---|---|
| Q-Learning | **−58.5** (worst) | hugs the cliff edge |
| SARSA | −35.2 | detours away from it |
| Expected SARSA | **−21.1** (best) | intermediate |

Q-Learning learns the *optimal* path — right along the cliff — and keeps falling off it
while exploring. SARSA learns a path that accounts for its own exploration risk, so it
collects more reward **while learning**. Expected SARSA wins on lower variance because it
averages over the policy instead of sampling one action from it.

That is §7.2's argument, reproduced rather than cited: our attacker is punished for
exploring (−50 when detected), so it should learn the stealthy route, and SARSA gives
that for free.

### Three implementation details that are load-bearing

1. **The `max` is over legal actions only.** Bootstrapping off an illegal action means
   chasing a return the agent can never collect, and the error propagates backwards
   through every earlier state. Nothing crashes — the values are just quietly too
   optimistic.
2. **`argmax` breaks ties at random.** At initialisation every value is 0.0, so a plain
   `argmax` returns action 0 in every state forever and the greedy policy is a constant.
3. **Zero initialisation is optimistic here, for free.** Nearly every reward in this
   environment is negative, so visited values drift below untried ones at 0.0 and untried
   actions look best. Change the reward scale and this silently disappears.

---

## 2. Four defects, and how each was found

This is the most useful section for a viva, because none of these four crashed. Three
were found by **measuring**, not by testing.

### 2.1 `GREEDY_EVAL` raised on first use — *found by a unit test*

The geometric ε schedule divides by `epsilon_start`. `GREEDY_EVAL`, the constant shipped
in the same module for evaluation runs, sets `epsilon_start = 0.0`. The first evaluation
would have crashed.

### 2.2 Eleven actions were unsatisfiable — *found by decoding a Q-table*

The tell was a policy readout showing `Q = 0.00` beside a spread of 120 in almost every
state. **An action that is never legal is never taken, so it is never updated, so it
keeps its optimistic 0.0 forever** — and an unmasked `argmax` over a row whose real values
have all gone negative will select it every time.

| Dead action | Why it could never be legal |
|---|---|
| `B_dmz: deploy_honeypot` | both honeypots are in Corp and Secure; Edge/DMZ have no slot |
| `exploit(ad-controller)` | Layer 4 guards the pivot, so `exploit` is barred from it |
| `lateral_move(fileserver, dev-box, ci-runner, honeypot-1)` | reachable only from *inside* Corp, where nothing in Corp is "deeper" than what red holds |
| `lateral_move(honeypot-2)` | reachable only from inside Secure, same argument |
| `steal_credentials(honeypot-1, honeypot-2)` | red can never *hold* a honeypot — touching one is an engagement, not a compromise |

`R_breach` went from 40 actions to 29; `B_dmz` from 11 to 10.

> **The test I got wrong twice.** My first satisfiability test sampled random rollouts and
> flagged deep-zone actions as dead — but blue was isolating the DMZ before red ever got a
> foothold, so they were merely *unreached*. The second used a scripted attacker, which
> beelines for the pivot and never widens inside Corp, so it flagged different live
> actions as dead. Only the third measured the right thing: **construct states directly**
> and ask whether each action's legality predicate is satisfiable at all. "Reachable by
> this opponent" and "satisfiable" are different claims.

### 2.3 The defender farmed the shaping reward — *found by decoding the policy*

The trained defender's greedy action was `tighten_ratelimit` in **93.9% of visited
states**. It was not defending; it was farming §5.3's flat +25 per layer restoration.

The arithmetic: `tighten_ratelimit` is legal whenever Layer 1 is down, and red re-breaches
Layer 1 with `slow_scan` at p=0.60 — about two steps. So blue earned roughly **+12 per
step** from the repair loop, against the **−10 per step** bleed the loop was supposed to
prevent. *Blue profited from red succeeding.*

Measured on the trained agent:

| | |
|---|---|
| Layer restorations per episode | **35.1** (max 92) |
| Reward from restorations | **+175,750** |
| Total return | **−115,338** |
| Share of \|total reward\| from the farm | **152%** |
| Attacker success | 48% — *worse than a random defender's 10%* |

**Fix:** pay the +25 only for the first restoration of each layer per episode. Repairing
again stays legal and still has real defensive value — red must spend actions breaching it
again — it simply is not new reward.

### 2.4 The attacker farmed the ladder the same way — *found by reasoning from 2.3*

The mirror image. §5.4 pays +10 for breaching Layer 1; if blue repairs it, red breaches
again and collected +10 every time. With blue repairing 35 times an episode, red was
quietly banking an unearned +350 from the same loop — **red profiting from blue
defending.** Same fix: the first fall of a layer pays, a re-breach does not.

> **The general lesson.** Both farms are the same failure: a shaping term that pays per
> *event* in a loop the two agents can drive together. If two opposed agents can both
> profit from repeating one interaction, the reward function is wrong, however sensible
> each number looks in isolation. This is §15's "careless shaping → degenerate policy",
> and it is worth checking every per-event reward for it.

---

## 3. Two corrections to the spec

### 3.1 "Layers 1–2 only" cannot mean "switch layers 3–6 off"

§12 specifies Phase 2 as "single blue agent … layers 1–2 only". Deactivating layers 3–6
makes the attacker **stronger**, not the task shallower, because the layers are what stand
in red's way. Measured with the same defender: **35.7%** attacker success with all six
layers active, **82.7%** with layers 3–6 off — red simply walks past the DMZ and the
defender cannot follow.

### 3.2 A curriculum stage must end at its own objective

The deeper cause. §7.4's stage table says stage 1 should teach red to "get a foothold in
the DMZ" — but the win condition always required reaching the crown jewel, so switching
layers off removed the obstacles while leaving the *same full-length journey*. A "shallow"
stage was the whole network with its defences disabled.

**Fix:** a stage's objective is to breach every **active** layer. Layer 6 is the exception
for a real reason rather than as a special case — `alter_credentials` is the move Layer 6
guards, so when it is in play the objective is executing that move, not breaching a layer.

The stages only became graded after this. Steps for a scripted attacker against a static
defence:

| Stage | Active layers | Steps to objective |
|---|---|---|
| 1 | 1–2 | **3.7** |
| 2 | 1–3 | **5.8** |
| 3 | 1–4 | **24.8** |
| 5 | 1–6 | **37.9** |

Before the fix every stage took about the same time, because every stage was the same
journey.

### 3.3 Prevention cannot be free

Once stages were genuinely shallow, `block` became the dominant action — and it cost
nothing. A preventive control that is both free and effective strictly dominates: blue
would blanket-block its zone and §5.3's tradeoff would have nothing left to trade.
Blocking now costs −0.5 per host per step, calibrated so covering all four Edge/DMZ hosts
costs about the same as one isolation. Blue has to choose *where* to spend prevention.

---

## 4. Methodology: two things this phase had to get right

### 4.1 The training curve is not the learning curve

The obvious curve plots return collected *while training*. Here that is misleading, and
measurably so. Eight of `B_dmz`'s ten actions are `block` or `isolate`, so even at ε = 0.05
it takes about a dozen expensive random containments per episode. Over one 3,000-episode
run, **online return fell from −365 to −622 while the greedy policy it had learned was far
better at −159.** The decline was the price of exploring.

So training takes periodic greedy **evaluation snapshots**, and those are the learning
curve. The online series is still recorded, because the gap between the two *is* the cost
of exploration — the defender's version of §7.2's cliff.

### 4.2 A single seed is not a result

With ε decaying over 70% of training, two runs of the **same configuration** scored 6.2%
and 76.5% attacker success. A four-seed sweep of another configuration gave 40.5%, 36.5%,
0.5%, 85.8% — mean 40.8%, **standard deviation 30.3 points.**

The instability has a cause worth stating rather than tuning away:

- **Blue's policy determines the state distribution it sees.** Isolating a DMZ host *ends
  the episode*, so a containment-happy defender only ever observes early-episode states
  while a passive one observes deep ones.
- **Most of blue's return is uncontrollable.** The −10/step bleed comes largely from Corp
  and Secure hosts that `B_dmz` has no action to touch — pure variance in every TD target.

Every number reported from Phase 2 onwards is a mean over seeds, with the spread quoted.
Quoting the best seed would be exactly the dishonesty `CLAUDE.md` §3.7 warns about.

---

## 5. Interview points

**"Walk me through the difference between Q-Learning and SARSA."**
> Same update; they differ in one term. Q-Learning bootstraps off the best next action,
> SARSA off the action it actually took. So Q-Learning learns the optimal policy
> regardless of how it explores, and SARSA learns the value of the policy it is really
> running, exploration included. I reproduced cliff walking to show it: Q-Learning learns
> the path along the cliff edge and keeps falling off while exploring, SARSA detours. I
> gave my attacker SARSA because it is punished for exploring — caught probing is −50 —
> so it should learn the stealthy route rather than the shortest one.

**"Tell me about a bug you found in your own reward function."**
> My defender's greedy action was "restore the perimeter" in 94% of states. It had learned
> to farm a shaping reward: +25 per restoration, and the attacker re-breached that layer
> every two steps, so it earned about +12 a step from the repair loop against the −10 a
> step it was supposed to be preventing. It was profiting from the attacker succeeding,
> and it scored worse than a random defender. I found it by decoding the Q-table into a
> readable policy rather than by looking at the reward curve, which just looked noisy. The
> fix was to pay the reward once per layer per episode — and then I checked the mirror
> image and found the attacker was farming the same loop in the other direction.

**"How do you know your results aren't noise?"**
> I didn't, at first — I reported a single-seed number and then found the same
> configuration scored 6.2% on one seed and 76.5% on another. The standard deviation
> across seeds was 30 points. Now every number is a mean over seeds with the spread
> quoted. The variance has a cause: the defender's own actions end episodes, so its policy
> determines which states it ever sees, and most of its reward comes from zones it cannot
> act in.

**"Why don't you just use stable-baselines?"**
> For this project the point is that I can defend every update rule as my own work — my
> rubric explicitly penalises calling a library without understanding it. Writing it
> myself is also how I found that my `max` had to be taken over the legal action set
> rather than the whole row, which a library would have hidden.

---

## 6. Quiz yourself

1. Expected SARSA had the best *online* return in cliff walking but Q-Learning learns the
   optimal policy. Which would you deploy, and what does that tell you about the
   difference between online and final performance?
2. Why must the `max` in the Q-Learning target be restricted to legal actions? Describe
   the specific way a wrong answer here stays invisible.
3. Blue farmed `tighten_ratelimit` and red farmed the Layer 1 rung. State the single
   property both rewards shared, in one sentence.
4. Why is `deploy_honeypot` absent from `B_dmz`'s action space but present in `B_corp`'s?
5. You see a flat learning curve. Name three diagnostics, in the order you would run them,
   and say what each one would rule out.
6. The defender's online return got *worse* over training while its policy got better.
   Explain how both can be true at once.
