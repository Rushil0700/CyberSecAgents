# Phase 1 — The simulated twin

**What exists:** eight modules under `src/marlsoc/`, 185 passing tests, a generated
`docker-compose.yml`, and an environment you can step.

```
config.py              run switches: three per agent, plus the availability-cost mode
env/topology.py        the map          -- who exists, who reaches whom, how hard
env/layers.py          the six layers   -- preconditions, reward ladder, noise
env/state.py           S                -- ground truth, hidden from everyone
env/detection.py       the alert noise  -- what blue gets to find out
env/observations.py    O_i(s)           -- what each agent is handed, + Q-table indexing
env/actions.py         A_i              -- per-agent action sets and the legality mask
env/rewards.py         R_i              -- both teams' reward functions
env/minicorp.py        P, step, reset   -- the transition function
env/compose_gen.py     the Docker lab, generated from topology.py
```

"Where is your transition function?" is answered by opening one file. That is the
one-file-per-MDP-symbol convention, and it is worth pointing at in a viva.

---

## The seven design decisions you should be able to defend

### 1. Gates are not targets

`edge-gateway` (Layer 1) and `mfa-service` (Layer 6) have `exploit_prob = 0.0` and a
separate `bypass_prob`. They cannot be owned, only passed.

**Why:** if they were ownable, red's cheapest policy would be to compromise them like
any other host, and the six heterogeneous layers would collapse into "exploit six hosts
in a row" — which is exactly the design `PROJECT.md` §3.1 argues *against*. One line of
data is what keeps the layer taxonomy real rather than decorative.

### 2. The layers are a partial order, not a chain

Layer 6 requires Layer 4, **not** Layer 5:

```
L1 → L2 → L3 → L4 ─┬→ L5 ─┬→ alter_credentials
                   └→ L6 ─┘
```

**Why:** once red holds the pivot it has network reach to `mfa-service`, so degrading
MFA is genuinely available before escalating privilege. A total order would make the
"six-layer decision problem" a single path with no decisions in it. With the branch, red
has to *learn* an ordering — and that interacts with detection, since
`escalate_privilege` is the loudest action in the game.

### 3. Ground truth has three values; the fourth is a belief

`state.py` knows only `CLEAN` / `COMPROMISED` / `ISOLATED`. `SUSPICIOUS` is manufactured
by `detection.py` and exists only in `observations.py`.

**Why:** §5.1 puts `suspicious` straight into blue's state vector. Built that way, a host
turns suspicious the instant it is attacked — MTTD is 1 step in every episode, false
positives are impossible, and the security/availability tradeoff has nothing to trade.

### 4. Blue's signal is buried in a base rate

A 2% per-step false-alert rate across twelve clean hosts produces alerts at a rate
comparable to one compromised host: measured over 400 steps, **48 true against 56 false**.

**Why:** blue cannot learn "isolate whatever alerted". It has to distinguish a *sustained*
signal on one host from a *scattered* one across many. This is the **base rate fallacy**,
and encoding it is what turns §5.3's tradeoff from a sentence into a property of the
environment.

The calibration that makes it work — steady-state accumulation is `p / (1 - decay)`:

| Situation | p/step | accumulates to | reads as |
|---|---|---|---|
| clean host | 0.02 | 0.13 | quiet |
| **compromised but idle** | 0.15 | **1.00** | **still quiet** |
| compromised, `slow_scan` | 0.23 | 1.53 | suspicious |
| compromised, `escalate_privilege` | 0.63 | 4.20 | loud |
| honeypot touched | 0.95 | 6.33 | screaming |

Read row three. **An attacker that sits still stays below the suspicion threshold.** Red
can genuinely hide by doing nothing, so `wait` is a real strategic option; *acting* is
what exposes it, and the noisiest layer is four times more visible than the quietest.
That is §7.2's cliff, built out of parameters rather than declared.

Two consequences in `observations.py`: a stealthy red shows as `SUSPICIOUS` at most and
**never** as `COMPROMISED`, so blue must either isolate on suspicion and pay, or wait for
a confirmation a careful attacker never gives. And a clean host still crosses the confirm
threshold roughly once in forty episodes — "confirmed" is not a synonym for "true".

### 5. The degenerate policy had to be priced out

§5.3 charges a false positive **once**, at −20. Do the arithmetic over twelve defended
hosts and a 250-step episode:

| Policy | `ONE_SHOT` | `PER_STEP` (default) |
|---|---|---|
| isolate everything at step 1 | **−240** | −6,240 |
| do nothing while red takes 5 hosts | −10,100 | −10,100 |
| isolate the 2 real intrusions at step 20 | — | **≈ −1,220** |

Under `ONE_SHOT` the lockdown is optimal by a factor of forty. Blue is **not**
malfunctioning when it learns it — it is correct, and the reward function is wrong. The
benefit of over-isolating is per-step while the penalty is one-shot. `PER_STEP` adds −2
per isolated host per step and restores the ordering you want: targeted ≫ lockdown ≫
passive. `ONE_SHOT` is kept so you can reproduce the degenerate policy as a before/after
result.

### 6. Each layer costs one binary digit, not a dimension of hosts

`R_breach` state: `(zone, footholds_bucket, creds, privilege, mfa, heat)`.

**Correction to the spec:** §5.2 says `4 × 4 × 2 × 2 × 2 × 3 = 768`. That product is
**384**. The *argument* survives untouched — §5.2 claims six layers cost 8× over two, and
two layers is `4 × 4 × 3 = 48`, with `48 × 8 = 384` exactly, since three binary flags is
2³. Fix the number on the slide; the point is unaffected and the corrected figure is
further inside budget.

| Agent | States | Actions |
|---|---|---|
| `B_dmz` | 768 | 10 |
| `B_corp` | 3,072 | 13 |
| `B_secure` | 192 | 9 |
| `R_breach` | 384 | 29 |
| `R_scout` | 96 | 9 |

Budget is 10,000. Every agent is inside it, and the test suite enforces it.

> **Updated in Phase 2.** These action-space sizes started at 11 and 40. Eleven actions
> turned out to be *unsatisfiable* — legal in no reachable state — and were removed. See
> `notes/phase2-first-curve.md`; the short version is that an action which is never legal
> is never updated, so it keeps its optimistic 0.0 initialisation forever and silently
> dominates any unmasked reading of the table.

### 7. Two bugs the unit tests could never have found

Both surfaced only from *running* episodes and measuring win rate per curriculum stage.

**Layer 4 was walkable.** The pivot sits in Corp, so once red held `intranet`,
`exploit(ad-controller)` was legal as ordinary intra-zone widening — and only
`lateral_move` awards Layer 4. Red took the pivot and walked straight through the layer
without breaching it, pinning mean depth at exactly 3.00 in every stage. Layer 4 is
micro-segmentation *inside* Corp, so `lateral_move` now covers a deeper zone **or the
pivot**, and `exploit` is barred from the pivot.

**Progress was derived from breached layers.** Under a curriculum stage where Layer 4 was
inactive it could never be *breached*, so Corp never counted as entered, the secure zone
stayed unscannable, and red could never discover the crown jewel. Stage 1–3 won 0% of
episodes while the strictly harder stage 1–4 won 100%. Where red has been is a fact about
footholds, not about which layers the curriculum happens to have switched on.

**The lesson worth saying out loud:** both bugs made red look *worse*, not better, and
neither raised an error. A learning curve would have flatlined and the natural conclusion
would have been "the agent needs more episodes."

---

## The feasibility result

| Condition | Red win rate | Mean layers breached |
|---|---|---|
| vs **static** defence, layers 1–2 | 100% (37.6 steps) | 2.0 / 2 |
| vs **static** defence, layers 1–6 | 100% (47.2 steps) | 6.0 / 6 |
| vs **random** blue, layers 1–6 | **0%** | **0.78 / 6** |

Red *can* win, so the environment is not broken. Random exploration *cannot find it* once
anyone is defending. That is the sparse-reward / long-horizon problem in §7.4 and §15,
demonstrated on day one rather than asserted — and it is the empirical case for both the
progressive ladder and the curriculum. Both rows are now tests, so if a future change
makes either false, the suite says so.

It also fixes your §9 headline: the static-firewall baseline reports ~100% attacker
success, consistent with the 94% figure in §0.

---

## Interview points

**"How did you keep tabular RL feasible with six layers of defence?"**
> Each layer contributes one binary flag to the attacker's state, not a new dimension of
> hosts. Going from two layers to six multiplied red's state space by exactly 8 — three
> binary flags — from 48 to 384, and cost the defenders nothing at all, because their
> state is scoped to their own zone. My largest agent is 3,072 states against a
> self-imposed budget of 10,000.

**"Isn't a six-layer environment just a long corridor?"**
> No — the preconditions are a partial order, not a total one. Layers 5 and 6 are
> independent once the pivot is held, so the agent has to discover which to do first, and
> that interacts with detection: privilege escalation is my noisiest action, so doing it
> early means carrying heat longer. The ordering red converges on is a learned result.

**"How do you stop the defender learning to just isolate everything?"**
> That policy is genuinely optimal under the reward function as originally specified —
> a one-shot −20 per false positive against a per-step benefit, so locking down twelve
> hosts costs −240 once and saves thousands. I made the availability cost per-step, which
> flips the ordering so targeted defence beats lockdown beats passivity. I kept the broken
> mode behind a config flag so I can show the degenerate policy as a before/after result.

**"Why is detection a hard problem in your environment?"**
> Because I gave clean hosts a 2% false-alert rate. Across twelve hosts that produces as
> many alerts as one genuinely compromised host — 56 false against 48 true over 400 steps.
> So the defender can't act on alerts, it has to act on patterns. And I capped detection
> at 0.85, never 1.0, so a stealthy attacker sitting idle stays below the suspicion
> threshold entirely and is never confirmed.

**"What went wrong and how did you find it?"**
> Two precondition bugs that made the attacker silently weaker with no error raised. I
> found them by measuring attacker win rate per curriculum stage against a static defence
> — one showed as depth pinned at exactly 3.0 in every stage, the other as a shallower
> stage being strictly harder than a deeper one, which is impossible if the layers are
> really in the way. Both are regression tests now. The general lesson is that in RL a
> broken environment doesn't crash, it just produces a flat curve you misread as needing
> more episodes.

---

## Quiz yourself

1. `escalate_privilege` has `success_prob = 0.35` and `noise = 6`; `slow_scan` has
   `bypass_prob = 0.60` and `noise = 1`. What does that pair of numbers do to red's
   preferred *ordering* of layers 5 and 6, and why does it matter that the order is
   partial?
2. Blue observes `SUSPICIOUS` on a host. Give two completely different true states that
   produce that observation, and say what it costs blue to guess wrong in each direction.
3. Why does `isolate()` count the false positive *before* writing the new status?
4. Red holds `ad-controller` and `mfa_degraded` is true, but `alter_credentials` is masked
   out. Name two different reasons that could be the case.
5. `LayerStatus` carries `active` and `breached` as separate sets. What specifically
   breaks in curriculum stage 1 if you merge them?
6. Detection probability is capped at 0.85 rather than 1.0. Which argument in §7.2 does
   that cap protect, and how?
