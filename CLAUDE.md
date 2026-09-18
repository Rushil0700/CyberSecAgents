# CLAUDE.md — MARL-SOC working agreement

Read this at the start of every session, before touching any file.
`PROJECT.md` is the full spec. This file is how we work and what we've decided.

---

## 1. Teaching mode — the most important section

Rushil is building this as a college mini-project **and** as preparation for a Dell
AI/GenAI internship interview. He is not looking for finished code handed to him.
He needs to understand every line well enough to defend it in a viva and in an
interview. This is permanent for the whole project, not a phase.

Every session, without being reminded:

1. **Explain before building.** Before writing a file, say in a short paragraph or two
   what it does, why it is designed that way, and what alternatives were rejected.
2. **Explain after building.** Walk through the non-obvious parts. For any line that
   implements an RL equation, show the equation and point at the line.
3. **Define jargon the first time it appears.** Never assume he knows a term.
4. **Flag interview material** inline as `INTERVIEW: <question>`, followed by the
   crisp thirty-second answer he should be able to say out loud.
5. **Quiz him** every few steps with one question about what was just built. If he gets
   it wrong, explain it a different way rather than just correcting him. Don't stack up
   more than one or two unanswered questions.
6. **Push back when he is wrong.** If he proposes something that won't work, say so
   directly and explain why. Do not implement a bad idea just because it was asked for.
7. **Checkpoint at phase boundaries.** Stop, summarise what exists, confirm before
   continuing.

Keep commentary conversational. Do not narrate trivial actions like creating a
directory — spend the explanation budget on things with real design content.

If explanations start thinning out, he will say *"you've stopped explaining — re-read
CLAUDE.md."* That means re-read this section.

---

## 2. Hard constraints — do not work around these

- **Airgapped, always.** Every Docker network is `internal: true`. The lab must never
  be able to reach the internet or any external host. If something appears to need
  external access, **stop and say so** rather than finding a workaround.
- **No real exploit code, no attack tooling.** All "vulnerabilities" are endpoints we
  write ourselves in our own toy Flask services — e.g. a login route that accepts a
  credential we planted. Attacker agents are state-machine navigators. No nmap, hydra,
  metasploit or equivalents, and no code that would function as a real exploit outside
  this lab. This is a decision-making study, not an intrusion study.
- **RL from scratch, NumPy only.** No stable-baselines, no RLlib, no Tianshou. The
  rubric penalises calling an RL library without understanding it. Every update rule
  must be defensible as our own work. PyTorch is allowed only for the optional DQN
  extension, and only in the final phase.
- **Tabular first.** Q-Learning, SARSA, Expected SARSA. Deep RL is the last optional
  phase, never the starting point.
- **State budget: ≤ ~10,000 states per agent.** If a design pushes past this, stop and
  say so — we redesign the discretization rather than accept the larger space.
- **Test as we go.** Unit tests for transition and reward logic, written alongside each
  module, not afterwards. He needs to trust the environment before trusting a curve.
- **Never hard-code tactics.** Domain knowledge enters through the action set, the
  reward function, legality constraints and the training curriculum — never as an
  `if host == "ad-controller"` rule. Hard-coded tactics fail the rubric's "not
  hard-coded" criterion and destroy every emergent-behaviour result.

---

## 3. Decisions made (2026-09-17)

These came out of the spec review. They amend `PROJECT.md`; where they conflict,
this section wins.

### 3.1 True state is hidden behind a noisy detection layer

`PROJECT.md` §5 puts `status ∈ {clean, suspicious, compromised, isolated}` directly in
blue's state vector. That makes detection free: MTTD is always 1 step and false
positives are impossible, which guts the security/availability tradeoff.

Instead, the environment keeps a private **true state** (`clean` / `compromised` /
`isolated`) and blue observes a **noisy status** derived from an alert process that can
be wrong in both directions. Compromised hosts emit alerts probabilistically (more when
the attacker acts loudly); clean hosts emit occasional false alerts. This lives in
`env/detection.py`, kept deliberately separate from `env/observations.py` so the true
state cannot leak into an agent's view by accident.

### 3.2 Availability cost is a config flag

`PROJECT.md` §5 penalises a false positive **once** at −20. That does not prevent the
degenerate "isolate everything" policy, because the benefit of over-isolating is
per-step while the penalty is one-shot: isolating all five hosts costs −100 once and
then avoids the −10/step compromise bleed forever.

`config.py` exposes `AvailabilityCost.ONE_SHOT` and `AvailabilityCost.PER_STEP`.
**Default is `PER_STEP`** (a per-step cost for every isolated host) so real runs have
clean curves. `ONE_SHOT` is kept so we can reproduce the degenerate policy on demand as
a before/after result.

### 3.3 Red learns with a curriculum; no hard-coded tactics

Red starts random and learns the DMZ → Corp → Secure chain from reward alone. It is
trained first against a **static** defence so it receives positive signal, then against
a learning blue. Warm-starting red's Q-table from a scripted expert is a **contingency
only** if red cannot succeed after tuning — and it costs the emergence claim for
whatever the script taught, so it must be flagged explicitly in the notes if used.

### 3.4 Alternating training is the main method; simultaneous IQL is a control

Training both teams simultaneously is non-stationary and will not converge. Main method
is alternating: freeze one team, train the other to convergence, swap, repeat — each
phase is then a stationary MDP. We **also** run one simultaneous-IQL configuration as a
control, because its failure to settle is the empirical demonstration that IQL is
theoretically unsound in the multi-agent setting.

### 3.5 Per-agent action spaces

`PROJECT.md` §5 computes `|A| = 12` for a five-host zone, but `B_dmz` has three hosts
(8 actions) and `B_secure` has four (10 actions). Action spaces are built per agent from
the topology; never assume a single shared action space.

### 3.6 Red's state discretization — resolved at Phase 1

Superseded by the six-layer `PROJECT.md` §5.2, which gives red progress features rather
than host bitmasks: `(current_zone, footholds_bucket, creds_held, privilege_escalated,
mfa_degraded, heat_level)`. Implemented in `env/observations.py`.

**Spec arithmetic correction:** §5.2 states `4 × 4 × 2 × 2 × 2 × 3 = 768`. The product is
**384**. The argument built on it is unaffected — §5.2's claim that six layers cost 8×
over two holds exactly, since two layers is `4 × 4 × 3 = 48` and three binary flags is
2³. Asserted in `tests/test_observations.py` so the slide and the code cannot drift.

### 3.9 Some hosts are gates, not targets (Phase 1)

`edge-gateway` (L1) and `mfa-service` (L6) have `exploit_prob = 0.0` and a separate
`bypass_prob`. They are passed, never owned. If they were ownable, red's cheapest policy
would be to compromise them like any other host and the six heterogeneous layers would
collapse into "exploit six hosts in a row" — losing §3.1's entire argument.

### 3.10 The layer preconditions are a partial order (Phase 1)

L6 requires L4, **not** L5. Once red holds the pivot it has network reach to
`mfa-service`, so degrading MFA is genuinely available before escalating privilege. A
total order would make the six-layer environment a single path with no decisions in it;
the branch forces red to *learn* an ordering.

### 3.11 Detection is calibrated so that stealth works (Phase 1)

Alert accumulation reaches `p / (1 - decay)`. A compromised host sitting idle settles at
1.00 against a suspicion threshold of 1.5 — it never even looks suspicious. Acting is
what exposes it, and the noisiest layer is 4× more visible than the quietest. A stealthy
red therefore never reads as `COMPROMISED`, only ever `SUSPICIOUS`, so blue must choose
between isolating on suspicion and paying, or waiting for a confirmation it will never
get. Do not "fix" the false-alert rate or the 0.85 detection ceiling: both are load-
bearing for §5.3's tradeoff and §7.2's cliff-walking argument.

### 3.7 Every agent has three independent switches

Each agent is configured by an `AgentConfig` with three orthogonal fields, because
"turn an agent off" means three different things and collapsing them would force a
rewrite of the training loop for every experiment:

- `enabled` — does the agent act at all? (ablations, baselines, demo control)
- `learning` — does it update its Q-table? (alternating training, frozen deployment)
- `policy` — `LEARNED` / `SCRIPTED` / `RANDOM` / `GREEDY` / `NOOP`

A `ScenarioConfig` bundles per-agent configs plus a `seed`. This single mechanism
covers the demo controls, the three baselines in `PROJECT.md` §9 and the agent
ablation in the experiment grid — "all blue agents disabled" *is* the static-firewall
baseline.

Episodes must be **exactly reproducible from a seed**, so a chosen episode can be
replayed on demand for a demo. Selecting a representative seeded episode to show is
legitimate; altering exploit probabilities or detection rates to manufacture an outcome
and then reporting it as a result is not. Demos pair two runs on the **same seed** with
one variable changed (e.g. blue disabled vs blue trained), and any single episode shown
is accompanied by the aggregate over many runs.

### 3.8 Progress reporting is built in from Phase 1

Every episode writes a row: `episode, phase, learner, red_return, blue_return, outcome,
steps, hosts_compromised, mttd, mttc, false_positives, honeypot_hits, epsilon`. Live
console summary every 500 episodes; per-run CSV and curve PNGs under
`artifacts/runs/<id>/`; a cross-phase arms-race plot at the end of alternating training.
Metrics cannot be retrofitted — the environment must record detection and containment
times as they happen.

---

## 4. Repo conventions

- **Commits:** small and frequent, messages explaining *why* rather than *what*.
- **`notes/`:** after each phase, a short markdown note on what we built and what he
  should be able to explain from it. This is his revision material.
- **Type hints and docstrings** on anything implementing an algorithm. The docstring
  states the update rule it implements, in the equation form used in `PROJECT.md` §7.
- **One file per MDP symbol** under `src/marlsoc/env/` — topology, state, detection,
  observations, actions, rewards — so that in a viva, "where is your transition
  function?" is answered by opening one file.
- **`topology.py` is pure data** and is the single source of truth for hosts, zones,
  the firewall matrix and exploit probabilities. Phase 4's `docker-compose.yml` is
  generated from it, so the twin and the lab cannot drift apart.

---

## 5. Where we are

**Phases 0 and 1 — complete.** The simulated twin runs. `config.py` plus eight modules
under `env/` (topology, layers, state, detection, observations, actions, rewards,
minicorp, compose_gen), 185 tests, and a generated `docker-compose.yml`. Write-up and
revision material: `notes/phase1-environment.md`.

Feasibility is established and test-enforced: red wins 100% against a static defence at
every curriculum stage, and 0% against even a random defender at mean depth 0.78 of 6.
Red can win; random exploration cannot find it — which is §7.4's sparse-reward argument
demonstrated rather than asserted.

**Phase 2 — complete.** Q-Learning, SARSA and Expected SARSA from scratch, validated
against Sutton & Barto's cliff walking; scripted opponent; training loop; metrics and
plots; policy inspector; `experiments/phase2.py`. 260 tests. Write-up:
`notes/phase2-first-curve.md`.

Headline: at curriculum stage 1–4 a single learned `B_dmz` takes attacker success from
**99.7% to 6.8%** (sd 7.6 over four seeds) and blue's return from **−1,315 to −224**.

### 3.12 A curriculum stage ends at its own objective (Phase 2)

§7.4's stage table says stage 1 teaches red to "get a foothold in the DMZ". Requiring the
crown jewel at every stage does not do that — switching layers off removes the obstacles
but leaves the same full-length journey, so a "shallow" stage is the whole network with
its defences disabled, which is *easier* for red. A stage's objective is now to breach
every **active** layer; Layer 6 is the exception because `alter_credentials` is the move
it guards. Only after this did the stages become graded (3.4 / 5.5 / 29.1 / 40.1 steps).

**Phase 2 runs at stage 1–4, not 1–2.** Measured: stage 1–2 is over in 3.4 steps and blue
never gets a turn (100% → 99.3%); stage 1–3 gives blue time but not value, so a defensive
stand costs more than the breach it prevents; stage 1–4 needs ~29 steps to the pivot,
which is long enough for containment to pay for itself.

### 3.13 Check every per-event reward for a farmable loop (Phase 2)

Two were found. Blue farmed §5.3's +25 layer restoration (93.9% of its greedy actions;
35.1 restorations an episode; 152% of its total reward) and red farmed the mirror image of
the same loop on §5.4's ladder. **If two opposed agents can both profit from repeating one
interaction, the reward is wrong however sensible each number looks alone.** Both are now
paid once per layer per episode. `block` likewise cannot be free: a preventive control
that is both free and effective strictly dominates.

### 3.14 `q_init` is load-bearing; do not set it to zero by accident (Phase 2)

Returns here are large and negative (≈ −150 typical, −2,500 bad), so initialising at 0.0
gives every untried action a ~+150 optimism bonus. Good during training, ruinous at
evaluation: after 4,000 episodes, **91% of visited states had a greedy action that had
never been updated** (68% of decisions by visit weight). That is why a trained defender
scored worse than always choosing `noop`, and why greedy scored worse than ε = 0.05. Use
`TabularLearner.untried_greedy_fraction` to check.

### 3.15 Decide termination before computing rewards (Phase 2)

`_check_termination` is what sets `events.red_won` for a stage objective, so computing
rewards first meant a stage win paid nothing at all. Only `alter_credentials` was
unaffected, which is why the full six-layer game hid it. **Reconcile the simplest baseline
against arithmetic you can do on paper** — 100% attacker success had to cost about −125
and was reading −38.6.

**Phase 3 — next.** Red learns with the curriculum, stages 1–5, against a static defence
(`PROJECT.md` §7.4). The stage machinery Phase 2 forced into correctness is what Phase 3
is built on.

Build order is `PROJECT.md` §12. Do not skip ahead, and do not build the Docker lab
(Phase 6) before a working learning curve (Phase 2) exists.

**Note on the spec:** `PROJECT.md` was revised from three zones to four and from a flat
chain to six heterogeneous layers. §3 above predates that revision in places; where §3
and the current `PROJECT.md` conflict on topology, `PROJECT.md` wins, but the *amendments*
in §3.1–3.4 and §3.7–3.8 still stand and are implemented.
