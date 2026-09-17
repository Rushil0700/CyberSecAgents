# MARL-SOC: Multi-Agent Reinforcement Learning for Autonomous Network Defense

**A-Z project bible.** Read this top to bottom once. Everything you need for the
course rubric *and* for the Dell interview is in here.

---

## 0. The 30-second pitch

> We build a fully isolated, simulated enterprise network inside Docker. Two
> attacker agents try to breach it and reach a customer database. Three defender
> agents — one per network zone — learn, through reinforcement learning, to detect
> and contain them. Neither side is programmed with tactics. Both sides learn by
> playing against each other. On top of the defenders sits an LLM copilot that
> explains every defensive decision in plain English.

**Course framing:** a general-sum Markov Game solved with Independent Q-Learning.

**CV / interview framing:** *"Built a multi-agent RL network-defense system with an
LLM copilot (RAG over live container logs), deployed on an isolated Docker network
with CI/CD. Reduced attacker success rate from 94% to X% and mean time-to-containment
from N to M steps versus a static firewall baseline."*

---

## 1. Safety and scope (read this first, say it in your demo)

This project **never touches a network we do not own**. This is enforced by design,
not by promise:

- The entire lab runs on a Docker bridge network declared `internal: true`, which
  means the containers have **no gateway to the outside world**. There is physically
  nowhere for a packet to go.
- There is **no real exploit code and no attack tooling** in this project. The
  "vulnerabilities" are endpoints *we wrote ourselves* in our own toy Flask services
  (e.g. a login route that accepts a planted weak credential). An "exploit" action is
  an HTTP request to our own container that we designed to succeed.
- The attacker agents are **state-machine navigators**, not exploit frameworks. The
  research contribution is the *decision-making*, not the intrusion technique.

Sentence for your proposal and your viva:

> "The environment is a fully isolated Docker network with no external connectivity.
> All services and their vulnerabilities are purpose-built by us for this study. No
> real systems are contacted at any point."

This mirrors how the academic benchmarks in this space work (CybORG / CAGE
Challenge, Microsoft CyberBattleSim) — they are all simulators, for exactly this
reason.

---

## 2. Why this project (mapping to the marks)

| Rubric component | Marks | How we hit it |
|---|---|---|
| Problem definition & motivation | 1.5 | Autonomous SOC response; real industry problem; clear threat model |
| AI agent & environment design | 2.0 | 5 agents, 13-host segmented enterprise, hand-built environment |
| RL/MDP formulation | 2.0 | Full ⟨S, A, P, R, γ, π⟩ in §5, with discretization justified |
| MAS / Markov Game formulation | 2.0 | General-sum Markov Game, §6; coop within team, compete across |
| Proposed methodology & algorithm | 1.5 | IQL with Q-Learning / SARSA / Expected SARSA comparison, §7 |
| Feasibility & expected outcomes | 1.0 | Phased plan §10, metrics §9, sim-twin keeps training cheap |

And the "Avoid" list from the guidelines:

- ✅ Not classification/regression — it's sequential decision-making.
- ✅ Not an LLM wrapper — the LLM is a *separate explanation layer*, the RL core is ours.
- ✅ Not single-agent with "multi-agent" in the title — 5 agents with genuine interdependence.
- ✅ Not hard-coded — tactics emerge from learning; baselines are the hard-coded thing we *beat*.
- ✅ Not library-calling — we implement Q-Learning, SARSA and Expected SARSA from scratch.

---

## 3. The environment: "MiniCorp"

A fake company network, ~13 containers, three zones, strict segmentation.

### 3.1 Topology

```
                    ┌───────────────────────────────────┐
  INTERNET          │  (no route — internal: true)      │
  (unreachable)     └───────────────────────────────────┘

  ╔═══════════════════ ZONE 1: DMZ ════════════════════╗
  ║  reverse-proxy   web-portal ★ENTRY   mail          ║
  ╚════════════════════════╤═══════════════════════════╝
                           │  (firewall: only proxy→intranet)
  ╔═══════════════ ZONE 2: CORPORATE LAN ══════════════╗
  ║  intranet   fileserver   dev-box   ci-runner       ║
  ║             ad-controller ★PIVOT                   ║
  ╚════════════════════════╤═══════════════════════════╝
                           │  (firewall: only ad-controller→db)
  ╔═══════════════ ZONE 3: SECURE ═════════════════════╗
  ║  db-primary ★TARGET    backup    honeypot-1/2      ║
  ╚════════════════════════════════════════════════════╝

  ┌──────────┐
  │  logger  │  ← receives logs from all zones
  └──────────┘     (feeds blue monitoring + LLM copilot RAG)
```

### 3.2 Hosts

| Host | Zone | Role | Planted weakness |
|---|---|---|---|
| `reverse-proxy` | DMZ | nginx front door | none (routing only) |
| `web-portal` | DMZ | **Attacker entry point** | weak admin credential |
| `mail` | DMZ | fake mail service | exposed config endpoint |
| `intranet` | Corp | internal web app | trusts DMZ session token |
| `fileserver` | Corp | file shares | world-readable share |
| `dev-box` | Corp | developer machine | reused credential from web-portal |
| `ci-runner` | Corp | build server | over-privileged service account |
| `ad-controller` | Corp | **Pivot to secure zone** | credential cache |
| `db-primary` | Secure | **CROWN JEWEL** | reachable only from ad-controller |
| `backup` | Secure | secondary target | weak backup credential |
| `honeypot-1/2` | Secure/Corp | decoys (blue deploys) | looks juicy, alerts on touch |
| `logger` | — | log aggregation | n/a |

**Design intent:** the attacker *must* traverse DMZ → Corp → Secure. That forced
path is what creates the lateral-movement decision problem worth learning.

### 3.3 Transitions, constraints, termination

- **Transitions:** an exploit action on host *h* succeeds with probability `p_h`
  (we set these; e.g. 0.9 for `web-portal`, 0.4 for `ad-controller`) **and** only if
  the attacker already holds a host with network reach to *h*. Reachability is
  defined by our firewall matrix.
- **Constraints:** attacker can only act from a compromised foothold; defenders can
  only act within their own zone; each isolate action costs availability.
- **Termination:** (a) attacker reaches `db-primary` → red win; (b) all attacker
  footholds isolated → blue win; (c) step limit `T = 100` → draw.

---

## 4. The agents

### 4.1 Red team (2 agents, cooperative with each other)

| Agent | Goal | Observes | Actions |
|---|---|---|---|
| `R_scout` | Find reachable hosts cheaply and quietly | Own zone reachability, discovered-host count, own "heat" level | `scan(zone)`, `probe(host)`, `wait` |
| `R_breach` | Compromise hosts, move laterally, reach `db-primary` | Discovered hosts, own footholds, heat level | `exploit(host)`, `lateral_move(host)`, `exfil`, `wait` |

They share a team reward — `R_scout` gets credit for discoveries that `R_breach`
later converts. That is your **credit assignment** discussion.

### 4.2 Blue team (3 agents, one per zone, cooperative)

| Agent | Defends | Observes | Actions |
|---|---|---|---|
| `B_dmz` | DMZ (3 hosts) | Per-host status in DMZ + zone alert level | `block(host)`, `isolate(host)`, `deploy_honeypot`, `noop` |
| `B_corp` | Corp LAN (5 hosts) | Per-host status in Corp + zone alert level | same |
| `B_secure` | Secure (4 hosts) | Per-host status in Secure + zone alert level | same |

**Why zone-split and not role-split (monitor/responder/deception)?** Three reasons,
and this is a good interview answer:

1. **Tractability.** A blue agent seeing all 13 hosts at 4 statuses each would need
   a Q-table of 4¹³ ≈ 67 million states. Scoped to one zone it's 4⁵ × 3 ≈ 3,000.
   Tabular RL becomes possible.
2. **Genuine partial observability.** Each defender sees only its own zone, which is
   realistic and makes this a proper POMDP-flavoured problem.
3. **Real interdependence.** If `B_dmz` fails, `B_corp` inherits an attacker. Their
   fates are coupled, which is exactly the "genuine interaction" the rubric demands.

---

## 5. RL formulation (the MDP)

Take the **`B_corp` defender** as the worked example.

### State S

```
s = ( status[intranet], status[fileserver], status[dev-box],
      status[ci-runner], status[ad-controller],   # 4 values each
      zone_alert_level )                          # 3 values: low/med/high
```

where `status ∈ {clean, suspicious, compromised, isolated}`.

|S| = 4⁵ × 3 = **3,072 states**. Comfortably tabular.

> **Discretization is the whole trick.** Raw telemetry (packet counts, CPU, log
> lines/sec) is continuous and unbounded. We bucket it into `{clean, suspicious,
> compromised, isolated}` using threshold rules on the alert stream. This is what
> makes Q-Learning applicable instead of needing a neural network.

### Actions A

`A = { block(h), isolate(h), deploy_honeypot, noop }` for the 5 hosts in zone
→ |A| = 5 + 5 + 1 + 1 = **12 actions**.

Q-table size = 3,072 × 12 ≈ **37k entries**. Trains in minutes.

### Reward R (blue)

| Event | Reward |
|---|---|
| Attacker reaches `db-primary` | **−100** (team-wide) |
| Each host currently compromised | **−10** per step |
| Correctly isolating a compromised host | **+50** |
| Isolating a *clean* host (false positive) | **−20** |
| Attacker engages a honeypot | **+30** |
| Per timestep | **−1** (urgency) |

> **The degenerate-policy trap.** Without the −20 false-positive penalty, blue
> learns the trivial policy "isolate everything immediately" — perfect security,
> zero availability. The penalty encodes the real security/availability tradeoff.
> **This is your single best analysis paragraph and a great interview story.**

### Reward R (red)

| Event | Reward |
|---|---|
| Reach `db-primary` | **+100** |
| Each new host compromised | **+10** |
| Detected / blocked | **−50** |
| Wasted action on honeypot | **−30** |
| Per timestep | **−1** |

### Transition P

Stochastic: `P(s'|s,a)` determined by exploit success probabilities `p_h`, detection
probability given alert level, and the firewall reachability matrix. We define these;
the agents never see them. That's model-free RL.

### Discount γ

**γ = 0.95.** Justification: containment is a multi-step endeavour — blocking a host
now pays off several steps later when the attacker fails to pivot. A high γ makes the
agent value that delayed payoff. Too low (0.5) and blue becomes myopic and reactive.
**Make γ one of your swept hyperparameters.**

### Policy π

ε-greedy over the Q-table, with ε decaying 1.0 → 0.05 over training.
`π(s) = argmax_a Q(s,a)` at evaluation time.

---

## 6. Multi-agent formulation (the Markov Game)

A **general-sum Markov Game** ⟨N, S, {Aᵢ}, {Rᵢ}, P, γ⟩:

- **N = 5** agents: `{R_scout, R_breach, B_dmz, B_corp, B_secure}`
- **S** — the true global state (all host statuses, all footholds). **No agent
  observes this.** Each agent sees its own local observation `oᵢ = Oᵢ(s)`.
- **Aᵢ** — per-agent action sets (§4). Joint action `a = (a₁,…,a₅)`.
- **Rᵢ** — team-shared within a team, opposed across teams.
- **Not zero-sum:** honeypot engagement is `+30` blue / `−30` red (zero-sum), but a
  false positive is `−20` blue / `0` red. The payoff matrix doesn't sum to zero, so
  it's **general-sum**, which is the more interesting and more honest classification.

### Cooperation and competition

- **Within a team — cooperation via shared reward.** All three blue agents receive
  the same team reward signal. Nobody is told to cooperate; coordination is the only
  way to maximise a shared return.
- **Across teams — competition.** Red's gain is largely blue's loss.

### Individual vs shared reward — your headline experiment

Run both. Predicted result: with **individual** reward, `B_dmz` becomes selfishly
trigger-happy (isolating aggressively is cheap for *it*, since the downstream cost
lands on `B_corp`). With **shared** reward, `B_dmz` learns restraint and lets
`B_corp` handle threats better suited to it. That behavioural difference *is* your
emergent-behaviour finding.

### Communication

Two configurations to compare:
- **No comms:** each agent sees only its zone.
- **With comms:** each agent additionally observes a 1-bit "neighbour zone under
  attack" flag (state space ×2 — still tiny).

Expect the comms version to show **anticipatory defence**: `B_corp` raising its guard
*before* the attacker arrives, purely because `B_dmz` is lit up.

### The non-stationarity problem (say this in the interview)

> When all agents learn simultaneously, each agent's environment includes the other
> agents — whose policies are changing. So the environment is **non-stationary**, and
> the Markov property that Q-Learning's convergence proof depends on is violated.
> Independent Q-Learning has *no convergence guarantee* here. We handle it with
> **alternating training**: freeze red, train blue to convergence, freeze blue, train
> red, repeat. Each phase is then a stationary MDP.

That paragraph alone will impress an interviewer. Most students don't know IQL is
theoretically unsound in the multi-agent setting.

---

## 7. Algorithms (and why each)

We implement all three **from scratch**. No `stable-baselines`, no `rllib`.

### Q-Learning (off-policy) — for the BLUE team

```
Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
```

**Why blue:** the defender can explore freely — a wrong action during training costs
nothing real. Off-policy learning of the optimal policy regardless of the
exploratory behaviour policy is exactly what we want.

### SARSA (on-policy) — for the RED team

```
Q(s,a) ← Q(s,a) + α [ r + γ·Q(s',a') − Q(s,a) ]     # a' actually taken
```

**Why red — and this is the elegant bit:** the attacker gets *punished for
exploring* (−50 when detected). This is precisely the **Cliff Walking** situation
from Sutton & Barto: Q-Learning learns the optimal path right along the cliff edge
and keeps falling off while exploring; SARSA learns a safer path that accounts for
its own exploration risk. An attacker that gets caught while probing should learn
the **stealthy** path, not the theoretically-optimal-but-noisy one. SARSA gives us
that naturally.

> This single design choice — off-policy for the defender, on-policy for the
> attacker, justified by the cliff-walking analogy — is the strongest technical
> point in your presentation. Lead with it on Slide 5.

### Expected SARSA — the third comparison

```
Q(s,a) ← Q(s,a) + α [ r + γ·Σ_a' π(a'|s')·Q(s',a') − Q(s,a) ]
```

Lower variance than SARSA because it averages over the policy instead of sampling
one action. Useful here since our rewards are stochastic (exploits succeed
probabilistically). Expect smoother convergence curves — good plot for the report.

### Optional deep extension

Once tabular works, swap one blue agent for a small **DQN** and compare. This is your
"comparison of different configurations" mark and your PyTorch line for Dell. Do this
**last** — do not start here.

---

## 8. Training architecture: the sim-twin

**The core engineering insight of this project.**

```
  ┌─────────────────────┐        ┌──────────────────────────┐
  │  SIMULATED TWIN     │        │  REAL DOCKER LAB         │
  │  pure Python        │        │  13 live containers      │
  │  ~10,000 eps/min    │  ───►  │  ~1 episode / 30s        │
  │                     │ deploy │                          │
  │  TRAIN HERE         │ Q-table│  DEMO + ONLINE LEARNING  │
  └─────────────────────┘        └──────────────────────────┘
```

- **Phase A — train in the twin.** A fast, pure-Python model of MiniCorp with
  identical state/action/reward semantics. Thousands of episodes in minutes.
- **Phase B — deploy to the real lab.** Load the learned Q-tables. Now `isolate(h)`
  actually runs `docker network disconnect` on a live container. Keep ε = 0.05 so it
  continues learning online.
- **Phase C — measure the sim-to-real gap.** Does the twin-trained policy still work
  against the real thing? Any gap is a *finding*, not a failure — it's the most
  interesting thing in your results section.

This is genuine **sim-to-real transfer**, the same pattern used in robotics. It's
also why we can afford a 13-container lab: the big lab is for the demo, the small
fast twin is for the learning.

---

## 9. Evaluation

### Baselines to beat (you need these — results mean nothing without them)

1. **Random defender** — floor.
2. **Static firewall** — the rules-only configuration, no defenders acting. This is
   "what a normal network does" and is your headline comparison.
3. **Greedy heuristic** — "isolate any host that looks suspicious." Beats random,
   but should lose badly on false positives. This is the one whose *shape* of failure
   you want to show.

### Metrics

| Metric | What it shows |
|---|---|
| Attacker success rate (% episodes reaching `db-primary`) | Headline number |
| Mean time to detection (MTTD), in steps | Blue responsiveness |
| Mean time to containment (MTTC) | Blue effectiveness |
| False positive rate (clean hosts isolated) | Availability cost |
| Hosts compromised before containment | Blast radius |
| Honeypot engagement rate | Did deception emerge? |
| Cumulative reward per episode (both teams) | Convergence |
| Policy stability across alternating phases | Equilibrium behaviour |

### Emergent behaviours to hunt for (and write up)

- **Chokepoint defence** — blue concentrating on `ad-controller` without being told
  it's the pivot. It should discover that from the reward structure alone.
- **Stealth pathing** — red learning slower, quieter routes as blue improves.
- **Deception strategy** — blue pre-deploying honeypots instead of reacting.
- **Anticipatory defence** — with comms on, `B_corp` reacting to `B_dmz`'s alerts.
- **Arms race** — plot both teams' win rates across alternating phases. Oscillation
  is the interesting result, not a bug.
- **Sacrificial containment** — blue deliberately conceding a low-value host to
  protect `db-primary`. If this emerges, lead your results with it.

### Experiment grid

| Sweep | Values |
|---|---|
| Algorithm | Q-Learning / SARSA / Expected SARSA |
| Reward structure | Individual vs shared |
| Communication | Off vs on |
| γ | 0.8 / 0.95 / 0.99 |
| α | 0.01 / 0.1 / 0.3 |
| ε decay | fast / slow |
| Agents | 3 blue vs 2 blue vs 1 blue (ablation) |

---

## 10. The LLM copilot layer (the Dell hook)

**Strictly bolted on top. It does not touch the RL loop.** That separation is
deliberate — it protects the course rubric while giving you the GenAI story.

```
  logger container  ──►  log chunks  ──►  embeddings  ──►  ChromaDB
                                                              │
  blue agent takes action ──────────────────────────────┐     │
                                                        ▼     ▼
                                              ┌──────────────────────┐
                                              │  LLM (RAG prompt)    │
                                              │  "why did we isolate │
                                              │   ci-runner?"        │
                                              └──────────┬───────────┘
                                                         ▼
                                              plain-English incident note
```

What it produces: for each defensive action, a short analyst-style justification
grounded in the actual log lines retrieved from the `logger` container — plus an
end-of-episode incident report.

**Vocabulary this earns you, honestly:** LLM, RAG, vector store, embeddings, chunking,
prompt engineering, AI copilot, grounding/hallucination control, responsible AI.
Every one of those is in the Dell JD.

Stack: LangChain or LlamaIndex + ChromaDB + any LLM API (Groq and Google AI Studio
have free tiers).

---

## 11. Tech stack

| Layer | Choice |
|---|---|
| Environment / twin | Python 3.11, Gymnasium-style API (our own `step()`/`reset()`) |
| RL | NumPy only. Hand-written Q-Learning / SARSA / Expected SARSA |
| Lab | Docker + Docker Compose, `internal: true` networks |
| Services | Flask (tiny, purpose-built, deliberately weak endpoints we wrote) |
| Orchestration | Docker SDK for Python (`docker` package) for isolate/block actions |
| Logging | Python logging → `logger` container → flat files |
| API | FastAPI (exposes agent state to the dashboard) |
| Dashboard | Plain HTML + Chart.js, or Grafana if you want the polish |
| Copilot | LangChain / LlamaIndex + ChromaDB + LLM API |
| CI | GitHub Actions — lint + unit tests + a smoke-train run |
| Deep extension | PyTorch (DQN), last |

---

## 12. Build plan

| Phase | Deliverable | When |
|---|---|---|
| **0** | Repo, `CLAUDE.md`, this document | Day 1 |
| **1** | Simulated twin: hosts, firewall matrix, transitions, rewards | Days 2–4 |
| **2** | Single blue agent, Q-Learning, vs scripted attacker. **First learning curve.** | Days 5–7 |
| **3** | All 5 agents, IQL, alternating training | Week 2 |
| **4** | Docker lab: 13 containers, 3 zones, airgapped | Week 2–3 |
| **5** | Sim-to-real deploy: real `isolate` actions on live containers | Week 3 |
| **6** | Experiment grid + plots + baselines | Week 4 |
| **7** | LLM copilot + dashboard | Week 4–5 |
| **8** | DQN extension, demo video, report | Week 5–6 |

**Next week's presentation needs only Phases 0–2.** A working learning curve from a
single agent is more than enough for a proposal presentation, and it proves
feasibility, which is a whole mark.

---

## 13. Interview glossary (Dell prep)

### Reinforcement learning

- **MDP** — ⟨S, A, P, R, γ⟩. The formal frame for sequential decision-making.
- **Markov property** — the future depends only on the current state, not the
  history. *Our discretization is what makes this approximately true.*
- **Model-free** — the agent never learns `P`; it learns values from experience.
- **TD learning** — update from the difference between predicted and observed
  return, bootstrapping off your own estimate rather than waiting for episode end.
- **Bellman equation** — `Q*(s,a) = E[r + γ·max_a' Q*(s',a')]`. Every update above
  is a sampled step toward satisfying it.
- **On-policy vs off-policy** — SARSA learns the value of the policy it's actually
  running (including its exploration); Q-Learning learns the optimal policy's value
  regardless. *Cliff walking is the canonical illustration, and it's why we assign
  SARSA to the attacker.*
- **Exploration/exploitation** — ε-greedy with decay. Early: gather information.
  Late: cash in.
- **Discount factor γ** — how much the future is worth now. High γ → far-sighted.
- **Credit assignment** — which of the last 40 actions actually caused the win?
  Harder with multiple agents (*who* on the team caused it?).
- **Reward shaping** — adding intermediate rewards to guide learning. Risky: shape
  carelessly and you get the degenerate "isolate everything" policy.

### Multi-agent

- **Markov Game** — MDP generalised to N agents with joint actions and per-agent rewards.
- **General-sum vs zero-sum** — ours is general-sum; payoffs don't cancel.
- **Independent Q-Learning (IQL)** — each agent runs single-agent Q-Learning and
  treats others as part of the environment. Simple, scalable, **theoretically
  unsound** (see non-stationarity).
- **Non-stationarity** — other agents' policies change, so the environment changes,
  breaking Q-Learning's convergence guarantee. Our fix: alternating training.
- **CTDE** (centralised training, decentralised execution) — train with global
  information, execute with local only. What QMIX/MADDPG do. Worth naming as future work.
- **POMDP** — agents observe partial state. Ours is zone-scoped, so: partially observable.
- **Emergent behaviour** — coordination that was never programmed.
- **Nash equilibrium** — no agent can improve by unilaterally changing policy. What
  the red/blue arms race oscillates around.

### GenAI

- **RAG** — retrieve relevant documents, stuff them into the prompt, generate a
  grounded answer. Reduces hallucination by grounding in real data.
- **Embeddings / vector store** — text as vectors; retrieve by cosine similarity.
- **Chunking** — splitting documents so retrieval returns focused context.
- **Grounding** — forcing the model to answer only from retrieved evidence.
- **Agent (LLM sense)** — an LLM that plans and calls tools in a loop. *Note the
  collision: "agent" means something different in RL. Being able to articulate that
  distinction cleanly is itself a good interview moment.*

### Engineering

- **Container vs VM** — containers share the host kernel; lighter, faster to start.
- **Docker network segmentation** — isolated bridge networks; `internal: true` means
  no external route.
- **Sim-to-real transfer** — train in simulation, deploy on the real system.
- **CI/CD** — automated lint/test/build on every push.

---

## 14. References

1. **Tan, M. (1993).** *Multi-Agent Reinforcement Learning: Independent vs.
   Cooperative Agents.* ICML. — The origin of Independent Q-Learning. Cite for §7.
2. **Sutton, R. & Barto, A. (2018).** *Reinforcement Learning: An Introduction*,
   2nd ed. — Ch. 6 for TD/SARSA/Q-Learning; §6.5 for the cliff-walking comparison
   that justifies our algorithm split.
3. **Standen, M. et al. (2021).** *CybORG: A Gym for the Development of Autonomous
   Cyber Agents.* — The academic benchmark for exactly this problem. Your closest
   related work.
4. **Microsoft (2021).** *CyberBattleSim.* — Simulation-based RL for lateral
   movement. Good for motivating the abstraction choice.
5. **Littman, M. (1994).** *Markov Games as a Framework for Multi-Agent
   Reinforcement Learning.* ICML. — Cite for §6, the Markov Game formalism.

Two or three of these satisfy the "initial literature search" submission. Use 1, 3
and 5 as the core trio.

---

## 15. Things that will go wrong (plan for them)

- **State space creep.** Every "let's also track X" multiplies the Q-table. Budget
  |S| ≤ ~10k per agent and defend that budget.
- **Blue converging to do-nothing.** If penalties are too harsh, `noop` looks safest.
  Fix by tuning the compromised-host penalty upward relative to false-positive cost.
- **Red never succeeding.** If blue is too strong early, red gets no positive signal
  to learn from. Fix: train red first against a *static* defence, then introduce
  learning blue.
- **Oscillation that never settles.** Expected in adversarial learning. Don't hide
  it — plot it and discuss it as equilibrium behaviour.
- **Docker eating your disk.** `docker system prune` regularly.
- **Scope creep.** The LLM copilot and DQN are *extensions*. If time runs short,
  Phases 1–6 alone are a complete, high-scoring project.
