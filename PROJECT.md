# MARL-SOC: Multi-Agent RL for Autonomous Network Defense, with an LLM Playbook Advisor

**A-Z project bible.** Read this top to bottom once. Everything for the course rubric
*and* the Dell interview is here.

---

## 0. The 30-second pitch

> We build a dummy company server network inside Docker, fully isolated from the
> internet, protected by **six layers of defence-in-depth** — perimeter, network
> segmentation, identity, internal segmentation, privilege, and application control.
> Two attacker (red) agents continuously try to breach it; three defender (blue)
> agents — one per network zone — learn to detect and block them. Both sides learn
> from their mistakes through reinforcement learning; neither is programmed with
> tactics. A local LLM, using RAG over a playbook **we write for our own toy system**,
> advises the agents on which moves to try — but the RL agents decide, and *learn
> whether to trust that advice*. The attacker's end goal is to punch through all six
> layers and **alter the admin login credentials**. The defenders learn to block them
> out.

**Course framing:** a general-sum Markov Game solved with Independent Q-Learning,
with curriculum learning and an LLM-guided-exploration comparison as the standout
experiments.

**CV / interview framing:** *"Built a multi-agent RL network-defense system against a
six-layer defence-in-depth architecture, where a local LLM (RAG over a security
playbook) guides agent exploration. Deployed on an isolated Docker network with
CI/CD. Curriculum learning solved the sparse-reward problem; LLM-guided exploration
cut episodes-to-convergence by X%; learned defenders reduced attacker success from
94% to Y% versus a static layered firewall."*

---

## 1. Safety and scope (read first, say it in your demo)

This project **never touches a network or system we do not own**, enforced by design:

- The whole lab runs on Docker bridge networks declared `internal: true` → the
  containers have **no route to the outside world**. Nowhere for a packet to go.
- **No real exploit code, no attack tooling** (no nmap/hydra/metasploit). Every
  "vulnerability" is an endpoint *we wrote* in our own toy Flask services (e.g. a
  login route that accepts a credential we planted). An "exploit" action is an HTTP
  request to our own container that we designed to succeed.
- The attacker agents are **state-machine navigators** over our own abstract action
  set, not an intrusion framework. The contribution is the *decision-making*.
- **The LLM's knowledge base is our own playbook** describing *this toy system's*
  abstract mechanics and action set — not a corpus of real-world hacking methods.

Say this in your viva:

> "The environment is a fully isolated Docker network with no external connectivity.
> All services, their vulnerabilities, and the LLM's knowledge base are purpose-built
> by us for this study. No real systems are contacted and no real exploit techniques
> are used at any point."

This mirrors the academic benchmarks (CybORG / CAGE Challenge, Microsoft
CyberBattleSim) — all simulators, for exactly this reason.

---

## 2. Why this project (mapping to the marks)

| Rubric component | Marks | How we hit it |
|---|---|---|
| Problem definition & motivation | 1.5 | Autonomous SOC response; real industry problem; layered threat model |
| AI agent & environment design | 2.0 | 5 agents, 6-layer segmented enterprise, hand-built environment |
| RL/MDP formulation | 2.0 | Full ⟨S, A, P, R, γ, π⟩ in §5, discretization + action masking justified |
| MAS / Markov Game formulation | 2.0 | General-sum Markov Game, §6; coop within team, compete across |
| Proposed methodology & algorithm | 1.5 | IQL + Q-Learning/SARSA/Expected SARSA + curriculum + LLM-guided exploration, §7 |
| Feasibility & expected outcomes | 1.0 | Phased plan §12, metrics §9, sim-twin keeps training cheap |

The guidelines' "Avoid" list, defused:

- ✅ Not classification/regression — sequential decision-making.
- ✅ **Not an LLM wrapper** — the LLM only *advises exploration*; RL learns the values
  and can learn to ignore bad advice. We show curves with the LLM turned off.
- ✅ Not single-agent-in-disguise — 5 agents with genuine interdependence.
- ✅ Not hard-coded — tactics emerge; the hard-coded thing is the *baseline we beat*.
- ✅ Not blind library-calling — Q-Learning, SARSA, Expected SARSA written from scratch.

---

## 3. The environment: "MiniCorp"

A dummy company network, ~14 containers, four zones, **six layers of
defence-in-depth**.

### 3.1 The six layers

Real defence-in-depth is not one control repeated — it is *different kinds* of
control stacked, so that no single attacker capability gets you through. Each layer
below requires a **different class of action** to beat.

| # | Layer | Type | Enforced by | What beats it |
|---|---|---|---|---|
| **1** | Perimeter / WAF | Rate limiting, IP reputation | `edge-gateway` | `slow_scan` — recon under the rate threshold |
| **2** | DMZ network boundary | Network segmentation | firewall matrix | Compromise a DMZ host with inward reach |
| **3** | Authentication gate | Identity | internal services | `steal_credentials` from a compromised host's cache |
| **4** | Corp internal segmentation | Network micro-segmentation | firewall matrix | `lateral_move` specifically to `ad-controller` |
| **5** | Privilege escalation gate | Privilege | `auth-server` ACL | `escalate_privilege` — noisy, high detection risk |
| **6** | Approval / MFA on credential change | Application control | `mfa-service` | Degrade `mfa-service`, *then* `alter_credentials` |

> **Why this beats "just add more zones":** more zones only makes the path longer.
> Heterogeneous layers make the attacker learn a *sequence of different move types in
> the right order* — a far richer decision problem, and a much better story in a viva.

### 3.2 Topology

```
                    ┌───────────────────────────────────┐
  INTERNET          │  (no route — internal: true)      │
  (unreachable)     └───────────────────────────────────┘

  ╔═══════════════ ZONE 0: EDGE ═══════════════════════╗
  ║  edge-gateway  (WAF: rate limit + IP reputation)   ║
  ╚════════════════════════╤═══════════════════════════╝
              ◄── LAYER 1: perimeter / WAF
  ╔═══════════════ ZONE 1: DMZ ════════════════════════╗
  ║  reverse-proxy   web-portal ★ENTRY   mail          ║
  ╚════════════════════════╤═══════════════════════════╝
              ◄── LAYER 2: DMZ network boundary
              ◄── LAYER 3: authentication gate
  ╔═══════════════ ZONE 2: CORPORATE LAN ══════════════╗
  ║  intranet   fileserver   dev-box   ci-runner       ║
  ║             ad-controller ★PIVOT                   ║
  ╚════════════════════════╤═══════════════════════════╝
              ◄── LAYER 4: internal micro-segmentation
              ◄── LAYER 5: privilege escalation gate
  ╔═══════════════ ZONE 3: SECURE ═════════════════════╗
  ║  auth-server ★TARGET (holds admin credentials)     ║
  ║  mfa-service ◄── LAYER 6: approval / MFA           ║
  ║  backup    honeypot-1/2                            ║
  ╚════════════════════════════════════════════════════╝

  ┌──────────┐
  │  logger  │  ← receives logs from all zones
  └──────────┘     (feeds blue monitoring + the LLM's RAG context)
```

**Layers ≠ agents.** Six layers, three blue defenders. One defender holds several
layers within its zone. Layers are environment *structure*; agents are *defenders*.
That decoupling is why we can go deep without adding agents or blowing up training.

### 3.3 Hosts

| Host | Zone | Role | Planted weakness |
|---|---|---|---|
| `edge-gateway` | Edge | WAF / rate limiter | rate threshold evadable by slow scanning |
| `reverse-proxy` | DMZ | nginx front door | none (routing only) |
| `web-portal` | DMZ | **Attacker entry point** | weak admin credential |
| `mail` | DMZ | fake mail service | exposed config endpoint |
| `intranet` | Corp | internal web app | trusts DMZ session token |
| `fileserver` | Corp | file shares | world-readable share |
| `dev-box` | Corp | developer machine | reused credential from web-portal |
| `ci-runner` | Corp | build server | over-privileged service account |
| `ad-controller` | Corp | **Pivot to secure zone** | credential cache |
| `auth-server` | Secure | **CROWN JEWEL — stores admin login** | reachable only via ad-controller |
| `mfa-service` | Secure | approval check on credential change | degradable under load |
| `backup` | Secure | secondary target | weak backup credential |
| `honeypot-1/2` | Corp/Secure | decoys (blue deploys) | looks juicy, alerts on touch |
| `logger` | — | log aggregation | n/a |

### 3.4 Transitions, constraints, termination

- **Transitions:** an action on host *h* succeeds with probability `p_h` (we set
  these) **and** only if every layer guarding *h* has already been satisfied —
  reachability from a held foothold, valid credentials if past Layer 3, escalated
  privilege if past Layer 5.
- **Action masking:** actions whose preconditions are unmet are masked out of the
  agent's choice set. This keeps the effective branching factor small and stops the
  agent wasting exploration on impossible moves.
- **Constraints:** attacker acts only from a compromised foothold; each defender acts
  only within its own zone; every isolate action costs availability.
- **The winning move:** holding `auth-server` with escalated privilege *and*
  `mfa-service` degraded, the action `alter_credentials` changes the admin login.
- **Termination:** (a) admin credentials altered → **red win**; (b) all attacker
  footholds isolated → **blue win**; (c) step limit **T = 250** → draw.

> T rose from 100 to 250 because six layers means a much longer minimum path. Watch
> your draw rate: if most episodes hit the limit, either T is too low or red is stuck.

---

## 4. The agents

Still **2 red + 3 blue**. Layers scale; agent count does not.

### 4.1 Red team (2 agents, cooperative with each other)

| Agent | Goal | Observes | Actions |
|---|---|---|---|
| `R_scout` | Find reachable hosts cheaply and quietly; beat Layer 1 | Own zone reachability, discovered-host count, heat level | `scan(zone)`, `slow_scan(zone)`, `probe(host)`, `wait` |
| `R_breach` | Breach layers 2–6, reach & alter admin creds | Discovered hosts, footholds, creds/privilege/MFA flags, heat | `exploit(h)`, `lateral_move(h)`, `steal_credentials(h)`, `escalate_privilege`, `degrade_mfa`, `alter_credentials`, `wait` |

Note how the action set maps 1:1 onto the layers: `slow_scan` → L1, `exploit`/
`lateral_move` → L2/L4, `steal_credentials` → L3, `escalate_privilege` → L5,
`degrade_mfa` → L6. **That mapping is the cleanest thing in your Slide 3.**

Shared team reward — `R_scout` earns credit for discoveries `R_breach` later
converts. That's your **credit-assignment** discussion.

### 4.2 Blue team (3 agents, one per defended zone, cooperative)

| Agent | Defends | Holds layers | Actions |
|---|---|---|---|
| `B_dmz` | Edge + DMZ (4 hosts) | 1, 2 | `block(h)`, `isolate(h)`, `tighten_ratelimit`, `deploy_honeypot`, `noop` |
| `B_corp` | Corp LAN (5 hosts) | 3, 4 | `block(h)`, `isolate(h)`, `rotate_credentials`, `deploy_honeypot`, `noop` |
| `B_secure` | Secure (5 hosts) | 5, 6 | `block(h)`, `isolate(h)`, `harden_mfa`, `deploy_honeypot`, `noop` |

Each blue agent's extra action is the one that reinforces *its own* layers — so
defenders are differentiated by the layers they hold, not just by geography.

**Why zone-split, not role-split?** Good interview answer:

1. **Tractability.** One agent seeing all 14 hosts at 4 statuses = 4¹⁴ ≈ 268M states.
   Scoped to a zone: a few thousand. Tabular RL becomes possible.
2. **Genuine partial observability.** Each defender sees only its zone → proper POMDP
   flavour.
3. **Real interdependence.** If `B_dmz` fails, `B_corp` inherits an attacker. Coupled
   fates = the "genuine interaction" the rubric demands.

---

## 5. RL formulation (the MDP)

### 5.1 Blue example — `B_corp`
```
s = ( status[intranet], status[fileserver], status[dev-box],
      status[ci-runner], status[ad-controller],   # 4 values each
      zone_alert_level )                          # 3: low / med / high
```
`status ∈ {clean, suspicious, compromised, isolated}` → |S| = 4⁵ × 3 = **3,072**.
Actions: `block(h)`/`isolate(h)` over 5 hosts + `rotate_credentials` +
`deploy_honeypot` + `noop` = **13**. Q-table ≈ 40k entries. Trains in minutes.

### 5.2 Red example — `R_breach` (note what the layers do to the state)
```
s = ( current_zone,        # 4
      footholds_bucket,    # 4  (0, 1, 2-3, 4+)
      creds_held,          # 2  ← Layer 3 flag
      privilege_escalated, # 2  ← Layer 5 flag
      mfa_degraded,        # 2  ← Layer 6 flag
      heat_level )         # 3
```
|S| = 4 × 4 × 2 × 2 × 2 × 3 = **768 states**.

> **The layers don't explode the state space — they structure it.** Each layer adds
> one binary flag, not a new dimension of hosts. Going from two layers to six cost us
> 8× on red's state space and *nothing* on blue's. Report this; it's a genuinely nice
> design result and a strong answer to "how did you keep tabular RL feasible?"

### 5.3 Reward R (blue)
| Event | Reward |
|---|---|
| Admin credentials altered (attacker wins) | **−100** (team-wide) |
| Each host currently compromised | **−10** / step |
| Correctly isolating a compromised host | **+50** |
| Isolating a *clean* host (false positive) | **−20** |
| Attacker engages a honeypot | **+30** |
| Restoring a breached layer (rotate creds / harden MFA / tighten rate limit) | **+25** |
| Per timestep | **−1** |

> **The degenerate-policy trap.** Drop the −20 false-positive penalty and blue learns
> the trivial "isolate everything" policy: perfect security, zero availability. That
> penalty encodes the real security/availability tradeoff. **Your single best analysis
> paragraph and a great interview story.**

### 5.4 Reward R (red) — progressive, one step per layer
| Event | Reward |
|---|---|
| Breach Layer 1 (past the WAF) | **+10** |
| Breach Layer 2 (foothold in DMZ) | **+20** |
| Breach Layer 3 (valid credentials) | **+30** |
| Breach Layer 4 (reach `ad-controller`) | **+40** |
| Breach Layer 5 (privilege escalated) | **+50** |
| Breach Layer 6 (MFA degraded) | **+60** |
| **Alter admin credentials** | **+100** |
| Detected / blocked | **−50** |
| Wasted action on a honeypot | **−30** |
| Per timestep | **−1** |

> **Why escalating rewards?** With one lonely +100 at the end of a six-layer chain,
> random exploration would essentially never reach it, red would get no learning
> signal, and the curve would flatline. This is the **sparse-reward / long-horizon
> problem**. The escalating ladder creates a reward gradient pulling red forward. See
> also curriculum learning, §7.4 — the other half of the fix.

### 5.5 Transition P, discount γ, policy π
- **P:** stochastic via `p_h`, detection probability given heat/alert level, and the
  layer precondition checks. We define these; agents never see them → model-free.
- **γ = 0.95.** With a six-layer path, payoffs are *far* downstream — an early
  `steal_credentials` only pays off many steps later. High γ is now essential, not
  just nice. Sweep 0.9 / 0.95 / 0.99 and expect 0.99 to do better than it would have
  with two layers. **Good result to report.**
- **π:** ε-greedy, ε decaying 1.0 → 0.05, over the *masked* action set. Eval:
  `π(s) = argmax_a Q(s,a)`.

---

## 6. Multi-agent formulation (the Markov Game)

**General-sum Markov Game** ⟨N, S, {Aᵢ}, {Rᵢ}, P, γ⟩:

- **N = 5**: `{R_scout, R_breach, B_dmz, B_corp, B_secure}`.
- **S** — true global state (host statuses, footholds, all six layer flags). **No
  agent observes it.** Each sees local `oᵢ = Oᵢ(s)`.
- **Aᵢ** — per-agent action sets (§4); joint action `a = (a₁..a₅)`.
- **Rᵢ** — shared within a team, opposed across teams.
- **General-sum, not zero-sum:** honeypot = +30 blue / −30 red (cancels), but a false
  positive = −20 blue / 0 red (doesn't). Payoffs don't sum to zero → general-sum.

### Cooperation & competition
- **Within team — cooperation via shared reward.** Nobody is told to cooperate;
  coordination is simply how you maximise a shared return.
- **Across teams — competition.** Red's gain ≈ blue's loss.

### Individual vs shared reward — headline experiment
Run both. Prediction: **individual** reward makes `B_dmz` selfishly trigger-happy
(isolating is cheap for *it*; downstream cost lands on `B_corp`). **Shared** reward
teaches restraint and hand-off. That behavioural difference *is* your
emergent-behaviour finding.

### Communication
- **No comms:** each agent sees only its zone.
- **With comms:** +1 bit "neighbour zone under attack" (state ×2, still tiny). Expect
  **anticipatory defence** — `B_corp` raising its guard *before* the attacker arrives.

### Non-stationarity (say this in the interview)
> With all agents learning at once, each agent's environment includes the others,
> whose policies keep changing — so it's **non-stationary**, violating the Markov
> property Q-Learning's convergence proof needs. Independent Q-Learning has **no
> convergence guarantee** here. We use **alternating training**: freeze red, train blue
> to convergence; freeze blue, train red; repeat. Each phase is then a stationary MDP.

Most students don't know IQL is theoretically unsound in MAS. Knowing it stands out.

---

## 7. Algorithms

Implement the learning algorithms **from scratch** — NumPy only, no
stable-baselines/RLlib.

### 7.1 Q-Learning (off-policy) → BLUE
```
Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
```
Blue can explore freely (a wrong action in training costs nothing real), so learning
the optimal policy regardless of the exploratory behaviour is exactly right.

### 7.2 SARSA (on-policy) → RED
```
Q(s,a) ← Q(s,a) + α [ r + γ·Q(s',a') − Q(s,a) ]   # a' actually taken
```
**The elegant bit:** the attacker is *punished for exploring* (−50 when detected) —
the **Cliff Walking** situation from Sutton & Barto. Q-Learning learns the optimal
path right on the cliff edge and keeps falling off while exploring; SARSA learns a
safer path that accounts for its own exploration risk. An attacker that gets caught
probing should learn the **stealthy** route. SARSA gives that for free — and with six
layers there is far more cliff to fall off, so the effect should be *larger* here
than in the textbook example.

> Off-policy for the defender, on-policy for the attacker, justified by cliff-walking:
> the strongest technical point in the deck. Lead Slide 5 with it.

### 7.3 Expected SARSA (third comparison)
```
Q(s,a) ← Q(s,a) + α [ r + γ·Σ_a' π(a'|s')·Q(s',a') − Q(s,a) ]
```
Lower variance (averages over the policy instead of sampling one action) — helpful
since our layer breaches succeed probabilistically. Smoother curves for the report.

### 7.4 Curriculum learning (required, because of the six layers)

**The problem:** with all six layers active from episode 1, red must chain many
correct moves of different types before anything good happens. Random exploration
almost never completes that chain, so red gets no signal and never learns.

**The fix:** train red against a progressively deeper stack.

| Stage | Active layers | Red must learn |
|---|---|---|
| 1 | 1–2 | Get a foothold in the DMZ |
| 2 | 1–3 | + steal credentials |
| 3 | 1–4 | + pivot to `ad-controller` |
| 4 | 1–5 | + escalate privilege |
| 5 | 1–6 | + degrade MFA, then alter credentials |

Promote to the next stage when red's success rate at the current stage exceeds a
threshold (say 70% over the last 200 episodes). Carry the Q-table forward between
stages — that transfer is the whole point.

**Plot the stage transitions on your learning curve.** You'll see a characteristic
sawtooth: success climbs, you add a layer, it drops, it climbs again. That plot is
the single most impressive figure in your final report, and "curriculum learning" is
a term that lands well in an interview.

### 7.5 Optional deep extension (last)
Swap one blue agent for a small **DQN** and compare. Your "comparison of
configurations" mark and your PyTorch line for Dell. Do this **last**.

---

## 8. Training architecture: the sim-twin

```
  ┌─────────────────────┐        ┌──────────────────────────┐
  │  SIMULATED TWIN     │        │  REAL DOCKER LAB         │
  │  pure Python        │  ───►  │  14 live containers      │
  │  ~10,000 eps/min    │ deploy │  ~1 episode / 60s        │
  │  TRAIN HERE         │ Q-table│  DEMO + ONLINE LEARNING  │
  └─────────────────────┘        └──────────────────────────┘
```
- **Phase A — train in the twin:** fast pure-Python model of MiniCorp with identical
  layer semantics. Thousands of episodes in minutes. Curriculum runs here.
- **Phase B — deploy to the real lab:** load the Q-tables; `isolate(h)` actually runs
  `docker network disconnect`, `tighten_ratelimit` actually reconfigures
  `edge-gateway`, and `alter_credentials` actually rewrites the admin record in
  `auth-server`. Keep ε = 0.05 for online learning.
- **Phase C — measure the sim-to-real gap.** Any gap is a *finding*, not a failure.

Genuine **sim-to-real transfer** (the robotics pattern). It's also why we can afford
14 containers and six layers: the big lab is for the demo, the small fast twin is for
the learning.

---

## 9. Evaluation

### Baselines to beat
1. **Random defender** — floor.
2. **Static six-layer firewall** — rules only, no defenders acting. "What a normal
   hardened network does." Headline comparison.
3. **Greedy heuristic** — "isolate anything suspicious." Beats random, should lose
   badly on false positives.

### Metrics
| Metric | Shows |
|---|---|
| Attacker success rate (% episodes altering admin creds) | Headline number |
| **Layers breached per episode (0–6)** | **Depth of intrusion — your best single graph** |
| Mean time to detection (steps) | Blue responsiveness |
| Mean time to containment | Blue effectiveness |
| False-positive rate (clean hosts isolated) | Availability cost |
| Honeypot engagement rate | Did deception emerge? |
| Draw rate (episodes hitting T=250) | Is the horizon right? |
| Cumulative reward per episode (both teams) | Convergence |
| Episodes-to-convergence, with vs without curriculum | Does curriculum help? |
| Episodes-to-convergence, LLM-guided vs ε-greedy | Does the LLM help? |

> **"Layers breached per episode" is your money graph.** It degrades gracefully
> (unlike binary win/lose), it shows blue pushing red back layer by layer over
> training, and it's instantly readable by a non-expert — including an interviewer.

### Emergent behaviours to hunt for
- **Chokepoint defence** — blue concentrating on `ad-controller` (Layer 4) without
  being told it's the pivot.
- **Layer prioritisation** — blue learning which layers are worth reinforcing first.
- **Stealth pathing** — red preferring `slow_scan` over `scan` as blue improves.
- **Deception strategy** — blue pre-deploying honeypots.
- **Anticipatory defence** — with comms, `B_corp` reacting to `B_dmz`'s alerts.
- **Arms race** — both teams' win rates across alternating phases; oscillation is the
  result, not a bug.
- **Sacrificial containment** — blue conceding a low-value host to protect
  `auth-server`. If it emerges, lead your results with it.

### Experiment grid
| Sweep | Values |
|---|---|
| Algorithm | Q-Learning / SARSA / Expected SARSA |
| Reward | Individual vs shared |
| Communication | Off vs on |
| **Curriculum** | **On vs off** |
| **Exploration** | **ε-greedy vs LLM-guided** |
| **Layers active** | **2 / 4 / 6 (difficulty ablation)** |
| γ | 0.9 / 0.95 / 0.99 |
| α | 0.01 / 0.1 / 0.3 |

---

## 10. The LLM playbook advisor (the Dell hook — RL stays in charge)

**The LLM advises which move to try; the RL agent decides and learns whether the
advice was any good.** It never overrides the policy. This is what keeps the project
an RL project, not an LLM wrapper.

### 10.1 The knowledge base (RAG corpus)
A **playbook we author** for our own toy system — two short markdown docs:
- *Attacker playbook:* for each abstract situation ("I hold a DMZ foothold, I have no
  credentials, Layer 3 is blocking me"), which of *our* abstract actions tend to pay
  off. Written purely in terms of MiniCorp's own mechanics and our six layers — no
  real CVEs, exploits, or commands.
- *Defender playbook:* which containment or layer-hardening move fits which alert
  pattern.

Chunk these, embed into a local vector store, retrieve by similarity to the agent's
current observation. **The six layers make this RAG genuinely useful** — there are
now six distinct situations to give distinct advice about, so retrieval has real work
to do rather than always returning the same chunk.

### 10.2 How it guides (guided exploration)
```
  agent needs to explore (prob ε)
        │
        ├─ plain mode:  pick a random action from the masked set   ← baseline
        │
        └─ LLM-guided:  observation ─► retrieve playbook chunks ─►
                        LLM proposes a candidate action ─►
                        (validate it against the action mask) ─►
                        agent takes it, gets the REAL reward,
                        updates Q(s,a) as normal
```
The LLM only changes *what we try while exploring*. Value learning is untouched, and
bad advice gets down-weighted by the Q-values. **Always validate the LLM's suggestion
against the action mask** — if it proposes something illegal, fall back to random.

The experiment: **does LLM-guided exploration reach good policies in fewer episodes
than random exploration?** With a six-layer chain this matters far more than it would
have with two — good guidance should shine exactly where random search struggles.
Plot both curves.

### 10.3 The explanation copilot (bonus)
After each defensive action the LLM writes a plain-English incident note grounded
(RAG) in the real log lines from `logger`, plus an end-of-episode report naming which
layers were breached and which held. Pure add-on; touches nothing in the loop.

### 10.4 Vocabulary this earns you (honestly)
LLM, local model, RAG, vector store, embeddings, chunking, retrieval, grounding /
hallucination control, prompt engineering, AI copilot, responsible AI — all in the
Dell JD.

Stack: local LLM via **Ollama** (e.g. Llama 3.x 8B) or a free API tier (Groq / Google
AI Studio) + **ChromaDB** + **LangChain/LlamaIndex**.

---

## 11. Tech stack

| Layer | Choice |
|---|---|
| Environment / twin | Python 3.11, Gymnasium-style API (our own `step()`/`reset()`) |
| RL | NumPy only. Hand-written Q-Learning / SARSA / Expected SARSA |
| Lab | Docker + Docker Compose, `internal: true` networks |
| Services | Flask (tiny, purpose-built, deliberately weak endpoints we wrote) |
| Layer enforcement | nginx rate-limit config (L1), Docker networks (L2/L4), our own token/privilege checks (L3/L5), `mfa-service` (L6) |
| Orchestration | Docker SDK for Python for isolate/block/harden + credential-change actions |
| Logging | Python logging → `logger` container → flat files |
| API | FastAPI (exposes agent + layer state to the dashboard) |
| Dashboard | HTML + Chart.js, or Grafana for polish |
| LLM advisor | Ollama (local) or free API + ChromaDB + LangChain/LlamaIndex |
| CI | GitHub Actions — lint + unit tests + smoke-train |
| Deep extension | PyTorch (DQN), last |

---

## 12. Build plan

| Phase | Deliverable | When |
|---|---|---|
| **0** | Repo, `CLAUDE.md`, this document | Day 1 |
| **1** | Simulated twin: hosts, **six-layer precondition model**, transitions, rewards, action masking | Days 2–5 |
| **2** | Single blue agent, Q-Learning, vs scripted attacker, **layers 1–2 only**. First learning curve. | Days 6–8 |
| **3** | Curriculum learning: red vs static defence, stages 1→5 | Week 2 |
| **4** | All 5 agents, IQL, alternating training, all six layers | Week 2–3 |
| **5** | LLM playbook advisor + guided-exploration experiment | Week 3 |
| **6** | Docker lab: 14 containers, 4 zones, six real layers, airgapped | Week 4 |
| **7** | Sim-to-real deploy: real isolate / harden / admin-credential change | Week 4–5 |
| **8** | Experiment grid + plots + baselines | Week 5 |
| **9** | Explanation copilot + dashboard; DQN extension; demo video; report | Week 5–6 |

**Next week's presentation needs only Phases 0–2**, and Phase 2 runs with just layers
1–2 active. A working learning curve on a two-layer version proves feasibility — a
full mark — and you present the six-layer architecture as the design. Nobody expects
it built yet.

---

## 13. Interview glossary (Dell prep)

### Reinforcement learning
- **MDP** — ⟨S,A,P,R,γ⟩; the frame for sequential decisions.
- **Markov property** — the future depends only on the current state; our
  discretization (and the layer flags) is what makes this approximately hold.
- **Model-free** — never learns `P`; learns values from experience.
- **TD learning** — update from predicted-vs-observed return, bootstrapping off your
  own estimate.
- **Bellman equation** — `Q*(s,a)=E[r+γ·max_a' Q*(s',a')]`; every update steps toward it.
- **On- vs off-policy** — SARSA values the policy it actually runs (exploration
  included); Q-Learning values the optimal one regardless. Cliff walking is why SARSA
  goes to the attacker.
- **Sparse reward / long horizon** — when the payoff is many correct steps away,
  random exploration never finds it. *Our six layers create exactly this; curriculum
  learning and progressive rewards are the fix.*
- **Curriculum learning** — train on an easy version, then progressively harder,
  carrying learned values forward.
- **Action masking** — removing invalid actions from the choice set; shrinks the
  effective branching factor.
- **γ** — how much the future is worth now; high γ → far-sighted. Matters more the
  deeper the layer stack.
- **Credit assignment** — which past action caused the win? Harder with a team and a
  long chain.
- **Reward shaping** — intermediate rewards to guide learning; careless shaping → the
  degenerate "isolate everything" policy.

### Multi-agent
- **Markov Game** — MDP for N agents, joint actions, per-agent rewards.
- **General-sum vs zero-sum** — ours is general-sum; payoffs don't cancel.
- **Independent Q-Learning** — each agent runs single-agent Q-Learning, others treated
  as environment. Simple, scalable, **theoretically unsound** (non-stationarity).
- **Non-stationarity** — others' policies change → environment changes → convergence
  guarantee breaks. Fix: alternating training.
- **CTDE** — centralised training, decentralised execution (QMIX/MADDPG). Future work.
- **POMDP** — partial observation; ours is zone-scoped.
- **Emergent behaviour** — coordination never programmed.
- **Nash equilibrium** — no agent gains by changing alone; what the arms race
  oscillates around.

### Security
- **Defence in depth** — layered, heterogeneous controls so no single capability
  grants access. *Our six layers.*
- **Lateral movement** — moving between hosts after initial access.
- **Privilege escalation** — gaining higher rights on a held host.
- **Micro-segmentation** — fine-grained internal network boundaries.
- **Blast radius** — how much is exposed once one host falls.

### GenAI
- **RAG** — retrieve relevant docs, put them in the prompt, generate a grounded
  answer; reduces hallucination.
- **Embeddings / vector store** — text as vectors; retrieve by cosine similarity.
- **Chunking** — splitting docs so retrieval returns focused context.
- **Grounding** — answering only from retrieved evidence.
- **Guided exploration** — using an external advisor (here the LLM) to bias which
  actions an RL agent tries. *Our novel bit.*
- **Agent (LLM sense)** — an LLM that plans and calls tools in a loop. **Note the word
  collision:** "agent" means something different in RL. Articulating that distinction
  cleanly is itself a good interview moment.

### Engineering
- **Container vs VM** — containers share the host kernel; lighter, faster.
- **Docker network segmentation / `internal: true`** — isolated bridge, no external
  route.
- **Sim-to-real transfer** — train in sim, deploy on the real system.
- **CI/CD** — automated lint/test/build on every push.

---

## 14. References
1. **Tan, M. (1993).** *Multi-Agent RL: Independent vs. Cooperative Agents.* ICML. —
   origin of Independent Q-Learning. §7.
2. **Sutton & Barto (2018).** *Reinforcement Learning: An Introduction*, 2nd ed. —
   Ch. 6 for TD/SARSA/Q-Learning; §6.5 cliff walking (the algorithm split).
3. **Standen, M. et al. (2021).** *CybORG: A Gym for Autonomous Cyber Agents.* — the
   benchmark for this problem; closest related work.
4. **Microsoft (2021).** *CyberBattleSim.* — simulation-based RL for lateral movement;
   motivates the abstraction.
5. **Littman, M. (1994).** *Markov Games as a Framework for Multi-Agent RL.* ICML. —
   the Markov Game formalism. §6.
6. **Bengio, Y. et al. (2009).** *Curriculum Learning.* ICML. — cite for §7.4.

Use 1, 3, 5 as the core trio for the "initial literature search" submission; add 6
once curriculum is in.

---

## 15. Things that will go wrong (plan for them)
- **Red never wins, curve flatlines.** The #1 risk now that there are six layers. This
  is sparse reward. Fixes, in order: turn curriculum on, check the progressive layer
  rewards are firing, verify action masking isn't masking a needed action, raise T.
- **Draw rate near 100%.** Episodes hitting T=250 without resolution → either raise T
  or the layer preconditions are too strict. Check the layer-breach histogram to see
  where red stalls.
- **State-space creep.** Every "let's also track X" multiplies the Q-table. Budget
  |S| ≤ ~10k per agent and defend it. Note layer *flags* are cheap; per-host detail is
  not.
- **Blue converging to do-nothing.** Too-harsh penalties make `noop` safest — raise the
  compromised-host penalty relative to false-positive cost.
- **Curriculum stage never promotes.** Threshold too high or the stage is genuinely
  too hard — inspect that stage in isolation before tuning blindly.
- **LLM advice that's useless.** If guided ≈ random, that's still a valid result;
  report it. Check the playbook actually covers all six layer situations.
- **Oscillation that never settles.** Expected in adversarial learning — plot it,
  discuss it as equilibrium behaviour.
- **Docker eating disk.** `docker system prune` often.
- **Scope creep.** LLM advisor, copilot, and DQN are *extensions*. Phases 1–4 + 8
  alone are a complete, high-scoring project.
