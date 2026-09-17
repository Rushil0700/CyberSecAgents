# Build log

Chronological record of what was built and why. Per-phase revision notes are in
`notes/phaseN-*.md`; commit messages carry the detailed reasoning.

---

## 2026-09-17 — Phase 0: repo and working agreement

- `git init`, `CLAUDE.md`, package skeleton, `pyproject.toml`.
- Reviewed `PROJECT.md` and recorded eight amendments in `CLAUDE.md` §3. The three that
  matter: true state must be hidden behind a noisy detection layer or MTTD is always 1
  step; the one-shot −20 false-positive penalty does not actually prevent the degenerate
  "isolate everything" policy; red's naive state space is ~201M, about 20,000× over
  budget.
- Decided agent control is three orthogonal switches (`enabled` / `learning` / `policy`)
  rather than one on/off flag, which makes the demo controls and the §9 baselines the
  same mechanism.

## 2026-09-17 — Phase 1a: the network map

- `env/topology.py` — 13 hosts, 3 zones, derived firewall matrix, exploit probabilities.
- `env/compose_gen.py` — generates `docker-compose.yml` from that topology.
- 23 tests, including one that walks every host pair asserting Docker connectivity
  equals the firewall policy, so the twin and the lab cannot drift.
- Detour from strict phase order: Rushil asked to start with Docker. Rather than build
  Phase 4 early, we wrote Phase 1's first file and generated the compose skeleton from
  it — the Docker artifact exists today without hand-writing anything that could
  disagree with the simulator later.

## 2026-09-17 — Phase 1b: the six-layer environment

`PROJECT.md` had been rewritten under the code: three zones became four, and the flat
exploit chain became six heterogeneous layers. `topology.py` was describing a network
that no longer existed, so the first job was re-mapping it — Edge zone, `auth-server` as
the crown jewel, `mfa-service`, and roles as data rather than booleans.

Then the rest of Phase 1: `layers.py`, `state.py`, `detection.py`, `observations.py`,
`actions.py`, `rewards.py`, `config.py`, `minicorp.py`. 185 tests. Full write-up and
revision material in `notes/phase1-environment.md`.

The decisions worth remembering:

- **Gates are not targets.** `edge-gateway` and `mfa-service` have `exploit_prob = 0`;
  left ownable, red would compromise them like any other host and the six heterogeneous
  layers would collapse into a six-host corridor.
- **The layers are a partial order.** L6 requires L4, not L5, so red must learn an
  ordering rather than be handed one.
- **Detection is calibrated so stealth works.** A compromised host sitting idle
  accumulates 1.00 alert points against a 1.5 threshold — it never even looks suspicious.
  Acting is what exposes it.
- **The availability cost is per-step by default**, because the one-shot version makes
  "isolate everything" genuinely optimal by a factor of forty.
- **Spec correction:** §5.2's `4*4*2*2*2*3 = 768` is actually 384. The 8x-for-six-layers
  argument built on it is unaffected.

Two bugs found by running episodes rather than by unit tests: red could take the pivot
with `exploit` and walk through Layer 4 unbreached, and progress derived from breached
layers made a shallow curriculum stage strictly harder than a deep one. Both made red
look *worse* with no error raised — the failure mode that reads as "needs more episodes".

Feasibility, now enforced by tests: red wins 100% against a static defence at every
curriculum stage, and 0% against even a random defender at mean depth 0.78 of 6. Red can
win; random exploration cannot find it. That is the sparse-reward argument demonstrated.

**Next:** Phase 2 — a single blue agent learning Q-Learning against a scripted attacker,
layers 1-2 only. The first learning curve, which is what the proposal presentation needs.
