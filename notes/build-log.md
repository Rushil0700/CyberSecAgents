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

**Next:** `env/state.py` (true episode state) and `env/detection.py` (the alert noise
model) — the amendment that turns this from a response problem into a detect-and-respond
problem.
