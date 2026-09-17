# Phase 1a — The network map (`topology.py`)

**What exists:** `src/marlsoc/env/topology.py`, `src/marlsoc/env/compose_gen.py`,
`docker-compose.yml` (generated), 23 passing tests.

---

## What this is

The map of MiniCorp: 13 hosts, 3 zones, and the firewall rules saying who can reach
whom. Pure data — no randomness, no simulation, no agents. It is the single source of
truth for both the simulated twin and the Docker lab.

## The three ideas you should be able to explain

### 1. The firewall matrix is *derived*, not typed out

A 13×13 matrix is 169 cells. One typo opens a path from the DMZ to the database, and
that failure is **invisible** — the attacker just looks unusually good, and you'd
conclude your agent learned well. So instead we declare the policy:

- inside a zone, every host reaches every other host
- across zones, only these three edges:
  `reverse-proxy → intranet`, `ad-controller → db-primary`, `ad-controller → backup`
- everyone ships logs one-way to `logger`

and compute the matrix from it. Three reviewable lines instead of 169 cells.

### 2. Reachability and credentials are different things

The spec says `dev-box` has "a reused credential from web-portal." That is *not* a
network path — it's a reason an attack on `dev-box` **succeeds more often** once you can
already reach it. So credential weakness lives in `exploit_prob` (0.80 for `dev-box`)
and the firewall stays purely about network paths. Mixing them would make the firewall
a lie.

### 3. Docker needs bridge networks, not multi-homing

Docker's unit of connectivity is the network, and there is no per-host rule inside one.
The obvious approach — put `reverse-proxy` on `net-corp` so it can reach `intranet` —
would also hand it `fileserver`, `dev-box`, `ci-runner` and `ad-controller`. That
flattens the corporate zone and destroys the chokepoint the whole project is about.

Fix: a dedicated two-host network per cross-zone hole.

```
net-dmz                  reverse-proxy, web-portal, mail, logger
net-corp                 intranet, fileserver, dev-box, ci-runner, ad-controller, logger
net-secure               db-primary, backup, logger
net-bridge-dmz-corp      reverse-proxy, intranet          ← the hole, exactly 1 host wide
net-bridge-corp-secure   ad-controller, db-primary, backup ← the pivot
```

The **logger is multi-homed into all three zone networks** rather than having its own.
If every host joined one shared log network, that network would become a flat any-to-any
path between all three zones through the log sink. This is a real and commonly-missed
misconfiguration, and a good thing to point at in a viva.

---

## Interview points from this file

**"How did you stop the twin and the real lab from drifting apart?"**
> I generate `docker-compose.yml` from the same Python module the simulator reads, and I
> have a test that walks every host pair and asserts Docker connectivity equals the
> firewall policy. They can't disagree, so the sim-to-real gap I measure is real
> transfer, not transcription error.

**"How is the lab isolated?"**
> Every network is `internal: true`, which means Docker creates no gateway. There is
> physically nowhere for a packet to go. A test asserts it on every network and also
> asserts no service publishes a port, since a published port would bridge back to the
> host's stack.

**"Why is your attacker forced through `ad-controller`?"**
> It's the only host with a network path into the secure zone. I test it by deleting it
> from the reachability graph and asserting `db-primary` becomes unreachable from the
> entry point. That's what makes "blue learned to defend the chokepoint" a real finding
> rather than a coincidence — the chokepoint is provably a chokepoint.

---

## Numbers to remember

| | |
|---|---|
| Hosts / zones / cross-zone edges | 13 / 3 / 3 |
| Entry → crown jewel | `web-portal` → … → `ad-controller` → `db-primary` |
| Easiest / hardest exploit | `web-portal` 0.90 / `ad-controller` 0.40 |
| Blue state spaces (DMZ/Corp/Secure) | 192 / 3,072 / 48 — budget is 10,000 |
| Blue action spaces (DMZ/Corp/Secure) | 8 / 12 / 10 |

## Open items

- **`B_secure` has only 2 defended hosts** (`db-primary`, `backup`) giving 48 states.
  §4.2 of the spec says 4 hosts. The other two were honeypots, which are deployed
  rather than present. 48 states is small but workable — decide at Phase 3 whether to
  add a host to the secure zone.
- The Flask services themselves are Phase 4. `docker-compose.yml` references
  `./services`, which does not exist yet — the file is a validated skeleton, not yet
  runnable.

## Quiz yourself

1. Why is the logger attached to three networks instead of having its own?
2. `reverse-proxy` has `exploit_prob=0.30` — lower than `mail` at 0.70. Why would we
   make the *bridge* host harder to compromise than a dead-end host?
3. What breaks if someone adds `("dev-box", "db-primary")` to `CROSS_ZONE_EDGES`?
