"""Decode a trained Q-table into a human-readable policy.

Run with::

    python experiments/inspect_policy.py artifacts/runs/phase2/B_dmz_qtable.npz B_dmz

Why this exists: a learning curve says an agent got better, not *what it learned*. In a
viva, "my defender learned to concentrate on the chokepoint" is a claim that needs
evidence, and the evidence is the greedy action in each state -- not the reward plot.
It is also the first place to look when a curve is disappointing, because a policy that
has collapsed onto a single action is a different problem from one that is merely
imprecise.

Only states the agent actually visited are reported. An unvisited state's greedy action
is whatever the random tie-break returned over a row of zeros, which is noise, and
presenting it as a learned decision would be dishonest.

**The mask has to be applied here too**, and the first version of this script did not
apply it -- which is how it reported nonsense with complete confidence. An illegal action
is never taken, so it is never updated, so it keeps its optimistic 0.0 initialisation
forever; over a row whose real values have all gone negative, an unmasked argmax
therefore selects an *illegal* action in essentially every state. The tell was ``Q=0.00``
next to a spread of 120: the "best" action was the one the agent had never been allowed
to try.

Blue's observation does not fully determine its own action mask -- ``tighten_ratelimit``
depends on whether Layer 1 is currently breached, which blue cannot observe. So only the
derivable part is reconstructed here: ``block`` and ``isolate`` are illegal on a host
that reads as ISOLATED, and observed-ISOLATED is exactly true-ISOLATED because blue is
what isolated it. Reinforcement actions are marked with a ``?`` since their legality
cannot be recovered from the observation alone. That the agent can be offered an action
whose availability it cannot predict from its own state is a genuine consequence of
partial observability, and worth a sentence in the report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from marlsoc.env import actions as act
from marlsoc.env import detection as det
from marlsoc.env import observations as obs
from marlsoc.env import topology as topo
from marlsoc.training.loop import dims_for

ALERT_NAMES = {det.ALERT_LOW: "low", det.ALERT_MEDIUM: "med", det.ALERT_HIGH: "high"}


def derivable_mask(agent: str, digits: tuple[int, ...]) -> np.ndarray:
    """The part of the legality mask recoverable from a blue observation.

    Conservative by construction: an action is marked legal unless the observation proves
    it is not. That can only over-report legality, never under-report it, so a policy
    read through this mask is at worst incomplete rather than wrong.
    """
    hosts = topo.defended_hosts(agent)
    isolated = {
        host.name for host, code in zip(hosts, digits[:-1], strict=True)
        if obs.ObservedStatus(code) is obs.ObservedStatus.ISOLATED
    }
    mask = np.ones(len(act.ACTION_SPACES[agent]), dtype=bool)
    for i, action in enumerate(act.ACTION_SPACES[agent]):
        if action.host in isolated:
            mask[i] = False          # nothing to block or isolate on a dead host
    return mask


def describe_blue_state(agent: str, digits: tuple[int, ...]) -> str:
    """Render a blue observation tuple as something readable."""
    hosts = topo.defended_hosts(agent)
    parts = []
    for host, code in zip(hosts, digits[:-1], strict=True):
        status = obs.ObservedStatus(code)
        if status is not obs.ObservedStatus.CLEAN:
            parts.append(f"{host.name}={status.name.lower()}")
    if not parts:
        parts.append("all clean")
    return f"{', '.join(parts)} | alert={ALERT_NAMES[digits[-1]]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument("agent")
    parser.add_argument("--top", type=int, default=20, help="most-visited states to show")
    args = parser.parse_args()

    data = np.load(args.table, allow_pickle=False)
    q, visits = data["q"], data["visits"]
    space = act.ACTION_SPACES[args.agent]
    dims = dims_for(args.agent)

    seen = np.flatnonzero(visits)
    print(f"{args.agent}: {q.shape[0]} states x {q.shape[1]} actions, "
          f"{seen.size} visited ({seen.size / q.shape[0]:.1%}), "
          f"{int(visits.sum()):,} updates, algorithm={data['algorithm']}\n")

    if seen.size == 0:
        print("nothing visited -- the agent never explored, which is a different problem "
              "from failing to learn")
        return

    # What the policy does overall, weighted by how often each state actually occurs.
    tally: Counter[str] = Counter()
    for state in seen:
        mask = derivable_mask(args.agent, obs.decode(int(state), dims))
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            continue
        best = int(legal[q[state, legal].argmax()])
        tally[str(space[best])] += int(visits[state])
    total = sum(tally.values())
    print("greedy action distribution, weighted by state visits:")
    for action, count in tally.most_common(10):
        print(f"  {count / total:6.1%}  {action}")

    flat = sum(1 for s in seen if np.ptp(q[s]) < 1e-9)
    print(f"\n{flat / seen.size:.1%} of visited states have a flat Q-row "
          f"(no preference learned there)")

    print(f"\nmost-visited states and what the agent does in them:")
    order = seen[np.argsort(-visits[seen])][: args.top]
    for state in order:
        digits = obs.decode(int(state), dims)
        mask = derivable_mask(args.agent, digits)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            continue
        best = int(legal[q[state, legal].argmax()])
        spread = float(np.ptp(q[state, legal]))
        flag = "?" if space[best].verb in act.REINFORCE_LAYER else " "
        print(f"  [{int(visits[state]):>6} visits] {describe_blue_state(args.agent, digits):<56}"
              f" -> {str(space[best]) + flag:<26} (Q={q[state, best]:8.2f}, spread={spread:6.2f})")


if __name__ == "__main__":
    main()
