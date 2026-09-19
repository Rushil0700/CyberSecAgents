# Phase 4 — the whole game: five agents, alternating training, the IQL control

Revision material for Phase 4. Read `notes/phase3-curriculum.md` first.

This is the first phase where the full Markov Game runs. Phases 2 and 3 trained a single
defender with `B_corp` and `B_secure` switched off; here all five agents are live, all six
layers are active, and both teams learn.

## The one piece of theory to be able to say out loud

**Independent Q-Learning is theoretically unsound in a multi-agent setting.**

Q-Learning's convergence proof assumes a **stationary** MDP — the transition and reward
functions do not change as you learn. In a multi-agent game each agent's *environment
includes the other agents*, and their policies are changing too. So from any one agent's
point of view the environment is **non-stationary**, the assumption is violated, and there
is no convergence guarantee. Each agent is chasing a target that moves because of them.

**Alternating training** restores the assumption. Freeze red, train blue to convergence;
freeze blue, train red; repeat. During each phase exactly one policy is changing, so the
learner faces a genuinely fixed opponent and its phase *is* a stationary MDP.

We run the unsound version too, deliberately — see "the control" below.

> **Viva question.** *Why not just train all your agents at once?*
> Because then each agent's environment contains the other agents, whose policies keep
> changing, so the environment is non-stationary and Q-Learning's convergence proof no
> longer applies — Independent Q-Learning has no convergence guarantee in a multi-agent
> setting. I train by alternating instead: freeze one team, train the other, swap. During
> each phase the opponent is fixed, so the learner is solving a stationary MDP and the
> proof holds again. I also run the simultaneous version as a control, because measuring
> that it doesn't settle is stronger evidence than citing that it shouldn't.

## What alternating training needed that the spec does not mention

**Exploration is rewound at every swap** (`epsilon_on_swap = 0.40`). An agent arriving
from the previous round is nearly greedy, and the opponent it is about to face has just
changed underneath it. A nearly greedy agent cannot discover the response, because every
action it would need to try is ranked below something that used to work. This is the same
argument that makes `epsilon_on_promote` necessary for the curriculum. Without it the arms
race stalls after round 1 — both teams keep replaying the counter they already had.

**Q-tables carry across rounds.** That carry-forward is what makes the run an arms race
rather than a sequence of unrelated trainings: round 3's blue inherits everything rounds 1
and 2 taught it, so the plot shows two policies *co-adapting*.

**The frozen opponent acts greedily.** It is standing in for a deployed policy, and an
opponent still exploring 5% of the time is a different — easier — opponent than the one
the learner will be scored against.

## The control: measuring that IQL does not settle

`train_iql` runs every agent learning simultaneously. It is not an inferior implementation
of alternating training; it is the thing alternating training exists to avoid, kept so the
claim can be measured rather than asserted.

`alternating.instability` is the mean absolute change in attacker success between
consecutive evaluations. A converging run drives it toward zero as the policies stop
moving. A non-stationary one does not, because every improvement by one side invalidates
what the other had learned. Both methods are evaluated on the same schedule so the number
compares like with like.

**Note what the claim is.** It is *not* "IQL scores worse" — it might not. It is "IQL does
not settle". The thing to look at on the plot is the **amplitude** of the oscillation, not
its level.

## Shared vs individual reward — §6's emergent-behaviour experiment

Until now all three defenders shared one reward, so cooperation was a *consequence* of the
return rather than an instruction: the DMZ defender pays for a corporate breach, which is
what makes hand-off worth learning.

`blue_reward_individual` charges each defender only for its own zone. The decisive detail
is that **the −100 for losing the crown jewel lands on `B_secure` alone**, because
`auth-server` is in `Zone.SECURE`. `B_dmz` therefore has *no stake at all* in the crown
jewel: isolating is cheap for it and the entire downstream cost of a missed intrusion is
charged to somebody else.

§6's prediction is that this makes `B_dmz` trigger-happy — more isolations, more false
positives — while shared reward teaches restraint. Nobody is told to cooperate or to
defect in either case; the behaviour falls out of which return each agent maximises.

**Attribution is exact, not heuristic.** Action spaces are built per zone, so the only
defender that *can* isolate a host in a zone is the one that owns it — ownership recovers
who acted without `StepEvents` carrying an actor field it needs for nothing else.

### The bug that attribution nearly hid

Repairs were first attributed by `LayerSpec.enforced_by`. That looks equivalent to the
action map and is not: Layer 3 (`AUTH`) has `enforced_by = None` because no single host
implements it, yet `B_corp` restores it with `rotate_credentials`. The first version paid
`B_corp` **nothing for its own repair action**. `actions.AGENT_REINFORCE` is the authority,
because it is what the environment actually executes.

## A measurement trap worth internalising

`blue_return` is `returns["B_corp"]`. Under shared reward that is the whole team's value;
under individual reward it is one defender's slice. **They are different quantities** and
a table with both in one column invites a comparison that means nothing.

This is the same category error as comparing red's return across shaping modes in Phase 3,
where potential-based shaping moved the scale by ~100 without any behaviour changing at
all.

**Compare behaviour, not price.** Attacker success, false positives, MTTD and layers
breached are properties of what the agents *did*, and they survive a change of reward
function. Returns do not.

> **Viva question.** *Your two reward schemes give very different returns — which is
> better?* That comparison isn't meaningful as stated, because the two returns measure
> different things: one is a shared team value and the other is a single agent's share. I
> compare them on quantities that don't depend on the reward function — attacker success,
> false positives per episode, mean time to detect. Those describe what the agents did
> rather than how I chose to price it.

## What to be able to explain from this phase

1. Why IQL has no convergence guarantee in a multi-agent setting, in terms of stationarity
   — and why freezing one team restores it.
2. Why the IQL control is kept, and that its claim is about *amplitude*, not level.
3. Why exploration must be rewound at each swap, and what stalls without it.
4. How shared vs individual reward produces a behavioural difference without anyone being
   instructed to cooperate or defect.
5. Why returns cannot be compared across reward structures, and which metrics can.
