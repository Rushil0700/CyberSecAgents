# Claude Code Kickoff Prompt

Paste everything between the lines into Claude Code as your **first message** in an
empty project folder. Put `PROJECT.md` in that folder first.

---

I'm building a college mini-project and I want to learn every part of it deeply,
because I'm also using it to prepare for a Dell AI/GenAI internship interview.
Read `PROJECT.md` in this folder first — it's the complete spec. Don't start coding
until you've read it and confirmed the plan back to me.

## What we're building

MARL-SOC: a multi-agent reinforcement learning system for autonomous network defense.
A dummy company server network runs inside a fully isolated Docker lab, protected by
**six layers of defence-in-depth**: perimeter/WAF, DMZ network boundary,
authentication, internal micro-segmentation, privilege escalation, and an
application-level MFA check. Two attacker (red) agents continuously try to breach all
six and **alter the admin login credentials**; three defender (blue) agents — one per
network zone — learn to detect and block them. Both sides learn from their mistakes
via reinforcement learning. A local LLM, using RAG over a security playbook we write
for our own toy system, advises the agents on which moves to try — but the RL agents
decide, and learn whether the advice was worth trusting.

Each of the six layers requires a *different class* of attacker action to beat (see
`PROJECT.md` §3.1). That mapping is deliberate — it's what makes this a rich
decision problem rather than just a long corridor.

## TEACHING MODE — the most important instruction

I am not looking for finished code handed to me. I want to **understand every line by
the end**. Permanently, for this whole project:

1. **Explain before you build.** Before writing a file, tell me in plain language what
   it does, why it's designed that way, and what alternatives you rejected. A short
   paragraph or two — reasoning, not a lecture.
2. **Explain after you build.** Walk me through the non-obvious parts. For any line
   implementing an RL equation, show me the equation and point at the line.
3. **Define jargon the first time it appears.** Never assume I know a term.
4. **Flag interview material.** When we hit something an interviewer would ask about,
   mark it — e.g. `💡 INTERVIEW: why SARSA for the attacker and Q-Learning for the
   defender?` — and give me the crisp 30-second spoken answer.
5. **Quiz me.** Every few steps, ask me one question about what we just did. If I'm
   wrong, explain it a different way rather than just correcting me.
6. **Tell me when I'm wrong.** If I suggest something that won't work, say so and
   explain why. Don't implement a bad idea just because I asked.
7. **Checkpoint before phases.** At each phase boundary, stop, summarise what exists,
   and confirm before continuing.

Keep commentary conversational. Don't narrate trivial actions ("creating a
directory") — save explanation for things with real design content.

## Hard constraints

- **Airgapped, always.** Every Docker network is `internal: true`. The lab must never
  reach the internet or any external host. If something seems to need external access,
  stop and tell me rather than working around it.
- **No real exploit code, no attack tooling.** All "vulnerabilities" are endpoints we
  write ourselves in our own toy Flask services (e.g. a login route that accepts a
  credential we planted). The attacker agents are state-machine navigators over our
  own abstract action set. This is a decision-making study, not an intrusion study. Do
  NOT add nmap/hydra/metasploit, and do NOT write code that would function as a real
  exploit outside this lab.
- **The LLM's knowledge base is OUR OWN playbook** describing this toy system's
  abstract mechanics and action set — NOT a corpus of real-world hacking techniques,
  CVEs, or shell commands. It advises only on moves that exist inside our simulation.
- **The LLM advises; the RL decides.** The LLM only proposes candidate actions during
  exploration. The RL agent still learns Q-values from real rewards and must be able
  to learn to ignore bad advice. Never let the LLM directly drive the policy — my
  rubric fails a project that "only uses an LLM without RL." We must be able to run
  the whole thing with the LLM turned off and still learn.
- **Write the RL from scratch.** NumPy only for the learning algorithms — no
  stable-baselines, RLlib, or Tianshou. I must be able to explain every update rule as
  my own work. (PyTorch allowed later, for the optional DQN extension only.)
- **Tabular first.** Q-Learning, SARSA, Expected SARSA. Deep RL is the final optional
  phase, not the start.
- **Keep state spaces small.** Budget ≤ ~10,000 states per agent. If a design pushes
  past that, stop and tell me — we redesign the discretization. Note that the six
  layers should be represented as **binary flags in the state**, not as extra host
  dimensions — that's what keeps red's state space at ~768 instead of exploding it.
- **Curriculum learning is mandatory, not optional.** Six layers creates a
  sparse-reward / long-horizon problem: under random exploration red would almost
  never complete the full chain and would never learn. Build the staged curriculum
  (`PROJECT.md` §7.4) as part of the training loop from the start, along with the
  progressive per-layer rewards. If red's learning curve flatlines, this is the first
  thing to check.
- **Action masking.** Actions whose layer preconditions aren't met must be masked out
  of the agent's choice set — including when validating an LLM suggestion. If the LLM
  proposes a masked action, fall back to a random legal one.
- **Test as we go.** Unit tests for the environment's transition and reward logic, and
  specifically for each of the six layer precondition checks. I need to trust the env
  before trusting the learning curves.

## Build order

Follow `PROJECT.md` §12. Do not skip ahead. Do not build the Docker lab (Phase 6)
before a working learning curve (Phase 2) exists, and do not build the LLM advisor
(Phase 5) before multi-agent IQL works (Phase 4).

Priority right now is **Phases 0–2**: I have a proposal presentation next week and
need a real learning curve to show feasibility. Phase 2 runs with **layers 1–2 only** —
build the full six-layer model in the environment, but train the first curve against
the shallow version. I present the six-layer architecture as the design; nobody
expects it trained yet.

## Repo conventions

- Create a `CLAUDE.md` at the repo root capturing the teaching-mode rules and the hard
  constraints above, so this behaviour persists across sessions. Do this first.
- Small, frequent commits; messages explain *why*, not just what.
- A `notes/` folder: after each phase, write a short markdown note on what we built
  and what I should be able to explain from it. This is my revision material.
- Type hints and docstrings on anything implementing an algorithm; the docstring
  states the update rule it implements.

## Start here

1. Read `PROJECT.md`.
2. Tell me your understanding of the project in your own words, and flag anything you
   think is wrong, risky, or will bite us later. I'd rather hear objections now than
   in week 4.
3. Propose the Phase 0 + Phase 1 file structure and explain the reasoning.
4. Wait for my go-ahead before writing code.

---

## Notes for you (not part of the prompt)

**Keeping teaching mode alive.** Claude Code reads `CLAUDE.md` at the start of every
session, which is why the prompt has it write the rules there. If explanations thin
out later, say *"you've stopped explaining — re-read CLAUDE.md."*

**Good things to say mid-project:**
- "Stop — explain that last function line by line."
- "Quiz me on this phase."
- "How would I explain this in a Dell interview?"
- "What breaks if we change γ to 0.5?"
- "Show me the equation this implements."
- "Turn the LLM advisor off and show me the learning curve without it."

**Don't let it run ahead of your understanding.** The whole point is that you can
defend every line in a viva and an interview. If you're lost, ask it to re-summarise
the repo state and the current phase before continuing.
