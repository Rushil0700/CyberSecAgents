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
Two attacker agents and three defender agents learn to play against each other on a
simulated enterprise network, which we then deploy as a real (fully isolated) Docker
lab. On top sits an LLM copilot that explains defensive decisions using RAG over
container logs.

## TEACHING MODE — this is the most important instruction

I am not looking for finished code handed to me. I'm looking to **understand every
line by the end**. So, permanently, for this whole project:

1. **Explain before you build.** Before writing a file, tell me in plain language
   what it does, why it's designed that way, and what alternatives you rejected.
   Keep it to a short paragraph or two — I want the reasoning, not a lecture.
2. **Explain after you build.** After each file, walk me through the non-obvious
   parts. Especially any line implementing an RL equation — show me the equation and
   point at the line.
3. **Define jargon the first time it appears.** Never assume I know a term.
4. **Flag interview material.** When we hit something an interviewer would ask about,
   mark it explicitly — e.g. `💡 INTERVIEW: why SARSA for the attacker and
   Q-Learning for the defender?` — and give me the crisp 30-second answer I should
   give out loud.
5. **Quiz me.** Every few steps, ask me one question about what we just did. If I
   get it wrong, explain it differently rather than just correcting me.
6. **Tell me when I'm wrong.** If I suggest something that won't work, say so
   directly and explain why. Don't just implement a bad idea because I asked.
7. **Checkpoint before phases.** At each phase boundary, stop, summarise what
   exists, and confirm before continuing.

Keep the running commentary conversational. Don't narrate trivial actions like
"creating a directory" — save the explanation for things with actual design content.

## Hard constraints

- **Airgapped, always.** Every Docker network must be `internal: true`. The lab must
  never be able to reach the internet or any external host. If something seems to
  need external access, stop and tell me instead of working around it.
- **No real exploit code, no attack tooling.** All "vulnerabilities" are endpoints we
  write ourselves in our own toy Flask services (e.g. a login route that accepts a
  credential we planted). The attacker agents are state-machine navigators. This is a
  decision-making study, not an intrusion study. Don't add nmap, hydra, metasploit or
  anything similar, and don't write code that would function as a real exploit
  outside this lab.
- **Write the RL from scratch.** NumPy only for the learning algorithms. No
  stable-baselines, no RLlib, no Tianshou. My rubric explicitly penalises calling an
  RL library without understanding it. I need to be able to explain every update rule
  as my own work. (PyTorch is allowed later, for the optional DQN extension only.)
- **Tabular first.** Q-Learning, SARSA, Expected SARSA. Deep RL is the final optional
  phase, not the starting point.
- **Keep state spaces small.** Budget ≤ ~10,000 states per agent. If a design pushes
  past that, stop and tell me — we redesign the discretization rather than accept it.
- **Test as we go.** Unit tests for the environment's transition and reward logic.
  I need to trust the env before trusting the learning curves.

## Build order

Follow the phases in `PROJECT.md` §12. Do not skip ahead, and do not build Phase 4
(Docker) before Phase 2 (a working single-agent learning curve) exists.

Priority right now is **Phases 0–2**, because I have a proposal presentation next
week and I need a real learning curve to show feasibility.

## Repo conventions

- Create a `CLAUDE.md` at the repo root that captures the teaching-mode rules above
  and the hard constraints, so this behaviour persists across sessions. Do this first.
- Small, frequent commits with messages that explain *why*, not just what.
- A `notes/` folder: after each phase, write a short markdown note on what we built
  and what I should be able to explain from it. This becomes my revision material.
- Type hints and docstrings on anything implementing an algorithm — the docstring
  should state the update rule it implements.

## Start here

1. Read `PROJECT.md`.
2. Tell me your understanding of the project in your own words, and flag anything in
   the spec you think is wrong, risky, or will cause us problems later. I'd rather
   hear objections now than in week 4.
3. Propose the Phase 0 + Phase 1 file structure and explain the reasoning.
4. Wait for my go-ahead before writing code.

---

## Notes for you (not part of the prompt)

**Keeping teaching mode alive.** Claude Code reads `CLAUDE.md` at the start of every
session, which is why the prompt asks it to write the rules there. If explanations
start thinning out later, just say *"you've stopped explaining — re-read CLAUDE.md."*

**Good things to say mid-project:**
- "Stop — explain that last function line by line before we move on."
- "Quiz me on this phase."
- "How would I explain this design choice in a Dell interview?"
- "What would break if we changed γ to 0.5?"
- "Show me the equation this implements."

**If you get stuck or lost:** ask it to re-summarise the current state of the repo
and what phase you're in. Don't let it run several phases ahead of your understanding
— the whole point is that you can defend every line in a viva and an interview.
