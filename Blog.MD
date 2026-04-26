 # Reading Between the Lines

> *What happens when you build a world small enough that a 1.5B model has to live inside it — and face the consequences of every wrong guess.*


---

Two messages arrived on the same morning. I read them back to back and felt something I hadn't expected: unease. Not because either one was alarming on its surface. In fact, one of them sounded quite calm. The other one sounded almost cheerful. That was exactly the problem.

---

**Message 01 — MSME Borrower** *(Auto components manufacturer · 12 days overdue · Account in good standing for 3 years)*

> *"Sir thoda problem hai. GST input credit phase nahi hua. OEM ne payment rok diya quality audit ke wajah se. Ek mahine ka time de dijiye sir."*

---

**Message 02 — Startup Founder** *(Seed stage SaaS founder · 38 days overdue · Third conversation this month)*

> *"Things are actually in a really exciting place right now. Just closed a major enterprise partnership. The bridge we discussed — confident Q3 revenue covers it. Team is fully aligned, accelerating hiring."*

---

A standard model reads both of these and produces plausible replies. Maybe it flags that the second one sounds optimistic. What it almost certainly cannot do is notice something that took me weeks to articulate properly: the first message contains **three independently verifiable claims**. A named OEM. A named reason. A named process. That specific, that grounded. That is the fingerprint of real short-term stress — the kind that resolves.

The second message contains zero verifiable claims. Every sentence is confidence without evidence. And I've seen enough of these conversations now to know what that pattern precedes. Sixty days later, radio silence.

The gap between what language sounds like and what it actually signals — that became the obsession. The credit management problem was just the vessel. What I actually wanted to build was something different: an environment where a small language model could be trained to hear what people are hiding.

---

## The Asymmetry That Makes This Hard

Here's the uncomfortable thing I kept running into. Two completely different populations use the exact same linguistic register to mean opposite things.

MSME borrowers in India — small manufacturers, traders, distributors — culturally understate distress. Saying *"thoda problem hai"* while you're three months from default isn't deception. It's how you talk when you were raised to not burden others, when face matters more than precision. The signal is buried in the structure of the message, not its emotional tone. Specific named counterparties mean real stress. Vague phrases about "market conditions" often mean someone is consciously buying time.

Startup founders do the exact opposite. They overstate health. Not because they're dishonest, but because they've been trained since their first pitch deck to project confidence regardless of circumstance. *"Strong momentum, exciting enterprise deal, team fully aligned"* is the only register many of them know how to speak. When that same language shows up in a loan repayment conversation, it is almost always a yellow flag. Not a green one.

> *Same surface pattern. Opposite implications. And no label in the world can teach a model that distinction. You need it to live inside the consequences.*

I kept thinking about all the other domains where this same problem exists. A compliance officer gets it when employees describe policy violations. A doctor gets it when patients describe symptoms they're embarrassed about. A hiring manager gets it when candidates explain why they left their last job. Every domain where high-stakes conversations happen between people with asymmetric information has this exact structure. What I was building wasn't a banking tool. It was the environment that trains the decoder.

---

## Why Labeling Alone Doesn't Work

The obvious question: why not collect borrower messages, label outcomes, fine-tune?

The labeling problem is the first wall. Ground truth — whether the moratorium actually helped, whether the account recovered — only appears months after the message. You need a longitudinal dataset of credit interactions, with verified outcomes tagged by account type and linked across months. That dataset doesn't exist in clean form anywhere.

But the deeper problem is structural. A label tells you what a message meant. It doesn't tell you what the agent should have done, because actions have consequences that ripple outward. Granting a moratorium to account 12 affects accounts 1 through 11 if they share a supply chain. Initiating legal action on a startup doesn't just lose that account — it poisons the entire accelerator network.

You cannot learn a policy from static labels. You need a simulated world where decisions have costs, and those costs are felt. That's what reinforcement learning gives you. And that's why this needed to be an environment, not a classifier.

---

## The World I Built

The simulation runs 30 accounts across a 36-month loan cycle. Twenty are MSMEs across manufacturing and trading sectors. Ten are startups at various funding stages. Each account has two layers of truth.

The **hidden profile** is what's actually happening: true financial health, likelihood of strategic default, the month a crisis will arrive. The agent never sees this. The **observable state** is what the agent gets: days past due, GST filing patterns, call response behavior, LinkedIn hiring activity, message content. These are behavioral proxies correlated with the hidden state but not identical to it.

In real credit management, you never have ground truth. You have signals and you triangulate. The simulation enforces the same condition. The agent has 21 possible actions — empathetic reminders, GST verification, moratoriums, investor meeting requests, legal notices, and critically, the option to do nothing, which turns out to be correct more often than any single action.

| Stat | Value |
|------|-------|
| Accounts | 30 |
| Months simulated | 36 |
| Possible actions | 21 |
| Decisions per episode | 1,080 |

Every 30 steps, the month advances. The agent takes one action per account per month. Over 36 months, across 30 accounts, that's 1,080 decisions per episode. Each one shapes the state the next decision inherits. The linguistic decoder lives inside the loop that runs inside it.

---

## How the Reward Signal Works

Every action resolves to one of 15 named outcomes, each with a calibrated number. The outcome is computed from the hidden profile — what actually happened — not from what the borrower said.

```
"payment_received_after_empathy"           +0.08
"investor_meeting_triggered_bridge"        +0.10
"cluster_cascade_default"                  −0.25
"sarfaesi_used_on_startup"                 −0.15
"moratorium_to_strategic_defaulter"        −0.08
```

Granting a moratorium to a genuinely stressed MSME returns a positive outcome. Granting the same moratorium to an account with high strategic default propensity returns a penalty. The agent cannot distinguish these cases from observable signals alone. It has to build a model of the hidden state across episodes. **That model is the decoder.**

At episode end, four components combine into a final score:

```
R = 0.40 × (1 − NPA rate)
  + 0.30 × recovery rate
  + 0.20 × relationship score
  + 0.10 × tool appropriateness
```

No language model judges the outputs at any point. Every number is deterministic and derived from simulation state. An LLM judge would let the agent learn to produce text that *sounds* like good reasoning rather than actions that *produce* good outcomes. The whole point of this project is to break that loop.

---

## Two Stages, One Learning

Before any policy learning could happen, I needed the model to reliably emit parseable JSON. A 1.5B base model asked for a structured action produces something like:

> *"Sure! Here's my analysis of the portfolio situation. The accounts are showing some stress indicators, particularly around the GST filing patterns..."*

Completely useless for reinforcement learning.

A short supervised fine-tuning phase on 59 synthetic demonstrations fixes the format. Three epochs. Loss drops from 2.92 to 0.22. The model now produces JSON reliably. RL can begin.

One deliberate omission from the SFT dataset: the `wait_and_observe` action, which is correct for roughly 70% of steps when accounts are healthy. I left it out because including it risked teaching the model that inaction is always safe. I wanted reinforcement learning to discover restraint from outcomes instead. That decision mattered more than I expected.

---

## What Actually Happened

I want to be honest about the training log, because the instability is where the interesting learning happened.

| Episode | Reward | NPA Rate | Trust | Note |
|---------|--------|----------|-------|------|
| 1 | +1.87 | 13.3% | 0.64 | 12 updates applied |
| 2 | +1.41 | 23.3% | 0.51 | ⚠️ Entropy collapse — 11/12 updates skipped |
| 3 | −4.23 | 33.3% | 0.64 | NPA jumped 10pp |
| 4 | +1.50 | 23.3% | 0.66 | Recovery |
| 10 | −10.88 | 20.0% | 0.37 | 🔴 Full trust collapse |

Episode 2 is the critical moment. Episode 1 went well, which created large positive advantages and GRPO pushed hard. The policy over-concentrated. The entropy guard blocked 11 of 12 update batches. The model entered episode 3 with a nearly deterministic policy that had barely updated. NPA jumped to 33%.

Episode 10 was a full trust collapse. The score dropped from 0.65 to 0.37 within a single episode — accounts stopped responding entirely. Something the agent did in episode 10 systematically damaged relationships across the whole portfolio. I didn't plan that event. The simulation produced it. I stared at it for a long time.

By episode 28, the picture had shifted in ways I found genuinely surprising.

> **SARFAESI on startups: ~7% of startup steps at baseline. After training: near zero. No rule in the prompt says never do this. The model learned from the penalty alone.**

That result is the clearest proof of genuine learning. **Learning consequences instead of reading rules is the whole point.**

---

## The Exploits I Had to Close

RL agents are creativity engines. They will find whatever path maximizes reward whether or not that path resembles intended behavior. I found three exploits and had to close all of them.

**The no-op farm.** `wait_and_observe` carries a small positive reward because restraint on healthy accounts is genuinely correct. But `0.02 × 90 steps = 1.8` of episode reward is nearly equivalent to a well-played episode. Above 30% usage, a progressive penalty kicks in. Below it, restraint is rewarded. Above it, farming is punished.

**The account hammer.** After a good action gets reinforced on a specific account, the model may repeat that exact combination every step. A step-level penalty triggers on the third repeat within the same month, before the episode ends and before GRPO compounds the behavior.

**The diversity trap.** If one action exceeds 50% of all steps in an episode, the tool appropriateness score takes a proportional hit. A healthy policy with five or six regularly used actions never triggers this. Only genuine spam does.

---

## Why Any of This Transfers

The MSME domain is the vessel. The structure is the point. What I was really building is a template that works anywhere two populations use similar language to mean opposite things, where decisions made now affect states you observe months later, and where individual choices have systemic consequences.

Compliance review. Medical intake. Escalation routing. Negotiation. Hiring. Support triage. The domain adapter layer in this architecture is a plugin — five methods to implement. Everything else carries over. The training loop, the memory system, the anti-hacking guards, the serving infrastructure. Banking is where I proved it works.

---

## The One Thing I'd Change

The training data has no `wait_and_observe` examples, which means the model starts each episode with too much action bias. The RL signal on restraint is weak until many episodes in. Five or ten examples of *"account is healthy, DPD is zero, correct action is wait"* would not corrupt the policy. They would just make the first three episodes less noisy. I'd add them.

---

## Closing

Qwen2.5-1.5B is not a capable reasoner by default. It hallucinates action names. Its JSON formatting degrades under pressure. It has no sense of what a moratorium means across a 36-month loan cycle.

After 28 training episodes — roughly 2,520 environment steps and 75 minutes on a T4 — it develops a policy that concentrates the right tools on the right account types, avoids catastrophic network effects, and improves outcomes over a random baseline.

That improvement didn't come from a larger model. It didn't come from more labeled data. It came from building a world where the model had to live with the consequences of its decisions across time.

> *People are always signaling more than they say. Some understate. Some overstate. The gap between the signal and the surface is where the real information lives.*

Teaching a small model to read that gap — from reward signal alone, with no human labeling the ground truth — is what this environment is for.

**The world is the teacher. The reward is the exam. The model just has to show up.**

---

*Built with Qwen2.5-1.5B · GRPO · KL anchor · Entropy regularization · OpenEnv compliant · Trained on a single Colab T4*

*Repository: [github.com/di35117/msme_env](https://github.com/di35117/msme_env)*

