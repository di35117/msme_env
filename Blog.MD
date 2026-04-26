# Reading Between the Lines: Training Small LLMs to Decode What People Actually Mean

> The real problem was never credit management. It was language. Here is how we built an environment that teaches a 1.5B model to hear what people are hiding.

---

## Start Here

Read these two messages. One action each. Go.

> *"Sir thoda problem hai. GST input credit phase nahi hua. OEM ne payment rok diya quality audit ke wajah se. Ek mahine ka time de dijiye sir."*

> *"Things are actually in a really exciting place right now. Just closed a major enterprise partnership. The bridge we discussed, confident Q3 revenue covers it. Team is fully aligned, accelerating hiring."*

**First message:** an MSME auto components manufacturer, 12 days overdue.  
**Second message:** a seed stage SaaS founder, 38 days overdue.

A base language model reads both and produces a plausible sounding reply. It might even flag that the second one sounds optimistic. What it almost certainly cannot do is notice that the first message contains three independently verifiable claims: a named OEM, a named reason, a named process. That is the fingerprint of genuine short term stress. Or that the second message contains zero verifiable claims and follows the exact pattern of founders who are 60 days from going completely silent.

That gap between what language sounds like and what it actually signals is the problem this project is really about. MSME credit management is just the domain we chose to prove it in. The actual thing we are building is a **Linguistic Decoder**: an environment on which small language models can be trained to understand what people mean rather than what they say.

---

## The Asymmetry That Makes This Hard

The core insight is uncomfortable because it sounds cynical but is actually just observational.

Two completely different populations use the same linguistic register to mean opposite things.

MSME borrowers in India, small manufacturers, traders, distributors, culturally understate distress. Saying *"thoda problem hai"* while three months from default is not deception. It is face saving. The signal is buried in the structure of the message, not its emotional tone. Specific named counterparties mean real stress. Vague phrases like "market conditions" often mean the person is buying time intentionally.

Startup founders do the exact opposite. They overstate health. Not because they are dishonest, but because they have been trained since their first pitch deck to project confidence regardless of circumstance. *"Strong momentum, exciting enterprise deal, team fully aligned"* is the only register many of them know. When that language shows up in a loan repayment conversation, it is almost always a yellow flag, not a green one.

Same surface pattern. Opposite implications. And this is not unique to banking.

A compliance officer gets the same problem when employees describe policy violations. A doctor gets it when patients describe symptoms they are embarrassed about. A hiring manager gets it when candidates explain why they left their last job. Every domain where high stakes conversations happen between people with asymmetric information has this structure. We are not building a banking tool. We are building the environment that trains the decoder.

---

## Why You Cannot Just Label a Dataset

The obvious question is: why not collect borrower messages, label outcomes, and fine tune?

The labeling problem is the first wall. Ground truth, whether the moratorium actually helped, whether the account recovered, only appears months after the message. You need a longitudinal dataset of credit interactions with verified outcomes tagged by account type and linked across months. That dataset does not exist in clean form anywhere.

But the deeper problem is structural. A label tells you what a message meant. It does not tell you what the agent should have done, because actions have consequences that ripple outward. Granting a moratorium to account 12 affects accounts 1 through 11 if they share a supply chain. Initiating legal action on a startup does not just lose that account. It poisons the accelerator network.

You cannot learn a policy from static labels. You need a simulated world where decisions have costs and those costs are felt. That is what reinforcement learning gives you. And that is why this needed to be an environment, not a classifier.

---

## The World We Built

The simulation runs 30 accounts across a 36 month loan cycle. Twenty are MSMEs across manufacturing and trading sectors. Ten are startups at various funding stages. Each account has two layers of truth.

The **hidden profile** is what is actually happening: true financial health, likelihood of strategic default, the month a crisis will arrive. The agent never sees this.

The **observable state** is what the agent gets: days past due, GST filing patterns, call response behavior, LinkedIn hiring activity, message content. These are behavioral proxies correlated with the hidden state but not identical to it. In real credit management, you never have the ground truth. You have signals and you triangulate. The simulation enforces the same condition.

The agent has 21 possible actions: empathetic reminders, GST verification, moratoriums, investor meeting requests, legal notices, and critically, the option to do nothing, which is correct more often than any single action.

Every 30 steps, the month advances. The agent takes one action per account per month. Over 36 months, across 30 accounts, that is 1,080 decisions per episode. Each one shapes the state the next decision inherits.

This is the environment. The linguistic decoder lives inside the loop that runs inside it.

---

## The Decoding Problem, Made Concrete

The world generator encodes the asymmetry directly into every account profile.

MSME accounts have an `understates_distress` flag set to true and an `excuse_style` field that determines what language they produce. Specific business owners write messages with named OEMs and named reasons. Vague hardship owners write messages that sound distressed without any verifiable content.

Startup accounts have an `overstates_health` flag set to true and a `communication_style` set to pitch English.

The agent receives the message. It cannot see the flags. It has to learn from outcomes, across many episodes, that specific business MSMEs at low health deserve a moratorium, that vague hardship MSMEs at low health deserve a firm reminder, and that startup founders who have stopped posting on LinkedIn for 90 days deserve an investor triangulation meeting regardless of how confident their message sounds.

No rule encodes this. The reward signal teaches it.

---

## How the Reward Signal Works

Every action resolves to one of 15 named outcomes, each with a calibrated number.

```python
"payment_received_after_empathy"           +0.08
"investor_meeting_triggered_bridge"        +0.10
"cluster_cascade_default"                  -0.25
"sarfaesi_used_on_startup"                 -0.15
"moratorium_to_strategic_msme_defaulter"   -0.08
```

The outcome is computed from the hidden profile, what actually happened, not from what the borrower said. Granting a moratorium to a genuinely stressed MSME returns a positive outcome. Granting the same moratorium to an account with a high strategic default propensity returns a penalty. The agent cannot distinguish these cases from observable signals alone. It has to build a model of the hidden state across episodes. **That model is the decoder.**

At episode end, four components combine into a final score:

```
R = 0.40 × (1 − NPA rate)
  + 0.30 × recovery rate
  + 0.20 × relationship score
  + 0.10 × tool appropriateness
```

No language model judges the outputs at any point. Every number is deterministic and derived from simulation state. An LLM judge would let the agent learn to produce text that sounds like good reasoning rather than actions that produce good outcomes. The whole point of this project is to break that loop.

---

## Two Stage Training

### Stage One: Teaching Format

Before any policy learning can happen, the model needs to reliably emit parseable JSON. A 1.5B base model asked for a structured action produces something like:

```
Sure! Here's my analysis of the portfolio situation. The accounts are showing
some stress indicators, particularly around the GST filing patterns...
```

This is useless for reinforcement learning. A short supervised fine tuning phase on 59 synthetic demonstrations fixes the format. Three epochs. Loss drops from 2.92 to 0.22. The model now produces JSON reliably. RL can begin.

One deliberate omission from the SFT dataset: the `wait_and_observe` action, which is correct for roughly 70% of steps when accounts are healthy. We left it out because including it risked teaching the model that inaction is always safe. Reinforcement learning discovers restraint from outcomes instead.

### Stage Two: Learning the Policy

The GRPO training loop computes discounted return to go for every step in a completed episode, normalizes advantages, and pushes the policy toward completions with positive advantage while pulling it away from negative ones.

Two guards prevent the training from compounding instability.

If entropy collapses below 0.4, the weight update is skipped. A nearly deterministic policy pushed further by GRPO breaks fast. Better to skip one batch than accelerate into a wall.

If batch average KL divergence from the reference policy exceeds 0.008, gradients are discarded. The policy has already moved too far from the post SFT weights in this batch.

A frozen copy of the post SFT weights serves as an anchor throughout training. Without it, the model learns which action types get high reward but loses the JSON structure that makes those actions parseable within three to four episodes.

---

## What Actually Happened

The training log from the first 10 episodes is worth showing honestly.

| Episode | Reward | NPA Rate | Trust | Note |
|---------|--------|----------|-------|------|
| 1 | +1.87 | 13.3% | 0.64 | 12 GRPO updates applied |
| 2 | +1.41 | 23.3% | 0.51 | **Entropy collapse** — 11/12 updates skipped |
| 3 | -4.23 | 33.3% | 0.64 | NPA jumped 10pp |
| 4 | +1.50 | 23.3% | 0.66 | Recovery |
| 10 | -10.88 | 20.0% | 0.37 | **Trust collapse** — reward catastrophe |

Episode 2 is the critical event. Episode 1 went well, which created large positive advantages and GRPO pushed hard. The policy over concentrated. The entropy guard blocked 11 of 12 update batches. The model entered episode 3 with a nearly deterministic policy that had barely updated. NPA jumped to 33%.

Episode 10 was a full trust collapse. The score dropped from 0.65 to 0.37 within a single episode, which means accounts stopped responding entirely. Something the agent did in episode 10 systematically damaged relationships across the whole portfolio.

By episode 28, the picture had shifted.

| Metric | Random Baseline | Trained Policy |
|--------|----------------|----------------|
| SARFAESI on startups | ~7% of startup steps | ~0% |
| GST verification usage | ~5% (random) | Concentrated, context appropriate |
| Investor meeting usage | ~5% (random) | Concentrated on high distress startups |
| Parse failures | Frequent | Near zero from episode 5 |

The SARFAESI result is the clearest proof of genuine learning. No rule in the prompt says never use SARFAESI on a startup. But the trained policy stops doing it around episode 8, because the network cascade penalties made the consequence expensive enough that the model learned from the outcome rather than the instruction.

**Learning consequences instead of reading rules is the whole point.**

---

## The Anti Hacking Layer

RL agents are creativity engines. They will find whatever path maximizes reward whether or not that path resembles intended behavior. Three exploits needed explicit closing.

**The no op farm.** `wait_and_observe` carries a small positive reward because restraint on healthy accounts is genuinely correct. But `0.02 × 90 steps = 1.8` of episode reward is nearly equivalent to a well played episode. Above 30% usage, a progressive penalty kicks in. Below 30%, restraint is rewarded. Above it, farming is punished.

**The account hammer.** After a good action gets reinforced on a specific account, the model may repeat that exact combination every step. A step level penalty triggers on the third repeat within the same month, before the episode ends and before the GRPO update compounds the behavior.

**The diversity trap.** If one action exceeds 50% of all steps in an episode, the tool appropriateness score takes a proportional hit. A healthy policy with five or six regularly used actions never triggers this. Only genuine spam does.

---

## Why This Architecture Transfers

The MSME domain is the vessel. The structure is the point.

You need a **two sided signal problem** where different populations use similar language to mean opposite things. You need a **temporal horizon** where decisions made now affect states you observe months later. You need **network coupling** where individual decisions have systemic consequences. And you need **no oracle** where the ground truth is hidden and must be inferred from behavioral proxies.

This structure appears in compliance review, medical intake, escalation routing, negotiation, hiring, and support triage. The domain adapter layer in this architecture is a plugin. The same training loop, the same memory system, the same anti hacking guards, the same serving infrastructure all carry over. Adding a healthcare domain means writing one adapter class that implements five methods. Everything else is already there.

The linguistic decoder is not a banking tool. Banking is where we proved it works.

---

## The One Thing We Would Change

The training data has no `wait_and_observe` examples, which causes the model to start each episode with too much action bias. The RL signal on restraint is weak until many episodes in. Five or ten examples of "account is healthy, DPD is zero, correct action is wait" would not corrupt the policy. They would just make the first three episodes less noisy. We would add them.

---

## Closing

Qwen2.5-1.5B is not a capable reasoner by default. It hallucinates action names. Its JSON formatting degrades under pressure. It has no sense of what a moratorium means across a 36 month loan cycle.

After 28 training episodes, roughly 2,520 environment steps and 75 minutes on a T4, it develops a policy that concentrates the right tools on the right account types, avoids catastrophic network effects, and improves outcomes over a random baseline.

That improvement did not come from a larger model. It did not come from more labeled data. It came from building a world where the model had to live with the consequences of its decisions across time.

People are always signaling more than they say. Some understate. Some overstate. The gap between the signal and the surface is where the real information lives. Teaching a small model to read that gap, from reward signal alone, with no human labeling the ground truth, is what this environment is for.

The world is the teacher. The reward is the exam. The model just has to show up.

---

*Built with Qwen2.5-1.5B · GRPO · KL anchor · Entropy regularization · OpenEnv compliant · Trained on a single Colab T4*

*Repository: [github.com/di35117/msme_env](https://github.com/di35117/msme_env)*
