# 🧠 Linguistic Decoding RL

**A reinforcement learning environment where an LLM agent learns to decode hidden financial stress from biased natural language and behavioral signals — then selects the correct intervention action.**

---

## Why This Exists

Imagine a loan officer reviewing a borrower's monthly check-in message. The borrower writes:

> *"Things have been a bit slow this quarter, but we're managing. Expect things to pick up once the season turns."*

On the surface — nothing alarming. But the loan officer also notices: this reply came after a 12-day silence, the last two meeting requests were declined, and only 40% of the requested documents were submitted. An experienced analyst reads all of this together and flags the account for a field visit. A junior analyst — or a naive model — reads the message, sees no explicit distress, and moves on.

That gap is the problem this environment is built to close.

In high-stakes lending and investment, entities rarely disclose stress directly. They manage perception — sometimes consciously, sometimes not.

- **MSME borrowers** tend to *understate* — masking overdue receivables, stretched supplier payments, and declining margins behind cautiously optimistic language. Admitting stress feels like inviting scrutiny they cannot afford.
- **Startup founders** tend to *overstate* — inflating ARR, projecting pipeline certainty, and reframing high burn as deliberate, strategic investment. The fundraising culture rewards confidence, even when the numbers don't support it.

The result is that the text alone is routinely misleading. What gives the true state away is the *combination* — what is said, how it is framed, when the reply came, what was left out, and how that pattern has shifted over the last several interactions.

---

## The Problem with Current Approaches

Most NLP-based risk systems treat this as a classification problem: feed the message into a model, predict a risk label, done. This fails in practice for a few reasons.

**First, language is adversarially biased.** The entities being assessed have strong incentives to sound healthy. A model trained on text alone learns to classify confident language as low-risk — which is exactly backwards when the speaker is a founder who has just been told their Series B fell through.

**Second, a single message is not enough context.** Stress rarely announces itself in one message. It leaks through patterns — a gradual increase in response latency, a slow decline in document completion, a shift in sentiment from collaborative to defensive. You need to track across time, not just classify in the moment.

**Third, static models cannot adapt to new intervention outcomes.** A classification model tells you the risk label. It cannot tell you whether to request financials, trigger a field visit, or escalate — and it cannot learn from what happened the last time you tried each of those actions on a similar profile.

---

## Why Reinforcement Learning

RL is a natural fit here because the problem is fundamentally sequential and consequential.

The agent does not just need to *identify* the hidden state — it needs to *act* on that identification, observe what happens, and update its understanding. A field visit either confirms or contradicts the stress estimate. A restructuring offer either stabilizes the entity or reveals that the situation is worse than it appeared. These outcomes are feedback, and RL is the framework that knows how to learn from them.

More specifically, GRPO trains the agent to optimize for the *quality of its reasoning and action selection over a full episode* — not just the next step. This matters because the consequences of misclassifying a `doubtful` entity as `watch` and doing nothing do not show up immediately. They compound. The reward structure is designed to reflect that: heavy penalties for inaction under confirmed stress, episodic bonuses for trajectories that actually reduce risk.

The environment also forces the agent to generalize across two domains with opposite bias directions — MSME understatement and startup overstatement — which prevents it from developing a naive prior like "positive language equals low risk."

---

## A Thought We Stumbled Into

While designing this, something broader surfaced.

The core skill we are training — reading intent and true state from biased, incomplete, socially managed language — is not specific to credit risk. It is the same skill a good doctor uses when a patient downplays pain. It is the same skill a manager uses when a team member says "I'm fine" and means something else entirely. It is what a therapist does, what a negotiator does, what any person who is genuinely good at working with other people does constantly and often unconsciously.

What we are really building is an environment for training **linguistic intent decoding** — the ability to model the gap between what someone says and what is actually true for them. And that raises an interesting question: could this generalize?

Could a model trained in environments like this — across many domains, many speaker types, many forms of bias — develop something closer to genuine social intelligence? Not sentiment analysis, not tone classification, but actual inference about hidden human state from observable language and behavior?

We think this direction is worth pursuing seriously. Language models are already extraordinarily capable at generating human-like text. What they are not yet good at is the inverse problem — reading the human on the other side of the conversation with the same depth that an experienced person would. Environments like this one are, we think, a step toward closing that gap.

---

## What the Agent Learns

At each step, the agent:

1. Receives a **natural language message** from the entity — written with speaker bias baked in
2. Observes **behavioral proxies** alongside the text: response latency, document completion rate, meeting cancellations, escalation avoidance score
3. Forms an estimate of the **hidden stress level**: `healthy → watch → substandard → doubtful → loss`
4. Selects a **policy action** from the intervention menu
5. Receives a **step reward** for action appropriateness and an **inference bonus** for correctly identifying the hidden state
6. Accumulates experience across episodes and refines its policy via GRPO

The ground truth is never revealed during the episode. The agent must earn its estimate entirely from what it can observe.

---

## Environment Design

### Domains

| Domain | Bias Direction | Sectors |
|--------|---------------|---------|
| **MSME** | Understatement | retail, manufacturing, agri-processing, logistics, hospitality |
| **Startup** | Overstatement | fintech, edtech, healthtech, SaaS, consumer |

The two domains are trained together intentionally. MSME and startup entities have *opposite* bias directions — a single-domain agent develops the wrong prior. Training across both forces the agent to condition on domain context before interpreting linguistic tone.

### Stress Levels (Hidden State)

```
healthy → watch → substandard → doubtful → loss
```

Each level maps to calibrated financial snapshots, behavioral profiles, and message templates. Speaker bias is injected at generation time to obscure the true level — the gap between what is said and what is real is the core inference challenge.

### Intervention Actions

| Action | Best Used When |
|--------|---------------|
| `request_audited_financials` | Early warning signals present |
| `trigger_field_visit` | Behavioral avoidance + document gaps |
| `offer_restructuring` | Confirmed stress, cooperative entity |
| `escalate_to_credit_committee` | Doubtful classification, high exposure |
| `schedule_follow_up` | Healthy or minor watch signals |
| `flag_for_npa` | Loss classification confirmed |
| `do_nothing` | Genuinely healthy, low-risk entity |

### Reward Structure

- **Step reward** — scored against an action-appropriateness matrix per true stress level (`-1.0 → +1.0`)
- **Inference bonus** — `+0.4` for correctly naming the hidden stress level; partial credit for adjacent-level estimates
- **Critical miss penalty** — `-0.5` for choosing `do_nothing` or `schedule_follow_up` on a `doubtful` or `loss` entity
- **Episode reward** — trajectory bonus for net stress reduction across steps, plus a terminal penalty if the entity ends the episode in a worse state than it started

---

## Architecture

```
Agent Policy (LLM)
        │
        ▼
OpenEnv Server          server/app.py
        │
        ▼
Environment Core        server/msmeEnv_environment.py
        │
        ▼
Domain Adapter Registry domains/__init__.py
        │
        ▼
MSME + Startup Adapter  domains/msme_startup/adapter.py
        │
        ├──▶ world_generator.py      — hidden state synthesis
        ├──▶ message_generator.py    — biased NL message generation
        ├──▶ reward.py               — step + episode reward logic
        ├──▶ network.py              — peer entity contagion effects
        └──▶ memory.py               — cross-step state accumulation
```

---

## Training Results

This is a preliminary run — 30 episodes, with each episode capped at 90 steps. The cap was a deliberate resource efficiency choice: enough steps to observe multi-turn behavioral drift and test whether the agent can track state across a dialogue, without the compute cost of full convergence runs. The goal at this stage was proof of concept — does the reward signal improve, does the policy stabilize, and does the agent develop differentiated strategies across domains.

We fine-tuned a **Qwen 1.5B** model with GRPO on this reinforcement learning task.

The answer to all three is yes.

### Reward Convergence & Loss Stability

| Reward Curve | Training Loss |
|:---:|:---:|
| ![training reward](./artifacts/training_reward.jpg) | ![policy loss](./artifacts/Policy%20Loss.jpg) |
| Mean episode reward across 30 training iterations | Policy loss and KL divergence across training |

Reward climbs steadily through the first 15 episodes, with the sharpest gains coming from the agent learning to avoid the highest-penalty actions — particularly `do_nothing` when behavioral signals are deteriorating. Loss stabilizes by mid-training, and KL divergence stays within acceptable bounds throughout, indicating the policy is updating without collapsing.

### Baseline vs. Trained Distribution

![Per-episode: Base vs Trained](./artifacts/per_episode_base_vs_train.jpg)

The untrained baseline clusters around low and negative rewards — it has no domain prior and defaults to generic, low-commitment actions regardless of signals. After training, the distribution shifts meaningfully toward positive reward, with the mass of episodes landing in the `+0.3 → +0.8` range. The long negative tail shrinks but does not disappear — critical misses on heavily biased `loss`-level entities remain the hardest problem.

### Training Metrics Dashboard

![Training metrics](./artifacts/training_metric.jpg)

This panel shows the RL signal and supporting stability metrics over 30 episodes: episode reward and rolling mean, GRPO policy loss, KL divergence vs anchor, completion token entropy, parse-failure rate, NPA rate per episode, average portfolio trust, and `wait_and_observe` usage.

### Reward Curve (Zoomed)

![Reward curve](./artifacts/reward_curve.png.jpg)

This view isolates the reward trend with a moving average, making it easier to see the step-change in performance in the second half of training.

---

## Training & Evaluation Workflow

### 1. Run Baseline & Train

```bash
# Establish baseline performance (untrained policy)
py -3 scripts/run_baseline_eval.py --episodes 30 --output artifacts/baseline_rewards.json

# Train with GRPO — 30 episodes, 90 steps per episode
py -3 train_grpo.py --episodes 30 --max_steps 90 --output_dir msme_rl_checkpoints
```

### 2. Generate Judge Artifacts

```bash
py -3 scripts/generate_judge_artifacts.py \
    --training_json msme_rl_checkpoints/reward_curve.json \
    --baseline_json artifacts/baseline_rewards.json \
    --output_dir artifacts
```

### 3. Evaluate & Verify

```bash
# Deterministic eval for reproducible submission scoring
py -3 scripts/run_deterministic_eval.py --seed 123 --episodes 5 \
    --output artifacts/deterministic_eval.json

# Pre-submission checklist
py -3 scripts/pre_submit_check.py
```

---

## Project Structure

```
msmeEnv/
├── README.md
├── openenv.yaml
├── pyproject.toml
├── __init__.py
├── train_grpo.py                        # GRPO training loop
├── world_generator.py                   # Hidden state synthesis
├── reward.py                            # Step + episode reward
├── network.py                           # Peer contagion effects
├── memory.py                            # Cross-step accumulation
├── message_generator.py                 # Biased NL message generation
├── server/
│   ├── app.py                           # FastAPI server (OpenEnv)
│   └── msmeEnv_environment.py           # Environment core logic
├── domains/
│   ├── __init__.py                      # Domain adapter registry
│   └── msme_startup/
│       └── adapter.py                   # MSME + Startup adapter
└── scripts/
    ├── run_baseline_eval.py
    ├── eval.py
    ├── generate_judge_artifacts.py
    ├── run_deterministic_eval.py
    └── pre_submit_check.py
```

---

## Key Design Decisions

**Why biased generation?** Real-world language is never neutral. Entities in financial distress have strong incentives to manage perception. Training on clean, unbiased text produces agents that perform well in evaluation and fail in deployment.

**Why behavioral proxies alongside text?** Language alone is gameable. Response latency, document gaps, and meeting avoidance are harder to fake — and often leak the true state precisely when the text is most polished. The agent is rewarded for integrating both channels, not just parsing the message.

**Why GRPO over PPO?** GRPO avoids the value network overhead and handles sparse, delayed reward signals better — which matters here because the episode reward often carries more signal than any individual step reward.

**Why cap episodes at 90 steps?** Full convergence runs are expensive and unnecessary for a preliminary proof. 90 steps is enough to test multi-turn tracking, behavioral drift detection, and domain-conditioned strategy without burning compute on marginal gains at this stage.

**What comes next?** Longer episodes, more startup-domain training data, and a harder bias injection regime where speaker bias is adversarially calibrated to maximally confuse the agent's current policy.

---

## Tags

`openenv` · `reinforcement-learning` · `linguistic-decoding` · `grpo` · `credit-risk` · `llm-agent` · `hidden-state-inference`