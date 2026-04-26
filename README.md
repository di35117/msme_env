# 🧠 Linguistic Decoding RL

**A reinforcement learning environment where an LLM agent learns to decode hidden financial stress from biased natural language and behavioral signals — then selects the correct intervention action.**

---

## Why This Exists

In high-stakes lending and investment, entities rarely disclose reality directly.

- **MSME borrowers** tend to *understate* stress — masking overdue receivables, stretched payments, and declining margins behind cautiously optimistic language.
- **Startup founders** tend to *overstate* health — inflating ARR, projecting pipeline certainty, and reframing burn as strategic investment.

Surface text is not enough. A capable analyst reads *between the lines* — cross-referencing what is said with how, when, and how completely it is said. This environment formalizes that skill as a learnable RL problem: given a stream of biased messages and behavioral signals, infer the true hidden state and act on it before it deteriorates further.

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

The answer to all three is yes.

### Reward Convergence & Loss Stability

| Reward Curve | Training Loss |
|:---:|:---:|
| ![reward](./req/reward.jpg) | ![loss](./req/loss.jpg) |
| Mean episode reward across 30 training iterations | Policy loss and KL divergence across training |

Reward climbs steadily through the first 15 episodes, with the sharpest gains coming from the agent learning to avoid the highest-penalty actions — particularly `do_nothing` when behavioral signals are deteriorating. Loss stabilizes by mid-training, and KL divergence stays within acceptable bounds throughout, indicating the policy is updating without collapsing.

### Baseline vs. Trained Distribution

![Base vs Trained](./req/based_vs_trained.jpg)

The untrained baseline clusters around low and negative rewards — it has no domain prior and defaults to generic, low-commitment actions regardless of signals. After training, the distribution shifts meaningfully toward positive reward, with the mass of episodes landing in the `+0.3 → +0.8` range. The long negative tail shrinks but does not disappear — critical misses on heavily biased `loss`-level entities remain the hardest problem.

### Action Distribution & Domain Strategy

![Action Distribution](./req/inferenece_action_distribtuon.jpg)

Post-training, the agent's action choices are no longer uniform. Two distinct strategies emerge by domain:

- **MSME**: the agent leans toward `trigger_field_visit` and `request_audited_financials` — tools that bypass linguistic framing and force direct evidence. This compensates for the understatement bias: when an MSME borrower sounds fine, the agent has learned not to take that at face value.
- **Startup**: the agent escalates more aggressively — higher use of `escalate_to_credit_committee` and `offer_restructuring`. Overstatement bias means the agent treats positive language as a weak signal and falls back on behavioral proxies to drive its decision.

### MSME vs. Startup Decoding Accuracy

![MSME vs Startup](./req/inference_msme_vs_startup.jpg)

MSME profiles are decoded more accurately than startup profiles at this training scale. Understatement produces a more consistent signal — the mismatch between cautious language and deteriorating behavioral proxies is a learnable pattern. Startup overstatement is harder: the language is often genuinely sophisticated and the behavioral signals are noisier. Further training and more startup-domain episodes will likely close this gap.

### Qualitative Before / After

| Inference Snapshot A | Inference Snapshot B |
|:---:|:---:|
| ![Before/After 1](./req/inference_before_after.png.jpg) | ![Before/After 2](./req/inference_before_after_2.jpg) |
| Agent reasoning pre-training: generic, non-committal, misses behavioral signals | Agent reasoning post-training: names hidden state explicitly, justifies action against observed drift |

The qualitative shift is the most telling result. Before training, the agent produces hedged, non-committal reasoning and frequently chooses `schedule_follow_up` regardless of signal severity. After training, it explicitly names a stress level estimate, cites specific behavioral evidence (latency, document gaps, avoidance), and selects an action proportionate to the inferred risk.

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