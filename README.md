---
title: Linguistic Decoding RL Environment
emoji: "🧠"
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - linguistic-decoding
---

# Linguistic Decoding RL

An OpenEnv environment where a small language model learns to decode hidden distress from language and behavior in partially observable financial communication.

## Problem Statement

Many borrowers do not state financial risk directly.
They hide it through communication style:

- **Understatement pattern:** "Small issue, will pay soon" while stress is high.
- **Overconfidence pattern:** "Great momentum, exciting pipeline" while runway is collapsing.

The core task is **linguistic decoding under hidden state**, not generic sentiment classification.
The agent must infer latent risk from language plus behavioral evidence and choose the right intervention over multiple steps.

## Why This Is an RL Problem

This task needs sequential decision making:

- The true financial state is hidden.
- The agent acts, then sees delayed consequences.
- Wrong actions can create cascading future risk.
- Reward is sparse if we only score at the end, so we combine step and episode rewards.

A static classifier cannot optimize intervention strategy across a horizon. RL can.

## Environment Design

The environment is structured as a **core decoder + domain sub-environments**:

### 1) Core Environment: Linguistic Decoding

Shared abstractions across all domains:

- Observation:
  - `message_text`
  - `speaker_profile`
  - `behavioral_signals`
  - `recent_actions`
  - `time_step`
- Hidden state:
  - true risk and trajectory (not visible to agent)
- Actions:
  - ask clarification, verify signal, soft reminder, firm reminder, restructure, escalate
- Transition:
  - hidden state evolves based on action quality and account dynamics

### 2) Sub-Environment A: MSME Module

- Distress often appears as understatement.
- Signals emphasize payment cadence, GST regularity, call response, and cluster effects.

### 3) Sub-Environment B: Startup Module

- Distress often appears as overconfidence.
- Signals emphasize runway proxies, investor update cadence, hiring activity, and ecosystem effects.

This split keeps the science focused (linguistic decoding) while preserving real-world specificity.

## Reward Design

### Step Rewards

- Positive for actions that reveal hidden state early and improve repayment discipline.
- Negative for inappropriate tool usage, missed distress signals, and avoidable escalation.

### Episode Rewards

Weighted objective over:

- final default/NPA rate
- recovery rate
- relationship preservation
- tool appropriateness by account type

## Training Setup

- Base model: small instruct LLM (for example, 1B-2B class)
- RL method: GRPO with TRL
- Environment API: OpenEnv-compliant `reset`, `step`, `state`
- Deployment target: Hugging Face Space

## Evaluation Plan

To satisfy judging requirements, we report:

- reward curve over episodes
- loss curve over training
- before vs after policy behavior on fixed deterministic scenarios
- baseline comparison (untrained or random policy)

## Why This Scope Is Strong for Hackathon

This framing reduces execution risk while staying ambitious:

- It keeps a clear novel claim: **asymmetric linguistic decoding under hidden state**.
- It avoids overloading the first milestone with too many world mechanics.
- It still allows expansion: MSME and startup remain first-class sub-environments.

## Alignment With OpenEnv Hackathon Criteria

- **Environment Innovation (40%)**: Novel partially observable communication-to-action loop.
- **Storytelling (30%)**: Simple and memorable narrative: decode two opposite communication masks.
- **Showing Improvement (20%)**: Deterministic trap scenarios make before/after behavior obvious.
- **Reward + Pipeline (10%)**: Verifiable reward components and reproducible training script.

## Deliverables Checklist

- [ ] OpenEnv-compliant environment hosted on HF Space
- [ ] Training script (TRL/Unsloth) runnable end-to-end
- [ ] Committed reward/loss plots (`.png` or `.jpg`)
- [ ] README with links to Space, training notebook/script, and short demo/writeup
- [ ] Short demo (<=2 minutes) showing clear learned behavior shift

## Quick Start (Local)

```bash
docker build -t linguistic-decoding-env:latest -f server/Dockerfile .
uvicorn server.app:app --reload
```

## Project Structure

```text
msmeEnv/
├── README.md
├── openenv.yaml
├── train_grpo.py
├── client.py
├── models.py
├── domains/
│   ├── base.py
│   └── msme_startup/
│       └── adapter.py
├── scripts/
│   ├── run_baseline_eval.py
│   └── generate_judge_artifacts.py
└── server/
    ├── app.py
    └── msmeEnv_environment.py
```

## Notes

- `train_grpo.py` stays at repo root as the primary training entrypoint.
- `scripts/run_baseline_eval.py` and `scripts/generate_judge_artifacts.py` are required for judge-ready evidence (base vs trained and reproducible artifact manifests).
- Keep `openenv.yaml` pointing to `server.app:app` to preserve OpenEnv compliance.
