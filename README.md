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

A reinforcement learning environment where an LLM learns to decode hidden state from language and behavior over time, then select the right intervention action.

This project is built around a concrete **MSME + startup credit demo** (India), while using a modular architecture that can generalize to other linguistic-decoding domains.

---

## Why This Exists

In high-stakes communication, people rarely state reality directly.

- MSME borrowers may **understate** stress.
- Startup founders may **overstate** health.
- Surface text can be misleading without behavioral and temporal context.

So this is not sentiment classification and not a chatbot.
It is a sequential decision problem with hidden state and delayed outcomes.

---

## What The Agent Learns

At each step, the agent:

1. observes messages + behavioral proxies,
2. chooses a policy action,
3. receives step reward and delayed episode reward,
4. updates behavior over episodes.

The learning target is robust linguistic decoding under partial observability.

---

## Architecture

### High-Level Flow

```text
Agent Policy (LLM)
        |
        v
OpenEnv Server (`server/app.py`)
        |
        v
Environment Core (`server/msmeEnv_environment.py`)
        |
        v
Domain Adapter Registry (`domains/__init__.py`)
        |
        v
MSME+Startup Adapter (`domains/msme_startup/adapter.py`)
        |
        +--> World generation (`world_generator.py`)
        +--> Reward logic (`reward.py`)
        +--> Network effects (`network.py`)
        +--> Message generation (`message_generator.py`)
        +--> Memory updates (`memory.py`)
```

### Design Principle

- **Core environment** handles episode lifecycle, state orchestration, OpenEnv contracts.
- **Domain adapter** encapsulates domain-specific semantics.
- **Domain logic modules** keep reward/world/network/message behavior explicit and testable.

This lets you keep a strong concrete demo while preserving extensibility.

---

## Domain Design (Current Demo)

Current active domain: `msme_startup`

- 20 MSME accounts + 10 startup accounts
- 36-month horizon
- 21+ action types
- two interacting topologies:
  - MSME cluster contagion
  - startup ecosystem propagation

This creates asymmetric decoding pressure:
- understatement vs overstatement
- same text pattern, different latent meaning by speaker profile and behavior.

---

## Reward Design

### Step Reward

Immediate feedback for action quality:
- positive for appropriate intervention and useful verification,
- negative for wrong tool usage, missed distress, avoidable cascades.

### Episode Reward

Hard-number objective (no LLM judge), combining:
- NPA/default behavior,
- recovery quality,
- relationship/trust preservation,
- tool appropriateness.

---

## OpenEnv Compliance

The environment remains OpenEnv-compliant:

- manifest: `openenv.yaml`
- app entrypoint: `server.app:app`
- standard contracts: `reset`, `step`, `state`
- compatible server wiring in `server/app.py`

---

## Training + Evaluation Workflow

### 1) Run random baseline

```bash
py -3 scripts/run_baseline_eval.py --episodes 30 --output artifacts/baseline_rewards.json
```

### 2) Train policy

```bash
py -3 train_grpo.py --episodes 50 --output_dir msme_rl_checkpoints
```

### 3) Generate judge artifacts

```bash
py -3 scripts/generate_judge_artifacts.py --training_json msme_rl_checkpoints/reward_curve.json --baseline_json artifacts/baseline_rewards.json --output_dir artifacts
```

### 4) Deterministic fixed-seed eval

```bash
py -3 scripts/run_deterministic_eval.py --seed 123 --episodes 5 --output artifacts/deterministic_eval.json
```

### 5) Validate domain registry wiring

```bash
py -3 scripts/check_domain_registry.py
```

### 6) Run baseline comparison report (no training required)

```bash
py -3 scripts/eval.py --episodes 5 --output artifacts/eval_report.json
```

Optional if pytest is installed:

```bash
py -3 -m pytest tests/test_domain_registry.py
```

### 7) Run pre-submit readiness checker

```bash
py -3 scripts/pre_submit_check.py
```

---

## Artifact Pack For Judges

Commit these generated files:

- `artifacts/training_reward_curve.png`
- `artifacts/training_loss_curve.png`
- `artifacts/reward_distribution_base_vs_trained.png`
- `artifacts/per_episode_base_vs_trained.png`
- `artifacts/judge_summary.json`
- `artifacts/judge_manifest.json`
- `artifacts/deterministic_eval.json`
- `artifacts/baseline_rewards.json`
- `artifacts/eval_report.json`

These cover the typical judging asks:
- reward improvement,
- policy loss behavior,
- base-vs-trained evidence,
- reproducibility manifest.

---

## Project Structure

```text
msmeEnv/
├── README.md
├── openenv.yaml
├── pyproject.toml
├── __init__.py
├── client.py
├── models.py
├── train_grpo.py
├── world_generator.py
├── reward.py
├── network.py
├── memory.py
├── message_generator.py
│
├── server/
│   ├── __init__.py
│   ├── app.py
│   └── msmeEnv_environment.py
│
├── domains/
│   ├── __init__.py
│   ├── base.py
│   └── msme_startup/
│       ├── __init__.py
│       └── adapter.py
│
├── scripts/
│   ├── run_baseline_eval.py
│   ├── eval.py
│   ├── run_deterministic_eval.py
│   ├── generate_judge_artifacts.py
│   ├── check_domain_registry.py
│   └── pre_submit_check.py
│
└── tests/
    └── test_domain_registry.py
```

---

## Local Quick Start

```bash
docker build -t linguistic-decoding-env:latest -f server/Dockerfile .
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

---

## Roadmap

- Add more domain adapters (compliance, support escalation, negotiation).
- Add deterministic benchmark suites per domain.
- Add side-by-side policy comparison dashboards in Space UI.

---

## Notes

- `world_generator.py` is still actively used (via the domain adapter).
- `train_grpo.py` remains the main training entrypoint at repo root.
- The adapter layer was added to generalize architecture without breaking current behavior.

## Evaluation Files (What each does)

- `scripts/eval.py`: compares random vs heuristic baselines and writes `artifacts/eval_report.json`.
- `scripts/run_baseline_eval.py`: generates baseline episode rewards only (`baseline_rewards.json`).
- `scripts/run_deterministic_eval.py`: fixed-seed reproducibility probe (`deterministic_eval.json`).
- `scripts/generate_judge_artifacts.py`: turns reward/loss JSON into judge-facing plots and summary files.

## Do We Need `inference.py`?

Not required for judging.

You only need an `inference.py` if you want a dedicated script to run a saved checkpoint policy for demo or offline comparison.  
For current submission goals, environment server + eval scripts are sufficient.
