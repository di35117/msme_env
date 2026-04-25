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

## Anti-Reward-Hacking Design

A reward function is only as good as the loopholes it doesn't have. The reward
in `reward.py` is built around a simple principle: **every shortcut that would
make the agent look good without actually helping a borrower has an explicit
counter-penalty**. Below are the four mechanisms a judge can verify by reading
`reward.py` directly — all values are constants in the same file.

### 1) Cluster-cascade penalty (prevents over-aggression)

`STEP_REWARDS["cluster_cascade_default"] = -0.25`

MSME borrowers in this environment are connected by a cluster topology
(`network.py`). If the agent over-uses coercive tools (e.g. `initiate_sarfaesi`)
on a borrower with high `cluster_centrality`, it can trigger a *cascade
default* across linked accounts — and the agent eats a `-0.25` step penalty
**per cascading account**, dwarfing the `-0.18` it would have gotten for letting
that single account become NPA. This rules out the "be maximally aggressive,
recover the loan, ignore the rest of the cluster" strategy.

A small *information bonus* (`+0.03`) is also given for running
`verify_gst_returns`, `pull_bank_statements`, or `check_industry_cluster_stress`
**before** acting on a high-centrality account, so the gradient nudges toward
"check first, then decide" instead of trigger-happy behavior.

### 2) SARFAESI-on-startup penalty (prevents wrong-tool abuse)

`STEP_REWARDS["sarfaesi_used_on_startup"] = -0.15`

SARFAESI is a collateral-recovery tool designed for asset-backed MSMEs. Using
it on a startup is *legally permissible but operationally insane* — startups
have no real collateral, founders walk, and ecosystem trust collapses. The
agent therefore incurs a `-0.15` step penalty *in addition to* the normal
account-NPA penalty whenever it fires SARFAESI on an `account_type == "startup"`
(see `compute_step_reward`, line 87). The correct startup-side action,
`schedule_investor_meeting_check_in`, carries a `+0.10` bonus when used on a
genuinely stressed startup (`investor_meeting_triggered_bridge`). The reward
gap (`+0.10` vs `-0.15` ≈ 0.25 swing per step) makes wrong-tool behavior
strictly dominated by right-tool behavior in expectation.

### 3) Episode-level shortcut penalty (`_compute_shortcut_penalty`)

`reward.py:350` runs four deterministic checks at episode end that catch the
most common RL exploits we saw during ablations:

| Pathology               | Trigger                                          | Coefficient |
|-------------------------|--------------------------------------------------|-------------|
| **No-op farming**       | `wait_and_observe` ratio above 30%               | `(ratio - 0.30) * 0.50` |
| **Malformed-JSON abuse**| `format_error` ratio                             | `ratio * 0.60` |
| **Action spamming**     | One action used in more than 35% of all steps    | `(ratio - 0.35) * 0.40` |
| **Account thrashing**   | More than 70% target-switches between steps      | `(ratio - 0.70) * 0.25` |

The total is capped at `0.25` and subtracted from the episode reward, so even
a perfect raw score is wiped out by extreme degenerate behavior. This prevents
the most common GRPO failure mode: the model finds a single line of safe text
that always parses, repeats it 60 times, and looks "correct" without making
any actual decisions.

### 4) Action-frequency cap on positive rewards

In `compute_episode_reward` (line 299) the positive contribution from any
single action type is capped after the third use:

```python
if action_frequency[action_type] <= 3:
    R += positive_reward
```

This means even if the model finds a genuinely high-reward action, it cannot
spam it 60 times to inflate the episode return — the 4th, 5th, ... uses
contribute 0 to the positive side while still being subject to negative
penalties. Combined with mechanism (3), this means **the only way to get a
high episode reward is to use a diverse set of contextually-appropriate
actions** — exactly the behavior the environment is supposed to teach.

### Auditability

The function `_build_anti_cheat_metrics` (`reward.py:386`) emits per-episode
counters for every one of the above (no-op rate, malformed rate, dominant
action share, switch ratio, SARFAESI-on-startup count, etc.) into the episode
summary. These are the same numbers a judge can re-derive by replaying any
saved episode log — so the anti-hacking claim is not just a design statement,
it is a runtime invariant exposed in `judge_summary.json`.

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

- `msme_rl_checkpoints/reward_curve.png` — headline reward curve
- `msme_rl_checkpoints/training_metrics.png` — **multi-metric dashboard (Reward / Loss / KL / Entropy / Parse-failure %)**
- `msme_rl_checkpoints/loss_curve.png`
- `msme_rl_checkpoints/reward_curve.json` — raw per-episode metrics for re-rendering
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
- **multiple monitored metrics (FAQ Q17): KL vs SFT reference + completion-token entropy + parse-failure % alongside reward**,
- base-vs-trained evidence,
- reproducibility manifest.

### What the multi-metric dashboard shows

`training_metrics.png` is a 2x3 grid produced by `_save_reward_plot` in
`train_grpo.py`:

| Panel | Metric                                | What a healthy run looks like |
|-------|---------------------------------------|-------------------------------|
| (0,0) | Per-episode reward                    | Trends upward over episodes  |
| (0,1) | Rolling-mean reward                   | Smoothed upward trend         |
| (0,2) | GRPO policy loss (per-episode mean)   | Hovers around 0, not exploding |
| (1,0) | KL vs frozen SFT reference            | Bounded — the KL anchor (`KL_COEF=0.05`) keeps the policy from drifting off the format manifold |
| (1,1) | Completion token entropy              | Stays clearly above 0 — entropy bonus (`ENT_COEF=0.01`) prevents mode collapse |
| (1,2) | Parse-failure %                       | Drops to ≈ 0% within the first few episodes thanks to the JSON prefill + extractor fallback |

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
