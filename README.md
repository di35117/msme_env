# MSME Linguistic Decoder RL Environment

This repository implements an OpenEnv-compatible reinforcement learning environment for a
domain-agnostic linguistic decoding agent, with MSME lending as the demonstration domain.

The goal is not to build a generic chatbot. The goal is to train a policy that learns how to act on
behavioral language signals under uncertainty, using verifiable environment rewards.

## Problem Framing

- **Core module:** Domain-agnostic linguistic decoding (intent, distress, credibility, risk cues)
- **Demo domain:** MSME + startup relationship-management decisions
- **Training objective:** Reduce defaults while preserving trust and action appropriateness

## Why This Is RL (Not Prompting)

- **State:** Portfolio observables, memory context, network alerts, month progression
- **Action:** Structured RM interventions (`action_type`, `account_id`, `parameters`)
- **Reward:** Step + episode rewards from deterministic, auditable environment logic
- **Episode:** Multi-step 36-month simulation horizon
- **Policy update:** SFT warm start + GRPO-style post-training loop

## Environment Highlights

- 20 MSME + 10 startup accounts in every episode
- Hidden state vs observable proxies
- Two linguistic behavior regimes:
  - MSME understatement in local-language communication
  - Startup overstatement in pitch-style communication
- Cluster/ecosystem contagion and cross-contamination effects
- Three-tier memory: working + episodic + semantic

## Project Structure

- `server/msmeEnv_environment.py`: environment step/reset logic
- `world_generator.py`: scenario generation and observables
- `reward.py`: verifiable reward engineering
- `memory.py`: three-tier memory system
- `network.py`: contagion and network propagation
- `train_grpo.py`: SFT warm start + RL training loop
- `eval.py`: baseline vs policy evaluation on fixed seeds
- `reward_audit.py`: reward-hacking diagnostics and audit checks
- `FAQ_MAPPING.md`: explicit mapping to hackathon FAQ expectations

## Quick Start

```bash
uv sync
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
python train_grpo.py --episodes 10 --port 8000 --output_dir ./msme_rl_checkpoints
python eval.py --episodes 5 --seed 42
```

## Evaluation Outputs

`eval.py` emits policy-level metrics useful for judging:

- NPA rate
- Recovery rate
- Relationship score
- Tool appropriateness
- Suspicious shortcut indicators

## Deployment

OpenEnv manifest:

- `openenv.yaml`
- runtime: FastAPI
- app: `server.app:app`

Use `openenv push` to deploy to Hugging Face Spaces.
