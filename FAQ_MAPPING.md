# Hackathon FAQ Mapping

This document maps FAQ expectations to concrete implementation in this repository.

## RL Environment vs Chatbot

- **FAQ 1, 5, 6, 7:** RL loop with `reset -> step -> reward -> update`
  - Implemented in `server/msmeEnv_environment.py`, `server/app.py`, `client.py`.
- **Why it matters:** Demonstrates true agent-environment interaction, not static prompting.

## Verifiable Rewards (RLVR)

- **FAQ 2, 3, 4, 26, 27:** Deterministic reward components
  - Implemented in `reward.py` (`compute_step_reward`, `compute_episode_reward`).
- **Why it matters:** Judges can audit reward logic directly.

## Reward-Hacking Mitigation

- **FAQ 12, 13, 43, 44, 57:** Shortcut detection and penalties
  - Implemented in `reward.py` (`detect_suspicious_shortcuts`) and `reward_audit.py`.
- **Why it matters:** Prevents proxy optimization from dominating real task outcomes.

## Curriculum / RLVE-style Adaptation

- **FAQ 14, 22, 23, 35:** Adaptive difficulty hooks
  - Implemented in `server/msmeEnv_environment.py` (`_compute_adaptive_difficulty`)
  - Integrated with `world_generator.py` (`difficulty_override`).
- **Why it matters:** Keeps tasks near capability frontier and avoids early stagnation.

## Warm Start Before RL

- **FAQ 16, 45:** SFT warm-start before RL updates
  - Implemented in `train_grpo.py` (`run_sft_warm_start`).
- **Why it matters:** Avoids near-zero-success rollouts at training start.

## Monitoring Beyond Headline Reward

- **FAQ 17, 52:** Additional training diagnostics
  - Implemented in `train_grpo.py` (`training_metrics.json`):
    - parse errors
    - distinct action count
    - top action ratio
    - reward breakdown components
- **Why it matters:** Helps detect drift and silent failure modes.

## Baseline Evaluation

- **FAQ 53, 54, 56:** Baseline policy evaluation before scaling
  - Implemented in `eval.py` (random vs heuristic policy comparison).
- **Why it matters:** Produces judge-friendly before/after evidence.

## Domain-Agnostic Decoder Positioning

- **FAQ fit:** The same RL mechanics can be transferred across domains.
  - MSME is used as demo deployment context.
  - Linguistic decoding architecture is described in `README.md`.
- **Why it matters:** Aligns innovation narrative with practical verifiability.
