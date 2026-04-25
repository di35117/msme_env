# Hackathon FAQ Mapping

This document maps FAQ expectations to concrete implementation in this repository.

## RL environment, not chatbot

- **FAQ 1, 5, 6, 7, 24:** RL loop with `reset -> step -> state/reward`
  - Implemented in `server/msmeEnv_environment.py` and served via `server/app.py`.
- **Why it matters:** Demonstrates multi-step agent-environment interaction, not one-shot prompting.

## Verifiable rewards (RLVR)

- **FAQ 2, 3, 4, 21, 26, 27:** deterministic reward functions and hard metrics
  - Implemented in `reward.py` (`compute_step_reward`, `compute_episode_reward`).
- **Why it matters:** judges can inspect reward logic directly without relying on an LLM judge.

## Reward-hacking awareness and auditing

- **FAQ 12, 13, 43, 44, 57:** explicit reward-hacking mindset
  - Implemented in reward constraints in `reward.py` (tool appropriateness caps/penalties).
  - Episode action-pattern audit helper in `reward_audit.py`.
- **Why it matters:** reduces Goodhart-style proxy exploitation.

## Curriculum and difficulty shaping

- **FAQ 14, 22, 23, 35, 46, 48:** curriculum and dynamic challenge
  - Implemented through start-month curriculum in `server/msmeEnv_environment.py`.
  - Weakness analysis hooks for adversarial episodes in `reward.py` (`analyze_agent_weaknesses`).
- **Why it matters:** keeps training signal informative and avoids early stagnation.

## SFT before RL

- **FAQ 16, 45, 53:** warm-start pipeline
  - Implemented in `train_grpo.py` (`run_sft_warm_start`) before GRPO updates.
- **Why it matters:** improves early rollout quality.

## Baseline and deterministic evaluation

- **FAQ 17, 52, 54, 56:** monitor beyond one metric and validate before scaling
  - `scripts/run_baseline_eval.py`: random-policy baseline rewards.
  - `scripts/eval.py`: random vs heuristic baseline comparison report.
  - `scripts/run_deterministic_eval.py`: fixed-seed deterministic probe report.
- **Why it matters:** provides reproducible evidence and sanity checks.

## Judge artifact generation

- **FAQ 17, 52, 53:** evaluation evidence and monitoring outputs
  - `scripts/generate_judge_artifacts.py` produces reward/loss and base-vs-trained plots plus summary/manifest JSON.
- **Why it matters:** creates judge-friendly, auditable deliverables.

## OpenEnv compliance checks

- **FAQ 6, 7, 18:** standardized environment contracts
  - Manifest in `openenv.yaml`.
  - Server app entrypoint `server.app:app`.
  - Validation script `scripts/check_domain_registry.py`.
- **Why it matters:** ensures portability and compatibility with OpenEnv tooling.

## Generalization narrative

- **FAQ fit:** same RL mechanics can transfer across domains.
  - Domain-adapter interface in `domains/base.py`.
  - Active module in `domains/msme_startup/adapter.py`.
- **Why it matters:** keeps concrete MSME demo while supporting architecture-level innovation.