"""
Run a random-policy baseline and save rewards for comparison plots.

Usage:
  py -3 scripts/run_baseline_eval.py --episodes 30 --output artifacts/baseline_rewards.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from server.msmeEnv_environment import MSMERLEnvironment
    from models import MSMERLAction
except (ModuleNotFoundError, ImportError):
    from msmeEnv.server.msmeEnv_environment import MSMERLEnvironment  # type: ignore
    from msmeEnv.models import MSMERLAction  # type: ignore


ACTION_TYPES: List[str] = [
    "send_empathetic_reminder",
    "send_firm_reminder",
    "send_legal_notice_section13",
    "call_promoter_founder",
    "call_guarantor_investor",
    "conduct_cluster_ecosystem_visit",
    "grant_moratorium",
    "restructure_emi",
    "offer_eclgs_topup",
    "offer_bridge_loan_extension",
    "accept_partial_payment",
    "waive_penal_interest",
    "initiate_sarfaesi",
    "refer_to_recovery_agent",
    "file_drt_case",
    "offer_one_time_settlement",
    "verify_gst_returns",
    "pull_bank_statements",
    "check_industry_cluster_stress",
    "request_investor_update_meeting",
    "check_startup_ecosystem_signals",
    "wait_and_observe",
]


def make_random_action() -> MSMERLAction:
    action_type = random.choice(ACTION_TYPES)
    account_id = random.randint(1, 30)
    params: Dict = {}
    if action_type in ("grant_moratorium", "offer_bridge_loan_extension"):
        params["months"] = random.choice([1, 2, 3])
    elif action_type == "restructure_emi":
        params["new_amount"] = random.choice([50000, 75000, 100000])
    elif action_type in ("accept_partial_payment", "offer_one_time_settlement"):
        params["amount"] = random.choice([50000, 100000, 200000])
    return MSMERLAction(
        action_type=action_type,
        account_id=account_id,
        parameters=params,
        reasoning="random baseline",
    )


def run_baseline(episodes: int, max_steps: int, seed: int) -> List[float]:
    random.seed(seed)
    env = MSMERLEnvironment()
    rewards: List[float] = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            obs = env.step(make_random_action())
            done = bool(obs.done)
            steps += 1

        breakdown = (obs.last_action_result or {}).get("episode_reward_breakdown")
        episode_reward = breakdown["total"] if breakdown else float(obs.episode_reward_so_far)
        rewards.append(float(episode_reward))
        print(f"baseline episode {ep}/{episodes}: reward={episode_reward:.4f} steps={steps}")

    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline random-policy evaluation.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max_steps_per_episode", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline_rewards.json"))
    args = parser.parse_args()

    rewards = run_baseline(args.episodes, args.max_steps_per_episode, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"episodes": list(range(1, len(rewards) + 1)), "rewards": rewards}, f, indent=2)
    print(f"saved baseline rewards -> {args.output}")


if __name__ == "__main__":
    main()

