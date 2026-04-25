"""
Run deterministic fixed-seed evaluation scenarios for reproducible checks.

This script creates a stable JSON report suitable for README evidence and
automated evaluation references.
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


def _msme_trap_action() -> MSMERLAction:
    return MSMERLAction(
        action_type="verify_gst_returns",
        account_id=7,
        parameters={},
        reasoning="deterministic msmE trap probe",
    )


def _startup_trap_action() -> MSMERLAction:
    return MSMERLAction(
        action_type="request_investor_update_meeting",
        account_id=22,
        parameters={},
        reasoning="deterministic startup optimism trap probe",
    )


def run_eval(seed: int, episodes: int, max_steps: int) -> Dict:
    random.seed(seed)
    env = MSMERLEnvironment()
    episode_rows: List[Dict] = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        # Apply fixed first two probe actions for reproducibility
        obs = env.step(_msme_trap_action())
        msmE_outcome = (obs.last_action_result or {}).get("outcome")
        obs = env.step(_startup_trap_action())
        startup_outcome = (obs.last_action_result or {}).get("outcome")

        # Continue with deterministic simple policy
        steps = 2
        while not obs.done and steps < max_steps:
            obs = env.step(
                MSMERLAction(
                    action_type="wait_and_observe",
                    account_id=((steps % 30) + 1),
                    parameters={},
                    reasoning="deterministic filler policy",
                )
            )
            steps += 1

        done = bool(obs.done)
        breakdown = (obs.last_action_result or {}).get("episode_reward_breakdown")
        if done and breakdown:
            total_reward = float(breakdown["total"])
            reward_source = "terminal_episode_reward"
        else:
            # Explicit fallback so partial rollouts are distinguishable in reports.
            total_reward = float(obs.episode_reward_so_far)
            reward_source = "partial_cumulative_reward"
        row = {
            "episode": ep,
            "steps": steps,
            "done": done,
            "msme_probe_outcome": msmE_outcome,
            "startup_probe_outcome": startup_outcome,
            "total_reward": total_reward,
            "reward_source": reward_source,
            "npa_rate": breakdown.get("npa_rate") if breakdown else None,
            "recovery_rate": breakdown.get("recovery_rate") if breakdown else None,
        }
        episode_rows.append(row)
        print(
            f"det-eval ep {ep}/{episodes}: reward={total_reward:.4f} "
            f"(done={done}, source={reward_source})"
        )

    avg_reward = sum(r["total_reward"] for r in episode_rows) / len(episode_rows)
    terminal_done_count = sum(1 for r in episode_rows if r["done"])
    return {
        "seed": seed,
        "episodes": episodes,
        "max_steps": max_steps,
        "avg_reward": avg_reward,
        "terminal_done_count": terminal_done_count,
        "rows": episode_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic evaluation scenarios.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=1080)
    parser.add_argument("--output", type=Path, default=Path("artifacts/deterministic_eval.json"))
    args = parser.parse_args()

    report = run_eval(args.seed, args.episodes, args.max_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"saved deterministic eval -> {args.output}")


if __name__ == "__main__":
    main()

