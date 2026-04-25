"""
Evaluation script for MSME linguistic decoder RL environment.

Compares baseline policies using fixed seeds and reports judge-friendly metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from models import MSMERLAction
    from server.msmeEnv_environment import MSMERLEnvironment
except (ModuleNotFoundError, ImportError):
    from msmeEnv.models import MSMERLAction  # type: ignore
    from msmeEnv.server.msmeEnv_environment import MSMERLEnvironment  # type: ignore


UNIVERSAL_ACTIONS = [
    "send_empathetic_reminder",
    "send_firm_reminder",
    "grant_moratorium",
    "restructure_emi",
    "accept_partial_payment",
    "pull_bank_statements",
    "wait_and_observe",
]
MSME_ACTIONS = UNIVERSAL_ACTIONS + [
    "verify_gst_returns",
    "check_industry_cluster_stress",
    "offer_eclgs_topup",
]
STARTUP_ACTIONS = UNIVERSAL_ACTIONS + [
    "request_investor_update_meeting",
    "check_startup_ecosystem_signals",
    "offer_bridge_loan_extension",
]


def _pick_target(observation: Dict) -> Tuple[int, str, Dict]:
    msme_accounts = observation.get("msme_accounts", [])
    startup_accounts = observation.get("startup_accounts", [])
    combined = msme_accounts + startup_accounts
    if not combined:
        return 1, "msme", {}
    target = sorted(combined, key=lambda x: x.get("dpd", 0), reverse=True)[0]
    return int(target.get("account_id", 1)), str(target.get("account_type", "msme")), target


def random_policy(observation: Dict, rng: random.Random) -> MSMERLAction:
    account_id, account_type, _ = _pick_target(observation)
    action_space = MSME_ACTIONS if account_type == "msme" else STARTUP_ACTIONS
    action = rng.choice(action_space)
    return MSMERLAction(action_type=action, account_id=account_id, parameters={}, reasoning="random baseline")


def heuristic_policy(observation: Dict, rng: random.Random) -> MSMERLAction:
    account_id, account_type, obs = _pick_target(observation)
    dpd = int(obs.get("dpd", 0))

    if account_type == "msme":
        gst = str(obs.get("gst_filing_status", ""))
        if dpd > 45:
            action = "restructure_emi"
        elif "not_filed" in gst or "delay" in gst:
            action = "verify_gst_returns"
        elif dpd > 15:
            action = "grant_moratorium"
        else:
            action = "send_empathetic_reminder"
    else:
        investor_update = str(obs.get("investor_update_sent", ""))
        if dpd > 45:
            action = "request_investor_update_meeting"
        elif "skipped" in investor_update:
            action = "check_startup_ecosystem_signals"
        elif dpd > 15:
            action = "offer_bridge_loan_extension"
        else:
            action = "send_empathetic_reminder"

    return MSMERLAction(action_type=action, account_id=account_id, parameters={}, reasoning="heuristic baseline")


def run_policy(policy_name: str, episodes: int, seed: int, max_steps: int) -> Dict:
    rng = random.Random(seed)
    env = MSMERLEnvironment()
    rows: List[Dict] = []

    for ep in range(episodes):
        obs_obj = env.reset()
        obs = obs_obj.__dict__
        done = False
        steps = 0
        while not done and steps < max_steps:
            if policy_name == "random":
                action = random_policy(obs, rng)
            else:
                action = heuristic_policy(obs, rng)
            obs_obj = env.step(action)
            obs = obs_obj.__dict__
            done = bool(obs.get("done", False))
            steps += 1

        breakdown = (obs.get("last_action_result") or {}).get("episode_reward_breakdown") or {}
        rows.append({
            "episode": ep + 1,
            "reward": float(breakdown.get("total", obs.get("episode_reward_so_far", 0.0))),
            "npa_rate": float(breakdown.get("npa_rate", 1.0)),
            "recovery_rate": float(breakdown.get("recovery_rate", 0.0)),
            "relationship_score": float(breakdown.get("relationship_score", 0.0)),
            "tool_appropriateness": float(breakdown.get("tool_appropriateness", 0.0)),
            "shortcut_penalty": float(breakdown.get("shortcut_penalty", 0.0)),
        })

    return {
        "policy": policy_name,
        "episodes": episodes,
        "avg_reward": round(mean([r["reward"] for r in rows]), 4),
        "avg_npa_rate": round(mean([r["npa_rate"] for r in rows]), 4),
        "avg_recovery_rate": round(mean([r["recovery_rate"] for r in rows]), 4),
        "avg_relationship_score": round(mean([r["relationship_score"] for r in rows]), 4),
        "avg_tool_appropriateness": round(mean([r["tool_appropriateness"] for r in rows]), 4),
        "avg_shortcut_penalty": round(mean([r["shortcut_penalty"] for r in rows]), 4),
        "episodes_raw": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline policies for MSME linguistic decoder RL env")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=1080)
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval_report.json"))
    args = parser.parse_args()

    random_report = run_policy("random", args.episodes, args.seed, args.max_steps)
    heuristic_report = run_policy("heuristic", args.episodes, args.seed + 17, args.max_steps)

    report = {
        "comparison": [random_report, heuristic_report],
        "notes": "Use this as baseline-vs-policy evidence. Add trained-policy row after checkpoint integration.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved evaluation report to: {args.output}")


if __name__ == "__main__":
    main()
