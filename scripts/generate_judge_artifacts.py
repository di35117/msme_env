"""
Generate judge-ready plots and manifest from training outputs.

Usage:
  py -3 scripts/generate_judge_artifacts.py ^
      --training_json msme_rl_checkpoints/reward_curve.json ^
      --baseline_json artifacts/baseline_rewards.json ^
      --output_dir artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _rolling_mean(values: List[float], window: int) -> List[float]:
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / max(1, len(chunk)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hackathon judging artifacts.")
    parser.add_argument("--training_json", type=Path, required=True)
    parser.add_argument("--baseline_json", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    data = _read_json(args.training_json)
    rewards: List[float] = data.get("rewards", [])
    losses: List[float] = data.get("losses", [])
    episodes: List[int] = data.get("episodes", list(range(1, len(rewards) + 1)))
    if not rewards:
        raise ValueError("training_json has no rewards.")

    baseline_rewards: List[float] = []
    if args.baseline_json and args.baseline_json.exists():
        baseline_rewards = _read_json(args.baseline_json).get("rewards", [])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Reward curve
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, alpha=0.4, linewidth=1.4, label="raw reward")
    trend = _rolling_mean(rewards, max(3, len(rewards) // 8))
    plt.plot(episodes, trend, linewidth=2.2, label="rolling mean")
    plt.title("Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "training_reward_curve.png", dpi=150)
    plt.close()

    # Loss curve
    if losses:
        x = list(range(1, len(losses) + 1))
        plt.figure(figsize=(10, 5))
        plt.plot(x, losses, linewidth=1.8, label="policy loss")
        plt.plot(x, _rolling_mean(losses, max(3, len(losses) // 10)), linewidth=2.0, label="loss trend")
        plt.title("Policy Loss")
        plt.xlabel("Update batch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / "training_loss_curve.png", dpi=150)
        plt.close()

    # Base vs trained
    if baseline_rewards:
        plt.figure(figsize=(10, 5))
        plt.hist(baseline_rewards, bins=12, alpha=0.55, label="base")
        plt.hist(rewards, bins=12, alpha=0.55, label="trained")
        plt.title("Reward Distribution: Base vs Trained")
        plt.xlabel("Reward")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / "reward_distribution_base_vs_trained.png", dpi=150)
        plt.close()

        n = min(len(baseline_rewards), len(rewards))
        plt.figure(figsize=(10, 5))
        plt.scatter(range(1, n + 1), baseline_rewards[:n], s=20, alpha=0.7, label="base")
        plt.scatter(range(1, n + 1), rewards[:n], s=20, alpha=0.7, label="trained")
        plt.title("Per-Episode: Base vs Trained")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True, linestyle="--", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / "per_episode_base_vs_trained.png", dpi=150)
        plt.close()

    base_avg = (sum(baseline_rewards) / len(baseline_rewards)) if baseline_rewards else None
    trained_avg = sum(rewards) / len(rewards)
    summary = {
        "episodes": len(rewards),
        "updates": len(losses),
        "base_avg_reward": base_avg,
        "trained_avg_reward": trained_avg,
        "abs_reward_improvement": (trained_avg - base_avg) if base_avg is not None else None,
        "final_reward": rewards[-1],
        "best_reward": max(rewards),
        "final_loss": losses[-1] if losses else None,
    }
    with (args.output_dir / "judge_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    manifest = {
        "required": [
            "training_reward_curve.png",
            "training_loss_curve.png",
            "judge_summary.json",
        ],
        "optional": [
            "reward_distribution_base_vs_trained.png",
            "per_episode_base_vs_trained.png",
        ],
    }
    with (args.output_dir / "judge_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"artifacts generated -> {args.output_dir}")


if __name__ == "__main__":
    main()

