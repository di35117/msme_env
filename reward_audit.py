"""
Reward audit helpers for MSME linguistic decoder RL.

Use this module to sanity-check whether reward trends reflect real behavior
improvement or likely proxy exploitation.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List


def audit_episode_history(episode_history: List[Dict]) -> Dict[str, float]:
    if not episode_history:
        return {
            "steps": 0,
            "distinct_actions": 0,
            "top_action_ratio": 0.0,
            "repeat_action_ratio": 0.0,
            "suspicious": 0,
        }

    actions = [s.get("action_type", "") for s in episode_history]
    counts = Counter(actions)
    top_count = counts.most_common(1)[0][1] if counts else 0
    repeats = sum(1 for i in range(1, len(actions)) if actions[i] == actions[i - 1])

    top_action_ratio = top_count / len(actions)
    repeat_ratio = repeats / max(1, len(actions) - 1)
    suspicious = 1 if top_action_ratio > 0.45 or repeat_ratio > 0.55 else 0

    return {
        "steps": len(actions),
        "distinct_actions": len(counts),
        "top_action_ratio": round(top_action_ratio, 4),
        "repeat_action_ratio": round(repeat_ratio, 4),
        "suspicious": suspicious,
    }
