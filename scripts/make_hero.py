"""
Build the README hero image and KPI scorecard from training + inference artifacts.

Reads:
    <run_dir>/reward_curve.json              (from train_grpo.py)
    <run_dir>/inference_before_after.json    (from inference_before_after.py)

Writes:
    <run_dir>/hero.png            — 4-panel header image for the README
    <run_dir>/kpi_scorecard.png   — single judge-screenshot-ready KPI card

Usage:
    python scripts/make_hero.py --run-dir /data/msme_rl_run3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _arrow(delta: float, lower_is_better: bool = False) -> Tuple[str, str]:
    """Return (symbol, color) for a delta given direction-of-good."""
    good = (delta < 0) if lower_is_better else (delta > 0)
    if abs(delta) < 1e-9:
        return "—", "#888888"
    if good:
        return "▼" if lower_is_better else "▲", "#2ca02c"
    return "▲" if lower_is_better else "▼", "#d62728"


def _pct(delta: float, denom: float) -> float:
    return (delta / abs(denom)) * 100 if denom else 0.0


# ---------------------------------------------------------------------------
# 1) KPI scorecard — pure text rendering, no axes
# ---------------------------------------------------------------------------

def _kpi_scorecard(rc: Dict[str, Any], inf: Dict[str, Any], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_eps         = len(rc.get("rewards", []))
    final_reward  = rc["rewards"][-1]      if rc.get("rewards") else 0.0
    final_npa     = rc["npa_rates"][-1]    if rc.get("npa_rates") else 0.0
    final_trust   = rc["trust_scores"][-1] if rc.get("trust_scores") else 0.0
    final_wait    = rc["wait_ratios"][-1]  if rc.get("wait_ratios") else 0.0

    inf_b = inf.get("before_rl", {}); inf_a = inf.get("after_rl", {})
    delta = inf.get("improvement_after_rl", {}) or {}

    base_r,  trained_r  = inf_b.get("mean_reward", 0.0), inf_a.get("mean_reward", 0.0)
    base_n,  trained_n  = inf_b.get("mean_npa", 0.0),    inf_a.get("mean_npa", 0.0)
    base_t,  trained_t  = inf_b.get("mean_trust", 0.0),  inf_a.get("mean_trust", 0.0)
    sarf_b = inf.get("sarfaesi_on_startup_count", {}).get("before", 0)
    sarf_a = inf.get("sarfaesi_on_startup_count", {}).get("after",  0)

    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")

    # Header band
    ax.add_patch(plt.Rectangle((0, 90), 100, 10, color="#1f3a5f"))
    ax.text(50, 95, "MSME-RL  ·  Hackathon Submission Scorecard",
            ha="center", va="center", color="white", fontsize=16, fontweight="bold")
    ax.text(50, 87.5,
            f"Qwen2.5-1.5B-Instruct  ·  GRPO  ·  {n_eps} episodes  ·  KL anchor + entropy bonus",
            ha="center", va="center", color="#444", fontsize=10)

    rows = [
        ("Cumulative reward", f"{base_r:+.2f}", f"{trained_r:+.2f}",
         delta.get("reward_delta", trained_r - base_r), False),
        ("Final NPA rate",    f"{base_n*100:.1f}%", f"{trained_n*100:.1f}%",
         (delta.get("npa_delta", trained_n - base_n)) * 100, True),
        ("Avg trust score",   f"{base_t:.2f}",   f"{trained_t:.2f}",
         delta.get("trust_delta", trained_t - base_t), False),
        ("SARFAESI on startups (5 ep × 90 steps = 450 actions)",
         f"{sarf_b}", f"{sarf_a}", float(sarf_a - sarf_b), True),
    ]

    # Column headers
    ax.text(4,  82, "Metric",       fontsize=11, fontweight="bold", color="#222")
    ax.text(46, 82, "Before RL",    fontsize=11, fontweight="bold", color="#222")
    ax.text(64, 82, "After RL",     fontsize=11, fontweight="bold", color="#222")
    ax.text(82, 82, "Δ",            fontsize=11, fontweight="bold", color="#222")
    ax.plot([2, 98], [80, 80], color="#cccccc", linewidth=0.8)

    y = 73
    for metric, before, after, d, lower_is_better in rows:
        sym, col = _arrow(d, lower_is_better=lower_is_better)
        ax.text(4,  y, metric, fontsize=10, color="#222")
        ax.text(46, y, before, fontsize=11, color="#666")
        ax.text(64, y, after,  fontsize=11, color="#222", fontweight="bold")
        unit = "pts" if "NPA" in metric else ""
        ax.text(82, y, f"{sym} {abs(d):.2f} {unit}".strip(),
                fontsize=11, color=col, fontweight="bold")
        y -= 8

    # Footer KPIs from training (not inference)
    ax.add_patch(plt.Rectangle((0, 0), 100, 18, color="#f5f7fa"))
    ax.text(50, 14,
            f"End-of-training (episode {n_eps}):  reward = {final_reward:+.2f}   ·   "
            f"NPA = {final_npa*100:.1f}%   ·   trust = {final_trust:.2f}   ·   "
            f"wait_and_observe = {final_wait*100:.0f}% of steps",
            ha="center", va="center", fontsize=10, color="#222")
    ax.text(50, 5,
            "Identical seeds.  Identical environment.  Same Qwen2.5-1.5B architecture loaded twice (off-the-shelf vs GRPO-fine-tuned).",
            ha="center", va="center", fontsize=8, color="#777", style="italic")

    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {out_png}")


# ---------------------------------------------------------------------------
# 2) Hero image — 2x2 banner: training reward, NPA curve, KPI summary, action shift
# ---------------------------------------------------------------------------

def _hero(rc: Dict[str, Any], inf: Dict[str, Any], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_eps    = len(rc.get("rewards", []))
    rewards  = rc.get("rewards", [])
    npa      = [r * 100 for r in rc.get("npa_rates", [])]
    trust    = rc.get("trust_scores", [])
    waits    = [r * 100 for r in rc.get("wait_ratios", [])]
    eps      = list(range(1, n_eps + 1))

    inf_b = inf.get("before_rl", {})
    inf_a = inf.get("after_rl", {})
    base_r, trained_r = inf_b.get("mean_reward", 0.0), inf_a.get("mean_reward", 0.0)
    base_n, trained_n = inf_b.get("mean_npa",    0.0), inf_a.get("mean_npa",    0.0)
    base_t, trained_t = inf_b.get("mean_trust",  0.0), inf_a.get("mean_trust",  0.0)

    action_dist = (inf.get("action_distribution") or {})
    bd = action_dist.get("before", {}) or {}
    td = action_dist.get("after",  {}) or {}
    diff = sorted([(a, td.get(a, 0) - bd.get(a, 0)) for a in set(list(bd) + list(td))],
                  key=lambda x: x[1])
    # Top 5 climbed, top 5 dropped
    if len(diff) > 10:
        diff = diff[:5] + diff[-5:]

    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.28)

    # (0,0) Reward curve with smoothing
    ax = fig.add_subplot(gs[0, 0])
    if rewards:
        ax.plot(eps, rewards, color="#1f77b4", linewidth=1.2, alpha=0.5, label="per-episode")
        if len(rewards) >= 5:
            w = min(8, len(rewards) // 3)
            sm = np.convolve(rewards, np.ones(w) / w, mode="valid")
            ax.plot(list(range(w, len(rewards) + 1)), sm,
                    color="#d62728", linewidth=2.6, label=f"rolling avg (w={w})")
        ax.set_title(f"Training reward over {n_eps} episodes", fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
        ax.grid(True, ls="--", alpha=0.4); ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "no reward data", ha="center", va="center"); ax.axis("off")

    # (0,1) Business outcomes: NPA + Trust + wait, three lines on twin axes
    ax = fig.add_subplot(gs[0, 1])
    if npa:
        ax.plot(eps, npa, color="#d62728", linewidth=2.2, marker="o", markersize=3,
                label="NPA % (lower better)")
    if waits:
        ax.plot(eps, waits, color="#1f77b4", linewidth=2.2, marker="o", markersize=3,
                label="wait_and_observe %")
    ax.set_xlabel("Episode"); ax.set_ylabel("Percent of steps / accounts")
    ax.grid(True, ls="--", alpha=0.4)
    if trust:
        ax2 = ax.twinx()
        ax2.plot(eps, trust, color="#2ca02c", linewidth=2.2, marker="o", markersize=3,
                 label="Trust [0–1]  (higher better)")
        ax2.set_ylabel("Trust", color="#2ca02c")
        ax2.set_ylim(0, 1.0)
        ax2.tick_params(axis="y", labelcolor="#2ca02c")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)
    else:
        ax.legend(fontsize=9)
    ax.set_title("Business outcomes learned during training", fontsize=12, fontweight="bold")

    # (1,0) KPI text panel — Before vs After RL on inference
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.text(50, 92, "Inference: Before RL  vs  After RL",
            ha="center", fontsize=13, fontweight="bold", color="#222")
    ax.text(50, 84, "(matched seeds, identical env, same Qwen2.5-1.5B)",
            ha="center", fontsize=9, color="#666", style="italic")
    rows = [
        ("Mean reward",       f"{base_r:+.2f}",   f"{trained_r:+.2f}",   trained_r - base_r,    False),
        ("Mean final NPA",    f"{base_n*100:.1f}%", f"{trained_n*100:.1f}%", (trained_n - base_n) * 100, True),
        ("Mean trust score",  f"{base_t:.2f}",   f"{trained_t:.2f}",   trained_t - base_t,    False),
    ]
    y = 70
    ax.text(8,  y, "metric",   fontsize=10, fontweight="bold")
    ax.text(48, y, "before",   fontsize=10, fontweight="bold")
    ax.text(66, y, "after",    fontsize=10, fontweight="bold")
    ax.text(82, y, "Δ",        fontsize=10, fontweight="bold")
    ax.plot([6, 94], [66, 66], color="#cccccc", linewidth=0.8)
    y -= 12
    for metric, before, after, d, lib in rows:
        sym, col = _arrow(d, lower_is_better=lib)
        ax.text(8,  y, metric, fontsize=11)
        ax.text(48, y, before, fontsize=12, color="#666")
        ax.text(66, y, after,  fontsize=12, fontweight="bold")
        ax.text(82, y, f"{sym} {abs(d):.2f}", fontsize=12, color=col, fontweight="bold")
        y -= 14

    # (1,1) Action shift — top 5 climbed + top 5 dropped
    ax = fig.add_subplot(gs[1, 1])
    if diff:
        actions, deltas = zip(*diff)
        deltas_pp = [d * 100 for d in deltas]
        colors = ["#d62728" if d < 0 else "#2ca02c" for d in deltas_pp]
        ys = np.arange(len(actions))
        ax.barh(ys, deltas_pp, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(ys); ax.set_yticklabels(actions, fontsize=9)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_xlabel("Δ usage (percentage points)")
        ax.set_title("Top action shifts after RL  (green ↑ used MORE  /  red ↓ used LESS)",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="x", ls="--", alpha=0.4)
        for i, d in enumerate(deltas_pp):
            ax.annotate(f"{d:+.1f}", xy=(d, i),
                        xytext=(4 if d >= 0 else -4, 0), textcoords="offset points",
                        ha="left" if d >= 0 else "right", va="center",
                        fontsize=8, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "no action distribution", ha="center", va="center"); ax.axis("off")

    fig.suptitle("MSME-RL  —  GRPO on a dual-strategy Indian banking env",
                 fontsize=18, fontweight="bold")
    fig.text(0.5, 0.945,
             "Same model architecture, same fixed seeds. Untrained vs GRPO-fine-tuned.",
             ha="center", fontsize=10, color="#555")
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {out_png}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build hero image and KPI scorecard.")
    parser.add_argument("--run-dir", required=True,
                        help="Directory containing reward_curve.json and inference_before_after.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    rc_path  = run_dir / "reward_curve.json"
    inf_path = run_dir / "inference_before_after.json"

    if not rc_path.exists():
        raise SystemExit(f"missing {rc_path} — run training first")
    rc = _load_json(rc_path)
    inf = _load_json(inf_path) if inf_path.exists() else {
        "before_rl": {}, "after_rl": {}, "improvement_after_rl": {},
        "sarfaesi_on_startup_count": {}, "action_distribution": {"before": {}, "after": {}},
    }
    if not inf_path.exists():
        print(f"  warning: {inf_path} missing — inference panels will be empty")

    print("Building scorecard + hero image…")
    _kpi_scorecard(rc, inf, run_dir / "kpi_scorecard.png")
    _hero(rc, inf, run_dir / "hero.png")
    print("\nDone. Add to README:")
    print(f"  ![Hero](artifacts/hero.png)")
    print(f"  ![KPI Scorecard](artifacts/kpi_scorecard.png)")


if __name__ == "__main__":
    main()
