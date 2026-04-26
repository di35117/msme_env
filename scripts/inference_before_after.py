"""
Before-vs-After inference comparison: untrained base model vs trained checkpoint.

This is the headline "did RL actually do anything?" plot. It loads the *exact
same architecture* twice — once with the off-the-shelf weights (BASE), once
with the GRPO-fine-tuned weights (TRAINED) — and runs both against the same
fixed-seed episodes of MSMERLEnvironment.

Usage:
    python scripts/inference_before_after.py \
        --base Qwen/Qwen3-1.7B \
        --trained /data/msme_rl_run2/episode_0030 \
        --episodes 5 \
        --output-dir /data/recovered_artifacts

Outputs (in --output-dir):
    inference_before_after.png             — 3-panel: reward / NPA / trust (Base vs Trained)
    inference_action_distribution_ba.png   — action mix per policy
    inference_before_after.json            — raw numbers, fully auditable

Memory: only ONE model is loaded at a time. The base model is freed before
the trained model is loaded, so this works on a single 16 GB GPU.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch

# Make repo importable when run from arbitrary CWDs.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from server.msmeEnv_environment import MSMERLEnvironment
from models import MSMERLAction
from train_grpo import VALID_ACTIONS, SYSTEM_PROMPT, build_agent_prompt, _snap_to_valid_action

JSON_PREFILL = '{"action_type": "'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _obs_to_dict(o: Any) -> Dict[str, Any]:
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "__dict__"):
        return dict(o.__dict__)
    return o


def _build_full_prompt(tokenizer, user_prompt: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )


def _parse_action(generated: str) -> MSMERLAction:
    clean = generated.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        clean = parts[1] if len(parts) > 1 else clean
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()

    candidates: List[str] = [clean]
    candidates.extend(re.findall(r"\{.*?\}", clean, flags=re.DOTALL))
    li, ri = clean.find("{"), clean.rfind("}")
    if li != -1 and ri != -1 and ri > li:
        candidates.append(clean[li:ri + 1])

    def _to_action(d: Dict[str, Any]) -> MSMERLAction:
        try:
            acc = int(str(d.get("account_id", 1)).strip())
        except Exception:
            acc = 1
        params = d.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
        return MSMERLAction(
            action_type=_snap_to_valid_action(str(d.get("action_type", "wait_and_observe"))),
            account_id=max(1, min(30, acc)),
            parameters=params,
            reasoning=str(d.get("reasoning", "")),
        )

    for c in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                d = loader(c)
                if isinstance(d, dict):
                    return _to_action(d)
            except Exception:
                pass

    am = re.search(r"(action_type|action)\s*[:=]\s*['\"]?([a-zA-Z0-9_]+)['\"]?", clean)
    aid = re.search(r"(account_id|account)\s*[:=]\s*([0-9]+)", clean)
    if am:
        return MSMERLAction(
            action_type=_snap_to_valid_action(am.group(2)),
            account_id=max(1, min(30, int(aid.group(2)) if aid else 1)),
            parameters={},
            reasoning="(regex-recovered)",
        )
    return MSMERLAction(
        action_type="wait_and_observe",
        account_id=1,
        parameters={},
        reasoning="(unparseable)",
    )


def make_llm_policy(model, tokenizer, device: str) -> Callable[[Dict[str, Any]], MSMERLAction]:
    """Build a policy callable from a HuggingFace model + tokenizer."""

    @torch.inference_mode()
    def policy(obs: Dict[str, Any]) -> MSMERLAction:
        prompt = build_agent_prompt(obs)
        full   = _build_full_prompt(tokenizer, prompt) + JSON_PREFILL
        inputs = tokenizer(full, return_tensors="pt").to(device)
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
        )
        suffix = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return _parse_action(JSON_PREFILL + suffix)

    return policy


def run_episode(policy, seed: int, max_steps: int = 60) -> Dict[str, Any]:
    _seed_everything(seed)
    env = MSMERLEnvironment()
    obs = _obs_to_dict(env.reset())

    trace: List[Dict[str, Any]] = []
    cum = 0.0
    for step_idx in range(max_steps):
        action = policy(obs)
        new_obs = _obs_to_dict(env.step(action))
        step_reward = new_obs.get("step_reward", 0.0) or 0.0
        cum += step_reward
        ps = new_obs.get("portfolio_summary", {}) or {}
        trace.append({
            "step":        step_idx,
            "action_type": action.action_type,
            "account_id":  action.account_id,
            "step_reward": step_reward,
            "cum_reward":  cum,
            "month":       ps.get("current_month"),
            "npa_rate":    ps.get("npa_rate"),
            "trust":       ps.get("avg_trust_score"),
        })
        obs = new_obs
        if new_obs.get("done", False):
            break

    final_ps = obs.get("portfolio_summary", {}) or {}
    return {
        "trace":          trace,
        "total_reward":   cum,
        "final_npa_rate": final_ps.get("npa_rate"),
        "final_trust":    final_ps.get("avg_trust_score"),
        "steps_taken":    len(trace),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_model(model_path: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n→ loading {model_path}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    mdl.eval()
    n = sum(p.numel() for p in mdl.parameters()) / 1e9
    print(f"  loaded in {time.time()-t0:.1f}s ({n:.2f}B params, {dtype})")
    return mdl, tok


def _free(model) -> None:
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_policy(label: str, policy, seeds: List[int], max_steps: int) -> List[Dict[str, Any]]:
    runs = []
    for i, s in enumerate(seeds, 1):
        t0 = time.time()
        r = run_episode(policy, seed=s, max_steps=max_steps)
        r["seed"] = s
        r["episode"] = i
        runs.append(r)
        print(
            f"  [{label}] ep {i}/{len(seeds)} (seed={s}): "
            f"reward={r['total_reward']:+.3f} | "
            f"NPA={(r['final_npa_rate'] or 0):.1%} | "
            f"trust={(r['final_trust'] or 0):.2f} | "
            f"{time.time()-t0:.1f}s"
        )
    return runs


def _plot_aggregate(base_runs, trained_runs, n_eps, out_png: Path) -> None:
    """Side-by-side bars per episode + a giant aggregate Δ summary on top."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    base_r    = [r["total_reward"] for r in base_runs]
    trained_r = [r["total_reward"] for r in trained_runs]
    base_n    = [(r["final_npa_rate"] or 0) * 100 for r in base_runs]
    trained_n = [(r["final_npa_rate"] or 0) * 100 for r in trained_runs]
    base_t    = [r["final_trust"]    or 0 for r in base_runs]
    trained_t = [r["final_trust"]    or 0 for r in trained_runs]

    # Aggregate deltas — these go in the title for at-a-glance impact.
    mr_b, mr_t = float(np.mean(base_r)),  float(np.mean(trained_r))
    mn_b, mn_t = float(np.mean(base_n)),  float(np.mean(trained_n))
    mt_b, mt_t = float(np.mean(base_t)),  float(np.mean(trained_t))

    def _pct(delta, denom):
        return (delta / abs(denom)) * 100 if denom else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    x = np.arange(n_eps); w = 0.4

    # ---- Panel 1: cumulative episode reward ----
    bars0a = axes[0].bar(x - w/2, base_r,    w, label="Before RL (base)",    color="#bbbbbb")
    bars0b = axes[0].bar(x + w/2, trained_r, w, label="After RL (trained)", color="#1f77b4")
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_title(
        f"Cumulative reward  (higher is better)\n"
        f"mean Δ = {mr_t-mr_b:+.2f}  ({_pct(mr_t-mr_b, mr_b):+.0f}%)",
        fontsize=12, fontweight="bold",
    )
    axes[0].set_ylabel("reward"); axes[0].set_xlabel("episode")
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"Ep {i+1}" for i in range(n_eps)])
    axes[0].grid(axis="y", ls="--", alpha=0.4); axes[0].legend(loc="best", fontsize=9)
    for b in list(bars0a) + list(bars0b):
        h = b.get_height()
        axes[0].annotate(f"{h:.1f}", xy=(b.get_x()+b.get_width()/2, h),
                         xytext=(0, 3 if h >= 0 else -10), textcoords="offset points",
                         ha="center", fontsize=8)

    # ---- Panel 2: final NPA rate ----
    bars1a = axes[1].bar(x - w/2, base_n,    w, label="Before RL (base)",    color="#bbbbbb")
    bars1b = axes[1].bar(x + w/2, trained_n, w, label="After RL (trained)", color="#d62728")
    axes[1].set_title(
        f"Final NPA %  (lower is better)\n"
        f"mean Δ = {mn_t-mn_b:+.1f} pts  ({_pct(mn_t-mn_b, mn_b):+.0f}%)",
        fontsize=12, fontweight="bold",
    )
    axes[1].set_ylabel("NPA %"); axes[1].set_xlabel("episode")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"Ep {i+1}" for i in range(n_eps)])
    axes[1].grid(axis="y", ls="--", alpha=0.4); axes[1].legend(loc="best", fontsize=9)
    for b in list(bars1a) + list(bars1b):
        h = b.get_height()
        axes[1].annotate(f"{h:.0f}%", xy=(b.get_x()+b.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=8)

    # ---- Panel 3: final trust score ----
    bars2a = axes[2].bar(x - w/2, base_t,    w, label="Before RL (base)",    color="#bbbbbb")
    bars2b = axes[2].bar(x + w/2, trained_t, w, label="After RL (trained)", color="#2ca02c")
    axes[2].set_title(
        f"Final trust score  (higher is better)\n"
        f"mean Δ = {mt_t-mt_b:+.2f}  ({_pct(mt_t-mt_b, mt_b):+.0f}%)",
        fontsize=12, fontweight="bold",
    )
    axes[2].set_ylabel("trust"); axes[2].set_xlabel("episode")
    axes[2].set_xticks(x); axes[2].set_xticklabels([f"Ep {i+1}" for i in range(n_eps)])
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(axis="y", ls="--", alpha=0.4); axes[2].legend(loc="best", fontsize=9)
    for b in list(bars2a) + list(bars2b):
        h = b.get_height()
        axes[2].annotate(f"{h:.2f}", xy=(b.get_x()+b.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=8)

    fig.suptitle(
        "MSME-RL: Before RL  vs  After RL  (matched seeds, identical env)",
        fontsize=14, fontweight="bold",
    )
    fig.text(0.5, 0.945,
             f"5 fixed-seed episodes per policy · same Qwen2.5-1.5B architecture · "
             "trained = GRPO checkpoint",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {out_png}")


def _plot_msme_vs_startup(base_runs, trained_runs, out_png: Path) -> None:
    """How does the trained policy treat MSME accounts (1-20) vs startup
    accounts (21-30)? Two pillars of the central thesis (dual strategies)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    MSME_RANGE    = range(1, 21)
    STARTUP_RANGE = range(21, 31)

    def _aid(s):
        try:
            return int(s.get("account_id", 0))
        except (TypeError, ValueError):
            return 0

    def _split_action_pct(runs, account_range):
        from collections import Counter
        c = Counter()
        total = 0
        for r in runs:
            for s in r["trace"]:
                if _aid(s) in account_range:
                    c[s["action_type"]] += 1
                    total += 1
        return {a: (c.get(a, 0) / max(1, total)) * 100 for a in VALID_ACTIONS}, total

    def _split_step_reward(runs, account_range):
        sums, counts = [], 0
        for r in runs:
            ep_sum = 0.0
            for s in r["trace"]:
                if _aid(s) in account_range:
                    ep_sum += s["step_reward"]
                    counts += 1
            sums.append(ep_sum)
        return float(np.mean(sums)) if sums else 0.0, counts

    base_msme_dist,    n_msme_b    = _split_action_pct(base_runs,    MSME_RANGE)
    trained_msme_dist, n_msme_t    = _split_action_pct(trained_runs, MSME_RANGE)
    base_stp_dist,     n_stp_b     = _split_action_pct(base_runs,    STARTUP_RANGE)
    trained_stp_dist,  n_stp_t     = _split_action_pct(trained_runs, STARTUP_RANGE)

    base_msme_r, _    = _split_step_reward(base_runs,    MSME_RANGE)
    trained_msme_r, _ = _split_step_reward(trained_runs, MSME_RANGE)
    base_stp_r, _     = _split_step_reward(base_runs,    STARTUP_RANGE)
    trained_stp_r, _  = _split_step_reward(trained_runs, STARTUP_RANGE)

    # Coercive vs information-gathering action buckets — the two clusters
    # the trained policy is supposed to swap probability mass between.
    COERCIVE = {"initiate_sarfaesi", "file_drt_case", "refer_to_recovery_agent",
                "send_legal_notice_section13"}
    INFO     = {
        "verify_gst_returns",
        "pull_bank_statements",
        "check_industry_cluster_stress",
        "request_investor_update_meeting",
        "check_startup_ecosystem_signals",
        "wait_and_observe",
        "call_guarantor_investor",
        "call_promoter_founder",
        "call_guarantor",
        "call_guarantor_intermediary",
        "conduct_cluster_ecosystem_visit",
    }

    def _bucket(dist, keys):
        return sum(dist.get(k, 0.0) for k in keys)

    cats   = ["MSME accts (1–20)", "Startup accts (21–30)"]
    base_c = [_bucket(base_msme_dist, COERCIVE), _bucket(base_stp_dist, COERCIVE)]
    trnd_c = [_bucket(trained_msme_dist, COERCIVE), _bucket(trained_stp_dist, COERCIVE)]
    base_i = [_bucket(base_msme_dist, INFO),     _bucket(base_stp_dist, INFO)]
    trnd_i = [_bucket(trained_msme_dist, INFO), _bucket(trained_stp_dist, INFO)]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    x = np.arange(2); w = 0.35

    axes[0].bar(x - w/2, base_c, w, label="Before RL", color="#bbbbbb")
    axes[0].bar(x + w/2, trnd_c, w, label="After RL",  color="#d62728")
    axes[0].set_title("% Coercive actions  (sarfaesi / drt / agent / sec13)\nlower is better — esp. on startups",
                      fontsize=11, fontweight="bold")
    axes[0].set_ylabel("% of decisions on these accounts")
    axes[0].set_xticks(x); axes[0].set_xticklabels(cats)
    axes[0].grid(axis="y", ls="--", alpha=0.4); axes[0].legend(fontsize=9)
    _top_c = max(base_c + trnd_c + [0.0])
    axes[0].set_ylim(0, max(8.0, _top_c * 1.15 + 1e-6))
    for i, (b, t) in enumerate(zip(base_c, trnd_c)):
        axes[0].annotate(f"Δ {t-b:+.1f} pts", xy=(i, max(b, t) + 1.5),
                         ha="center", fontsize=9, fontweight="bold")

    axes[1].bar(x - w/2, base_i, w, label="Before RL", color="#bbbbbb")
    axes[1].bar(x + w/2, trnd_i, w, label="After RL",  color="#2ca02c")
    axes[1].set_title("% Information / restraint actions\nhigher is better",
                      fontsize=11, fontweight="bold")
    axes[1].set_ylabel("% of decisions on these accounts")
    axes[1].set_xticks(x); axes[1].set_xticklabels(cats)
    axes[1].grid(axis="y", ls="--", alpha=0.4); axes[1].legend(fontsize=9)
    _top_i = max(base_i + trnd_i + [0.0])
    axes[1].set_ylim(0, max(8.0, _top_i * 1.15 + 1e-6))
    for i, (b, t) in enumerate(zip(base_i, trnd_i)):
        axes[1].annotate(f"Δ {t-b:+.1f} pts", xy=(i, max(b, t) + 1.5),
                         ha="center", fontsize=9, fontweight="bold")

    axes[2].bar(x - w/2, [base_msme_r, base_stp_r], w, label="Before RL", color="#bbbbbb")
    axes[2].bar(x + w/2, [trained_msme_r, trained_stp_r], w, label="After RL", color="#1f77b4")
    axes[2].axhline(0, color="black", lw=0.6)
    axes[2].set_title("Mean step-reward earned per episode\nfrom this account class",
                      fontsize=11, fontweight="bold")
    axes[2].set_ylabel("step-reward sum (per episode)")
    axes[2].set_xticks(x); axes[2].set_xticklabels(cats)
    axes[2].grid(axis="y", ls="--", alpha=0.4); axes[2].legend(fontsize=9)
    # If a class gets no targeted steps, pct bars are 0 — show counts for debugging.
    fig.text(
        0.5,
        0.02,
        f"Steps in eval traces — MSME: base n={n_msme_b} trained n={n_msme_t} | "
        f"Startup: base n={n_stp_b} trained n={n_stp_t}",
        ha="center",
        fontsize=9,
        color="#444",
    )

    fig.suptitle("Dual-Strategy Learning  —  MSME (understatement)  vs  Startup (overstatement)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_png}")


def _plot_action_shift(base_runs, trained_runs, out_png: Path) -> None:
    """Single horizontal bar chart showing which actions INCREASED and which
    DECREASED after RL. The clearest single visual that learning happened."""
    from collections import Counter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def dist(runs):
        c = Counter(); total = 0
        for r in runs:
            for s in r["trace"]:
                c[s["action_type"]] += 1; total += 1
        return {a: (c.get(a, 0) / max(1, total)) * 100 for a in VALID_ACTIONS}

    bd, td = dist(base_runs), dist(trained_runs)
    diff = sorted(
        [(a, td[a] - bd[a]) for a in VALID_ACTIONS],
        key=lambda x: x[1],
    )
    actions, deltas = zip(*diff)
    colors = ["#d62728" if d < 0 else "#2ca02c" for d in deltas]

    fig, ax = plt.subplots(figsize=(11, 8))
    ys = np.arange(len(actions))
    ax.barh(ys, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(ys); ax.set_yticklabels(actions, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Δ usage  (percentage points: trained − base)", fontsize=10)
    ax.set_title(
        "Policy shift after RL  —  what the agent learned to use MORE (green)  vs  LESS (red)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", ls="--", alpha=0.4)
    for i, d in enumerate(deltas):
        ax.annotate(f"{d:+.1f}", xy=(d, i),
                    xytext=(4 if d >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if d >= 0 else "right", va="center",
                    fontsize=8, fontweight="bold")
    fig.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_png}")


def _plot_action_dist(base_runs, trained_runs, out_png: Path) -> None:
    from collections import Counter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def dist(runs):
        c = Counter(); total = 0
        for r in runs:
            for s in r["trace"]:
                c[s["action_type"]] += 1
                total += 1
        return {a: c.get(a, 0) / max(1, total) for a in VALID_ACTIONS}

    bd = dist(base_runs); td = dist(trained_runs)
    ordered = sorted(VALID_ACTIONS, key=lambda a: -td[a])
    xs = np.arange(len(ordered)); w = 0.4

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(xs - w/2, [bd[a]*100 for a in ordered], w, label="Before RL (base)", color="#bbbbbb")
    ax.bar(xs + w/2, [td[a]*100 for a in ordered], w, label="After RL (trained)", color="#1f77b4")
    ax.set_xticks(xs); ax.set_xticklabels(ordered, rotation=60, ha="right")
    ax.set_ylabel("% of all steps"); ax.set_title("Action distribution: Before RL vs After RL")
    ax.grid(axis="y", ls="--", alpha=0.4); ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_png}")
    return bd, td


def main() -> None:
    parser = argparse.ArgumentParser(description="Before-vs-After RL inference comparison.")
    parser.add_argument("--base",       type=str, required=True, help="HF model id or local path of the untrained base model")
    parser.add_argument("--trained",    type=str, required=True, help="Local path to the trained checkpoint dir")
    parser.add_argument("--episodes",   type=int, default=5,     help="Number of fixed-seed episodes to run per policy")
    parser.add_argument("--max-steps",  type=int, default=60,    help="Max steps per episode")
    parser.add_argument("--seed",       type=int, default=42,    help="Base seed (each episode uses seed+i)")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    seeds = [args.seed + i for i in range(args.episodes)]

    # ---- BEFORE RL ----------------------------------------------------------
    print("\n" + "=" * 72)
    print("BEFORE RL — untrained base model")
    print("=" * 72)
    base_model, base_tok = _load_model(args.base, device)
    base_policy = make_llm_policy(base_model, base_tok, device)
    base_runs   = _run_policy("base", base_policy, seeds, args.max_steps)
    base_mean   = sum(r["total_reward"] for r in base_runs) / len(base_runs)
    print(f"\n  Mean reward (BEFORE RL): {base_mean:+.3f}")
    _free(base_model); del base_tok, base_policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  GPU memory free after release: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    # ---- AFTER RL -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("AFTER RL — trained checkpoint")
    print("=" * 72)
    trained_model, trained_tok = _load_model(args.trained, device)
    trained_policy = make_llm_policy(trained_model, trained_tok, device)
    trained_runs   = _run_policy("trained", trained_policy, seeds, args.max_steps)
    trained_mean   = sum(r["total_reward"] for r in trained_runs) / len(trained_runs)
    print(f"\n  Mean reward (AFTER  RL): {trained_mean:+.3f}")
    _free(trained_model)

    # ---- Plots --------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Plotting")
    print("=" * 72)
    aggregate_png    = out_dir / "inference_before_after.png"
    action_png       = out_dir / "inference_action_distribution_before_after.png"
    msme_startup_png = out_dir / "inference_msme_vs_startup.png"
    action_shift_png = out_dir / "inference_action_shift.png"
    _plot_aggregate(base_runs, trained_runs, args.episodes, aggregate_png)
    bd, td = _plot_action_dist(base_runs, trained_runs, action_png)
    _plot_msme_vs_startup(base_runs, trained_runs, msme_startup_png)
    _plot_action_shift(base_runs, trained_runs, action_shift_png)

    # ---- Audit JSON ---------------------------------------------------------
    sarfaesi_on_startup = lambda runs: sum(
        1 for r in runs for s in r["trace"]
        if s["action_type"] == "initiate_sarfaesi" and s["account_id"] in range(21, 31)
    )
    def _mean(xs): return float(sum(xs) / max(1, len(xs)))
    base_mean_npa    = _mean([(r["final_npa_rate"] or 0) for r in base_runs])
    trained_mean_npa = _mean([(r["final_npa_rate"] or 0) for r in trained_runs])
    base_mean_trust  = _mean([(r["final_trust"]    or 0) for r in base_runs])
    trained_mean_trust = _mean([(r["final_trust"]  or 0) for r in trained_runs])

    audit = {
        "config": {
            "base":        args.base,
            "trained":     args.trained,
            "seed":        args.seed,
            "episodes":    args.episodes,
            "max_steps":   args.max_steps,
        },
        "before_rl": {
            "per_episode": [{"seed": r["seed"], "reward": r["total_reward"], "npa": r["final_npa_rate"], "trust": r["final_trust"]} for r in base_runs],
            "mean_reward": base_mean,
            "mean_npa":    base_mean_npa,
            "mean_trust":  base_mean_trust,
        },
        "after_rl": {
            "per_episode": [{"seed": r["seed"], "reward": r["total_reward"], "npa": r["final_npa_rate"], "trust": r["final_trust"]} for r in trained_runs],
            "mean_reward": trained_mean,
            "mean_npa":    trained_mean_npa,
            "mean_trust":  trained_mean_trust,
        },
        "improvement_after_rl": {
            "reward_delta": trained_mean - base_mean,
            "npa_delta":    trained_mean_npa - base_mean_npa,
            "trust_delta":  trained_mean_trust - base_mean_trust,
        },
        "sarfaesi_on_startup_count": {
            "before": sarfaesi_on_startup(base_runs),
            "after":  sarfaesi_on_startup(trained_runs),
        },
        "action_distribution": {"before": bd, "after": td},
    }
    audit_path = out_dir / "inference_before_after.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"  Saved → {audit_path}")

    # ---- Summary ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Mean reward — BEFORE RL: {base_mean:+.3f}")
    print(f"Mean reward — AFTER  RL: {trained_mean:+.3f}")
    print(f"Improvement:             {trained_mean - base_mean:+.3f}")
    print(f"SARFAESI-on-startup (before / after): "
          f"{audit['sarfaesi_on_startup_count']['before']} / {audit['sarfaesi_on_startup_count']['after']}")
    print(f"\nAll outputs → {out_dir}/")


if __name__ == "__main__":
    main()
