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

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(n_eps); w = 0.4

    axes[0].bar(x - w/2, base_r,    w, label="Before RL (base)", color="#bbbbbb")
    axes[0].bar(x + w/2, trained_r, w, label="After RL (trained)", color="#1f77b4")
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_title("Cumulative episode reward (higher is better)")
    axes[0].set_ylabel("reward"); axes[0].set_xlabel("episode")
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"Ep {i+1}" for i in range(n_eps)])
    axes[0].grid(axis="y", ls="--", alpha=0.4); axes[0].legend()

    axes[1].bar(x - w/2, base_n,    w, label="Before RL (base)", color="#bbbbbb")
    axes[1].bar(x + w/2, trained_n, w, label="After RL (trained)", color="#d62728")
    axes[1].set_title("Final NPA rate (lower is better)")
    axes[1].set_ylabel("NPA %"); axes[1].set_xlabel("episode")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"Ep {i+1}" for i in range(n_eps)])
    axes[1].grid(axis="y", ls="--", alpha=0.4); axes[1].legend()

    axes[2].bar(x - w/2, base_t,    w, label="Before RL (base)", color="#bbbbbb")
    axes[2].bar(x + w/2, trained_t, w, label="After RL (trained)", color="#2ca02c")
    axes[2].set_title("Final trust score (higher is better)")
    axes[2].set_ylabel("trust"); axes[2].set_xlabel("episode")
    axes[2].set_xticks(x); axes[2].set_xticklabels([f"Ep {i+1}" for i in range(n_eps)])
    axes[2].grid(axis="y", ls="--", alpha=0.4); axes[2].legend()

    fig.suptitle("MSME-RL: Before RL vs After RL (matched seeds)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {out_png}")


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
    aggregate_png = out_dir / "inference_before_after.png"
    action_png    = out_dir / "inference_action_distribution_before_after.png"
    _plot_aggregate(base_runs, trained_runs, args.episodes, aggregate_png)
    bd, td = _plot_action_dist(base_runs, trained_runs, action_png)

    # ---- Audit JSON ---------------------------------------------------------
    sarfaesi_on_startup = lambda runs: sum(
        1 for r in runs for s in r["trace"]
        if s["action_type"] == "initiate_sarfaesi" and s["account_id"] in range(21, 31)
    )
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
        },
        "after_rl": {
            "per_episode": [{"seed": r["seed"], "reward": r["total_reward"], "npa": r["final_npa_rate"], "trust": r["final_trust"]} for r in trained_runs],
            "mean_reward": trained_mean,
        },
        "improvement_after_rl": trained_mean - base_mean,
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
