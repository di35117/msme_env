"""
Score multiple GRPO checkpoints on the same fixed-seed eval episodes.
Pick the best by composite score: lower NPA, higher trust (and optional reward).

Usage:
  python scripts/score_checkpoints.py \\
    --base Qwen/Qwen2.5-1.5B-Instruct \\
    --checkpoint-dir ./msme_rl_run5 \\
    --episodes 5 \\
    --seed 42

Prints a ranked table; use the top path for --trained in inference_before_after.py.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_ib_path = _REPO / "scripts" / "inference_before_after.py"
_spec = importlib.util.spec_from_file_location("inference_before_after", _ib_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load {_ib_path}")
_ib = importlib.util.module_from_spec(_spec)
sys.modules["inference_before_after"] = _ib
_spec.loader.exec_module(_ib)
_free = _ib._free
_load_model = _ib._load_model
_run_policy = _ib._run_policy
make_llm_policy = _ib.make_llm_policy


def main() -> None:
    p = argparse.ArgumentParser(description="Rank episode_* checkpoints by eval NPA/trust.")
    p.add_argument("--base", type=str, required=True, help="HF id or path (same arch as checkpoints)")
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing episode_0002, episode_0010, ... subfolders",
    )
    p.add_argument("--episodes", type=int, default=5, help="Eval episodes per checkpoint")
    p.add_argument("--max-steps", type=int, default=60, help="Max env steps per eval episode")
    p.add_argument("--seed", type=int, default=42, help="Base seed (episode i uses seed+i)")
    p.add_argument("--glob", type=str, default="episode_*", help="Subdir glob under checkpoint-dir")
    args = p.parse_args()

    ckpt_root = Path(args.checkpoint_dir).resolve()
    subdirs = sorted(ckpt_root.glob(args.glob))
    subdirs = [d for d in subdirs if d.is_dir()]
    if not subdirs:
        print(f"No checkpoints matching {ckpt_root}/{args.glob}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [args.seed + i for i in range(args.episodes)]

    print("Loading base model once for reference (not scored)...")
    base_m, base_t = _load_model(args.base, device)
    base_pol = make_llm_policy(base_m, base_t, device)
    base_runs = _run_policy("base", base_pol, seeds, args.max_steps)
    base_npa = sum((r.get("final_npa_rate") or 0) for r in base_runs) / len(base_runs)
    base_tr = sum((r.get("final_trust") or 0) for r in base_runs) / len(base_runs)
    base_rw = sum(r["total_reward"] for r in base_runs) / len(base_runs)
    _free(base_m)
    del base_t, base_pol
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Base reference | mean_reward={base_rw:+.3f} mean_NPA={base_npa:.1%} mean_trust={base_tr:.3f}\n")

    rows: list[tuple[float, str, float, float, float]] = []
    for d in subdirs:
        print("=" * 60)
        m, t = _load_model(str(d), device)
        pol = make_llm_policy(m, t, device)
        runs = _run_policy(str(d.name), pol, seeds, args.max_steps)
        mn_pa = sum((r.get("final_npa_rate") or 0) for r in runs) / len(runs)
        mn_tr = sum((r.get("final_trust") or 0) for r in runs) / len(runs)
        mn_rw = sum(r["total_reward"] for r in runs) / len(runs)
        # Lower is better: NPA dominates, then reward, trust as tie-breaker
        score = mn_pa * 10.0 - mn_tr * 0.5 - mn_rw * 0.02
        rows.append((score, str(d), mn_pa, mn_tr, mn_rw))
        _free(m)
        del t, pol
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows.sort(key=lambda x: x[0])
    print("\n" + "=" * 60)
    print("RANKED (best first). score = 10*NPA - 0.5*trust - 0.02*reward (lower is better)")
    print("=" * 60)
    for i, (sc, path, npa, tru, rw) in enumerate(rows, 1):
        print(f"{i:2d}. {Path(path).name:16s}  score={sc:.4f}  NPA={npa:.1%}  trust={tru:.3f}  reward={rw:+.3f}")
        print(f"     {path}")


if __name__ == "__main__":
    main()
