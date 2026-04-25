"""
Pre-submit checker for hackathon packaging readiness.

This is intentionally non-training: it validates structure, key files,
and already-generated artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _check_exists(path: Path, label: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"Missing {label}: {path}")


def _check_non_empty(path: Path, label: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"Missing {label}: {path}")
        return
    if path.stat().st_size == 0:
        errors.append(f"Empty {label}: {path}")


def main() -> None:
    errors: list[str] = []

    # Core OpenEnv files
    _check_non_empty(ROOT / "openenv.yaml", "OpenEnv manifest", errors)
    _check_non_empty(ROOT / "README.md", "README", errors)
    _check_non_empty(ROOT / "server" / "app.py", "server app", errors)
    _check_non_empty(ROOT / "server" / "msmeEnv_environment.py", "environment core", errors)
    _check_non_empty(ROOT / "train_grpo.py", "training script", errors)

    # Eval scripts
    for script in [
        "scripts/run_baseline_eval.py",
        "scripts/run_deterministic_eval.py",
        "scripts/generate_judge_artifacts.py",
        "scripts/check_domain_registry.py",
        "scripts/eval.py",
    ]:
        _check_non_empty(ROOT / script, f"script {script}", errors)

    # Domain architecture files
    for path in [
        "domains/base.py",
        "domains/__init__.py",
        "domains/msme_startup/adapter.py",
    ]:
        _check_non_empty(ROOT / path, f"domain file {path}", errors)

    # Artifacts expected before final submission
    artifacts = [
        "artifacts/baseline_rewards.json",
        "artifacts/deterministic_eval.json",
        "artifacts/judge_summary.json",
        "artifacts/judge_manifest.json",
        "artifacts/training_reward_curve.png",
        "artifacts/training_loss_curve.png",
    ]
    for rel in artifacts:
        _check_exists(ROOT / rel, f"artifact {rel}", errors)

    # If manifest exists, validate JSON parse and required keys
    manifest_path = ROOT / "artifacts" / "judge_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if "required" not in manifest:
                errors.append("judge_manifest.json missing 'required' key")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Invalid judge_manifest.json: {exc}")

    if errors:
        print("PRE-SUBMIT CHECK: FAIL")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)

    print("PRE-SUBMIT CHECK: PASS")


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()

