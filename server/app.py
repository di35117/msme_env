"""
FastAPI application for the MSME-RL Environment.

Exposes MSMERLEnvironment over HTTP and WebSocket endpoints.
Compatible with EnvClient and OpenEnv training infrastructure.

Also serves a single-page dashboard at GET / that:
  - displays the latest training plots (training_metrics.png, reward_curve.png, ...)
  - lets a visitor manually drive the environment via /reset and /step

Endpoints:
    GET  /         — HTML dashboard (live demo + training plots)
    GET  /web      — alias for /
    GET  /static/* — static assets (HTML + plots)
    POST /reset    — Start a new 36-month portfolio episode
    POST /step     — Execute an RM action (action_type, account_id, parameters)
    GET  /state    — Get current environment state
    GET  /schema   — Get action/observation schemas
    WS   /ws       — WebSocket for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
    uv run --project . server
"""

import shutil
from pathlib import Path

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

from threading import Lock
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from ..models import MSMERLAction, MSMERLObservation
    from .msmeEnv_environment import MSMERLEnvironment
except (ModuleNotFoundError, ImportError):
    from models import MSMERLAction, MSMERLObservation
    from server.msmeEnv_environment import MSMERLEnvironment


app = create_app(
    MSMERLEnvironment,
    MSMERLAction,
    MSMERLObservation,
    env_name="msmeEnv",
    max_concurrent_envs=4,
)


# ---------------------------------------------------------------------------
# Static dashboard (single-page HTML + plot images).
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _THIS_DIR / "static"
_PLOTS_DIR = _STATIC_DIR / "plots"
_DATA_DIR = _STATIC_DIR / "data"
_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _sync_plots_into_static() -> None:
    """Copy any training plots from the repo root into static/plots/ at startup.

    The trainer writes plots like ``training_metrics.png`` into the current
    working directory (or the directory passed to ``run_training``). For the HF
    Space deployment we want them under ``/static/plots/`` so the dashboard can
    embed them.

    Runs at import time so the Space picks up whatever artefacts ship with the
    repo. If you train fresh after deployment, drop the new PNGs into
    ``server/static/plots/`` (or re-run this function).
    """
    repo_root = _THIS_DIR.parent
    for plot_name in (
        "training_metrics.png",
        "reward_curve.png",
        "loss_curve.png",
        "training_reward_curve.png",
        "training_loss_curve.png",
        "per_episode_base_vs_trained.png",
        "reward_distribution_base_vs_trained.png",
        "inference_comparison.png",
        "inference_action_distribution.png",
        "inference_before_after.png",
        "inference_action_distribution_before_after.png",
        "inference_msme_vs_startup.png",
        "inference_action_shift.png",
    ):
        candidates = [
            repo_root / plot_name,
            repo_root / "artifacts" / plot_name,
            repo_root / "msme_rl_checkpoints_risk30_q25" / plot_name,
        ]
        for src in candidates:
            if src.exists() and src.is_file():
                dst = _PLOTS_DIR / plot_name
                try:
                    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                        shutil.copy2(src, dst)
                except Exception:
                    pass
                break


def _sync_dashboard_json_into_static() -> None:
    """Copy eval + training JSON into static/data/ for the dashboard demo UI."""
    repo_root = _THIS_DIR.parent
    dst_inf = _DATA_DIR / "inference_before_after.json"
    dst_rc = _DATA_DIR / "reward_curve.json"
    dst_js = _DATA_DIR / "judge_summary.json"

    def _copy_if_newer(src: Path, dst: Path) -> None:
        if not src.is_file():
            return
        try:
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(src, dst)
        except Exception:
            pass

    for name in ("inference_before_after.json",):
        for src in (repo_root / "artifacts" / name, repo_root / name):
            _copy_if_newer(src, dst_inf)

    rc_sources: list[Path] = []
    for src in (repo_root / "artifacts" / "reward_curve.json", repo_root / "reward_curve.json"):
        if src.is_file():
            rc_sources.append(src)
    try:
        run_dirs = [p for p in repo_root.glob("msme_rl_run*") if p.is_dir()]
        run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for d in run_dirs:
            rj = d / "reward_curve.json"
            if rj.is_file():
                rc_sources.insert(0, rj)
                break
    except Exception:
        pass
    for src in rc_sources:
        _copy_if_newer(src, dst_rc)
        if dst_rc.exists():
            break

    for src in (repo_root / "artifacts" / "judge_summary.json",):
        _copy_if_newer(src, dst_js)


_sync_plots_into_static()
_sync_dashboard_json_into_static()

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
# Hugging Face Spaces often iframes under /web — mirror static here too.
app.mount("/web/static", StaticFiles(directory=str(_STATIC_DIR)), name="web_static")


@app.get("/", include_in_schema=False)
def dashboard_root():
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/web", include_in_schema=False)
def dashboard_web():
    # HF Spaces routes the Space iframe to base_path: /web (per openenv.yaml).
    # Serve the HTML directly here instead of redirecting, so iframing works.
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/web/", include_in_schema=False)
def dashboard_web_slash():
    return FileResponse(str(_STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Stateful demo session for the dashboard.
#
# The default OpenEnv /reset and /step endpoints create a *new* environment
# on every HTTP request and close it after, which means session state is not
# preserved between calls. That's correct for distributed training but useless
# for a "click a button and step through an episode" demo.
#
# Below is a minimal single-session demo wrapper: there is exactly one
# persistent ``MSMERLEnvironment`` instance, protected by a lock. Good enough
# for one or two judges hitting the Space at the same time.
# ---------------------------------------------------------------------------

_demo_env: MSMERLEnvironment | None = None
_demo_lock = Lock()


def _ensure_demo_env() -> MSMERLEnvironment:
    global _demo_env
    if _demo_env is None:
        _demo_env = MSMERLEnvironment()
    return _demo_env


def _obs_to_dict(obs: MSMERLObservation) -> Dict[str, Any]:
    """Convert an observation dataclass/Pydantic model to a plain JSON dict."""
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "to_dict"):
        return obs.to_dict()
    if hasattr(obs, "__dict__"):
        return dict(obs.__dict__)
    return obs  # type: ignore[return-value]


class DemoActionRequest(BaseModel):
    action_type: str
    account_id: int
    parameters: Dict[str, Any] | None = None
    reasoning: str | None = "(manual demo)"


@app.post("/demo/reset", include_in_schema=False)
def demo_reset():
    """Reset the persistent demo environment and return the initial observation."""
    with _demo_lock:
        env = _ensure_demo_env()
        try:
            obs = env.reset()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Demo reset failed: {e!s}") from e
        return _obs_to_dict(obs)


@app.post("/demo/step", include_in_schema=False)
def demo_step(req: DemoActionRequest):
    """Step the persistent demo environment and return the new observation."""
    with _demo_lock:
        env = _ensure_demo_env()
        try:
            action = MSMERLAction(
                action_type=req.action_type,
                account_id=req.account_id,
                parameters=req.parameters or {},
                reasoning=req.reasoning or "(manual demo)",
            )
            obs = env.step(action)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Demo step failed: {e!s}") from e
        return _obs_to_dict(obs)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for: uv run --project . server

    For production with multiple workers:
        uvicorn msmeEnv.server.app:app --workers 4
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)