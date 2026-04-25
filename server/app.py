"""
FastAPI application for the MSME-RL Environment.

Exposes MSMERLEnvironment over HTTP and WebSocket endpoints.
Compatible with EnvClient and OpenEnv training infrastructure.

Endpoints:
    POST /reset  — Start a new 36-month portfolio episode
    POST /step   — Execute an RM action (action_type, account_id, parameters)
    GET  /state  — Get current environment state
    GET  /schema — Get action/observation schemas
    WS   /ws     — WebSocket for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
    uv run --project . server
"""

from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import MSMERLAction, MSMERLObservation
    from .msmeEnv_environment import MSMERLEnvironment
except (ModuleNotFoundError, ImportError):  # <-- Added ImportError here
    from models import MSMERLAction, MSMERLObservation
    from server.msmeEnv_environment import MSMERLEnvironment


app = create_app(
    MSMERLEnvironment,
    MSMERLAction,
    MSMERLObservation,
    env_name="msmeEnv",
    max_concurrent_envs=4,  # Multiple training workers
)


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