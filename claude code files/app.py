"""
FastAPI application for the Linguistic Decoding RL environment.

Exposes LinguisticDecodingEnvironment over HTTP and WebSocket endpoints.
Compatible with EnvClient and OpenEnv training infrastructure.

Domain is selected per-request via the `domain` field in reset/step payloads.
Currently registered domains: "msme_startup" (default).

Endpoints:
    POST /reset  — Start a new episode (pass {"domain": "msme_startup"})
    POST /step   — Execute an action
    GET  /state  — Get current environment state
    GET  /schema — Get action/observation schemas
    WS   /ws     — WebSocket for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
    uv run --project . server
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

try:
    from ..core.models import LinguisticDecodingAction, LinguisticDecodingObservation
    from .environment import LinguisticDecodingEnvironment
except (ModuleNotFoundError, ImportError):
    from core.models import LinguisticDecodingAction, LinguisticDecodingObservation
    from server.environment import LinguisticDecodingEnvironment


app = create_app(
    LinguisticDecodingEnvironment,
    LinguisticDecodingAction,
    LinguisticDecodingObservation,
    env_name="linguistic_decoding",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for: uv run --project . server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
