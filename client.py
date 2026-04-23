"""
MSME-RL Environment Client.

Persistent WebSocket connection to the MSME-RL environment server.
Enables efficient multi-step interactions across a 36-month loan cycle
with 30 mixed MSME + startup accounts.
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MSMERLAction, MSMERLObservation


class MSMERLEnv(EnvClient[MSMERLAction, MSMERLObservation, State]):
    """
    Client for the MSME-RL Environment.

    Teaches a 1.7B language model to manage a mixed portfolio of 20 MSME accounts
    and 10 startup accounts across a 36-month loan cycle — learning to decode:
      - MSME owners who UNDERSTATE their problems (Hindi/Hinglish)
      - Startup founders who OVERSTATE their health (pitch-deck English)

    Example:
        >>> with MSMERLEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(f"Episode started. Month: {result.observation.month}")
        ...
        ...     # Agent decides to verify GST before acting on MSME account 7
        ...     action = MSMERLAction(
        ...         action_type="verify_gst_returns",
        ...         account_id=7,
        ...         parameters={},
        ...         reasoning="Account 7 shows OEM delay. GST verification before moratorium decision.",
        ...     )
        ...     result = client.step(action)
        ...     print(f"Outcome: {result.observation.last_action_result['outcome']}")
        ...     print(f"Reward: {result.reward:.3f}")
        ...     print(f"Semantic memory: {result.observation.semantic_memory_context}")

    Example with Docker:
        >>> client = MSMERLEnv.from_docker_image("msmeEnv-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     msme_accounts  = result.observation.msme_accounts
        ...     startup_accounts = result.observation.startup_accounts
        ...     print(f"Portfolio: {len(msme_accounts)} MSME + {len(startup_accounts)} startup")
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: MSMERLAction) -> Dict:
        """Convert MSMERLAction to JSON payload."""
        return {
            "action_type": action.action_type,
            "account_id":  action.account_id,
            "parameters":  action.parameters,
            "reasoning":   action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MSMERLObservation]:
        """Parse server response into StepResult[MSMERLObservation]."""
        obs_data = payload.get("observation", {})

        observation = MSMERLObservation(
            episode=obs_data.get("episode", 1),
            month=obs_data.get("month", 1),
            msme_accounts=obs_data.get("msme_accounts", []),
            startup_accounts=obs_data.get("startup_accounts", []),
            portfolio_summary=obs_data.get("portfolio_summary", {}),
            working_memory=obs_data.get("working_memory", ""),
            semantic_memory_context=obs_data.get("semantic_memory_context", ""),
            episodic_memory_context=obs_data.get("episodic_memory_context", ""),
            last_action_result=obs_data.get("last_action_result"),
            active_cluster_alerts=obs_data.get("active_cluster_alerts", []),
            active_ecosystem_alerts=obs_data.get("active_ecosystem_alerts", []),
            step_reward=obs_data.get("step_reward", 0.0),
            episode_reward_so_far=obs_data.get("episode_reward_so_far", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )