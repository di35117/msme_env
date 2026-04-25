"""
MSME + startup domain adapter.

This adapter wraps the existing production logic so we can adopt a
generalized architecture incrementally without breaking the current
OpenEnv runtime.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..base import DomainAdapter

try:
    from ...models import ACTION_TYPES
    from ...world_generator import generate_portfolio
    from ...reward import (
        classify_action_outcome,
        compute_step_reward,
        compute_episode_reward,
    )
    from ...network import (
        propagate_msme_cluster_effect,
        propagate_startup_ecosystem_effect,
        check_cross_contamination,
    )
except (ModuleNotFoundError, ImportError):
    from models import ACTION_TYPES
    from world_generator import generate_portfolio
    from reward import (
        classify_action_outcome,
        compute_step_reward,
        compute_episode_reward,
    )
    from network import (
        propagate_msme_cluster_effect,
        propagate_startup_ecosystem_effect,
        check_cross_contamination,
    )


class MSMEStartupAdapter(DomainAdapter):
    @property
    def domain_id(self) -> str:
        return "msme_startup"

    @property
    def total_entities(self) -> int:
        return 30

    @property
    def time_horizon(self) -> int:
        return 36

    @property
    def valid_actions(self) -> List[str]:
        # Literal members from ACTION_TYPES are available at runtime as tuple-like values.
        return list(ACTION_TYPES.__args__) if hasattr(ACTION_TYPES, "__args__") else []  # type: ignore[attr-defined]

    def generate_world(
        self,
        episode: int,
        adversarial_weaknesses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        # Existing world generator already handles curriculum using episode number.
        return generate_portfolio(episode=episode)

    def classify_outcome(
        self,
        action_type: str,
        account_type: str,
        hidden_profile: Dict[str, Any],
        current_month: int,
        action_params: Dict[str, Any],
    ) -> str:
        return classify_action_outcome(
            action_type=action_type,
            account_type=account_type,
            hidden_profile=hidden_profile,
            current_month=current_month,
            action_params=action_params,
        )

    def compute_step_reward(
        self,
        action_type: str,
        account_type: str,
        outcome: str,
        hidden_profile: Dict[str, Any],
    ) -> float:
        return compute_step_reward(
            action_type=action_type,
            account_type=account_type,
            outcome=outcome,
            hidden_profile=hidden_profile,
        )

    def compute_episode_reward(
        self,
        hidden_profiles: Dict[int, Dict[str, Any]],
        episode_history: List[Dict[str, Any]],
        episode_num: int,
        final_month: int,
    ) -> Dict[str, float]:
        return compute_episode_reward(
            hidden_profiles=hidden_profiles,
            episode_history=episode_history,
            episode_num=episode_num,
            final_month=final_month,
        )

    def propagate_effects(
        self,
        hidden_profiles: Dict[int, Dict[str, Any]],
        account_id: int,
        account_type: str,
        action_type: str,
        outcome: str,
        trust_delta: float,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[int]]:
        cluster_effects: Dict[str, Any] = {}
        ecosystem_effects: Dict[str, Any] = {}
        cross_contamination: List[int] = []

        if account_type == "msme":
            effect_type = self._map_msme_effect(action_type, outcome)
            if effect_type:
                cluster_effects = propagate_msme_cluster_effect(
                    hidden_profiles,
                    account_id,
                    effect_type,
                    effect_strength=abs(trust_delta) * 2,
                )
                if effect_type in ("sarfaesi", "npa"):
                    cross_contamination = check_cross_contamination(
                        hidden_profiles,
                        account_id,
                        hidden_profiles.get(account_id, {}).get("cluster_centrality", 0.5),
                    )
        elif account_type == "startup":
            effect_type = self._map_startup_effect(action_type, outcome)
            if effect_type:
                ecosystem_effects = propagate_startup_ecosystem_effect(
                    hidden_profiles,
                    account_id,
                    effect_type,
                    effect_strength=abs(trust_delta) * 2,
                )

        return cluster_effects, ecosystem_effects, cross_contamination

    @staticmethod
    def _map_msme_effect(action_type: str, outcome: str) -> Optional[str]:
        if action_type in ("initiate_sarfaesi", "send_legal_notice_section13"):
            return "sarfaesi"
        if outcome in ("account_npa_no_intervention", "cluster_cascade_default"):
            return "npa"
        if outcome in ("payment_received_after_moratorium", "payment_received_after_empathy"):
            return "recovery"
        if action_type == "grant_moratorium":
            return "moratorium"
        return None

    @staticmethod
    def _map_startup_effect(action_type: str, outcome: str) -> Optional[str]:
        if action_type == "initiate_sarfaesi":
            return "harsh_action"
        if outcome == "ecosystem_cascade_ghosting":
            return "ghost_detected"
        if outcome == "investor_meeting_triggered_bridge":
            return "bridge_arranged"
        if outcome in ("payment_received_after_moratorium", "payment_received_after_empathy"):
            return "recovery"
        return None

