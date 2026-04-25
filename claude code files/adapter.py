"""
MSMEStartupAdapter — Domain Module A+B.

Wires the MSME+Startup domain logic into the core DomainAdapter interface.

This file is intentionally thin: it delegates to the domain modules
(world_generator, reward, network, message_generator) that contain
the actual logic. The adapter is the boundary layer, not the brain.

Domain properties:
  - 30 speakers (20 MSME + 10 startup)
  - 36 time steps (months)
  - 21 action types
  - Two network topologies: MSME cluster + startup ecosystem
  - Asymmetric linguistic decoding: understatement vs overstatement
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from ..base import DomainAdapter
from .world_generator import generate_world as _generate_world
from .reward import (
    classify_action_outcome,
    compute_step_reward as _compute_step_reward,
    compute_episode_reward as _compute_episode_reward,
    analyze_agent_weaknesses,
)
from .network import (
    propagate_msme_cluster_effect,
    propagate_startup_ecosystem_effect,
    apply_network_effects,
    collect_active_alerts,
)
from .message_generator import generate_rm_message


# ---------------------------------------------------------------------------
# Valid action types for this domain
# ---------------------------------------------------------------------------

VALID_ACTIONS = [
    # Communication
    "send_empathetic_reminder",
    "send_firm_reminder",
    "send_legal_notice_section13",
    "call_promoter_founder",
    "call_guarantor_investor",
    "conduct_cluster_ecosystem_visit",
    # Financial restructuring
    "grant_moratorium",
    "restructure_emi",
    "offer_eclgs_topup",
    "offer_bridge_loan_extension",
    "accept_partial_payment",
    "waive_penal_interest",
    # Recovery
    "initiate_sarfaesi",
    "refer_to_recovery_agent",
    "file_drt_case",
    "offer_one_time_settlement",
    # Information gathering
    "verify_gst_returns",
    "pull_bank_statements",
    "check_industry_cluster_stress",
    "request_investor_update_meeting",
    "check_startup_ecosystem_signals",
    # No-op
    "wait_and_observe",
]

# NPA DPD thresholds (RBI standard)
NPA_DPD_THRESHOLD = 90


class MSMEStartupAdapter(DomainAdapter):
    """
    Domain adapter for the MSME + Startup credit management domain.

    Teaches a 1.7B LM to decode two opposite linguistic strategies
    from reward signal alone:
      - MSME owners who UNDERSTATE problems (Hindi/Hinglish/Marathi)
      - Startup founders who OVERSTATE health (pitch-deck English)
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def domain_id(self) -> str:
        return "msme_startup"

    @property
    def total_speakers(self) -> int:
        return 30  # 20 MSME + 10 startup

    @property
    def time_steps(self) -> int:
        return 36  # months

    @property
    def valid_actions(self) -> List[str]:
        return VALID_ACTIONS

    # ------------------------------------------------------------------
    # World generation
    # ------------------------------------------------------------------

    def generate_world(
        self,
        episode: int,
        adversarial_weaknesses: Optional[Dict[str, float]] = None,
    ) -> Dict[int, Dict]:
        return _generate_world(
            episode=episode,
            adversarial_weaknesses=adversarial_weaknesses or {},
        )

    # ------------------------------------------------------------------
    # Observable state
    # ------------------------------------------------------------------

    def build_speakers_observation(
        self,
        hidden_profiles: Dict[int, Dict],
        time_step: int,
        episode_history: List[Dict],
    ) -> List[Dict]:
        """
        Strip hidden fields and build observable speaker states.
        MSME and startup share the same schema at this level;
        domain_state carries the split for prompt context.
        """
        HIDDEN_FIELDS = {
            "true_financial_health", "true_runway_months",
            "strategic_default_propensity", "ghosting_propensity",
            "investor_bridge_probability", "crisis_trigger_month",
            "crisis_trigger", "went_npa", "amount_recovered",
            "founder_optimism_bias", "pivot_risk",
        }

        speakers = []
        for speaker_id, profile in sorted(hidden_profiles.items()):
            observable = {
                k: v for k, v in profile.items()
                if k not in HIDDEN_FIELDS
            }
            observable["id"] = speaker_id
            # Generate time-step-appropriate message
            observable["last_message"] = self._get_current_message(
                profile, time_step
            )
            speakers.append(observable)
        return speakers

    def build_portfolio_summary(
        self,
        hidden_profiles: Dict[int, Dict],
        time_step: int,
    ) -> Dict:
        msme = [p for p in hidden_profiles.values() if p.get("account_type") == "msme"]
        startup = [p for p in hidden_profiles.values() if p.get("account_type") == "startup"]
        npa_count = sum(1 for p in hidden_profiles.values() if p.get("went_npa"))

        return {
            "month": time_step,
            "total_accounts": len(hidden_profiles),
            "msme_count": len(msme),
            "startup_count": len(startup),
            "npa_count": npa_count,
            "avg_trust_score": round(
                sum(p.get("trust_score", 0.5) for p in hidden_profiles.values())
                / max(1, len(hidden_profiles)),
                3,
            ),
        }

    def build_domain_state(
        self,
        hidden_profiles: Dict[int, Dict],
        time_step: int,
    ) -> Dict:
        """Split speakers into typed lists for the agent's prompt context."""
        HIDDEN_FIELDS = {
            "true_financial_health", "true_runway_months",
            "strategic_default_propensity", "ghosting_propensity",
            "investor_bridge_probability", "crisis_trigger_month",
            "crisis_trigger", "went_npa", "amount_recovered",
            "founder_optimism_bias", "pivot_risk",
        }

        msme_accounts, startup_accounts = [], []
        for speaker_id, profile in sorted(hidden_profiles.items()):
            obs = {k: v for k, v in profile.items() if k not in HIDDEN_FIELDS}
            obs["id"] = speaker_id
            obs["last_message"] = self._get_current_message(profile, time_step)
            if profile.get("account_type") == "msme":
                msme_accounts.append(obs)
            else:
                startup_accounts.append(obs)

        return {
            "msme_accounts": msme_accounts,
            "startup_accounts": startup_accounts,
        }

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def classify_outcome(
        self,
        action_type: str,
        target_id: int,
        hidden_profiles: Dict[int, Dict],
        time_step: int,
        parameters: Dict[str, Any],
    ) -> str:
        if action_type not in VALID_ACTIONS:
            return "malformed_json_format"

        profile = hidden_profiles.get(target_id, {})
        account_type = profile.get("account_type", "msme")

        return classify_action_outcome(
            action_type=action_type,
            account_type=account_type,
            hidden_profile=profile,
            current_month=time_step,
            action_params=parameters,
        )

    def compute_step_reward(
        self,
        action_type: str,
        target_id: int,
        outcome: str,
        hidden_profiles: Dict[int, Dict],
    ) -> float:
        profile = hidden_profiles.get(target_id, {})
        account_type = profile.get("account_type", "msme")
        return _compute_step_reward(
            action_type=action_type,
            account_type=account_type,
            outcome=outcome,
            hidden_profile=profile,
        )

    # ------------------------------------------------------------------
    # Network effects
    # ------------------------------------------------------------------

    def propagate_network_effects(
        self,
        hidden_profiles: Dict[int, Dict],
        action_type: str,
        target_id: int,
        outcome: str,
    ) -> Tuple[Dict[int, Dict], List[str]]:
        profile = hidden_profiles.get(target_id, {})
        account_type = profile.get("account_type", "msme")

        all_effects = []

        if account_type == "msme":
            effect_type = self._msme_effect_type(action_type, outcome)
            if effect_type:
                effects = propagate_msme_cluster_effect(
                    hidden_profiles, target_id, effect_type
                )
                if effects:
                    all_effects.append(effects)
                    hidden_profiles = apply_network_effects(hidden_profiles, effects)
        else:
            effect_type = self._startup_effect_type(action_type, outcome)
            if effect_type:
                effects = propagate_startup_ecosystem_effect(
                    hidden_profiles, target_id, effect_type
                )
                if effects:
                    all_effects.append(effects)
                    hidden_profiles = apply_network_effects(hidden_profiles, effects)

        alerts = collect_active_alerts(all_effects)
        return hidden_profiles, alerts

    # ------------------------------------------------------------------
    # Episode reward
    # ------------------------------------------------------------------

    def compute_episode_reward(
        self,
        hidden_profiles: Dict[int, Dict],
        episode_history: List[Dict],
        episode_num: int,
    ) -> Dict[str, float]:
        return _compute_episode_reward(
            hidden_profiles=hidden_profiles,
            episode_history=episode_history,
            episode_num=episode_num,
        )

    # ------------------------------------------------------------------
    # Weakness analysis
    # ------------------------------------------------------------------

    def analyze_weaknesses(
        self,
        episode_histories: List[List[Dict]],
    ) -> Dict[str, float]:
        return analyze_agent_weaknesses(episode_histories)

    # ------------------------------------------------------------------
    # Message generation
    # ------------------------------------------------------------------

    def generate_message(
        self,
        action_type: str,
        target_id: int,
        hidden_profiles: Dict[int, Dict],
        observable: Dict,
        parameters: Dict[str, Any],
    ) -> str:
        profile = hidden_profiles.get(target_id, {})
        account_type = profile.get("account_type", "msme")
        return generate_rm_message(
            action_type=action_type,
            account_type=account_type,
            account_profile=profile,
            observable=observable,
            action_params=parameters,
        )

    # ------------------------------------------------------------------
    # Time step advancement
    # ------------------------------------------------------------------

    def advance_time_step(
        self,
        hidden_profiles: Dict[int, Dict],
        time_step: int,
        episode_history: List[Dict],
    ) -> Dict[int, Dict]:
        """
        Advance hidden profiles to next month.
          - Apply trajectory (health declining/improving per month)
          - Activate crisis triggers
          - Classify NPA at DPD >= 90
          - Evolve DPD based on payment outcomes
        """
        for speaker_id, profile in hidden_profiles.items():
            # Trajectory evolution
            trajectory = profile.get("health_trajectory", "stable")
            health = profile.get("true_financial_health", 0.5)
            if trajectory == "declining":
                health = max(0.0, health - random.uniform(0.01, 0.04))
            elif trajectory == "recovering":
                health = min(1.0, health + random.uniform(0.01, 0.03))
            profile["true_financial_health"] = round(health, 4)

            # Crisis trigger
            crisis_month = profile.get("crisis_trigger_month")
            if crisis_month and time_step >= crisis_month:
                profile["true_financial_health"] = min(
                    profile["true_financial_health"], 0.30
                )

            # DPD evolution (simplified — full version uses payment outcomes)
            if health < 0.25:
                profile["dpd"] = profile.get("dpd", 0) + random.randint(10, 20)
            elif health > 0.60:
                profile["dpd"] = max(0, profile.get("dpd", 0) - random.randint(0, 5))

            # NPA classification
            if profile.get("dpd", 0) >= NPA_DPD_THRESHOLD and not profile.get("went_npa"):
                profile["went_npa"] = True

        return hidden_profiles

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_current_message(self, profile: Dict, time_step: int) -> str:
        """
        Return a time-step-appropriate message from this speaker.
        In production this calls the LLM; here it returns the
        pre-generated message stored in the profile.
        """
        messages = profile.get("messages", {})
        # Prefer message for this exact month, fall back to closest
        if time_step in messages:
            return messages[time_step]
        # Fallback: use latest available
        available = sorted(k for k in messages if k <= time_step)
        if available:
            return messages[available[-1]]
        return profile.get("default_message", "")

    def _msme_effect_type(self, action_type: str, outcome: str) -> Optional[str]:
        if action_type == "initiate_sarfaesi":
            return "sarfaesi"
        if outcome in ("account_npa_no_intervention",):
            return "npa"
        if outcome in ("payment_received_after_moratorium", "payment_received_after_empathy"):
            return "recovery"
        if action_type == "grant_moratorium":
            return "moratorium"
        return None

    def _startup_effect_type(self, action_type: str, outcome: str) -> Optional[str]:
        if action_type in ("initiate_sarfaesi", "refer_to_recovery_agent", "file_drt_case"):
            return "harsh_action"
        if outcome in ("ghost_detected_too_late", "ecosystem_cascade_ghosting"):
            return "ghost_detected"
        if outcome == "investor_meeting_triggered_bridge":
            return "bridge_arranged"
        if outcome in ("payment_received_after_moratorium", "payment_received_after_empathy"):
            return "recovery"
        return None
