# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MSME-RL Environment Implementation.

A mixed portfolio of 20 MSME + 10 startup accounts across a 36-month loan cycle.
The agent learns to decode two opposite linguistic strategies from reward signal alone:
  - MSME owners who UNDERSTATE their problems in Hindi/Hinglish
  - Startup founders who OVERSTATE their health in pitch-deck English

Zero hardcoded rules. Policy discovered purely from reward signal.

Calibrated to published data:
  - RBI Annual Report FY24 (MSME NPA rates by sector)
  - NASSCOM / CIBIL 2023 (startup default rates)
  - SIDBI MSME Pulse Report (cluster contagion factor = 2.3)
  - IBA study 2023 (moratorium vs SARFAESI recovery rates)
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MSMERLAction, MSMERLObservation
    from ..domains import get_adapter
    from ..world_generator import build_msme_observable, build_startup_observable
    from ..network import (
        apply_network_effects,
        collect_active_alerts,
    )
    from ..reward import (
        _is_appropriate_tool,
    )
    from ..memory import MemoryManager
    from ..message_generator import generate_rm_message
except (ModuleNotFoundError, ImportError):  # FIXED: Catching both error types safely
    from models import MSMERLAction, MSMERLObservation
    from domains import get_adapter
    from world_generator import build_msme_observable, build_startup_observable
    from network import (
        apply_network_effects,
        collect_active_alerts,
    )
    from reward import (
        _is_appropriate_tool,
    )
    from memory import MemoryManager
    from message_generator import generate_rm_message

# Max actions per episode before environment auto-advances to avoid infinite loops
MAX_STEPS_PER_EPISODE = 36 * 30  


class MSMERLEnvironment(Environment):
    """
    The MSME-RL environment: teaching a 1.7B language model to read between
    the lines of Indian business communication.

    World state:
      - 20 MSME accounts with hidden financial health (understaters)
      - 10 startup accounts with hidden runway (overstaters)
      - Two network topologies: MSME clusters + startup ecosystems
      - Endogenous trust model: agent actions change repayment probabilities
      - Three-tier memory: episodic + semantic + working memory

    The agent's task: manage this mixed portfolio for 36 months,
    minimizing NPA rate and maximizing recovery rate and relationship scores.

    Reward signal: NPA rate + recovery rate + relationship score + tool appropriateness.
    All hard numbers. No LLM judge.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the MSME-RL environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_num = 0

        # Current episode state
        self._portfolio: Dict = {}              # full portfolio (hidden + observable)
        self._hidden_profiles: Dict = {}        # true hidden state
        self._observable_states: Dict = {}      # observable signals per account
        self._current_month: int = 1
        self._episode_history: List[Dict] = []  # all steps this episode
        self._active_cluster_alerts: List[str] = []
        self._active_ecosystem_alerts: List[str] = []
        self._step_count_this_episode: int = 0
        self._episode_cumulative_reward: float = 0.0

        # Memory — persists across episodes (this is the learning signal)
        self._memory = MemoryManager()

        # Episode histories for adversarial curriculum
        self._all_episode_histories: List[List[Dict]] = []
        # Domain adapter for generalized architecture (defaults to current domain)
        self._domain_adapter = get_adapter("msme_startup")

    # -------------------------------------------------------------------------
    # RESET
    # -------------------------------------------------------------------------

    def reset(self) -> MSMERLObservation:
        """
        Reset the environment — start a new 36-month portfolio episode.
        Generates a fresh portfolio of 20 MSME + 10 startup accounts.
        """
        self._episode_num += 1
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Generate new portfolio
        self._portfolio = self._domain_adapter.generate_world(episode=self._episode_num)
        self._hidden_profiles = self._portfolio["hidden_profiles"]
        self._observable_states = self._portfolio["observable_states"]

        # FIXED: Horizon Curriculum. Start near crisis (Month 34) and walk back to Month 1
        # as episodes increment to prevent sparse reward wandering.
        start_month = max(1, 34 - (self._episode_num // 5) * 4)
        
        self._current_month = start_month
        self._episode_history = []
        self._active_cluster_alerts = []
        self._active_ecosystem_alerts = []
        self._step_count_this_episode = 0
        self._episode_cumulative_reward = 0.0

        # Build initial working memory
        working_mem = self._memory.working.refresh(
            month=self._current_month,
            episode=self._episode_num,
            hidden_profiles=self._hidden_profiles,
            observable_states=self._observable_states,
            recent_actions=[],
            active_cluster_alerts=[],
            active_ecosystem_alerts=[],
        )

        return MSMERLObservation(
            episode=self._episode_num,
            month=self._current_month,
            msme_accounts=self._get_msme_observables(),
            startup_accounts=self._get_startup_observables(),
            portfolio_summary=self._build_portfolio_summary(),
            working_memory=working_mem,
            semantic_memory_context="(episode start — no patterns retrieved yet)",
            episodic_memory_context=(
                "(episode start — no past cases yet)"
                if self._episode_num == 1
                else f"(patterns available from {self._episode_num - 1} prior episodes)"
            ),
            active_cluster_alerts=[],
            active_ecosystem_alerts=[],
            step_reward=0.0,
            episode_reward_so_far=0.0,
            done=False,
            reward=0.0,
            metadata={
                "portfolio_id": self._portfolio["episode_id"],
                "msme_count": len(self._portfolio["msme_ids"]),
                "startup_count": len(self._portfolio["startup_ids"]),
                "episode": self._episode_num,
            },
        )

    # -------------------------------------------------------------------------
    # STEP
    # -------------------------------------------------------------------------

    def step(self, action: MSMERLAction) -> MSMERLObservation:  # type: ignore[override]
        """
        Execute one RM action in the environment.

        The action targets a specific account (account_id).
        The environment:
          1. Validates the action
          2. Classifies the outcome using hidden profile state
          3. Applies endogenous trust update
          4. Propagates network effects (cluster/ecosystem)
          5. Computes step reward
          6. Updates memory (episodic + semantic)
          7. Advances month if all accounts have been acted on
          8. Returns new observation with memory context injected
        """
        self._state.step_count += 1
        self._step_count_this_episode += 1

        account_id   = action.account_id
        action_type  = action.action_type
        params       = action.parameters
        step_started_at = time.perf_counter()

        # Validate account
        if account_id not in self._hidden_profiles:
            return self._error_observation(
                f"Invalid account_id {account_id}. Valid: 1-30."
            )

        # Validate action type explicitly to prevent bypasses in direct mode.
        valid_actions = set(self._domain_adapter.valid_actions or [])
        if action_type not in valid_actions:
            return self._error_observation(
                f"Invalid action_type '{action_type}'. Must be one of configured domain actions."
            )

        hidden_profile = self._hidden_profiles[account_id]
        account_type   = hidden_profile.get("account_type", "msme")
        observable     = self._observable_states[account_id]

        # Validate action parameters with strict schema/range checks.
        params_error = self._validate_action_parameters(action_type, account_type, params)
        if params_error:
            return self._error_observation(params_error)

        # 1. Classify outcome using hidden state
        outcome = self._domain_adapter.classify_outcome(
            action_type=action_type,
            account_type=account_type,
            hidden_profile=hidden_profile,
            current_month=self._current_month,
            action_params=params,
        )

        # 2. Compute step reward
        step_reward = self._domain_adapter.compute_step_reward(
            action_type=action_type,
            account_type=account_type,
            outcome=outcome,
            hidden_profile=hidden_profile,
        )
        guardrail_penalty, guardrail_flags = self._compute_behavior_penalties(account_id, action_type)
        budget_penalty = self._compute_budget_penalty(action.reasoning)
        step_reward = round(step_reward - guardrail_penalty - budget_penalty, 4)

        # 3. Endogenous trust update — agent actions change repayment probability
        trust_delta = self._apply_trust_update(
            account_id, account_type, action_type, outcome, hidden_profile
        )

        # 4. Propagate network effects
        cluster_effects    = {}
        ecosystem_effects  = {}
        cross_contamination = []

        cluster_effects, ecosystem_effects, cross_contamination = self._domain_adapter.propagate_effects(
            hidden_profiles=self._hidden_profiles,
            account_id=account_id,
            account_type=account_type,
            action_type=action_type,
            outcome=outcome,
            trust_delta=trust_delta,
        )

        # Apply network effects to hidden profiles
        if cluster_effects:
            self._hidden_profiles = apply_network_effects(self._hidden_profiles, cluster_effects)
        if ecosystem_effects:
            self._hidden_profiles = apply_network_effects(self._hidden_profiles, ecosystem_effects)

        # Collect alerts
        new_cluster_alerts   = collect_active_alerts([cluster_effects])
        new_ecosystem_alerts = collect_active_alerts([ecosystem_effects])
        if cross_contamination:
            new_cluster_alerts.append(
                f"MSME cascade may affect startup accounts: {cross_contamination}"
            )

        self._active_cluster_alerts   = (self._active_cluster_alerts + new_cluster_alerts)[-5:]
        self._active_ecosystem_alerts = (self._active_ecosystem_alerts + new_ecosystem_alerts)[-5:]

        # 5. Generate RM message for this action
        rm_message = generate_rm_message(
            action_type=action_type,
            account_type=account_type,
            account_profile=hidden_profile,
            observable=observable,
            action_params=params,
        )

        # 6. Record step in memory
        industry_or_stage = hidden_profile.get("industry") or hidden_profile.get("stage", "")
        self._memory.record_step(
            episode=self._episode_num,
            month=self._current_month,
            account_id=account_id,
            account_type=account_type,
            action_type=action_type,
            outcome=outcome,
            reward=step_reward,
            hidden_profile=hidden_profile,
            observable=observable,
            trust_delta=trust_delta,
        )

        # 7. Log to episode history (used by reward.compute_episode_reward for NPA,
        # ToolFit, anti-cheat — action_type + account_type drive _is_appropriate_tool).
        step_record = {
            "episode": self._episode_num,
            "month": self._current_month,
            "account_id": account_id,
            "account_type": account_type,
            "action_type": action_type,
            "outcome": outcome,
            "reward": step_reward,
            "trust_delta": trust_delta,
            "rm_message": rm_message,
            "reasoning": action.reasoning,
            "guardrail_penalty": guardrail_penalty,
            "budget_penalty": budget_penalty,
            "guardrail_flags": guardrail_flags,
        }
        self._episode_history.append(step_record)
        self._episode_cumulative_reward += step_reward

        # 8. Update observable state for this account
        if account_type == "msme":
            self._observable_states[account_id] = build_msme_observable(
                self._hidden_profiles[account_id]
            )
        else:
            self._observable_states[account_id] = build_startup_observable(
                self._hidden_profiles[account_id]
            )

        # 9. Advance month after every 30 actions (one action per account per month)
        # Horizon is exactly 36 months. When the 30th action of month 36 completes,
        # the episode must end with done=True and month reported as 36 — never
        # incrementing to a fictitious "month 37" (that corrupted logs, NPA, and RL).
        done = False
        episode_reward_breakdown = None
        if self._step_count_this_episode % 30 == 0:
            if self._current_month >= 36:
                # Completed 30/30 actions in the final month: terminate here.
                # _simulate_spontaneous_defaults() expects self._current_month == M+1
                # so "prior" month is M (same pattern as a normal month advance).
                self._current_month += 1
                self._simulate_spontaneous_defaults()
                self._current_month = 36
                done = True
                episode_reward_breakdown = self._domain_adapter.compute_episode_reward(
                    self._hidden_profiles,
                    self._episode_history,
                    episode_num=self._episode_num,
                    final_month=36,
                )
                self._all_episode_histories.append(self._episode_history)
            else:
                self._current_month += 1
                self._memory.working.refresh(
                    month=self._current_month,
                    episode=self._episode_num,
                    hidden_profiles=self._hidden_profiles,
                    observable_states=self._observable_states,
                    recent_actions=self._episode_history[-10:],
                    active_cluster_alerts=self._active_cluster_alerts,
                    active_ecosystem_alerts=self._active_ecosystem_alerts,
                )
                self._simulate_spontaneous_defaults()

        # 10. Build memory context for this observation
        episodic_ctx, semantic_ctx = self._memory.build_context(
            account_id=account_id,
            account_type=account_type,
            observable=self._observable_states[account_id],
            industry_or_stage=industry_or_stage,
            action_type=action_type,
        )

        working_mem = self._memory.working.refresh(
            month=self._current_month,
            episode=self._episode_num,
            hidden_profiles=self._hidden_profiles,
            observable_states=self._observable_states,
            recent_actions=self._episode_history[-10:],
            active_cluster_alerts=self._active_cluster_alerts,
            active_ecosystem_alerts=self._active_ecosystem_alerts,
        )

        final_reward = (
            episode_reward_breakdown["total"] if episode_reward_breakdown else step_reward
        )

        return MSMERLObservation(
            episode=self._episode_num,
            month=self._current_month,
            msme_accounts=self._get_msme_observables(),
            startup_accounts=self._get_startup_observables(),
            portfolio_summary=self._build_portfolio_summary(),
            working_memory=working_mem,
            semantic_memory_context=semantic_ctx,
            episodic_memory_context=episodic_ctx,
            last_action_result={
                "account_id":       account_id,
                "account_type":     account_type,
                "action_type":      action_type,
                "outcome":          outcome,
                "step_reward":      step_reward,
                "trust_delta":      trust_delta,
                "rm_message":       rm_message,
                "cluster_effects":  bool(cluster_effects),
                "ecosystem_effects": bool(ecosystem_effects),
                "cross_contamination": cross_contamination,
                "episode_reward_breakdown": episode_reward_breakdown,
                "guardrail_penalty": guardrail_penalty,
                "budget_penalty": budget_penalty,
                "guardrail_flags": guardrail_flags,
            },
            active_cluster_alerts=self._active_cluster_alerts,
            active_ecosystem_alerts=self._active_ecosystem_alerts,
            step_reward=step_reward,
            episode_reward_so_far=self._episode_cumulative_reward,
            done=done,
            reward=final_reward,
            metadata={
                "step":             self._step_count_this_episode,
                "month":            self._current_month,
                "episode":          self._episode_num,
                "is_appropriate_tool": _is_appropriate_tool(action_type, account_type),
                "episode_history_length": len(self._episode_history),
                "semantic_patterns_known": self._memory.semantic.pattern_count,
                "episodic_records_total": self._memory.episodic.total_records,
                "step_latency_ms": round((time.perf_counter() - step_started_at) * 1000, 2),
            },
        )

    # -------------------------------------------------------------------------
    # STATE PROPERTY
    # -------------------------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    # -------------------------------------------------------------------------
    # PRIVATE HELPERS
    # -------------------------------------------------------------------------

    def _apply_trust_update(
        self,
        account_id: int,
        account_type: str,
        action_type: str,
        outcome: str,
        hidden_profile: Dict,
    ) -> float:
        """
        Update trust_score endogenously based on action+outcome.
        The agent literally shapes the probability distribution it is optimizing.
        Returns the trust_delta applied.
        """
        TRUST_DELTAS = {
            # Positive outcomes build trust
            "payment_received_after_empathy":             +0.10,
            "payment_received_after_moratorium":          +0.12,
            "behavioral_signal_check_revealed_distress":  +0.03,
            "investor_meeting_triggered_bridge":          +0.15,
            "information_verified_genuine_stress":        +0.05,
            "cluster_ecosystem_discipline_improved":      +0.08,
            "ghost_prevented_early_intervention":         +0.10,

            # Negative outcomes damage trust
            "account_npa_no_intervention":                -0.30,
            "cluster_cascade_default":                    -0.40,
            "ecosystem_cascade_ghosting":                 -0.35,
            "sarfaesi_before_restructuring_attempted":    -0.25,
            "moratorium_to_strategic_msme_defaulter":     -0.05,
            "pitch_optimism_taken_at_face_value":         -0.10,
            "sarfaesi_used_on_startup":                   -0.50,
            "ghost_detected_too_late":                    -0.20,
            "empathy_on_strategic_defaulter":             -0.08,
        }

        delta = TRUST_DELTAS.get(outcome, 0.0)

        profile = self._hidden_profiles[account_id]
        current_trust = profile.get("trust_score", 0.5)
        new_trust = max(0.0, min(1.0, current_trust + delta))
        profile["trust_score"] = round(new_trust, 4)

        # Also update financial health endogenously
        if outcome in ("payment_received_after_moratorium", "payment_received_after_empathy"):
            current_health = profile.get("true_financial_health", 0.5)
            profile["true_financial_health"] = min(1.0, current_health + 0.05)

        if outcome == "investor_meeting_triggered_bridge":
            profile["true_runway_months"] = min(24, profile.get("true_runway_months", 6) + 4)

        return round(delta, 4)

    def _map_to_network_effect(
        self,
        action_type: str,
        outcome: str,
        account_type: str,
    ) -> Optional[str]:
        """Map action+outcome to a network effect type."""
        if account_type == "msme":
            if action_type in ("initiate_sarfaesi", "send_legal_notice_section13"):
                return "sarfaesi"
            if outcome in ("account_npa_no_intervention", "cluster_cascade_default"):
                return "npa"
            if outcome in ("payment_received_after_moratorium", "payment_received_after_empathy"):
                return "recovery"
            if action_type == "grant_moratorium":
                return "moratorium"

        if account_type == "startup":
            if action_type == "initiate_sarfaesi":
                return "harsh_action"
            if outcome == "ecosystem_cascade_ghosting":
                return "ghost_detected"
            if outcome == "investor_meeting_triggered_bridge":
                return "bridge_arranged"
            if outcome in ("payment_received_after_moratorium", "payment_received_after_empathy"):
                return "recovery"

        return None

    def _simulate_spontaneous_defaults(self) -> None:
        """
        At month boundary, simulate spontaneous defaults for severely distressed accounts
        that the agent didn't act on this month.
        Calibrated to published NPA rates (RBI FY24, NASSCOM 2023).
        """
        acted_this_month = {
            s["account_id"]
            for s in self._episode_history
            if s["month"] == self._current_month - 1
        }

        for acc_id, profile in self._hidden_profiles.items():
            if acc_id in acted_this_month:
                continue

            health = profile.get("true_financial_health", 0.5)
            runway = profile.get("true_runway_months", 12)
            crisis_month = profile.get("crisis_trigger_month")
            account_type = profile.get("account_type", "msme")

            # Check if account hits crisis without intervention
            in_crisis = crisis_month and self._current_month >= crisis_month

            if account_type == "msme" and in_crisis and health < 0.30:
                # Mark as NPA
                profile["went_npa"] = True
                profile["amount_recovered"] = int(
                    profile.get("outstanding_principal", 1_000_000) * 0.31  # SARFAESI recovery rate
                )
                self._active_cluster_alerts.append(
                    f"Account {acc_id} went NPA without intervention — cluster at risk"
                )

            elif account_type == "startup" and in_crisis and runway <= 2:
                profile["went_npa"] = True
                profile["amount_recovered"] = 0  # No collateral for startups
                profile["ghosting_propensity"] = 1.0
                self._active_ecosystem_alerts.append(
                    f"Account {acc_id} (startup) defaulted — ecosystem confidence hit"
                )

    def _get_msme_observables(self) -> List[Dict]:
        """Return observable states for all MSME accounts (sorted by DPD desc)."""
        msme_ids = self._portfolio.get("msme_ids", list(range(1, 21)))
        result = [self._observable_states[i] for i in msme_ids if i in self._observable_states]
        return sorted(result, key=lambda x: x.get("dpd", 0), reverse=True)

    def _get_startup_observables(self) -> List[Dict]:
        """Return observable states for all startup accounts (sorted by DPD desc)."""
        startup_ids = self._portfolio.get("startup_ids", list(range(21, 31)))
        result = [self._observable_states[i] for i in startup_ids if i in self._observable_states]
        return sorted(result, key=lambda x: x.get("dpd", 0), reverse=True)

    def _build_portfolio_summary(self) -> Dict:
        """Build a high-level portfolio summary visible to the agent."""
        all_dpds = [obs.get("dpd", 0) for obs in self._observable_states.values()]
        npa_count = sum(
            1 for p in self._hidden_profiles.values() if p.get("went_npa", False)
        )
        avg_trust = (
            sum(p.get("trust_score", 0.5) for p in self._hidden_profiles.values())
            / max(1, len(self._hidden_profiles))
        )

        return {
            "total_accounts":     len(self._hidden_profiles),
            "msme_accounts":      len(self._portfolio.get("msme_ids", [])),
            "startup_accounts":   len(self._portfolio.get("startup_ids", [])),
            "npa_count":          npa_count,
            "npa_rate":           round(npa_count / max(1, len(self._hidden_profiles)), 4),
            "avg_dpd":            round(sum(all_dpds) / max(1, len(all_dpds)), 1),
            "max_dpd":            max(all_dpds) if all_dpds else 0,
            "accounts_with_dpd":  sum(1 for d in all_dpds if d > 0),
            "avg_trust_score":    round(avg_trust, 3),
            "current_month":      self._current_month,
            "episode":            self._episode_num,
            "steps_this_episode": self._step_count_this_episode,
            "cumulative_reward":  round(self._episode_cumulative_reward, 4),
        }

    def _error_observation(self, message: str) -> MSMERLObservation:
        """Return an error observation (invalid action)."""
        # Use explicit format_error tag so reward logic can track malformed/invalid policy outputs.
        self._episode_history.append({
            "episode": self._episode_num,
            "month": self._current_month,
            "account_id": -1,
            "account_type": "unknown",
            "action_type": "format_error",
            "outcome": "malformed_json_format",
            "reward": -0.15,
            "reasoning": f"error:{message}",
        })
        self._episode_cumulative_reward += -0.15
        return MSMERLObservation(
            episode=self._episode_num,
            month=self._current_month,
            msme_accounts=self._get_msme_observables(),
            startup_accounts=self._get_startup_observables(),
            portfolio_summary=self._build_portfolio_summary(),
            working_memory=f"ERROR: {message}",
            step_reward=-0.15,
            done=False,
            reward=-0.15,
            metadata={"error": message},
        )

    def _validate_action_parameters(
        self,
        action_type: str,
        account_type: str,
        params: Dict[str, Any],
    ) -> Optional[str]:
        """Strict parameter schema/range validation per action."""
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return "Invalid parameters: must be a JSON object."

        allowed_by_action: Dict[str, set[str]] = {
            "grant_moratorium": {"months"},
            "offer_bridge_loan_extension": {"months"},
            "restructure_emi": {"new_amount"},
            "accept_partial_payment": {"amount"},
            "offer_one_time_settlement": {"amount"},
        }
        allowed = allowed_by_action.get(action_type, set())
        unknown_keys = set(params.keys()) - allowed
        if unknown_keys:
            return f"Invalid parameters for {action_type}: unexpected keys {sorted(list(unknown_keys))}."

        if action_type in ("grant_moratorium", "offer_bridge_loan_extension"):
            months = params.get("months")
            if months is not None:
                if not isinstance(months, int) or months < 1 or months > 6:
                    return f"Invalid months for {action_type}: expected integer in [1, 6]."

        if action_type == "grant_moratorium" and account_type == "startup":
            months = params.get("months")
            if months is not None and months > 3:
                return "Invalid months for startup moratorium: expected <= 3."

        if action_type == "restructure_emi":
            new_amount = params.get("new_amount")
            if new_amount is not None:
                if not isinstance(new_amount, (int, float)) or new_amount < 1000 or new_amount > 2_000_000:
                    return "Invalid new_amount for restructure_emi: expected numeric in [1000, 2000000]."

        if action_type in ("accept_partial_payment", "offer_one_time_settlement"):
            amount = params.get("amount")
            if amount is not None:
                if not isinstance(amount, (int, float)) or amount <= 0:
                    return f"Invalid amount for {action_type}: expected positive numeric value."

        return None

    def _compute_behavior_penalties(self, account_id: int, action_type: str) -> Tuple[float, List[str]]:
        """
        Penalize repeated action abuse patterns:
        - same action repeatedly on same account in same month
        - excessive consecutive actions on same account (account hammering)
        """
        flags: List[str] = []
        penalty = 0.0
        same_month_steps = [s for s in self._episode_history if s.get("month") == self._current_month]
        same_account_steps = [s for s in same_month_steps if s.get("account_id") == account_id]
        same_action_same_account = [
            s for s in same_account_steps if s.get("action_type") == action_type
        ]

        if len(same_action_same_account) >= 2:
            penalty += 0.04
            flags.append("repeat_action_same_account")

        recent = same_month_steps[-6:]
        if len([s for s in recent if s.get("account_id") == account_id]) >= 4:
            penalty += 0.03
            flags.append("account_hammering")

        return round(penalty, 4), flags

    @staticmethod
    def _compute_budget_penalty(reasoning: str) -> float:
        """
        Penalize excessively long reasoning payloads (token/runtime budget proxy).
        """
        if not reasoning:
            return 0.0
        chars = len(reasoning)
        if chars <= 700:
            return 0.0
        # Soft linear penalty capped at 0.05.
        return round(min(0.05, (chars - 700) / 8000.0), 4)