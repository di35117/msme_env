# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reward Functions for MSME-RL.

All rewards are deterministic numbers — no LLM judge needed.
NPA rate and recovery rate are hard outputs verifiable by judges.

Step rewards: immediate feedback per action.
Episode rewards: NPA rate + recovery rate + relationship score + tool appropriateness.
"""

import random
from collections import Counter
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# STEP-LEVEL REWARD TABLE  (15 distinct outcome types)
# ---------------------------------------------------------------------------

STEP_REWARDS: Dict[str, float] = {
    # --- Universal good outcomes ---
    "payment_received_after_empathy":             +0.08,
    "payment_received_after_moratorium":          +0.06,
    "behavioral_signal_check_revealed_distress":  +0.05,
    "cluster_ecosystem_discipline_improved":      +0.07,
    "investor_meeting_triggered_bridge":          +0.10,
    "information_verified_genuine_stress":        +0.04,
    "ghost_prevented_early_intervention":         +0.06,
    "malformed_json_format":                      -0.15,
    # --- Universal bad outcomes ---
    "account_npa_no_intervention":                -0.18,
    "cluster_cascade_default":                    -0.25,
    "ecosystem_cascade_ghosting":                 -0.20,
    "unnecessary_action_on_current_account":      -0.03,
    "tool_execution_failed":                      -0.02, # FIXED: Penalty for tool failure

    # --- MSME-specific penalties ---
    "sarfaesi_before_restructuring_attempted":    -0.12,
    "moratorium_to_strategic_msme_defaulter":     -0.08,
    "gst_skipped_before_moratorium_decision":     -0.05,
    "cluster_centrality_ignored":                 -0.06,
    "empathy_on_strategic_defaulter":             -0.07,

    # --- Startup-specific penalties ---
    "sarfaesi_used_on_startup":                   -0.15,
    "pitch_optimism_taken_at_face_value":         -0.07,
    "ghost_detected_too_late":                    -0.10,
    "ecosystem_signals_not_checked":              -0.05,
    "investor_not_looped_in_before_restructure":  -0.04,
}

# Baseline: no reward, no penalty
DEFAULT_STEP_REWARD = 0.0


# ---------------------------------------------------------------------------
# STEP-LEVEL REWARD COMPUTATION
# ---------------------------------------------------------------------------

def compute_step_reward(
    action_type: str,
    account_type: str,       # "msme" or "startup"
    outcome: str,            # outcome key from STEP_REWARDS
    hidden_profile: Dict,    # used for context
) -> float:
    """
    Compute step-level reward for a single action outcome.

    Args:
        action_type: The action taken (e.g. "initiate_sarfaesi")
        account_type: "msme" or "startup"
        outcome: Outcome identifier string
        hidden_profile: Hidden profile for context (not exposed to agent)

    Returns:
        Float reward value
    """
    base_reward = STEP_REWARDS.get(outcome, DEFAULT_STEP_REWARD)

    # Type-specific amplification
    if account_type == "startup" and action_type == "initiate_sarfaesi":
        # SARFAESI on startup: extra penalty (wrong tool, destroys ecosystem trust)
        base_reward += STEP_REWARDS["sarfaesi_used_on_startup"]

    if account_type == "msme" and action_type == "initiate_sarfaesi":
        strategic_default = hidden_profile.get("strategic_default_propensity", 0) > 0.5
        if not strategic_default:
            # SARFAESI on a genuinely stressed MSME before restructuring
            base_reward += STEP_REWARDS["sarfaesi_before_restructuring_attempted"]
        # else: SARFAESI on strategic defaulter is acceptable

    # Cluster centrality bonus: information-gathering before high-centrality decisions
    if action_type in ("verify_gst_returns", "pull_bank_statements", "check_industry_cluster_stress"):
        cluster_centrality = hidden_profile.get("cluster_centrality", 0)
        if cluster_centrality > 0.7:
            base_reward += 0.03  # Bonus for checking before acting on high-centrality accounts

    return round(base_reward, 4)


def classify_action_outcome(
    action_type: str,
    account_type: str,
    hidden_profile: Dict,
    current_month: int,
    action_params: Dict,
) -> str:
    """
    Simulate the outcome of an action given hidden profile state.
    Returns an outcome key from STEP_REWARDS.

    This is where the environment's hidden state drives the reward signal.
    """
    if action_type == "format_error":
        return "malformed_json_format"
    health   = hidden_profile.get("true_financial_health", 0.5)
    runway   = hidden_profile.get("true_runway_months", 12)
    strategic_default = hidden_profile.get("strategic_default_propensity", 0) > 0.5
    crisis_month = hidden_profile.get("crisis_trigger_month")
    in_crisis = crisis_month is not None and current_month >= crisis_month

    # -------------------------------------------------------------------------
    # MSME outcomes
    # -------------------------------------------------------------------------
    if account_type == "msme":
        if action_type == "initiate_sarfaesi" and not strategic_default:
            return "sarfaesi_before_restructuring_attempted"

        if action_type == "initiate_sarfaesi" and strategic_default:
            return "cluster_ecosystem_discipline_improved"

        if action_type == "grant_moratorium":
            if strategic_default:
                return "moratorium_to_strategic_msme_defaulter"
            if health > 0.35 or in_crisis:
                return "payment_received_after_moratorium"
            return "account_npa_no_intervention"

        if action_type == "send_empathetic_reminder":
            if health > 0.5 and not strategic_default:
                return "payment_received_after_empathy"
            if strategic_default:
                return "empathy_on_strategic_defaulter"
            return "behavioral_signal_check_revealed_distress"

        if action_type == "send_firm_reminder":
            if strategic_default:
                return "payment_received_after_empathy"   # firm works on strategic
            if health < 0.3:
                return "account_npa_no_intervention"
            return "payment_received_after_empathy"

        if action_type == "verify_gst_returns":
            # FIXED: Inject stochastic tool failure (15% chance)
            if random.random() < 0.15:
                return "tool_execution_failed"
            if health > 0.4:
                return "information_verified_genuine_stress"
            return "behavioral_signal_check_revealed_distress"

        if action_type == "check_industry_cluster_stress":
            cluster_centrality = hidden_profile.get("cluster_centrality", 0.5)
            if cluster_centrality > 0.6:
                return "cluster_ecosystem_discipline_improved"
            return "information_verified_genuine_stress"

        if action_type in ("offer_eclgs_topup", "restructure_emi"):
            if health > 0.35 and not strategic_default:
                return "payment_received_after_moratorium"
            return "moratorium_to_strategic_msme_defaulter"

        if action_type in ("refer_to_recovery_agent", "file_drt_case"):
            if strategic_default:
                return "cluster_ecosystem_discipline_improved"
            return "sarfaesi_before_restructuring_attempted"

    # -------------------------------------------------------------------------
    # STARTUP outcomes
    # -------------------------------------------------------------------------
    if account_type == "startup":
        ghosting = hidden_profile.get("ghosting_propensity", 0.22)
        bridge_prob = hidden_profile.get("investor_bridge_probability", 0.35)

        if action_type == "initiate_sarfaesi":
            return "sarfaesi_used_on_startup"

        if action_type == "check_startup_ecosystem_signals":
            # FIXED: Inject stochastic tool failure (15% chance)
            if random.random() < 0.15:
                return "tool_execution_failed"
            if runway <= 6:
                return "behavioral_signal_check_revealed_distress"
            return "information_verified_genuine_stress"

        if action_type == "request_investor_update_meeting":
            if runway <= 8 and bridge_prob > 0.4:
                return "investor_meeting_triggered_bridge"
            if runway <= 8:
                return "behavioral_signal_check_revealed_distress"
            return "cluster_ecosystem_discipline_improved"

        if action_type == "offer_bridge_loan_extension":
            if runway > 6:
                return "payment_received_after_moratorium"
            if bridge_prob > 0.5:
                return "investor_meeting_triggered_bridge"
            return "pitch_optimism_taken_at_face_value"

        if action_type == "grant_moratorium":
            if runway <= 3:
                return "pitch_optimism_taken_at_face_value"
            return "payment_received_after_moratorium"

        if action_type == "send_empathetic_reminder":
            if runway <= 4 and ghosting > 0.35:
                return "ghost_detected_too_late"
            return "payment_received_after_empathy"

        if action_type == "extend_credit" or action_type == "accept_partial_payment":
            if runway <= 5:
                return "pitch_optimism_taken_at_face_value"
            return "payment_received_after_empathy"

        if action_type in ("refer_to_recovery_agent", "file_drt_case"):
            return "ecosystem_cascade_ghosting"

    # Default — neutral
    return "information_verified_genuine_stress"


# ---------------------------------------------------------------------------
# EPISODE-LEVEL REWARD COMPUTATION
# ---------------------------------------------------------------------------

def compute_episode_reward(
    hidden_profiles: Dict[int, Dict],
    episode_history: List[Dict],
    episode_num: int, # FIXED: Added for curriculum phasing
    final_month: int = 36,
) -> Dict[str, float]:
    """
    Compute final episode reward from hard numbers.
    No LLM judge. Fully verifiable.

    Components:
      - NPA rate: fraction of accounts that went NPA (primary signal, 40%)
      - Recovery rate: amount recovered / amount disbursed (30%)
      - Relationship score: mean final trust scores (20%)
      - Tool appropriateness: right tool per account type (10%)

    Returns:
        Dict with component scores and total R
    """
    total_accounts = len(hidden_profiles)
    if total_accounts == 0:
        return {"total": 0.0}

    # NPA count
    npa_accounts = sum(
        1 for p in hidden_profiles.values()
        if p.get("went_npa", False)
    )
    npa_rate = npa_accounts / total_accounts

    # Recovery rate
    total_disbursed = sum(
        p.get("loan_amount", 0) for p in hidden_profiles.values()
    )
    total_recovered = sum(
        p.get("amount_recovered", 0) for p in hidden_profiles.values()
    )
    recovery_rate = total_recovered / total_disbursed if total_disbursed > 0 else 0.5

    # Relationship score
    trust_scores = [
        p.get("trust_score", 0.5) for p in hidden_profiles.values()
    ]
    relationship_score = sum(trust_scores) / len(trust_scores) if trust_scores else 0.5

    # FIXED: Prevent Goodharting Tool Appropriateness
    appropriate_actions = 0
    action_frequency = {}
    
    total_actions = max(1, len(episode_history))
    for step in episode_history:
        action_type  = step.get("action_type", "")
        account_type = step.get("account_type", "")
        
        # Track frequency to penalize spamming
        action_frequency[action_type] = action_frequency.get(action_type, 0) + 1
        
        if _is_appropriate_tool(action_type, account_type):
            if action_frequency[action_type] <= 3: # Cap rewards for the same action
                appropriate_actions += 1
            else:
                appropriate_actions -= 0.5 # Actively penalize spamming the same tool

    tool_appropriateness = max(0.0, appropriate_actions / total_actions)

    # Anti-hacking penalty (independent signal):
    # penalize exploit-like behavior such as no-op farming, repeated invalid/format
    # actions, single-action spamming, and account-thrashing patterns.
    shortcut_penalty = _compute_shortcut_penalty(episode_history)

    # Anti-cheat audit metrics for deterministic inspection/reporting.
    anti_cheat_metrics = _build_anti_cheat_metrics(episode_history)

    # Curriculum with smooth transition to avoid brittle early binary behavior.
    if episode_num < 30:
        progress = max(0.0, min(1.0, episode_num / 30.0))
        survival_signal = 1.0 if npa_rate == 0 else (0.1 - npa_rate)
        shaped_signal = (
            0.40 * (1.0 - npa_rate) +
            0.30 * recovery_rate +
            0.20 * relationship_score +
            0.10 * tool_appropriateness
        )
        R = ((1.0 - progress) * survival_signal) + (progress * shaped_signal)
    else:
        # Phase 2: Complex shaping
        R = (
            0.40 * (1.0 - npa_rate)          +
            0.30 * recovery_rate              +
            0.20 * relationship_score         +
            0.10 * tool_appropriateness
        )
        # Keep this explicit and bounded so the anti-hack channel is visible
        # while not completely dominating primary objective metrics.
        R -= min(0.25, shortcut_penalty)

    return {
        "total":                round(R, 4),
        "npa_rate":             round(npa_rate, 4),
        "recovery_rate":        round(recovery_rate, 4),
        "relationship_score":   round(relationship_score, 4),
        "tool_appropriateness": round(tool_appropriateness, 4),
        "shortcut_penalty":     round(shortcut_penalty, 4),
        "anti_cheat_metrics":   anti_cheat_metrics,
        "npa_count":            npa_accounts,
        "total_accounts":       total_accounts,
    }


def _compute_shortcut_penalty(episode_history: List[Dict[str, Any]]) -> float:
    """Compute penalty for common reward-hacking / degenerate behavior patterns."""
    if not episode_history:
        return 0.0

    total_steps = len(episode_history)
    action_counter = Counter(step.get("action_type", "") for step in episode_history)
    no_op_count = action_counter.get("wait_and_observe", 0)
    malformed_count = action_counter.get("format_error", 0)

    # 1) No-op farming.
    no_op_ratio = no_op_count / total_steps
    no_op_penalty = max(0.0, no_op_ratio - 0.30) * 0.50

    # 2) Format/invalid output abuse.
    malformed_penalty = (malformed_count / total_steps) * 0.60

    # 3) Action spamming (same action repeated excessively).
    dominant_ratio = max(action_counter.values()) / total_steps if action_counter else 0.0
    spam_penalty = max(0.0, dominant_ratio - 0.35) * 0.40

    # 4) Account thrashing: rapid target switching across consecutive steps.
    switches = 0
    prev_account = None
    for step in episode_history:
        account_id = step.get("account_id")
        if prev_account is not None and account_id != prev_account:
            switches += 1
        prev_account = account_id
    switch_ratio = switches / max(1, total_steps - 1)
    # A high switch ratio can indicate shallow probing with no follow-through.
    thrash_penalty = max(0.0, switch_ratio - 0.70) * 0.25

    return round(no_op_penalty + malformed_penalty + spam_penalty + thrash_penalty, 4)


def _build_anti_cheat_metrics(episode_history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Deterministic anti-cheat counters for judge-facing audits."""
    if not episode_history:
        return {
            "no_op_ratio": 0.0,
            "format_error_ratio": 0.0,
            "dominant_action_ratio": 0.0,
            "account_switch_ratio": 0.0,
            "same_account_repeat_ratio": 0.0,
        }

    total_steps = len(episode_history)
    action_counter = Counter(step.get("action_type", "") for step in episode_history)
    no_op_ratio = action_counter.get("wait_and_observe", 0) / total_steps
    format_error_ratio = action_counter.get("format_error", 0) / total_steps
    dominant_action_ratio = max(action_counter.values()) / total_steps if action_counter else 0.0

    switches = 0
    repeats_same_account = 0
    prev_account = None
    for step in episode_history:
        account_id = step.get("account_id")
        if prev_account is not None:
            if account_id != prev_account:
                switches += 1
            else:
                repeats_same_account += 1
        prev_account = account_id

    denom = max(1, total_steps - 1)
    return {
        "no_op_ratio": round(no_op_ratio, 4),
        "format_error_ratio": round(format_error_ratio, 4),
        "dominant_action_ratio": round(dominant_action_ratio, 4),
        "account_switch_ratio": round(switches / denom, 4),
        "same_account_repeat_ratio": round(repeats_same_account / denom, 4),
    }


def _is_appropriate_tool(action_type: str, account_type: str) -> bool:
    """
    Return True if the action is appropriate for the given account type.
    Core check: SARFAESI on startup = inappropriate; investor meeting on startup = appropriate.
    """
    MSME_ONLY = {
        "send_legal_notice_section13",
        "initiate_sarfaesi",
        "offer_eclgs_topup",
        "verify_gst_returns",
        "check_industry_cluster_stress",
    }
    STARTUP_ONLY = {
        "request_investor_update_meeting",
        "check_startup_ecosystem_signals",
        "offer_bridge_loan_extension",
    }
    UNIVERSAL = {
        "send_empathetic_reminder",
        "send_firm_reminder",
        "call_promoter_founder",
        "call_guarantor_investor",
        "grant_moratorium",
        "restructure_emi",
        "accept_partial_payment",
        "waive_penal_interest",
        "refer_to_recovery_agent",
        "file_drt_case",
        "offer_one_time_settlement",
        "pull_bank_statements",
        "conduct_cluster_ecosystem_visit",
        "wait_and_observe",
    }

    if action_type in UNIVERSAL:
        return True
    if action_type in MSME_ONLY and account_type == "msme":
        return True
    if action_type in STARTUP_ONLY and account_type == "startup":
        return True
    # Wrong tool for account type
    return False


# ---------------------------------------------------------------------------
# ADVERSARIAL CURRICULUM
# ---------------------------------------------------------------------------

def analyze_agent_weaknesses(episode_histories: List[List[Dict]]) -> Dict[str, float]:
    """
    Analyze agent action patterns across completed episodes to identify weaknesses.
    Used by adversarial curriculum controller.

    Returns weakness scores (higher = more exploited by adversary).
    """
    if not episode_histories:
        return {}

    sarfaesi_on_startup_rate  = 0.0
    investor_check_rate        = 0.0
    gst_verify_before_morat    = 0.0
    face_value_trust_rate      = 0.0
    total_msme_actions         = 0
    total_startup_actions      = 0

    for history in episode_histories:
        for step in history:
            action_type  = step.get("action_type", "")
            account_type = step.get("account_type", "")

            if account_type == "msme":
                total_msme_actions += 1
                if action_type == "initiate_sarfaesi":
                    sarfaesi_on_startup_rate += 0   # counting for MSME track separately
                if action_type == "verify_gst_returns":
                    gst_verify_before_morat += 1

            if account_type == "startup":
                total_startup_actions += 1
                if action_type == "initiate_sarfaesi":
                    sarfaesi_on_startup_rate += 1
                if action_type == "request_investor_update_meeting":
                    investor_check_rate += 1
                outcome = step.get("outcome", "")
                if outcome == "pitch_optimism_taken_at_face_value":
                    face_value_trust_rate += 1

    weaknesses = {}
    if total_startup_actions > 0:
        weaknesses["sarfaesi_on_startup"] = sarfaesi_on_startup_rate / total_startup_actions
        weaknesses["investor_meeting_rate"] = investor_check_rate / total_startup_actions
        weaknesses["face_value_acceptance"] = face_value_trust_rate / total_startup_actions
    if total_msme_actions > 0:
        weaknesses["gst_verify_rate"] = gst_verify_before_morat / total_msme_actions

    return weaknesses


def should_apply_adversarial_curriculum(
    episode_num: int,
    weaknesses: Dict[str, float],
) -> bool:
    """Enable adversarial curriculum after episode 40."""
    return episode_num >= 40 and bool(weaknesses)