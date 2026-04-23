# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Network Effects for MSME-RL.

Two distinct network topologies, both simulated simultaneously:

1. MSME Cluster Networks — tight geographic / industry clusters.
   SARFAESI on account 7 reaches all cluster members within 2 weeks.
   Calibrated: 1 default → avg 2.3 connected defaults (SIDBI MSME Pulse).

2. Startup Ecosystem Networks — loose accelerator / investor networks.
   Harsh treatment reaches the accelerator cohort within 1 month.
   Slower radius, wider reach than MSME clusters.

3. Cross-contamination — MSME supply chain cascade can reach startup accounts
   that share the same OEM/FMCG supply chain.
"""

from typing import Any, Dict, List, Optional, Set, Tuple


# Calibration constants
MSME_CLUSTER_CONTAGION_FACTOR   = 2.3    # SIDBI MSME Pulse
SARFAESI_TRUST_COLLAPSE         = -0.45
SARFAESI_ON_STARTUP_PENALTY     = -0.60   # Wrong tool — much worse
CLUSTER_RUMOR_PROPAGATION_WEEKS = 2
ECOSYSTEM_RUMOR_PROPAGATION_WEEKS = 4

CROSS_CONTAMINATION_THRESHOLD   = 0.6    # cluster contagion strength for spill to startups


def propagate_msme_cluster_effect(
    hidden_profiles: Dict[int, Dict],
    trigger_account_id: int,
    effect_type: str,           # "sarfaesi" | "moratorium" | "npa" | "recovery"
    effect_strength: float = 1.0,
) -> Dict[int, Dict]:
    """
    Propagate an event from one MSME account to its cluster members.
    Returns updated trust_score and repayment_probability deltas for connected accounts.

    Args:
        hidden_profiles: Current state of all account profiles
        trigger_account_id: The account that triggered the event
        effect_type: The type of event to propagate
        effect_strength: Scaling factor (0-1)

    Returns:
        Dict mapping account_id → {trust_delta, repayment_prob_delta, alert_message}
    """
    trigger = hidden_profiles.get(trigger_account_id)
    if not trigger or trigger.get("account_type") != "msme":
        return {}

    cluster_members: List[int] = trigger.get("cluster_members", [])
    cluster_centrality: float  = trigger.get("cluster_centrality", 0.5)

    effects: Dict[int, Dict] = {}

    # Define propagation deltas per effect type
    PROPAGATION = {
        "sarfaesi": {
            "trust_delta":            -0.35 * cluster_centrality,
            "repayment_prob_delta":   -0.28 * cluster_centrality,
            "alert": f"Account {trigger_account_id} received SARFAESI notice — cluster on high alert",
        },
        "npa": {
            "trust_delta":            -0.20 * cluster_centrality,
            "repayment_prob_delta":   -0.22 * cluster_centrality,
            "alert": f"Account {trigger_account_id} went NPA — cluster distress spreading",
        },
        "moratorium": {
            "trust_delta":            +0.12 * cluster_centrality,
            "repayment_prob_delta":   +0.08 * cluster_centrality,
            "alert": f"Account {trigger_account_id} got moratorium — cluster confidence improved",
        },
        "recovery": {
            "trust_delta":            +0.15 * cluster_centrality,
            "repayment_prob_delta":   +0.10 * cluster_centrality,
            "alert": f"Account {trigger_account_id} recovered — cluster stabilising",
        },
    }

    prop = PROPAGATION.get(effect_type, {})
    if not prop:
        return {}

    # Contagion decays with distance (all direct connections are distance=1 here)
    for member_id in cluster_members:
        if member_id not in hidden_profiles:
            continue
        member = hidden_profiles[member_id]
        if member.get("account_type") != "msme":
            continue

        trust_delta          = prop["trust_delta"] * effect_strength
        repayment_prob_delta = prop["repayment_prob_delta"] * effect_strength

        effects[member_id] = {
            "trust_delta":          round(trust_delta, 4),
            "repayment_prob_delta": round(repayment_prob_delta, 4),
            "alert_message":        prop["alert"],
            "propagation_weeks":    CLUSTER_RUMOR_PROPAGATION_WEEKS,
        }

    return effects


def propagate_startup_ecosystem_effect(
    hidden_profiles: Dict[int, Dict],
    trigger_account_id: int,
    effect_type: str,           # "harsh_action" | "recovery" | "ghost_detected" | "bridge_arranged"
    effect_strength: float = 1.0,
) -> Dict[int, Dict]:
    """
    Propagate event from one startup account to its ecosystem network.

    Args:
        hidden_profiles: Current state of all account profiles
        trigger_account_id: The account that triggered the event
        effect_type: The type of event to propagate
        effect_strength: Scaling factor (0-1)

    Returns:
        Dict mapping account_id → {trust_delta, ghosting_propensity_delta, alert_message}
    """
    trigger = hidden_profiles.get(trigger_account_id)
    if not trigger or trigger.get("account_type") != "startup":
        return {}

    ecosystem_network: List[int]   = trigger.get("ecosystem_network", [])
    ecosystem_centrality: float    = trigger.get("ecosystem_centrality", 0.5)

    PROPAGATION = {
        "harsh_action": {                   # e.g. SARFAESI on startup — catastrophic
            "trust_delta":                  -0.50 * ecosystem_centrality,
            "ghosting_propensity_delta":    +0.30 * ecosystem_centrality,
            "alert": f"Harsh action on {trigger_account_id} spreading through accelerator network",
        },
        "recovery": {
            "trust_delta":                  +0.15 * ecosystem_centrality,
            "ghosting_propensity_delta":    -0.10 * ecosystem_centrality,
            "alert": f"Account {trigger_account_id} recovery improving ecosystem trust",
        },
        "ghost_detected": {
            "trust_delta":                  -0.18 * ecosystem_centrality,
            "ghosting_propensity_delta":    +0.15 * ecosystem_centrality,
            "alert": f"Ghosting by {trigger_account_id} rippling through ecosystem",
        },
        "bridge_arranged": {
            "trust_delta":                  +0.20 * ecosystem_centrality,
            "ghosting_propensity_delta":    -0.12 * ecosystem_centrality,
            "alert": f"Bridge arranged for {trigger_account_id} — ecosystem optimism rising",
        },
    }

    prop = PROPAGATION.get(effect_type, {})
    if not prop:
        return {}

    effects: Dict[int, Dict] = {}
    for member_id in ecosystem_network:
        if member_id not in hidden_profiles:
            continue
        member = hidden_profiles[member_id]
        if member.get("account_type") != "startup":
            continue

        effects[member_id] = {
            "trust_delta":               round(prop["trust_delta"] * effect_strength, 4),
            "ghosting_propensity_delta": round(prop["ghosting_propensity_delta"] * effect_strength, 4),
            "alert_message":             prop["alert"],
            "propagation_weeks":         ECOSYSTEM_RUMOR_PROPAGATION_WEEKS,
        }

    return effects


def check_cross_contamination(
    hidden_profiles: Dict[int, Dict],
    trigger_msme_account_id: int,
    contagion_strength: float,
) -> List[int]:
    """
    Check if an MSME cluster cascade is strong enough to contaminate
    connected startup accounts (shared supply chain scenario).

    Returns list of startup account IDs to partially affect.
    """
    if contagion_strength < CROSS_CONTAMINATION_THRESHOLD:
        return []

    trigger = hidden_profiles.get(trigger_msme_account_id)
    if not trigger:
        return []

    # Industry linkages — construction/auto-ancillary often link to startup supply chains
    high_linkage_industries = {"auto_ancillary", "construction", "fmcg"}
    if trigger.get("industry") not in high_linkage_industries:
        return []

    # Find startup accounts that could be affected (shared supply chain)
    affected_startups = []
    for acc_id, profile in hidden_profiles.items():
        if profile.get("account_type") != "startup":
            continue
        # Startups in b2b_saas / fintech serving these industries could be affected
        if profile.get("sector") in {"b2b_saas", "fintech"}:
            affected_startups.append(acc_id)

    # Limit cross-contamination to 1-2 accounts
    return affected_startups[:2]


def apply_network_effects(
    hidden_profiles: Dict[int, Dict],
    effects: Dict[int, Dict],
) -> Dict[int, Dict]:
    """
    Apply computed network effect deltas to hidden profiles.
    Returns updated hidden_profiles.
    """
    for account_id, delta in effects.items():
        if account_id not in hidden_profiles:
            continue
        profile = hidden_profiles[account_id]

        # Apply trust delta (clamp to [0, 1])
        current_trust = profile.get("trust_score", 0.5)
        new_trust = max(0.0, min(1.0, current_trust + delta.get("trust_delta", 0)))
        profile["trust_score"] = round(new_trust, 4)

        # Apply repayment probability delta for MSME
        if "repayment_prob_delta" in delta:
            current_health = profile.get("true_financial_health", 0.5)
            profile["true_financial_health"] = max(
                0.0, min(1.0, current_health + delta["repayment_prob_delta"] * 0.3)
            )

        # Apply ghosting propensity delta for startups
        if "ghosting_propensity_delta" in delta:
            current_ghost = profile.get("ghosting_propensity", 0.22)
            profile["ghosting_propensity"] = max(
                0.0, min(1.0, current_ghost + delta["ghosting_propensity_delta"])
            )

    return hidden_profiles


def collect_active_alerts(
    all_effects: List[Dict[int, Dict]],
    account_type_filter: Optional[str] = None,
) -> List[str]:
    """
    Collect all unique alert messages from a list of effect dicts.
    Used to build the cluster_alerts / ecosystem_alerts in the observation.
    """
    seen: Set[str] = set()
    alerts: List[str] = []
    for effect_dict in all_effects:
        for _, effect in effect_dict.items():
            msg = effect.get("alert_message", "")
            if msg and msg not in seen:
                seen.add(msg)
                alerts.append(msg)
    return alerts