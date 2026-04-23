# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Three-Tier Memory System for MSME-RL.

A 36-month loan cycle across 30 accounts with two different signal structures
is unmanageable in a single context window.  Three-tier memory solves this
and turns a technical limitation into a research contribution.

Tier 1 — Episodic Memory: individual interaction outcomes (what happened)
Tier 2 — Semantic Memory: distilled patterns with confidence scores (what works)
Tier 3 — Working Memory: current-month compact state (< 2K tokens)

Patterns are NOT written by a human — the agent discovers them from reward
signal across episodes.  Confidence scores update via Bayesian-style updates.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# EPISODIC MEMORY
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    Stores individual interaction→outcome records for each account.
    Separate schemas for MSME and startup interactions.
    """

    def __init__(self, max_per_account: int = 50):
        self.max_per_account = max_per_account
        self._records: Dict[int, List[Dict]] = defaultdict(list)

    def add(self, record: Dict) -> None:
        """
        Store one interaction record.

        Required fields:
            episode, month, account_id, account_type, action_type, outcome, reward
        Optional (MSME):
            industry, message_summary, verify_gst_result, trust_delta, cluster_effect
        Optional (Startup):
            founder_message_tone, behavioral_signals_checked,
            signals_contradicted_message, ecosystem_effect
        """
        acc_id = record.get("account_id", 0)
        bucket = self._records[acc_id]
        bucket.append(record)
        # Cap per-account records
        if len(bucket) > self.max_per_account:
            self._records[acc_id] = bucket[-self.max_per_account:]

    def retrieve_similar(
        self,
        account_type: str,
        industry_or_stage: str,
        action_type: str,
        n: int = 3,
    ) -> List[Dict]:
        """
        Retrieve N most similar past interaction records for context injection.
        Matching: same account_type + industry/stage + action_type.
        Falls back to same account_type + action_type if no exact matches.
        """
        matches = []
        fallback = []
        for records in self._records.values():
            for rec in reversed(records):
                if rec.get("account_type") != account_type:
                    continue
                if (
                    rec.get("industry") == industry_or_stage
                    or rec.get("stage") == industry_or_stage
                ) and rec.get("action_type") == action_type:
                    matches.append(rec)
                elif rec.get("action_type") == action_type:
                    fallback.append(rec)
                if len(matches) >= n:
                    break
            if len(matches) >= n:
                break

        result = matches[:n] if matches else fallback[:n]
        return result

    def format_for_context(self, records: List[Dict]) -> str:
        """Format episodic records as compact context text."""
        if not records:
            return "(no relevant past cases)"
        lines = []
        for r in records:
            account_type = r.get("account_type", "unknown")
            if account_type == "msme":
                lines.append(
                    f"• [Ep{r.get('episode','?')} M{r.get('month','?')}] "
                    f"{r.get('industry','?')} MSME | "
                    f"action={r.get('action_type','?')} | "
                    f"outcome={r.get('outcome','?')} | "
                    f"trust_delta={r.get('trust_delta',0):+.2f} | "
                    f"reward={r.get('reward',0):.3f}"
                )
                if r.get("cluster_effect"):
                    lines[-1] += f" | cluster: {r['cluster_effect']}"
            else:
                lines.append(
                    f"• [Ep{r.get('episode','?')} M{r.get('month','?')}] "
                    f"{r.get('stage','?')} startup | "
                    f"tone={r.get('founder_message_tone','?')} | "
                    f"signals_vs_message={r.get('signals_contradicted_message','?')} | "
                    f"action={r.get('action_type','?')} | "
                    f"outcome={r.get('outcome','?')} | "
                    f"reward={r.get('reward',0):.3f}"
                )
        return "\n".join(lines)

    @property
    def total_records(self) -> int:
        return sum(len(v) for v in self._records.values())


# ---------------------------------------------------------------------------
# SEMANTIC MEMORY
# ---------------------------------------------------------------------------

_BUILTIN_PATTERNS: Dict[str, Dict] = {
    # --- MSME patterns (initial priors — refined by reward) ---
    "msme+auto_ancillary+OEM_delay+gst_filing_regular": {
        "signal": "genuine_stress",
        "confidence": 0.60,
        "recommended_action": "grant_moratorium",
        "source": "prior",
    },
    "msme+textile+LC_stuck+third_consecutive_excuse": {
        "signal": "strategic_default_risk",
        "confidence": 0.55,
        "recommended_action": "verify_guarantor",
        "source": "prior",
    },
    "msme+high_cluster_centrality+legal_notice_risk": {
        "signal": "cascade_risk",
        "confidence": 0.65,
        "recommended_action": "check_industry_cluster_stress",
        "source": "prior",
    },
    "msme+13_months_clean+sudden_silence": {
        "signal": "genuine_crisis",
        "confidence": 0.60,
        "recommended_action": "call_promoter_founder",
        "source": "prior",
    },

    # --- Startup patterns (initial priors) ---
    "startup+pitch_optimism+linkedin_hiring_stopped": {
        "signal": "distress_behind_confidence",
        "confidence": 0.55,
        "recommended_action": "check_startup_ecosystem_signals",
        "source": "prior",
    },
    "startup+missed_2_investor_updates+optimistic_message": {
        "signal": "imminent_default_risk",
        "confidence": 0.60,
        "recommended_action": "request_investor_update_meeting",
        "source": "prior",
    },
    "startup+mrr_declining_3_months+exciting_language": {
        "signal": "discount_optimism_heavily",
        "confidence": 0.55,
        "recommended_action": "pull_bank_statements",
        "source": "prior",
    },
    "startup+cofounder_job_hunting+payment_delay": {
        "signal": "company_dissolving",
        "confidence": 0.65,
        "recommended_action": "call_guarantor_investor",
        "source": "prior",
    },

    # --- Cross-type patterns ---
    "msme_cluster_cascade+connected_startup_same_supply_chain": {
        "signal": "ecosystem_contagion_risk",
        "confidence": 0.50,
        "recommended_action": "check_startup_ecosystem_signals",
        "source": "prior",
    },
}


class SemanticMemory:
    """
    Stores distilled patterns discovered from reward signal across episodes.
    Patterns are NOT hardcoded — they emerge from training.
    Initial priors are set but refined by Bayesian-style updates.
    """

    def __init__(self):
        # Key: pattern string, Value: {signal, confidence, count, wins, losses}
        self._patterns: Dict[str, Dict] = {}
        # Load priors
        for key, val in _BUILTIN_PATTERNS.items():
            self._patterns[key] = {
                **val,
                "count":  10,   # pseudo-count for priors
                "wins":   int(10 * val["confidence"]),
                "losses": int(10 * (1 - val["confidence"])),
            }

    def update(
        self,
        pattern_key: str,
        outcome_positive: bool,
        reward: float,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Update confidence for a pattern key based on observed outcome.
        Uses Laplace-smoothed Bayesian update.
        """
        if pattern_key not in self._patterns:
            self._patterns[pattern_key] = {
                "signal": "discovered_from_training",
                "confidence": 0.50,
                "recommended_action": "",
                "source": "discovered",
                "count": 0,
                "wins": 1,     # Laplace smoothing
                "losses": 1,
            }

        pat = self._patterns[pattern_key]
        pat["count"] = pat.get("count", 0) + 1

        if outcome_positive and reward > 0:
            pat["wins"] = pat.get("wins", 1) + 1
        else:
            pat["losses"] = pat.get("losses", 1) + 1

        # Update confidence
        wins = pat.get("wins", 1)
        losses = pat.get("losses", 1)
        pat["confidence"] = round(wins / (wins + losses), 3)

        # Update recommended action if we have metadata
        if metadata and metadata.get("action_type") and outcome_positive:
            pat["recommended_action"] = metadata["action_type"]

    def retrieve(
        self,
        account_type: str,
        observable_signals: Dict,
        n: int = 3,
    ) -> List[Tuple[str, Dict]]:
        """
        Retrieve N most relevant patterns for an account's observable state.
        Matching is keyword-based against the pattern key.
        """
        keywords = _extract_keywords(account_type, observable_signals)
        scored = []
        for key, pat in self._patterns.items():
            score = sum(1 for kw in keywords if kw in key)
            if score > 0:
                scored.append((score, key, pat))
        scored.sort(key=lambda x: (-x[0], -x[2]["confidence"]))
        return [(key, pat) for _, key, pat in scored[:n]]

    def format_for_context(
        self,
        retrieved: List[Tuple[str, Dict]],
    ) -> str:
        """Format semantic patterns as compact context text."""
        if not retrieved:
            return "(no semantic patterns matched)"
        lines = []
        for key, pat in retrieved:
            lines.append(
                f"• [{pat['source']}] {key}\n"
                f"  → signal={pat['signal']} | "
                f"confidence={pat['confidence']:.2f} | "
                f"action={pat.get('recommended_action','?')}"
            )
        return "\n".join(lines)

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)


def _extract_keywords(account_type: str, signals: Dict) -> List[str]:
    """Extract matching keywords from observable signals for semantic lookup."""
    kws = [account_type]

    # MSME keywords
    if account_type == "msme":
        kws.append(signals.get("industry", ""))
        gst = signals.get("gst_filing_status", "")
        if "regular" in gst or "on_time" in gst:
            kws.append("gst_filing_regular")
        elif "delay" in gst or "not_filed" in gst:
            kws.append("gst_filing_irregular")
        msg = signals.get("last_message", "").lower()
        if "oem" in msg or "ओईएम" in msg:
            kws.append("OEM_delay")
        if "lc" in msg or "letter of credit" in msg:
            kws.append("LC_stuck")
        if "gst" in msg:
            kws.append("gst_mentioned")
        payment_hist = signals.get("payment_history", [])
        if payment_hist:
            clean_months = sum(1 for p in payment_hist if p == "on_time")
            if clean_months >= 4 and signals.get("dpd", 0) > 0:
                kws.append("sudden_silence")
            if clean_months >= 10:
                kws.append("13_months_clean")
        cluster_info = signals.get("cluster_accounts_behavior", "")
        if "2_of" in cluster_info or "3_of" in cluster_info:
            kws.append("high_cluster_centrality")

    # Startup keywords
    if account_type == "startup":
        kws.append(signals.get("stage", ""))
        kws.append(signals.get("sector", ""))
        linkedin = signals.get("linkedin_hiring_posts", "")
        if "none" in linkedin or "slowing" in linkedin:
            kws.append("linkedin_hiring_stopped")
        investor_update = signals.get("investor_update_sent", "")
        if "skipped_last_2" in investor_update:
            kws.append("missed_2_investor_updates")
        elif "skipped_last" in investor_update:
            kws.append("missed_investor_update")
        mrr = signals.get("mrr_last_3_months", [])
        if len(mrr) == 3 and mrr[2] < mrr[1] < mrr[0]:
            kws.append("mrr_declining_3_months")
        cofounder = signals.get("cofounder_linkedin_activity", "")
        if "job_hunting" in cofounder:
            kws.append("cofounder_job_hunting")
        if signals.get("dpd", 0) > 0:
            kws.append("payment_delay")
        msg = signals.get("last_message", "").lower()
        if any(w in msg for w in ["exciting", "great", "strong", "momentum", "accelerated"]):
            kws.append("pitch_optimism")
            kws.append("optimistic_message")
            kws.append("exciting_language")

    return [kw for kw in kws if kw]


# ---------------------------------------------------------------------------
# WORKING MEMORY
# ---------------------------------------------------------------------------

class WorkingMemory:
    """
    Compact current-month state under 2,000 tokens.
    Refreshed every month with both account types represented.
    """

    def __init__(self):
        self._state: Dict = {}

    def refresh(
        self,
        month: int,
        episode: int,
        hidden_profiles: Dict[int, Dict],
        observable_states: Dict[int, Dict],
        recent_actions: List[Dict],
        active_cluster_alerts: List[str],
        active_ecosystem_alerts: List[str],
    ) -> str:
        """
        Build and cache current working memory context string.
        Returns the formatted string (< 2K tokens).
        """
        # DPD distribution
        dpd_buckets = {"current": 0, "1-30": 0, "31-60": 0, "60+": 0}
        high_risk_accounts = []
        upcoming_npa_risk = []

        for acc_id, obs in observable_states.items():
            dpd = obs.get("dpd", 0)
            if dpd == 0:
                dpd_buckets["current"] += 1
            elif dpd <= 30:
                dpd_buckets["1-30"] += 1
            elif dpd <= 60:
                dpd_buckets["31-60"] += 1
                high_risk_accounts.append(acc_id)
            else:
                dpd_buckets["60+"] += 1
                upcoming_npa_risk.append(acc_id)

        # Recent action summary (last 5)
        recent = recent_actions[-5:] if len(recent_actions) > 5 else recent_actions
        action_lines = [
            f"  M{a['month']} Acc#{a['account_id']} ({a.get('account_type','?')}): "
            f"{a.get('action_type','?')} → {a.get('outcome','?')} (r={a.get('reward',0):.3f})"
            for a in recent
        ]

        # Portfolio split
        msme_count    = sum(1 for p in hidden_profiles.values() if p.get("account_type") == "msme")
        startup_count = sum(1 for p in hidden_profiles.values() if p.get("account_type") == "startup")
        avg_trust = (
            sum(p.get("trust_score", 0.5) for p in hidden_profiles.values()) / max(1, len(hidden_profiles))
        )

        lines = [
            f"=== WORKING MEMORY | Episode {episode} | Month {month}/36 ===",
            f"Portfolio: {msme_count} MSME + {startup_count} Startup | avg_trust={avg_trust:.2f}",
            f"DPD Distribution: current={dpd_buckets['current']} | 1-30dpd={dpd_buckets['1-30']} | "
            f"31-60dpd={dpd_buckets['31-60']} | 60+dpd={dpd_buckets['60+']}",
        ]

        if high_risk_accounts:
            lines.append(f"High-risk accounts (31-60 DPD): {high_risk_accounts}")
        if upcoming_npa_risk:
            lines.append(f"⚠ NPA risk (60+ DPD): {upcoming_npa_risk} — act this month")

        if active_cluster_alerts:
            lines.append("MSME Cluster Alerts:")
            lines.extend(f"  • {a}" for a in active_cluster_alerts[:3])

        if active_ecosystem_alerts:
            lines.append("Startup Ecosystem Alerts:")
            lines.extend(f"  • {a}" for a in active_ecosystem_alerts[:3])

        if action_lines:
            lines.append("Recent actions:")
            lines.extend(action_lines)

        self._state = {
            "month": month,
            "episode": episode,
            "dpd_buckets": dpd_buckets,
            "high_risk": high_risk_accounts,
            "npa_risk": upcoming_npa_risk,
            "avg_trust": avg_trust,
        }

        return "\n".join(lines)

    def get_state(self) -> Dict:
        return self._state


# ---------------------------------------------------------------------------
# MEMORY MANAGER  (convenience wrapper)
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Orchestrates all three memory tiers for an MSME-RL training session.
    """

    def __init__(self):
        self.episodic  = EpisodicMemory(max_per_account=50)
        self.semantic  = SemanticMemory()
        self.working   = WorkingMemory()

    def record_step(
        self,
        episode: int,
        month: int,
        account_id: int,
        account_type: str,
        action_type: str,
        outcome: str,
        reward: float,
        hidden_profile: Dict,
        observable: Dict,
        trust_delta: float = 0.0,
    ) -> None:
        """Record one step into episodic + update semantic memory."""
        record: Dict = {
            "episode": episode,
            "month": month,
            "account_id": account_id,
            "account_type": account_type,
            "action_type": action_type,
            "outcome": outcome,
            "reward": reward,
            "trust_delta": trust_delta,
        }
        if account_type == "msme":
            record["industry"] = hidden_profile.get("industry", "")
            record["message_summary"] = observable.get("last_message", "")[:80]
            record["gst_status"] = observable.get("gst_filing_status", "")
            record["cluster_effect"] = f"{observable.get('cluster_accounts_behavior', '')}"
        else:
            record["stage"] = hidden_profile.get("stage", "")
            record["founder_message_tone"] = "highly_optimistic"   # always for startups
            record["behavioral_signals_checked"] = [
                k for k in ["linkedin_hiring_posts", "investor_update_sent", "github_commit_frequency"]
                if k in observable
            ]
            record["signals_contradicted_message"] = (
                observable.get("linkedin_hiring_posts", "") in ("none_in_90_days", "slowing_down")
                or observable.get("investor_update_sent", "") in ("skipped_last_month", "skipped_last_2_months")
            )

        self.episodic.add(record)

        # Update semantic memory
        keywords = _extract_keywords(account_type, observable)
        pattern_key = "+".join([account_type] + [kw for kw in keywords[1:4] if kw])
        self.semantic.update(
            pattern_key=pattern_key,
            outcome_positive=(reward > 0),
            reward=reward,
            metadata={"action_type": action_type},
        )

    def build_context(
        self,
        account_id: int,
        account_type: str,
        observable: Dict,
        industry_or_stage: str,
        action_type: str,
    ) -> Tuple[str, str]:
        """
        Build episodic + semantic context strings for memory injection.
        Returns (episodic_context, semantic_context).
        """
        # Episodic: similar past cases
        past_cases = self.episodic.retrieve_similar(
            account_type, industry_or_stage, action_type, n=3
        )
        episodic_ctx = self.episodic.format_for_context(past_cases)

        # Semantic: matched patterns
        matched_patterns = self.semantic.retrieve(account_type, observable, n=3)
        semantic_ctx = self.semantic.format_for_context(matched_patterns)

        return episodic_ctx, semantic_ctx