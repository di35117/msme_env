# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the MSME-RL Environment.

Two borrower types. Two opposite linguistic strategies.
One agent learns to decode both from reward signal alone.
"""

from typing import Any, Dict, List, Literal, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# ACTION MODEL
# ---------------------------------------------------------------------------

ACTION_TYPES = Literal[
    # Communication
    "send_empathetic_reminder",
    "send_firm_reminder",
    "send_legal_notice_section13",   # SARFAESI — wrong for startups
    "call_promoter_founder",
    "call_guarantor_investor",
    "conduct_cluster_ecosystem_visit",

    # Financial restructuring
    "grant_moratorium",
    "restructure_emi",
    "offer_eclgs_topup",              # MSME government scheme
    "offer_bridge_loan_extension",    # startup-specific
    "accept_partial_payment",
    "waive_penal_interest",

    # Recovery
    "initiate_sarfaesi",
    "refer_to_recovery_agent",
    "file_drt_case",
    "offer_one_time_settlement",

    # Information gathering
    "verify_gst_returns",             # MSME health check
    "pull_bank_statements",
    "check_industry_cluster_stress",
    "request_investor_update_meeting",  # startup triangulation
    "check_startup_ecosystem_signals",  # LinkedIn, GitHub, accelerator data

    # No-op (agent decides to wait and observe)
    "wait_and_observe",
]


class MSMERLAction(Action):
    """Action for the MSME-RL environment."""

    action_type: str = Field(..., description="One of 21 action types or wait_and_observe")
    account_id: int = Field(..., description="Target account (1-30; 1-20 MSME, 21-30 startup)")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific params: e.g. months=2 for grant_moratorium",
    )
    reasoning: str = Field(
        default="",
        description="Agent's chain-of-thought before selecting this action",
    )


# ---------------------------------------------------------------------------
# OBSERVABLE STATE SCHEMAS
# ---------------------------------------------------------------------------

class MSMEObservableState(dict):
    """
    Observable signals for one MSME account.
    True financial health is HIDDEN — agent only sees these proxies.
    """
    pass


class StartupObservableState(dict):
    """
    Observable signals for one startup account.
    True runway is HIDDEN — agent only sees behavioral proxies.
    """
    pass


# ---------------------------------------------------------------------------
# OBSERVATION MODEL
# ---------------------------------------------------------------------------

class MSMERLObservation(Observation):
    """Full portfolio observation delivered to the agent each step."""

    # Current episode/time context
    episode: int = Field(default=1, description="Training episode number")
    month: int = Field(default=1, description="Current month in 36-month loan cycle")

    # Portfolio state — agent sees observable signals only
    msme_accounts: List[Dict] = Field(
        default_factory=list,
        description="Observable state for each of the 20 MSME accounts",
    )
    startup_accounts: List[Dict] = Field(
        default_factory=list,
        description="Observable state for each of the 10 startup accounts",
    )

    # Portfolio-level summary
    portfolio_summary: Dict = Field(
        default_factory=dict,
        description="High-level portfolio stats visible to agent",
    )

    # Three-tier memory injected as context
    working_memory: str = Field(
        default="",
        description="Compact current-month state (< 2K tokens)",
    )
    semantic_memory_context: str = Field(
        default="",
        description="Relevant patterns retrieved from semantic memory",
    )
    episodic_memory_context: str = Field(
        default="",
        description="Similar past cases retrieved from episodic memory",
    )

    # Last action feedback
    last_action_result: Optional[Dict] = Field(
        default=None,
        description="Outcome of the last action taken",
    )

    # Network effect alerts
    active_cluster_alerts: List[str] = Field(
        default_factory=list,
        description="Active MSME cluster stress alerts",
    )
    active_ecosystem_alerts: List[str] = Field(
        default_factory=list,
        description="Active startup ecosystem stress alerts",
    )

    # Reward signal
    step_reward: float = Field(default=0.0, description="Reward for last action")
    episode_reward_so_far: float = Field(
        default=0.0, description="Cumulative episode reward"
    )

    # Terminal
    done: bool = Field(default=False, description="Episode complete (month 36 reached)")
    reward: Optional[float] = Field(default=None)
    metadata: Dict = Field(default_factory=dict)