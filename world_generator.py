# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
World Generator for MSME-RL.

Generates a mixed portfolio of 20 MSME + 10 startup accounts at episode start.
Hidden profiles drive simulation logic.  Agent sees only observable signals.

Parameters calibrated to published data:
  - MSME NPA rates: RBI Annual Report FY24
  - Startup default rates: NASSCOM / CIBIL 2023
  - Cluster contagion: SIDBI MSME Pulse Report
  - Startup ghosting rate: public NBFC filing disclosures
"""

import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


# ---------------------------------------------------------------------------
# CALIBRATION CONSTANTS  (from published data)
# ---------------------------------------------------------------------------

NPA_RATES = {
    "auto_ancillary": 0.092,   # RBI FY24
    "textile":         0.114,
    "pharma":          0.061,
    "fmcg":            0.073,
    "construction":    0.128,
    "food_processing": 0.084,
    "pre_seed":        0.18,   # NASSCOM / CIBIL 2023
    "seed":            0.14,
    "series_a":        0.08,
    "series_b":        0.05,
    "edtech":          0.15,   # Added for SFT alignment
}

MSME_CLUSTER_CONTAGION_FACTOR = 2.3  # SIDBI MSME Pulse: 1 default → 2.3 connected defaults
STARTUP_GHOSTING_BASE = 0.22         # public NBFC filings

MORATORIUM_RECOVERY_RATE = 0.67      # IBA study 2023
SARFAESI_RECOVERY_RATE   = 0.31


# ---------------------------------------------------------------------------
# MSME PROFILE TEMPLATES
# ---------------------------------------------------------------------------

MSME_INDUSTRIES = ["auto_ancillary", "textile", "pharma", "fmcg", "construction", "food_processing"]
MSME_LANGUAGES  = ["hindi", "hinglish", "marathi", "english"]
MSME_EXCUSE_STYLES = [
    "specific_business",   # real details — named OEM, named reason
    "vague_hardship",      # generic excuses
    "regulatory_blame",    # GST, government policy
    "market_conditions",   # sector-wide blame
]
MSME_COLLATERAL = ["plant_machinery", "commercial_property", "stock_pledge", "gold"]

# Representative MSME business names
MSME_BUSINESSES = [
    ("Sharma Auto Components Pvt Ltd",   "Rajesh Sharma"),
    ("Gupta Textile Mills",              "Suresh Gupta"),
    ("Patel Pharma Distributors",        "Kiran Patel"),
    ("Singh FMCG Wholesale",             "Harpreet Singh"),
    ("Reddy Construction Works",         "Venkat Reddy"),
    ("Mehta Food Processing",            "Dilip Mehta"),
    ("Kumar Auto Parts",                 "Santosh Kumar"),
    ("Joshi Weaving Industries",         "Prakash Joshi"),
    ("Verma Chemical Suppliers",         "Anil Verma"),
    ("Agarwal Trading Company",          "Ramesh Agarwal"),
    ("Rao Engineering Works",            "Srinivasa Rao"),
    ("Mishra Fabrication Pvt Ltd",       "Dinesh Mishra"),
    ("Tiwari Agro Products",             "Manoj Tiwari"),
    ("Chauhan Textiles",                 "Hemant Chauhan"),
    ("Desai Auto Ancillary",             "Nitin Desai"),
    ("Pandey Construction Corp",         "Rakesh Pandey"),
    ("Iyer Pharma Services",             "Krishnamurthy Iyer"),
    ("Bose Food Industries",             "Subhash Bose"),
    ("Saxena Packaging Works",           "Vijay Saxena"),
    ("Nair Rubber Products",             "Gopalan Nair"),
]

# ---------------------------------------------------------------------------
# STARTUP PROFILE TEMPLATES
# ---------------------------------------------------------------------------

STARTUP_STAGES  = ["pre_seed", "seed", "series_a", "series_b"]
STARTUP_SECTORS = ["b2b_saas", "fintech", "d2c", "deeptech", "edtech"]

STARTUP_COMPANIES = [
    ("FinStack Technologies Pvt Ltd",   "Arjun Mehta",     "b2b_saas",   "series_a"),
    ("CreditBridge AI",                 "Priya Nair",      "fintech",    "seed"),
    ("RetailOS India",                  "Rohit Kapoor",    "b2b_saas",   "series_b"),
    ("NanoMed Diagnostics",             "Ananya Rao",      "deeptech",   "pre_seed"),
    ("DealerEdge Technologies",         "Vikram Bhat",     "b2b_saas",   "seed"),
    ("TrustPay Networks",               "Neha Joshi",      "fintech",    "series_a"),
    ("FreshBox Direct",                 "Aditya Singh",    "d2c",        "seed"),
    ("QuantAI Labs",                    "Rahul Sharma",    "deeptech",   "pre_seed"),
    ("InsureQ Technologies",            "Deepa Krishnan",  "fintech",    "series_a"),
    ("AgriSense Platforms",             "Sunil Verma",     "b2b_saas",   "seed"),
]

INVESTOR_BACKERS = {
    "pre_seed":  ["angel_syndicate", "founders_fund", "none"],
    "seed":      ["sequoia_seed", "accel_seed", "blume", "100x_vc"],
    "series_a":  ["sequoia", "accel", "lightspeed", "matrix"],
    "series_b":  ["tiger_global", "softbank", "elevation", "kalaari"],
}


# ---------------------------------------------------------------------------
# GENERATORS
# ---------------------------------------------------------------------------

def _rng_seed(episode: int, account_id: int, field: str) -> float:
    """Deterministic-ish value for a given episode+account+field triple."""
    h = hash(f"{episode}-{account_id}-{field}") % 10_000
    return h / 10_000


def generate_msme_profile(
    account_id: int,
    episode: int,
    cluster_id: int,
    cluster_members: List[int],
    difficulty: float,
) -> Dict[str, Any]:
    """
    Generate a hidden MSME profile aligned with SFT anchor scenarios.
    """
    idx = (account_id - 1) % len(MSME_BUSINESSES)
    business_name, promoter = MSME_BUSINESSES[idx]
    
    # ALIGNMENT: Explicitly assign industries and centralities based on SFT data
    if account_id <= 6:
        industry = "auto_ancillary"
        cluster_centrality = 0.90 if account_id == 1 else 0.1 + _rng_seed(episode, account_id, "clust") * 0.3
    elif account_id <= 12:
        industry = "textile"
        cluster_centrality = 0.85 if account_id == 7 else 0.1 + _rng_seed(episode, account_id, "clust") * 0.2
    else:
        industry = "pharma"
        cluster_centrality = 0.90 if account_id == 13 else 0.1 + _rng_seed(episode, account_id, "clust") * 0.3

    if difficulty < 0.4:
        base_health = 0.85 if _rng_seed(episode, account_id, "h") > 0.5 else 0.15
        trajectory = "stable"
        strategic_default = False
    else:
        base_health = 0.35 + _rng_seed(episode, account_id, "health") * 0.55
        trajectory  = "declining" if _rng_seed(episode, account_id, "traj") > 0.55 else "stable"
        if trajectory == "declining":
            base_health = max(0.15, base_health - 0.20)
        strategic_default = _rng_seed(episode, account_id, "strat") < (0.12 * difficulty)

    npa_rate = NPA_RATES.get(industry, 0.09)
    crisis_month = None
    if _rng_seed(episode, account_id, "crisis") < npa_rate * 3:
        crisis_month = int(6 + _rng_seed(episode, account_id, "cm") * 24)

    language = MSME_LANGUAGES[(account_id + episode) % len(MSME_LANGUAGES)]
    excuse_style = MSME_EXCUSE_STYLES[(account_id) % len(MSME_EXCUSE_STYLES)]

    loan_amount = int(1_000_000 + _rng_seed(episode, account_id, "loan") * 4_000_000)

    return {
        "account_id": account_id,
        "account_type": "msme",
        "business_name": business_name,
        "promoter": promoter,
        "industry": industry,
        "true_financial_health": round(base_health, 3),
        "health_trajectory": trajectory,
        # Bool strategic_default → {0.0, 0.8}; downstream uses >0.5 as "strategic" flag.
        "strategic_default_propensity": round(float(strategic_default) * 0.8, 3),
        "crisis_trigger_month": crisis_month,
        "trust_score": round(0.5 + _rng_seed(episode, account_id, "trust") * 0.3, 3),
        "response_to_pressure": round(_rng_seed(episode, account_id, "press"), 3),
        "response_to_empathy": round(0.4 + _rng_seed(episode, account_id, "emph") * 0.5, 3),
        "cluster_id": cluster_id,
        "cluster_centrality": round(cluster_centrality, 3),
        "cluster_members": cluster_members,
        "communication_language": language,
        "excuse_style": excuse_style,
        "understates_distress": True,
        "loan_amount": loan_amount,
        "emi_amount": int(loan_amount * 0.033),
        "outstanding_principal": int(loan_amount * (0.5 + _rng_seed(episode, account_id, "princ") * 0.4)),
        "collateral_type": MSME_COLLATERAL[account_id % len(MSME_COLLATERAL)],
        "guarantor_strength": round(0.4 + _rng_seed(episode, account_id, "guar") * 0.5, 3),
        "months_since_origination": int(1 + _rng_seed(episode, account_id, "orig") * 20),
        "payment_history": _generate_msme_payment_history(base_health, strategic_default, episode, account_id),
    }


def generate_startup_profile(
    account_id: int,
    episode: int,
    ecosystem_network: List[int],
    difficulty: float,
) -> Dict[str, Any]:
    """
    Generate a hidden startup profile aligned with SFT ecosystem scenarios.
    """
    idx = (account_id - 21) % len(STARTUP_COMPANIES)
    company, founder, _, _ = STARTUP_COMPANIES[idx]

    # ALIGNMENT: Match SFT domains explicitly 
    if account_id <= 25:
        sector = "b2b_saas"
        stage = "seed"
    else:
        sector = "edtech"
        stage = "series_a"

    if difficulty < 0.4:
        true_runway = 20 if _rng_seed(episode, account_id, "r") > 0.5 else 2
    else:
        true_runway = int(1 + _rng_seed(episode, account_id, "runway") * 18)

    founder_optimism_bias = 0.5 + _rng_seed(episode, account_id, "opt") * 0.45

    npa_rate = NPA_RATES.get(stage, 0.10)
    crisis_trigger_month = None
    if true_runway <= 6 or _rng_seed(episode, account_id, "stcrisis") < npa_rate * 2:
        crisis_trigger_month = int(4 + _rng_seed(episode, account_id, "cmon") * 16)

    burn_rate = int(2_000_000 + _rng_seed(episode, account_id, "burn") * 8_000_000)
    mrr_base  = int(burn_rate * (0.1 + _rng_seed(episode, account_id, "mrr") * 0.3))
    mrr_growth = [
        round(-0.05 - _rng_seed(episode, account_id, f"mg{i}") * 0.15, 3)
        if true_runway < 6 else
        round(-0.02 + _rng_seed(episode, account_id, f"mg{i}") * 0.08, 3)
        for i in range(3)
    ]

    backer_pool = INVESTOR_BACKERS.get(stage, ["angel_syndicate"])
    investor = backer_pool[account_id % len(backer_pool)]

    loan_amount = int(2_000_000 + _rng_seed(episode, account_id, "sloan") * 8_000_000)

    return {
        "account_id": account_id,
        "account_type": "startup",
        "company": company,
        "founder": founder,
        "stage": stage,
        "sector": sector,
        "true_runway_months": true_runway,
        "founder_optimism_bias": round(founder_optimism_bias, 3),
        "investor_bridge_probability": round(0.2 + _rng_seed(episode, account_id, "bridge") * 0.6, 3),
        "pivot_risk": round(_rng_seed(episode, account_id, "pivot"), 3),
        "crisis_trigger_month": crisis_trigger_month,
        "trust_score": round(0.5 + _rng_seed(episode, account_id, "strust") * 0.3, 3),
        "ghosting_propensity": round(STARTUP_GHOSTING_BASE + _rng_seed(episode, account_id, "ghost") * 0.3, 3),
        "ecosystem_centrality": round(_rng_seed(episode, account_id, "ecos"), 3),
        "ecosystem_network": ecosystem_network,
        "communication_style": "pitch_english",
        "overstates_health": True,
        "burn_rate_monthly": burn_rate,
        "mrr": mrr_base,
        "mrr_growth_last_3m": mrr_growth,
        "investor_backing": investor,
        "loan_amount": loan_amount,
        "emi_amount": int(loan_amount * 0.028),
        "outstanding_principal": int(loan_amount * (0.5 + _rng_seed(episode, account_id, "sprinc") * 0.4)),
        "collateral_type": "none_clean",
        "months_since_origination": int(1 + _rng_seed(episode, account_id, "sorig") * 14),
        "payment_history": _generate_startup_payment_history(true_runway, episode, account_id),
    }


def _generate_msme_payment_history(
    financial_health: float,
    strategic_default: bool,
    episode: int,
    account_id: int,
) -> List[str]:
    history = []
    for i in range(5):
        r = _rng_seed(episode, account_id, f"ph{i}")
        if financial_health > 0.7:
            history.append("on_time")
        elif financial_health > 0.5:
            history.append("on_time" if r > 0.25 else f"{int(r * 20) + 3}_days_late")
        elif financial_health > 0.3:
            history.append("on_time" if r > 0.4 else f"{int(r * 25) + 5}_days_late")
        else:
            history.append(f"{int(r * 30) + 10}_days_late" if r > 0.2 else "on_time")
        if strategic_default and i >= 3:
            history[-1] = f"{int(r * 45) + 15}_days_late"
    return history


def _generate_startup_payment_history(
    runway_months: int,
    episode: int,
    account_id: int,
) -> List[str]:
    history = []
    for i in range(5):
        r = _rng_seed(episode, account_id, f"sph{i}")
        if runway_months > 9:
            history.append("on_time")
        elif runway_months > 5:
            history.append("on_time" if r > 0.30 else f"{int(r * 15) + 3}_days_late")
        else:
            history.append(f"{int(r * 20) + 5}_days_late" if r > 0.15 else "on_time")
    return history


# ---------------------------------------------------------------------------
# CLUSTER / ECOSYSTEM NETWORK ASSIGNMENT
# ---------------------------------------------------------------------------

def assign_msme_clusters(account_ids: List[int]) -> Dict[int, Tuple[int, List[int]]]:
    """
    Group MSME accounts into geographic industry clusters.
    ALIGNED: Groups directly match the SFT sector brackets.
    """
    clusters = {}
    
    # 1. Auto Ancillary Cluster (1-6)
    c1 = [acc for acc in account_ids if acc <= 6]
    # 2. Textile Cluster (7-12)
    c2 = [acc for acc in account_ids if 7 <= acc <= 12]
    # 3. Pharma Cluster (13-20)
    c3 = [acc for acc in account_ids if acc >= 13]

    for cluster_id, chunk in enumerate([c1, c2, c3]):
        for acc_id in chunk:
            clusters[acc_id] = (cluster_id, [x for x in chunk if x != acc_id])
            
    return clusters


def assign_startup_ecosystem(account_ids: List[int]) -> Dict[int, List[int]]:
    ecosystem = {}
    for acc_id in account_ids:
        connected = [x for x in account_ids if x != acc_id]
        ecosystem[acc_id] = connected[:3]
    return ecosystem


# ---------------------------------------------------------------------------
# TOP-LEVEL PORTFOLIO GENERATOR
# ---------------------------------------------------------------------------

def generate_portfolio(episode: int) -> Dict[str, Any]:
    difficulty = min(1.0, episode / 50.0)

    msme_ids   = list(range(1, 21))
    startup_ids = list(range(21, 31))

    msme_clusters      = assign_msme_clusters(msme_ids)
    startup_ecosystem  = assign_startup_ecosystem(startup_ids)

    hidden_profiles: Dict[int, Dict] = {}
    observable_states: Dict[int, Dict] = {}

    for acc_id in msme_ids:
        cluster_id, cluster_members = msme_clusters[acc_id]
        profile = generate_msme_profile(acc_id, episode, cluster_id, cluster_members, difficulty)
        hidden_profiles[acc_id] = profile
        observable_states[acc_id] = build_msme_observable(profile)

    for acc_id in startup_ids:
        ecosystem = startup_ecosystem[acc_id]
        profile = generate_startup_profile(acc_id, episode, ecosystem, difficulty)
        hidden_profiles[acc_id] = profile
        observable_states[acc_id] = build_startup_observable(profile)

    return {
        "hidden_profiles": hidden_profiles,
        "observable_states": observable_states,
        "msme_ids": msme_ids,
        "startup_ids": startup_ids,
        "msme_clusters": msme_clusters,
        "startup_ecosystem": startup_ecosystem,
        "episode_id": str(uuid4()),
    }


# ---------------------------------------------------------------------------
# OBSERVABLE SIGNAL BUILDERS
# ---------------------------------------------------------------------------

_MSME_MESSAGES = {
    "hindi": [
        "Sir thoda problem hai. GST input credit phase nahi hua abhi. OEM ne bhi payment rok diya quality audit ke wajah se. Ek mahine ka time de dijiye sir.",
        "Sir, auto sector mein kaafi dikkat chal rahi hai. Meri payment rok gayi hai. Aap samjhenge meri situation.",
        "Namaskar sir. GST return mein kuch technical issue aaya. Agle hafte tak clear ho jayega. Main zaroor dunga.",
        "Sir bahut takleef hai. OEM ne payment rok diya. Thoda time de dijiye please.",
        "Sir koi baat nahi. GST atak gaya hai bas. Sab theek ho jayega jald hi.",
    ],
    "hinglish": [
        "Sir thoda GST issue aa gaya hai. OEM payment bhi pending hai. Give me 30 days please.",
        "Yaar GST input credit stuck hai. Supply chain mein problem. Will sort out soon.",
        "Sir, market conditions bahut tough hain. Cash flow issue hai. Please adjust karo.",
    ],
    "marathi": [
        "Sir, GST credit stack zala ahe. OEM payment late ahe. Mahina dya please.",
        "Sir thoda problem ahe. Payments yenyala vel lagel. Samjun ghya.",
    ],
    "english": [
        "Dear Sir, due to GST input credit delays and OEM payment issues, I require a brief extension. Will settle shortly.",
        "Sir, market conditions are tough in our sector. Requesting a 30-day grace period.",
    ],
}

_STARTUP_MESSAGES = [
    "Hey, we're actually in a really exciting place right now. Just closed a partnership with a major enterprise client. The bridge we discussed — we're confident Q3 revenue covers it comfortably. Team is fully aligned and we've actually accelerated hiring for the next phase.",
    "Things are going really well! We just hit a key milestone with our enterprise pipeline. The Series A momentum is strong and we expect to be cash-flow positive by Q3. Happy to share more details on our progress.",
    "We're seeing incredible traction on the enterprise side. The product-market fit is stronger than ever. We have a strong pipeline that will more than cover the repayment. The team is energized.",
    "Really exciting times! Just finalizing a large enterprise contract that will significantly boost our MRR. Q4 looks very strong. Bridge timing should work out perfectly.",
    "We're in a great place strategically. Just closed a key partnership and our runway is solid. The business fundamentals are stronger than they've ever been.",
]


def build_msme_observable(profile: Dict) -> Dict:
    health = profile["true_financial_health"]
    payment_history = profile["payment_history"]

    last_payment = payment_history[-1] if payment_history else "on_time"
    if last_payment == "on_time":
        dpd = 0
    else:
        try:
            dpd = int(last_payment.split("_")[0])
        except (ValueError, IndexError):
            dpd = 0

    if health > 0.65:
        gst_status = "filed_on_time"
    elif health > 0.45:
        gst_status = "filed_with_delay_last_month"
    elif health > 0.25:
        gst_status = "filed_with_delay_last_2_months"
    else:
        gst_status = "not_filed_last_month"

    if health > 0.6:
        cf_trend = "stable"
    elif health > 0.4:
        cf_trend = "declining_8pct"
    elif health > 0.25:
        cf_trend = "declining_15pct"
    else:
        cf_trend = "declining_30pct"

    strat = profile.get("strategic_default_propensity", 0)
    if strat > 0.5:
        call_response = "not_answering"
    elif health < 0.35:
        call_response = "answered_third_try"
    elif health < 0.55:
        call_response = "answered_second_try"
    else:
        call_response = "answered_immediately"

    guar = profile.get("guarantor_strength", 0.5)
    guarantor_status = "contactable" if guar > 0.4 else "unreachable"

    lang = profile.get("communication_language", "hindi")
    messages = _MSME_MESSAGES.get(lang, _MSME_MESSAGES["english"])
    msg_idx = profile["account_id"] % len(messages)
    last_message = messages[msg_idx]

    industry = profile["industry"]
    industry_stress = NPA_RATES.get(industry, 0.09) * 8

    cluster_members = profile.get("cluster_members", [])
    cluster_late_count = max(0, int(len(cluster_members) * (1 - health)))

    return {
        "account_id": profile["account_id"],
        "account_type": "msme",
        "business_name": profile["business_name"],
        "promoter": profile["promoter"],
        "industry": profile["industry"],
        "communication_language": lang,
        "dpd": dpd,
        "emi_amount": profile["emi_amount"],
        "outstanding_principal": profile["outstanding_principal"],
        "last_message": last_message,
        "call_response": call_response,
        "gst_filing_status": gst_status,
        "bank_statement_cash_flow_trend": cf_trend,
        "payment_history": payment_history,
        "industry_stress_index": round(industry_stress, 3),
        "cluster_accounts_behavior": f"{cluster_late_count}_of_{len(cluster_members)}_connected_also_late",
        "guarantor_reachability": guarantor_status,
        "last_physical_visit": f"month_{max(1, profile['months_since_origination'] - 6)}",
        "collateral_type": profile["collateral_type"],
        "loan_amount": profile["loan_amount"],
    }


def build_startup_observable(profile: Dict) -> Dict:
    runway = profile["true_runway_months"]
    optimism_bias = profile.get("founder_optimism_bias", 0.75)

    payment_history = profile["payment_history"]
    last_payment = payment_history[-1] if payment_history else "on_time"
    if last_payment == "on_time":
        dpd = 0
    else:
        try:
            dpd = int(last_payment.split("_")[0])
        except (ValueError, IndexError):
            dpd = 0

    if runway <= 4:
        linkedin_hiring    = "none_in_90_days"
        github_commits     = "declining_40pct"
        investor_updates   = "skipped_last_2_months"
        accelerator_status = "demo_day_missed"
        glassdoor_trend    = "recent_negative_reviews"
        call_response      = "replied_via_whatsapp_avoided_voice"
        mrr_trend = [
            int(profile["mrr"] * 1.10),
            int(profile["mrr"] * 1.04),
            profile["mrr"],
        ]
    elif runway <= 8:
        linkedin_hiring    = "slowing_down"
        github_commits     = "declining_15pct"
        investor_updates   = "skipped_last_month"
        accelerator_status = "active"
        glassdoor_trend    = "mixed"
        call_response      = "answered_with_delay"
        mrr_trend = [
            int(profile["mrr"] * 1.05),
            int(profile["mrr"] * 1.01),
            profile["mrr"],
        ]
    else:
        linkedin_hiring    = "active_postings"
        github_commits     = "stable"
        investor_updates   = "sent_on_time"
        accelerator_status = "active"
        glassdoor_trend    = "positive"
        call_response      = "answered_immediately"
        mrr_trend = [
            int(profile["mrr"] * 0.92),
            int(profile["mrr"] * 0.96),
            profile["mrr"],
        ]

    msg_idx = profile["account_id"] % len(_STARTUP_MESSAGES)
    last_message = _STARTUP_MESSAGES[msg_idx]

    cofounder_signal = (
        "one_cofounder_job_hunting"
        if runway <= 5 and profile.get("ghosting_propensity", 0) > 0.3
        else "stable"
    )

    ecosystem_network = profile.get("ecosystem_network", [])
    ecosystem_defaults = max(0, int(len(ecosystem_network) * (1 - runway / 12)))

    return {
        "account_id": profile["account_id"],
        "account_type": "startup",
        "company": profile["company"],
        "founder": profile["founder"],
        "stage": profile["stage"],
        "sector": profile["sector"],
        "dpd": dpd,
        "emi_amount": profile["emi_amount"],
        "outstanding_principal": profile["outstanding_principal"],
        "last_message": last_message,
        "call_response": call_response,
        "mrr_last_3_months": mrr_trend,
        "linkedin_hiring_posts": linkedin_hiring,
        "github_commit_frequency": github_commits,
        "investor_update_sent": investor_updates,
        "accelerator_status": accelerator_status,
        "glassdoor_trend": glassdoor_trend,
        "cofounder_linkedin_activity": cofounder_signal,
        "payment_history": payment_history,
        "ecosystem_accounts_behavior": f"{ecosystem_defaults}_portfolio_company_defaulted",
        "investor_backing": profile["investor_backing"],
        "loan_amount": profile["loan_amount"],
    }