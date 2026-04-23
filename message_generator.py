# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Message Generator for MSME-RL.

The same 1.7B model that learned the policy generates the outbound RM message
in the appropriate language and register.

MSME messages: Hindi / Hinglish / Marathi — relationship-preserving, empathetic.
Startup messages: English — firm but relationship-preserving.

Language, tone, and framing are NOT templated.
They emerge from pretraining knowledge + RL-trained policy.

This module provides deterministic templates for the simulation layer.
The actual LLM call replaces these templates during training.
"""

from typing import Dict, Optional


# ---------------------------------------------------------------------------
# RM MESSAGE TEMPLATES — MSME (Hindi / Hinglish)
# ---------------------------------------------------------------------------

MSME_TEMPLATES: Dict[str, Dict[str, str]] = {
    "grant_moratorium": {
        "hindi": (
            "{promoter} ji,\n\n"
            "Aapka hamare bank ke saath {months_relation} saal ka relationship hamein appreciate "
            "karta hai. {industry} sector ki current challenges aur payment situation ko samajhte "
            "hue, hum aapko {months} mahine ka moratorium de rahe hain. Is period mein koi EMI "
            "nahi hogi.\n\n"
            "Ek request: GST returns ka last 3 mahine ka screenshot share karein record ke liye. "
            "{resume_month} se normal schedule resume hogi.\n\n"
            "Koi dikkat ho toh mujhe directly call karein.\n\n"
            "[RM Name] | [Bank] | [Branch]"
        ),
        "hinglish": (
            "Hi {promoter} ji,\n\n"
            "Aapka situation fully samjha. {industry} sector mein sab facing same challenges. "
            "We are giving you {months} month moratorium — no EMI pressure during this period.\n\n"
            "Please share GST returns for last 3 months. Will help us process this faster.\n\n"
            "Call me anytime.\n\n"
            "[RM Name] | [Bank]"
        ),
        "marathi": (
            "{promoter} ji,\n\n"
            "Tumchi paristhiti samajali. {months} mahinyanche moratorium deto aahe. "
            "Ya kaladhat koni EMI bharayachi nahi.\n\n"
            "GST returns 3 mahinyache pathvaya. {resume_month} la normal schedule suru hoil.\n\n"
            "[RM Name] | [Bank] | [Branch]"
        ),
        "english": (
            "Dear {promoter},\n\n"
            "We appreciate your longstanding relationship with us. In view of the current "
            "challenges facing the {industry} sector, we are pleased to offer a {months}-month "
            "moratorium on your EMI. No payment will be due during this period.\n\n"
            "Please share your GST returns for the last 3 months for our records. "
            "Normal repayment will resume from {resume_month}.\n\n"
            "Do not hesitate to call me directly.\n\n"
            "[RM Name] | [Bank] | [Branch]"
        ),
    },

    "send_empathetic_reminder": {
        "hindi": (
            "{promoter} ji,\n\n"
            "Aapka EMI {days} din se pending hai. Samjhta hun ki {industry} sector mein "
            "abhi kaafi challenge hai. Agar koi temporary issue hai, bata dijiye — "
            "hum milke koi solution nikalte hain.\n\n"
            "Iss mahine mein connect karein please.\n\n"
            "[RM Name] | [Bank]"
        ),
        "hinglish": (
            "Sir {promoter},\n\n"
            "EMI {days} days pending. Sector mein tough time chal raha hai, pata hai. "
            "Let's talk — maybe we can work something out.\n\n"
            "Call me this week.\n\n"
            "[RM Name]"
        ),
        "english": (
            "Dear {promoter},\n\n"
            "Your EMI is {days} days overdue. I understand the {industry} sector is under "
            "considerable pressure right now. If you're facing a temporary cash flow issue, "
            "please connect with me this week so we can explore solutions together.\n\n"
            "[RM Name] | [Bank]"
        ),
        "marathi": (
            "{promoter} ji,\n\n"
            "EMI {days} divas pending aahe. Sector madhe problem ahe samajle. "
            "Bolu ya ek wela — kahi solution kadhata yeyl.\n\n"
            "[RM Name] | [Bank]"
        ),
    },

    "send_firm_reminder": {
        "hindi": (
            "{promoter} ji,\n\n"
            "Aapka EMI {days} din se overdue hai. Yeh ek formal reminder hai. "
            "Please {deadline} tak payment karein warna further action lena padega.\n\n"
            "Agar koi genuine issue hai, abhi contact karein.\n\n"
            "[RM Name] | [Bank] | [Branch]"
        ),
        "hinglish": (
            "{promoter} Sir,\n\n"
            "EMI {days} days overdue. Please pay by {deadline} to avoid further action. "
            "If genuine issue, call immediately.\n\n"
            "[RM Name] | [Bank]"
        ),
        "english": (
            "Dear {promoter},\n\n"
            "This is a formal reminder that your EMI instalment is {days} days overdue. "
            "Please ensure payment by {deadline} to avoid penal interest and further action.\n\n"
            "If you are facing a genuine hardship, please contact me immediately.\n\n"
            "[RM Name] | [Bank] | [Branch]"
        ),
        "marathi": (
            "{promoter} ji,\n\n"
            "EMI {days} divas overdue aahe. {deadline} paryant bharayla sanga. "
            "Khara adachaN aslyas lagech fon kara.\n\n"
            "[RM Name] | [Bank]"
        ),
    },

    "verify_gst_returns": {
        "hindi": (
            "{promoter} ji,\n\n"
            "Account review ke liye last 3 months ke GST returns GSTR-3B share karein. "
            "Isse aapki file strong hogi aur koi bhi support jaldi process hogi.\n\n"
            "[RM Name] | [Bank]"
        ),
        "hinglish": (
            "{promoter} Sir,\n\n"
            "Please share last 3 months GST GSTR-3B for account review. "
            "Will help us support your case faster.\n\n"
            "[RM Name]"
        ),
        "english": (
            "Dear {promoter},\n\n"
            "For our account review, please share your GSTR-3B returns for the last 3 months. "
            "This will help us process any support requests more efficiently.\n\n"
            "[RM Name] | [Bank]"
        ),
        "marathi": (
            "{promoter} ji,\n\n"
            "Account review sathi last 3 mahinyache GST GSTR-3B pathva. "
            "Yamulle tumchi case lavakar process hoil.\n\n"
            "[RM Name] | [Bank]"
        ),
    },

    "send_legal_notice_section13": {
        "hindi": (
            "{promoter} ji,\n\n"
            "Yeh SARFAESI Act 2002, Section 13(2) ke antargat formal notice hai. "
            "Aapka account {dpd} DPD par hai. Please {days} dino mein payment karein "
            "warna asset action shuru hoga.\n\n"
            "[Legal Department] | [Bank]"
        ),
        "english": (
            "Dear {promoter},\n\n"
            "This is a formal notice under SARFAESI Act 2002, Section 13(2). "
            "Your account stands at {dpd} DPD. Please clear your dues within {days} days "
            "failing which we will be constrained to proceed under the Act.\n\n"
            "[Legal Department] | [Bank]"
        ),
    },
}

# ---------------------------------------------------------------------------
# RM MESSAGE TEMPLATES — STARTUP (English)
# ---------------------------------------------------------------------------

STARTUP_TEMPLATES: Dict[str, str] = {
    "send_empathetic_reminder": (
        "Hi {founder},\n\n"
        "Good to connect. I saw your update about the enterprise pipeline — "
        "sounds like things are moving.\n\n"
        "That said, I wanted to flag that our repayment is {days} days overdue. "
        "I'm sure it's just timing — would it help to get on a quick call this week "
        "to align on the schedule?\n\n"
        "[RM Name] | [Bank]"
    ),

    "send_firm_reminder": (
        "Hi {founder},\n\n"
        "Following up on the outstanding EMI — now {days} days overdue. "
        "I need this cleared by {deadline}.\n\n"
        "Happy to discuss payment timing but need clarity on this before we can "
        "consider any further flexibility.\n\n"
        "[RM Name] | [Bank]"
    ),

    "request_investor_update_meeting": (
        "Hi {founder},\n\n"
        "Thanks for the update — great to hear about the enterprise momentum.\n\n"
        "Given where we are on the repayment schedule, I'd like to set up a quick call "
        "this week — even 20 minutes — to align on Q4 timelines. It would also help to "
        "loop in your lead investor briefly so we're all on the same page.\n\n"
        "Thursday 3pm work? Happy to do a video call.\n\n"
        "[RM Name] | [Bank]"
    ),

    "check_startup_ecosystem_signals": (
        "Hi {founder},\n\n"
        "Quick check-in. I noticed your LinkedIn activity has been quiet and wanted "
        "to make sure everything is on track.\n\n"
        "Can we set up a 15-minute call this week? Just want to make sure we're aligned "
        "ahead of the next payment date.\n\n"
        "[RM Name] | [Bank]"
    ),

    "offer_bridge_loan_extension": (
        "Hi {founder},\n\n"
        "I've reviewed your account and I'd like to explore a {months}-month bridge "
        "extension with you. This would give you runway to close the enterprise deal "
        "without payment pressure.\n\n"
        "Can we get on a call this week with you and your CFO to align on the terms?\n\n"
        "[RM Name] | [Bank]"
    ),

    "grant_moratorium": (
        "Hi {founder},\n\n"
        "We're offering a {months}-month moratorium on your EMI given the current "
        "fundraising timeline. No repayment due during this period.\n\n"
        "We'll need a brief investor confirmation by {deadline} to process this formally. "
        "Please loop in your lead investor on email.\n\n"
        "[RM Name] | [Bank]"
    ),

    "call_promoter_founder": (
        "Hi {founder},\n\n"
        "I'd like to schedule a call this week — 20 minutes to review where things stand "
        "on the repayment side and make sure we're aligned.\n\n"
        "What time works best for you?\n\n"
        "[RM Name] | [Bank]"
    ),
}


# ---------------------------------------------------------------------------
# GENERATOR FUNCTION
# ---------------------------------------------------------------------------

def generate_rm_message(
    action_type: str,
    account_type: str,
    account_profile: Dict,
    observable: Dict,
    action_params: Optional[Dict] = None,
) -> str:
    """
    Generate an RM message for the given action and account.

    During simulation: uses deterministic templates.
    During training: this call is replaced by the fine-tuned LLM inference.

    Args:
        action_type: The action being taken
        account_type: "msme" or "startup"
        account_profile: Full account profile
        observable: Observable signals for this account
        action_params: Action-specific parameters (e.g. months=2 for moratorium)

    Returns:
        The generated RM message string
    """
    params = action_params or {}

    if account_type == "msme":
        language = account_profile.get("communication_language", "hindi")
        templates = MSME_TEMPLATES.get(action_type, {})
        template = templates.get(language, templates.get("english", ""))

        if not template:
            template = (
                f"Dear {account_profile.get('promoter', 'Sir')},\n\n"
                f"Regarding your account — please contact us at the earliest.\n\n[RM Name] | [Bank]"
            )

        months_relation = account_profile.get("months_since_origination", 12) // 12 or 1
        months = params.get("months", 2)
        resume_month = f"Month {observable.get('dpd', 0) + months * 30}"
        dpd = observable.get("dpd", 0)
        days = max(7, params.get("days", 30))
        deadline_day = params.get("deadline_days", 30)

        return template.format(
            promoter=account_profile.get("promoter", "Sir"),
            industry=account_profile.get("industry", "your sector"),
            months_relation=months_relation,
            months=months,
            resume_month=resume_month,
            dpd=dpd,
            days=days,
            deadline=f"{deadline_day} days from today",
        )

    else:  # startup
        template = STARTUP_TEMPLATES.get(action_type, "")
        if not template:
            template = (
                f"Hi {account_profile.get('founder', 'Team')},\n\n"
                f"Please get in touch regarding your account at the earliest.\n\n[RM Name] | [Bank]"
            )

        months = params.get("months", 2)
        days = max(5, observable.get("dpd", 0))
        deadline_day = params.get("deadline_days", 14)

        return template.format(
            founder=account_profile.get("founder", "Team"),
            company=account_profile.get("company", "your company"),
            months=months,
            days=days,
            deadline=f"{deadline_day} days from today",
        )


def get_message_language_tag(account_type: str, account_profile: Dict) -> str:
    """Return language/register tag for the generated message."""
    if account_type == "msme":
        lang = account_profile.get("communication_language", "hindi")
        return f"msme_{lang}"
    return "startup_pitch_english"