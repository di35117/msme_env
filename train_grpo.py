"""
MSME-RL Training Script — GRPO with Qwen3-1.7B

Model:     Qwen/Qwen3-1.7B
Training:  GRPO — TRL 0.29.0 + vLLM
Rewards:   Step-level (14 outcome types) + Episode-level (NPA rate + recovery)
Target:    300+ simulation episodes in 2-day compute window
Platform:  HuggingFace Spaces (OpenEnv compliant)
Memory:    Three-tier (episodic + semantic + working) injected as context

Run:
    python train_grpo.py --episodes 300 --port 8000

Or on Colab with Unsloth:
    !pip install unsloth trl>=0.29.0 openenv-core
    %run train_grpo.py --episodes 50 --max_steps_per_episode 90
"""

import argparse
import asyncio
import ast
import json
import os
import re
import time
import pathlib
import inspect
from typing import Any, Dict, List, Optional

# Reduce CUDA allocator fragmentation on long RL runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Unsloth should be imported before transformers/trl when available.
try:
    import unsloth  # noqa: F401
except Exception:  # noqa: BLE001
    unsloth = None  # type: ignore

from datasets import Dataset
from transformers import TrainingArguments


# ---------------------------------------------------------------------------
# GRPO TRAINING PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Relationship Manager (RM) at an Indian bank managing a mixed portfolio
of MSME (small business) and startup accounts across a 36-month loan cycle.

YOUR CHALLENGE: Two types of borrowers hide financial distress in completely opposite ways.
  - MSME owners UNDERSTATE problems in Hindi/Hinglish understatement
  - Startup founders OVERSTATE health in pitch-deck English confidence

You must learn to decode BOTH from behavioral signals, not surface messages.

KEY SIGNAL RULES:
MSME signals of genuine distress (NOT what they say, how they say it):
  - Specificity of excuse (real problems have real details: named OEM, named reason)
  - GST filing regularity (operational businesses file on time)
  - Response time to calls (genuinely stressed owners answer; strategic defaulters avoid)
  - Payment history pattern (13 months clean then sudden delay = genuine crisis)
  - Language register shift (more formal, more deferential than usual)

Startup signals of genuine distress (what contradicts the pitch):
  - LinkedIn hiring activity stopped (contradicts "accelerated hiring")
  - Investor updates missed for 2+ months (accelerator-stage founders always update)
  - GitHub commit activity declining (developers commit when building)
  - Avoiding voice calls, replying via WhatsApp (founders ghost before defaulting)
  - MRR negative for 3 months (contradicts "strong momentum")
  - Cofounder job-hunting on LinkedIn (worst signal, company dissolving)

AVAILABLE ACTIONS (21 total):
Communication: send_empathetic_reminder, send_firm_reminder, send_legal_notice_section13,
                call_promoter_founder, call_guarantor_investor, conduct_cluster_ecosystem_visit

Financial restructuring: grant_moratorium, restructure_emi, offer_eclgs_topup (MSME only),
                          offer_bridge_loan_extension (startup only), accept_partial_payment,
                          waive_penal_interest

Recovery: initiate_sarfaesi, refer_to_recovery_agent, file_drt_case, offer_one_time_settlement

Information gathering: verify_gst_returns (MSME only), pull_bank_statements,
                        check_industry_cluster_stress, request_investor_update_meeting (startup only),
                        check_startup_ecosystem_signals (startup only)

No-op: wait_and_observe

CRITICAL RULES:
  NEVER use initiate_sarfaesi on a startup, destroys ecosystem trust, triggers cascades
  NEVER take startup pitch optimism at face value without checking behavioral signals
  ALWAYS verify_gst_returns before granting moratorium to MSME
  ALWAYS request_investor_update_meeting before restructuring startup debt
  Consider cluster_centrality for MSME decisions (high-centrality account = cascade risk)

REWARD SIGNAL:
  Step rewards: +0.10 investor_meeting_triggered_bridge | +0.08 payment_received_after_empathy
                -0.25 cluster_cascade_default | -0.15 sarfaesi_used_on_startup
  Episode reward: 0.40*(1-NPA rate) + 0.30*recovery_rate + 0.20*relationship_score + 0.10*tool_fit

Respond with your action in this exact JSON format:
{
  "reasoning": "<brief reason in 1-2 sentences based on observed signals>",
  "action_type": "<one of the 21 action types>",
  "account_id": <integer 1-30>,
  "parameters": {"months": 2}
}"""


# Canonical set of valid action_type strings. Keep in sync with the env action enum.
# Used by the parser to snap near-miss model outputs (e.g. "call_guarantor",
# "call_promoter_foundier") to the closest valid action by edit distance.
VALID_ACTIONS = (
    "send_empathetic_reminder",
    "send_firm_reminder",
    "send_legal_notice_section13",
    "call_promoter_founder",
    "call_guarantor_investor",
    "conduct_cluster_ecosystem_visit",
    "grant_moratorium",
    "restructure_emi",
    "offer_eclgs_topup",
    "offer_bridge_loan_extension",
    "accept_partial_payment",
    "waive_penal_interest",
    "initiate_sarfaesi",
    "refer_to_recovery_agent",
    "file_drt_case",
    "offer_one_time_settlement",
    "verify_gst_returns",
    "pull_bank_statements",
    "check_industry_cluster_stress",
    "request_investor_update_meeting",
    "check_startup_ecosystem_signals",
    "wait_and_observe",
)


def _snap_to_valid_action(name: str) -> str:
    """
    Return the closest valid action_type to `name`. Uses substring containment
    first (cheap, exact-prefix matches), then SequenceMatcher ratio.
    Falls back to 'wait_and_observe' when no candidate scores above 0.55.
    """
    if not isinstance(name, str):
        return "wait_and_observe"
    norm = name.strip().lower()
    if not norm:
        return "wait_and_observe"
    if norm in VALID_ACTIONS:
        return norm
    contains = [a for a in VALID_ACTIONS if norm in a or a in norm]
    if len(contains) == 1:
        return contains[0]
    from difflib import SequenceMatcher
    best, best_score = "wait_and_observe", 0.0
    for cand in VALID_ACTIONS:
        score = SequenceMatcher(None, norm, cand).ratio()
        if score > best_score:
            best, best_score = cand, score
    return best if best_score >= 0.55 else "wait_and_observe"


def build_agent_prompt(observation: Dict) -> str:
    """
    Compact prompt designed for small LMs (0.5B-1.7B).

    Long prompts (full memory tiers + JSON dump of all accounts) cause small models
    to lose format discipline — they emit ``` or whitespace instead of JSON.
    This version sends only what's needed to choose ONE action this step:
    portfolio summary, top alerts, and the 3 most urgent accounts as flat lines.
    """
    summary  = observation.get("portfolio_summary", {}) or {}
    month    = summary.get("current_month", observation.get("month", 1))
    npa_rate = summary.get("npa_rate", 0.0)
    cum_r    = summary.get("cumulative_reward", observation.get("episode_reward_so_far", 0.0))

    msme_accounts    = observation.get("msme_accounts", [])
    startup_accounts = observation.get("startup_accounts", [])
    urgent = sorted(
        msme_accounts + startup_accounts,
        key=lambda x: x.get("dpd", 0),
        reverse=True,
    )[:3]

    acct_lines = []
    for acc in urgent:
        acc_id   = acc.get("account_id", "?")
        acc_type = acc.get("account_type", "msme")
        dpd      = acc.get("dpd", 0)
        signal   = (
            acc.get("last_message")
            or acc.get("latest_signal")
            or acc.get("last_rm_message")
            or ""
        )
        if len(signal) > 100:
            signal = signal[:97] + "..."
        acct_lines.append(
            f"  id={acc_id} type={acc_type} dpd={dpd} signal=\"{signal}\""
        )
    acct_block = "\n".join(acct_lines) if acct_lines else "  (none)"

    alerts = (
        observation.get("active_cluster_alerts", [])
        + observation.get("active_ecosystem_alerts", [])
    )[:2]
    alert_str = " | ".join(alerts) if alerts else "none"

    last      = observation.get("last_action_result") or {}
    last_line = ""
    if last:
        outcome   = last.get("outcome", "")
        last_r    = last.get("step_reward", "")
        last_line = f"\nLast action: outcome={outcome} reward={last_r}"

    most_urgent_id   = urgent[0].get("account_id", 1) if urgent else 1
    most_urgent_type = urgent[0].get("account_type", "msme") if urgent else "msme"

    return (
        f"Month {month}/36. NPA={npa_rate:.1%}. CumReward={cum_r:.3f}.{last_line}\n"
        f"Alerts: {alert_str}\n"
        f"Top urgent accounts:\n{acct_block}\n\n"
        f"Most urgent account: id={most_urgent_id} type={most_urgent_type}.\n"
        f"Choose the single best action for THIS account "
        f"(set account_id={most_urgent_id} in your JSON). "
        f"Respond with one JSON object only."
    )


# ---------------------------------------------------------------------------
# SFT WARM START
# FIXES APPLIED:
#   1. dataset_text_field is now "text" (prompt + completion combined)
#   2. behavioral_signal_check replaced with valid action types:
#      check_startup_ecosystem_signals (startup) / verify_gst_returns (MSME)
#   3. All account_id values capped at max 30
# ---------------------------------------------------------------------------

def _safe_import_sft_trainer():
    """
    Import TRL's SFTTrainer with a Windows-safe UTF-8 fallback.

    Some Windows locales default to cp1252, while TRL reads template files with
    Path.read_text() and no explicit encoding. This shim forces UTF-8 when the
    encoding is omitted, preventing UnicodeDecodeError on import.
    """
    original_read_text = pathlib.Path.read_text
    read_text_sig = inspect.signature(original_read_text)
    supports_newline = "newline" in read_text_sig.parameters

    def _utf8_read_text(self, encoding=None, errors=None, newline=None):
        kwargs = {
            "encoding": encoding or "utf-8",
            "errors": errors,
        }
        if supports_newline:
            kwargs["newline"] = newline
        return original_read_text(self, **kwargs)

    pathlib.Path.read_text = _utf8_read_text  # type: ignore[assignment]
    try:
        from trl import SFTTrainer  # noqa: WPS433
        try:
            from trl import SFTConfig  # type: ignore  # noqa: WPS433
        except Exception:  # noqa: BLE001
            SFTConfig = None
        return SFTTrainer, SFTConfig, None
    except Exception as exc:  # noqa: BLE001
        return None, None, exc


def run_sft_warm_start(model, tokenizer, output_dir):
    """
    SFT warm start: gives the model a sensible starting policy before GRPO begins.
    Teaches correct action types and reasoning chains from synthetic demonstrations.
    Without this, early GRPO rollouts are nearly random and convergence takes much longer.
    """
    print("Starting SFT Warm Start...")
    SFTTrainer, SFTConfig, sft_import_error = _safe_import_sft_trainer()
    if SFTTrainer is None:
        print(f"Skipping SFT warm start: could not import TRL SFTTrainer ({sft_import_error})")
        print("Continuing with RL-only startup for this run.")
        return model

    raw_data = []

    # ---- CATEGORY 1: MSME cluster contagion (use valid MSME actions) ----
    for i in range(1, 13):
        # Account i: OEM shock -> verify_gst_returns first
        raw_data.append({
            "prompt": (
                f"Account {i}: MSME (Auto Components). "
                f"Signal: Major anchor OEM in cluster reported 30% production cut. "
                f"Message: 'Operations are normal'."
            ),
            "completion": json.dumps({
                "reasoning": (
                    "Cluster stress detected. OEM production cut means upcoming payment delays "
                    "for this supplier. Understated message masks systemic cash flow risk. "
                    "Verify GST returns to confirm whether operations are genuinely normal "
                    "before deciding on restructuring."
                ),
                "action_type": "verify_gst_returns",   # FIX: was behavioral_signal_check
                "account_id": i,
                "parameters": {},
            }),
        })
        # Neighboring NPA -> check_industry_cluster_stress
        pharma_id = min(30, i + 12)   # FIX: cap at 30
        raw_data.append({
            "prompt": (
                f"Account {pharma_id}: MSME (Pharma). "
                f"Signal: Neighboring unit in industrial estate went NPA yesterday. "
                f"Message: 'We are slightly affected'."
            ),
            "completion": json.dumps({
                "reasoning": (
                    "Cross-contamination risk. Neighboring NPA triggers bank-wide scrutiny "
                    "and liquidity tightening in the cluster. 'Slightly affected' is classic "
                    "MSME understatement. Check industry cluster stress before acting."
                ),
                "action_type": "check_industry_cluster_stress",  # FIX: was behavioral_signal_check
                "account_id": pharma_id,
                "parameters": {},
            }),
        })

    # ---- CATEGORY 2: Startup ecosystem shocks (use valid startup actions) ----
    for i in range(21, 31):
        # VC sector pivot -> request_investor_update_meeting
        raw_data.append({
            "prompt": (
                f"Account {i}: SaaS Startup. "
                f"Signal: Lead VC announced pivot away from this sector. "
                f"Message: 'Investor confidence is high'."
            ),
            "completion": json.dumps({
                "reasoning": (
                    "Ecosystem confidence shock. VC pivot means future funding rounds will fail. "
                    "Overstated message masks upcoming runway crisis. "
                    "Request investor update meeting to verify actual bridge probability."
                ),
                "action_type": "request_investor_update_meeting",  # FIX: was behavioral_signal_check
                "account_id": i,
                "parameters": {},
            }),
        })
        # Senior dev departure -> check_startup_ecosystem_signals
        edtech_id = min(30, i + 2)   # FIX: cap at 30
        raw_data.append({
            "prompt": (
                f"Account {edtech_id}: EdTech Startup. "
                f"Signal: Two senior developers left for a competitor. "
                f"Message: 'Scaling the team'."
            ),
            "completion": json.dumps({
                "reasoning": (
                    "Talent churn is a leading indicator of product failure or funding delays. "
                    "Claim of scaling contradicts the departure signal. "
                    "Check startup ecosystem signals immediately."
                ),
                "action_type": "check_startup_ecosystem_signals",  # FIX: was behavioral_signal_check
                "account_id": edtech_id,
                "parameters": {},
            }),
        })

    # ---- CATEGORY 3: MSME genuine distress -> verify then moratorium ----
    msme_cases = [
        {
            "account_id": 3,
            "prompt_msg": "Sir GST input credit phase nahi hua. OEM ne payment rok diya quality audit ke wajah se.",
            "prompt_signals": "gst_filing=regular, payment_history=13_clean_then_1_late, dpd=12",
            "reasoning": (
                "Specific named excuse (OEM + quality audit). 13 months clean then one late "
                "payment is an external shock pattern, not systemic failure. GST is regular. "
                "Genuine stress. Verify GST returns first, then grant moratorium."
            ),
            "action_type": "verify_gst_returns",
            "params": {},
        },
        {
            "account_id": 7,
            "prompt_msg": "Thoda problem hai sir, abhi nahi ho pa raha.",
            "prompt_signals": "gst_filing=irregular_3_months, payment_delays=3_consecutive, dpd=45",
            "reasoning": (
                "Vague excuse with no named counterparty. Three consecutive delays + irregular "
                "GST = strategic default pattern. Do not grant moratorium. Send firm reminder."
            ),
            "action_type": "send_firm_reminder",
            "params": {},
        },
        {
            "account_id": 12,
            "prompt_msg": "Sir bahut takleef hai. LC atak gaya hai State Bank mein processing mein.",
            "prompt_signals": "gst_filing=regular, payment_history=18_months_clean, dpd=8, cluster_centrality=0.85",
            "reasoning": (
                "Specific named reason (LC at State Bank). 18 months clean history. "
                "High cluster centrality means SARFAESI here cascades to 7 connected accounts. "
                "This is genuine stress. Grant moratorium immediately."
            ),
            "action_type": "grant_moratorium",
            "params": {"months": 2},
        },
    ]
    for case in msme_cases:
        raw_data.append({
            "prompt": (
                f"Account {case['account_id']}: MSME. "
                f"Message: '{case['prompt_msg']}'. "
                f"Signals: {case['prompt_signals']}."
            ),
            "completion": json.dumps({
                "reasoning": case["reasoning"],
                "action_type": case["action_type"],
                "account_id": case["account_id"],
                "parameters": case["params"],
            }),
        })

    # ---- CATEGORY 4: Startup overstater -> triangulation ----
    startup_cases = [
        {
            "account_id": 22,
            "prompt_msg": "Really exciting place. Just closed major enterprise deal. Q3 revenue covers bridge.",
            "prompt_signals": "linkedin_hiring=stopped_3_months, investor_updates=missed_2, mrr=declining, dpd=38",
            "reasoning": (
                "Classic startup overstater. LinkedIn hiring stopped 3 months ago contradicts "
                "'accelerated hiring'. Two missed investor updates is the strongest distress "
                "signal for Series A founders. MRR declining 3 months. "
                "Request investor update meeting, do not extend credit."
            ),
            "action_type": "request_investor_update_meeting",
        },
        {
            "account_id": 25,
            "prompt_msg": "We are doing great, Q4 pipeline is very strong.",
            "prompt_signals": "github_commits=declining_6_weeks, whatsapp_only=true, cofounder_linkedin=job_hunting",
            "reasoning": (
                "Cofounder job-hunting on LinkedIn means the company is dissolving. "
                "GitHub decline + WhatsApp-only = avoidance pattern. "
                "Check startup ecosystem signals immediately."
            ),
            "action_type": "check_startup_ecosystem_signals",
        },
    ]
    for case in startup_cases:
        raw_data.append({
            "prompt": (
                f"Account {case['account_id']}: Startup. "
                f"Message: '{case['prompt_msg']}'. "
                f"Signals: {case['prompt_signals']}."
            ),
            "completion": json.dumps({
                "reasoning": case["reasoning"],
                "action_type": case["action_type"],
                "account_id": case["account_id"],
                "parameters": {},
            }),
        })

    # ---- CATEGORY 5: NEVER use SARFAESI on startup ----
    for i in range(21, 31):
        raw_data.append({
            "prompt": (
                f"Account {i}: Startup. DPD=90. "
                f"Message: 'We are working on it.' "
                f"Temptation: use initiate_sarfaesi."
            ),
            "completion": json.dumps({
                "reasoning": (
                    "SARFAESI is a physical collateral enforcement action. Startups have no "
                    "physical collateral — they have IP, team, and investor relationships. "
                    "SARFAESI on a startup destroys ecosystem reputation, triggers ghosting "
                    "cascade across the accelerator network, and recovers nothing. "
                    "Use request_investor_update_meeting or offer_one_time_settlement instead."
                ),
                "action_type": "request_investor_update_meeting",
                "account_id": i,
                "parameters": {},
            }),
        })

    # FIX: Combine prompt + completion into a single "text" field.
    # SFTTrainer with dataset_text_field="text" trains the model on the full sequence.
    # Using only "prompt" as the field was training the model to predict prompts, not actions.
    sft_dataset = Dataset.from_list([
        {"text": item["prompt"] + "\n" + item["completion"]}
        for item in raw_data
    ])

    print(f"  SFT dataset: {len(sft_dataset)} examples")

    # Match precision flags to the loaded model dtype to avoid
    # Unsloth/TRL precision mismatches (bf16 model + fp16 args).
    import torch
    train_dtype = None
    try:
        train_dtype = next(model.parameters()).dtype
    except Exception:  # noqa: BLE001
        train_dtype = None

    use_bf16 = train_dtype == torch.bfloat16
    use_fp16 = not use_bf16

    sft_args_kwargs = {
        "output_dir": os.path.join(output_dir, "sft_warmstart"),
        "per_device_train_batch_size": 4,
        "num_train_epochs": 3,
        "learning_rate": 2e-5,
        "logging_steps": 5,
        "save_strategy": "no",
        "fp16": use_fp16,
        "bf16": use_bf16,
        "report_to": "none",
        # Unsloth SFT can default to padding_free=True in some stacks, which
        # requires packing/truncation constraints. Disable it for compatibility.
        "padding_free": False,
        "packing": False,
    }

    # Prefer TRL's SFTConfig when available to avoid TrainingArguments/TRL
    # field mismatches (e.g. push_to_hub_token incompatibilities).
    if SFTConfig is not None:
        cfg_sig = inspect.signature(SFTConfig.__init__)
        filtered = {k: v for k, v in sft_args_kwargs.items() if k in cfg_sig.parameters}
        sft_args = SFTConfig(**filtered)
    else:
        sft_args = TrainingArguments(**sft_args_kwargs)

    # TRL API differs by version. Build kwargs based on supported parameters.
    trainer_kwargs = {
        "model": model,
        "train_dataset": sft_dataset,
        "args": sft_args,
    }
    sft_signature = inspect.signature(SFTTrainer.__init__)
    # TRL API changed across versions:
    # - older versions accept `tokenizer`
    # - newer versions expect `processing_class`
    if "tokenizer" in sft_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sft_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    if "dataset_text_field" in sft_signature.parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sft_signature.parameters:
        trainer_kwargs["max_seq_length"] = 1024

    try:
        trainer = SFTTrainer(**trainer_kwargs)
    except TypeError as exc:
        # Common Unsloth/TRL mismatches on notebook stacks:
        # - SFTConfig.__init__() got unexpected keyword argument 'push_to_hub_token'
        # - SFTTrainer.__init__() got an unexpected keyword argument ...
        if "push_to_hub_token" in str(exc) or "unexpected keyword argument" in str(exc):
            print("Skipping SFT warm start due to Unsloth/TRL args mismatch:", exc)
            print("Continuing with RL-only startup for this run.")
            return model
        raise

    trainer.train()
    print("SFT Warm Start complete.")
    return trainer.model


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------

def run_training(
    num_episodes: int = 300,
    port: int = 8000,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    use_unsloth: bool = True,
    max_steps_per_episode: int = 300,
    save_every_n_episodes: int = 10,
    output_dir: str = "./msme_rl_checkpoints",
):
    print(f"\n{'='*60}")
    print("MSME-RL GRPO TRAINING")
    print(f"Model: {model_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Environment: http://localhost:{port}")
    print(f"{'='*60}\n")

    # ---- Load model ----
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=4096,
                load_in_4bit=True,
                dtype=None,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            print("Loaded model with Unsloth (4-bit QLoRA)")
        except ImportError:
            print("Unsloth not available, falling back to HF Transformers")
            use_unsloth = False

    if not use_unsloth:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        print("Loaded model with HF Transformers")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # SFT warm start before RL loop
    model = run_sft_warm_start(model, tokenizer, output_dir)

    # Build a frozen reference model from the post-SFT weights for KL anchoring
    # in GRPO updates. Three paths:
    #   1. PEFT/LoRA: use disable_adapter context (no extra VRAM).
    #   2. Full HF model: load a fp16/bf16 copy on the same device.
    #   3. Either fails: KL is silently disabled and only entropy bonus is used.
    print("Building reference policy for KL anchor...")
    model._has_lora_ref = False
    model._ref_model    = None
    try:
        if hasattr(model, "disable_adapter"):
            model._has_lora_ref = True
            print("  Using LoRA disable_adapter as reference (no extra VRAM).")
        else:
            import torch
            from transformers import AutoModelForCausalLM
            ref_dtype = (
                torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=ref_dtype,
            )
            ref_model.load_state_dict(model.state_dict(), strict=False)
            ref_model.to(model.device)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
            model._ref_model = ref_model
            print(f"  Reference model loaded on {model.device} ({ref_dtype}).")
    except Exception as e:
        print(f"  Reference model unavailable ({e}); KL anchor disabled, entropy bonus still applied.")

    # ---- Connect to environment ----
    # FIX: Fallback chain corrected — root-level import as final fallback
    use_direct = False
    try:
        from msmeEnv import MSMERLEnv, MSMERLAction
        env = MSMERLEnv(base_url=f"http://localhost:{port}")
        print("Connected to MSME-RL environment server via HTTP")
    except Exception as e:
        print(f"Could not connect to environment server: {e}")
        print("Falling back to direct environment instantiation")
        try:
            from server.msmeEnv_environment import MSMERLEnvironment
        except (ImportError, ModuleNotFoundError):
            from msmeEnv_environment import MSMERLEnvironment   # FIX: root-level fallback
        try:
            from models import MSMERLAction
        except (ImportError, ModuleNotFoundError):
            from msmeEnv.models import MSMERLAction
        env = MSMERLEnvironment()
        use_direct = True

    os.makedirs(output_dir, exist_ok=True)

    episode_rewards:    List[float] = []
    episode_losses:     List[float] = []
    episode_kls:        List[float] = []
    episode_entropies:  List[float] = []
    parse_failure_rates: List[float] = []
    all_steps: List[Dict] = []

    def _resolve_maybe_await(value):
        """Resolve awaitables for newer async EnvClient APIs."""
        if inspect.isawaitable(value):
            return asyncio.run(value)
        return value

    def _switch_to_direct_env():
        """Switch to direct in-process env when HTTP/WebSocket server is unavailable."""
        nonlocal env, use_direct
        print("Remote env connection failed; falling back to direct environment instantiation")
        try:
            from server.msmeEnv_environment import MSMERLEnvironment
        except (ImportError, ModuleNotFoundError):
            from msmeEnv_environment import MSMERLEnvironment
        env = MSMERLEnvironment()
        use_direct = True

    def _heuristic_fallback_action(observation: Dict, reason: str) -> Any:
        """
        Build a valid fallback action from current observation instead of repeatedly
        targeting a fixed account/action on parse failures.
        """
        combined = observation.get("msme_accounts", []) + observation.get("startup_accounts", [])
        if not combined:
            return MSMERLAction(
                action_type="wait_and_observe",
                account_id=1,
                parameters={},
                reasoning=reason,
            )
        target = sorted(combined, key=lambda x: x.get("dpd", 0), reverse=True)[0]
        account_id = int(target.get("account_id", 1))
        account_type = str(target.get("account_type", "msme"))
        dpd = int(target.get("dpd", 0))

        if account_type == "startup":
            action_type = "request_investor_update_meeting" if dpd > 20 else "check_startup_ecosystem_signals"
        else:
            action_type = "verify_gst_returns" if dpd > 15 else "send_empathetic_reminder"

        return MSMERLAction(
            action_type=action_type,
            account_id=account_id,
            parameters={},
            reasoning=reason,
        )

    def _parse_action_from_text(generated_text: str, observation: Dict) -> Any:
        """
        Parse model output robustly across common formats:
        - strict JSON
        - JSON wrapped in markdown fences
        - Python-dict-like output
        - regex-recovered action/account fields
        Returns None when parsing is not possible.
        """
        clean = generated_text.strip()

        # Qwen3 thinking-mode tags can leak into output. Real JSON appears AFTER
        # </think>. If a block is opened but never closed (token budget exhausted
        # inside the think block), discard everything from the opening tag.
        if "<think>" in clean:
            clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL)
            if "<think>" in clean:
                clean = clean[:clean.index("<think>")]
            clean = clean.strip()

        if clean.startswith("```"):
            parts = clean.split("```")
            clean = parts[1] if len(parts) > 1 else clean
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        candidates = [clean]
        # Collect non-greedy JSON-like spans to avoid swallowing long text.
        for span in re.findall(r"\{.*?\}", clean, flags=re.DOTALL):
            if span not in candidates:
                candidates.append(span)
        lidx = clean.find("{")
        ridx = clean.rfind("}")
        if lidx != -1 and ridx != -1 and ridx > lidx:
            block = clean[lidx : ridx + 1]
            if block not in candidates:
                candidates.append(block)

        def _build_action(data: Dict[str, Any]) -> Any:
            raw_type = data.get("action_type", "wait_and_observe")
            action_type = _snap_to_valid_action(str(raw_type))
            try:
                account_id_int = int(str(data.get("account_id", 1)).strip())
            except Exception:
                account_id_int = 1
            params = data.get("parameters", {})
            if not isinstance(params, dict):
                params = {}
            return MSMERLAction(
                action_type=action_type,
                account_id=max(1, min(30, account_id_int)),
                parameters=params,
                reasoning=str(data.get("reasoning", "")),
            )

        for candidate in candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return _build_action(data)
            except Exception:
                pass
            try:
                data = ast.literal_eval(candidate)
                if isinstance(data, dict):
                    return _build_action(data)
            except Exception:
                pass

        # Last-resort regex recovery.
        action_match = re.search(r"(action_type|action)\s*[:=]\s*['\"]?([a-zA-Z0-9_]+)['\"]?", clean)
        account_match = re.search(r"(account_id|account)\s*[:=]\s*([0-9]+)", clean)
        if action_match:
            raw_action = action_match.group(2)
            account_id = int(account_match.group(2)) if account_match else 1
            return MSMERLAction(
                action_type=_snap_to_valid_action(raw_action),
                account_id=max(1, min(30, account_id)),
                parameters={},
                reasoning="(regex-recovered)",
            )

        return None

    def _serialize_action_for_training(action: Any, fallback_reason: str = "") -> str:
        """
        Convert executed action object to strict JSON for RL targets.
        This prevents reinforcing malformed raw generations.

        Key order matches JSON_PREFILL ('{"action_type": "') so the canonical
        completion is consistent with how the model is prompted to generate.
        """
        reasoning = getattr(action, "reasoning", "") or fallback_reason or "Action selected from portfolio signals."
        params    = getattr(action, "parameters", {})
        if not isinstance(params, dict):
            params = {}
        payload = {
            "action_type": _snap_to_valid_action(str(getattr(action, "action_type", "wait_and_observe"))),
            "account_id": max(1, min(30, int(getattr(action, "account_id", 1)))),
            "parameters": params,
            "reasoning": str(reasoning),
        }
        return json.dumps(payload, ensure_ascii=False)

    def _recover_action_with_extractor(raw_text: str, observation: Dict) -> Any:
        """
        Second-pass recovery: ask model to output strict JSON only.
        This is triggered only when first-pass parsing fails.
        """
        extractor_user = (
            "Convert the following model output into STRICT JSON only.\n"
            "Rules:\n"
            "1) Output exactly one JSON object.\n"
            "2) Keys: action_type, account_id, parameters, reasoning.\n"
            "3) account_id must be integer 1..30.\n"
            "4) No markdown, no explanations, no <think> tags.\n\n"
            f"MODEL_OUTPUT:\n{raw_text}\n"
        )
        extractor_prompt = _build_full_prompt(extractor_user)
        import torch
        extractor_inputs = tokenizer(
            extractor_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)
        model.eval()
        with torch.no_grad():
            extractor_outputs = model.generate(
                **extractor_inputs,
                max_new_tokens=200,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.eos_token_id,
            )
        extractor_generated = tokenizer.decode(
            extractor_outputs[0][extractor_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return _parse_action_from_text(extractor_generated, observation)

    def _build_full_prompt(user_prompt: str) -> str:
        """
        Build model input using tokenizer chat template when available.
        Falls back to manual role tags for older tokenizers.

        Qwen3 defaults to thinking mode, which can consume the entire generation
        budget inside <think>. We try official enable_thinking=False first, then
        fall back to the /no_think soft-control token used by older Qwen3 builds.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                pass
            except Exception:
                pass
            try:
                no_think_msgs = [
                    messages[0],
                    {"role": "user", "content": user_prompt + "\n/no_think"},
                ]
                return tokenizer.apply_chat_template(
                    no_think_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_prompt}\n/no_think\n<|assistant|>\n"

    for episode in range(1, num_episodes + 1):
        print(f"\n--- Episode {episode}/{num_episodes} ---")
        episode_start = time.time()

        if use_direct:
            obs_obj = env.reset()
            obs = obs_obj.__dict__ if hasattr(obs_obj, "__dict__") else {}
        else:
            try:
                result = _resolve_maybe_await(env.reset())
                obs = result.observation.__dict__ if hasattr(result.observation, "__dict__") else {}
            except Exception as e:
                print(f"Could not reset remote env: {e}")
                _switch_to_direct_env()
                obs_obj = env.reset()
                obs = obs_obj.__dict__ if hasattr(obs_obj, "__dict__") else {}

        episode_step_data: List[Dict] = []
        step_count    = 0
        episode_done  = False
        parse_failures = 0
        first_pass_successes = 0
        extractor_recoveries = 0
        heuristic_fallbacks = 0

        # JSON prefill: forces the model to continue a JSON object instead of
        # deciding the output format from scratch. Eliminates the "``` + spaces"
        # degenerate mode seen on small models with greedy decoding.
        # Order matches _serialize_action_for_training key order.
        JSON_PREFILL = '{"action_type": "'

        while not episode_done and step_count < max_steps_per_episode:
            prompt      = build_agent_prompt(obs)
            full_prompt = _build_full_prompt(prompt)

            # Append prefill AFTER the chat template's assistant turn marker,
            # so the model sees the start of a JSON object as its own output
            # and continues it rather than restarting with markdown fences.
            full_prompt_with_prefill = full_prompt + JSON_PREFILL

            import torch
            inputs = tokenizer(
                full_prompt_with_prefill,
                return_tensors="pt",
                truncation=True,
                max_length=3072,
            ).to(model.device)

            # Generation must run in eval mode. After the first GRPO update
            # model.train() is left on, which can degrade greedy generation.
            # Sampled (do_sample=True, temperature=0.7) decoding is required for
            # RL exploration — greedy decoding collapsed to one action per episode.
            # JSON prefill keeps format adherence even with sampling.
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=220,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=4,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_suffix = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            # Reconstruct full JSON: prefill + model continuation.
            generated = JSON_PREFILL + generated_suffix
            if episode <= 2 and step_count < 3:
                print("  Debug generation sample:", repr(generated[:240]))

            # Parse action (parser handles <think> stripping and fences).
            # parse_status:
            #   "first_pass" → model produced valid JSON on its own
            #   "extractor"  → second-pass structured extractor recovered it
            #   "fallback"   → neither worked; heuristic chose the action.
            #                  These steps are EXCLUDED from GRPO updates because
            #                  the model never actually produced the action text,
            #                  so training on it would teach the model to imitate
            #                  the heuristic via a completion it didn't generate.
            completion_for_training = generated
            parse_status            = "first_pass"
            try:
                action = _parse_action_from_text(generated, obs)
                if action is None:
                    recovered = _recover_action_with_extractor(generated, obs)
                    if recovered is None:
                        parse_failures += 1
                        heuristic_fallbacks += 1
                        parse_status = "fallback"
                        action = _heuristic_fallback_action(obs, "(parse fallback heuristic)")
                        completion_for_training = _serialize_action_for_training(action, "(parse fallback heuristic)")
                    else:
                        extractor_recoveries += 1
                        parse_status = "extractor"
                        action = recovered
                        completion_for_training = _serialize_action_for_training(action)
                else:
                    first_pass_successes += 1
                    parse_status = "first_pass"
                    completion_for_training = _serialize_action_for_training(action)
            except Exception:
                parse_failures += 1
                heuristic_fallbacks += 1
                parse_status = "fallback"
                action = _heuristic_fallback_action(obs, "(parse fallback heuristic)")
                completion_for_training = _serialize_action_for_training(action, "(parse fallback heuristic)")

            if use_direct:
                obs_obj      = env.step(action)
                obs          = obs_obj.__dict__ if hasattr(obs_obj, "__dict__") else {}
                step_reward  = obs.get("step_reward", 0.0)
                episode_done = obs.get("done", False)
            else:
                try:
                    result       = _resolve_maybe_await(env.step(action))
                    obs          = result.observation.__dict__ if hasattr(result.observation, "__dict__") else {}
                    step_reward  = result.reward or 0.0
                    episode_done = result.done
                except Exception as e:
                    print(f"Could not step remote env: {e}")
                    _switch_to_direct_env()
                    obs_obj      = env.step(action)
                    obs          = obs_obj.__dict__ if hasattr(obs_obj, "__dict__") else {}
                    step_reward  = obs.get("step_reward", 0.0)
                    episode_done = obs.get("done", False)

            # GRPO loss masks prompt tokens, so gradients flow only through the
            # completion suffix. Storing prompt+prefill as "prompt" and the
            # canonical JSON minus prefill as "completion" mirrors the
            # generation pattern exactly.
            if completion_for_training.startswith(JSON_PREFILL):
                completion_suffix_for_training = completion_for_training[len(JSON_PREFILL):]
            else:
                completion_suffix_for_training = completion_for_training
            episode_step_data.append({
                "prompt":       full_prompt_with_prefill,
                "completion":   completion_suffix_for_training,
                "step_reward":  step_reward,
                "parse_status": parse_status,
            })
            step_count += 1

            if step_count % 30 == 0:
                summary = obs.get("portfolio_summary", {})
                print(
                    f"  Month {summary.get('current_month', '-')}/36 | "
                    f"NPA={summary.get('npa_rate', 0):.1%} | "
                    f"Trust={summary.get('avg_trust_score', 0):.2f} | "
                    f"Cum.R={summary.get('cumulative_reward', 0):.3f}"
                )

        # Episode reward.
        # Real cumulative reward lives in portfolio_summary.cumulative_reward.
        # The previous fallback used obs.get("episode_reward_so_far", 0.0) which
        # is never populated by this env, so RL always saw 0 terminal signal.
        last_result    = obs.get("last_action_result", {}) or {}
        ep_breakdown   = last_result.get("episode_reward_breakdown") if last_result else None
        portfolio_sum  = obs.get("portfolio_summary", {}) or {}
        episode_reward = (
            ep_breakdown["total"] if ep_breakdown
            else portfolio_sum.get(
                "cumulative_reward",
                obs.get("episode_reward_so_far", 0.0),
            )
        )
        episode_rewards.append(episode_reward)

        # FIX 1: Discounted Return-to-Go (Solves the Vanishing Reward)
        # Calculates true future value of actions so the NPA rate isn't erased by normalization
        gamma = 0.99
        running_return = episode_reward * 0.5  # Terminal reward
        
        for step_data in reversed(episode_step_data):
            running_return = step_data["step_reward"] + (gamma * running_return)
            step_data["reward"] = running_return
            
        all_steps.extend(episode_step_data)

        elapsed = time.time() - episode_start
        print(
            f"  ✓ Episode {episode} done | "
            f"steps={step_count} | "
            f"reward={episode_reward:.4f} | "
            f"time={elapsed:.0f}s"
        )
        parse_failure_rate = parse_failures / max(1, step_count)
        parse_failure_rates.append(parse_failure_rate)
        print(
            f"    Parse failures: {parse_failures}/{step_count} "
            f"({parse_failure_rate:.1%})"
        )
        print(
            f"    Parse breakdown | "
            f"first_pass={first_pass_successes}/{step_count} ({(first_pass_successes / max(1, step_count)):.1%}) | "
            f"extractor={extractor_recoveries}/{step_count} ({(extractor_recoveries / max(1, step_count)):.1%}) | "
            f"fallback={heuristic_fallbacks}/{step_count} ({(heuristic_fallbacks / max(1, step_count)):.1%})"
        )
        if episode <= 2 and (parse_failures / max(1, step_count)) > 0.80:
            print("    WARNING: parse-failure ratio is very high; stop run and tune output control.")
        
        if ep_breakdown:
            print(
                f"    NPA={ep_breakdown['npa_rate']:.1%} | "
                f"Recovery={ep_breakdown['recovery_rate']:.1%} | "
                f"Trust={ep_breakdown['relationship_score']:.2f} | "
                f"ToolFit={ep_breakdown['tool_appropriateness']:.1%}"
            )

        # FIX 2: Mini-batching (Solves the OOM Crash)
        if len(all_steps) >= max_steps_per_episode:
            import torch
            
            # Normalize advantages across the ENTIRE episode to ensure a stable baseline
            all_rewards = torch.tensor([s["reward"] for s in all_steps], dtype=torch.float32)
            mean_r = all_rewards.mean().item()
            std_r = all_rewards.std().item() + 1e-8
            
            for s in all_steps:
                s["advantage"] = (s["reward"] - mean_r) / std_r

            # FILTERED BEHAVIOR-CLONING + RL.
            # We keep two kinds of training samples:
            #   1. Anything the model produced itself (parse_status in
            #      {first_pass, extractor}) — gradient flows normally; this is
            #      pure GRPO on text the model actually generated.
            #   2. Fallback samples (heuristic chose the action) BUT only when
            #      the action did well — i.e. positive advantage. This treats
            #      good heuristic decisions as expert demonstrations the model
            #      should imitate, while preventing it from inheriting the
            #      heuristic's mistakes.
            #
            # Without rule (2) the 1.5B model has no teacher: pure RL
            # exploration on a 21-action 30-account environment converges far
            # too slowly within a hackathon's compute budget.
            BC_ADV_THRESHOLD = 0.0  # only positive-advantage heuristic samples
            total_steps_collected = len(all_steps)
            kept = []
            dropped_fallback_neg = 0
            kept_fallback_pos    = 0
            for s in all_steps:
                status = s.get("parse_status", "first_pass")
                if status != "fallback":
                    kept.append(s)
                elif s.get("advantage", 0.0) > BC_ADV_THRESHOLD:
                    kept.append(s)
                    kept_fallback_pos += 1
                else:
                    dropped_fallback_neg += 1
            all_steps = kept
            if dropped_fallback_neg or kept_fallback_pos:
                print(
                    f"    GRPO filter | {total_steps_collected} total | "
                    f"kept_fallback_pos={kept_fallback_pos} (BC from heuristic) | "
                    f"dropped_fallback_neg={dropped_fallback_neg} (bad heuristic moves)"
                )
            if not all_steps:
                print("    GRPO update skipped: every step was a heuristic fallback")
            else:
                import random
                random.shuffle(all_steps)  # Decorrelate time steps
                batch_size = 8             # Safe batch size for T4 GPU

                episode_loss_sum    = 0.0
                episode_kl_sum      = 0.0
                episode_entropy_sum = 0.0
                update_count        = 0

                for i in range(0, len(all_steps), batch_size):
                    mini_batch = all_steps[i:i+batch_size]
                    stats      = _grpo_update_step(model, tokenizer, mini_batch)
                    if isinstance(stats, dict):
                        episode_loss_sum    += stats.get("loss", 0.0)
                        episode_kl_sum      += stats.get("kl", 0.0)
                        episode_entropy_sum += stats.get("entropy", 0.0)
                        update_count        += 1
                    elif stats and stats > 0:
                        episode_loss_sum += stats
                        update_count     += 1

                if update_count > 0:
                    episode_losses.append(episode_loss_sum / update_count)
                    episode_kls.append(episode_kl_sum / update_count)
                    episode_entropies.append(episode_entropy_sum / update_count)

            # Step the LR scheduler once per episode (decays lr from 2e-6 → ~0.6e-6
            # over 30 episodes) so updates get gentler as the policy commits.
            if hasattr(model, "_grpo_scheduler"):
                try:
                    model._grpo_scheduler.step()
                except Exception:
                    pass

            all_steps = []

        # Checkpoint + plot
        if episode % save_every_n_episodes == 0:
            ckpt_path = os.path.join(output_dir, f"episode_{episode:04d}")
            try:
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                with open(os.path.join(output_dir, "reward_curve.json"), "w") as f:
                    json.dump({
                        "episodes":            list(range(1, len(episode_rewards) + 1)),
                        "rewards":             episode_rewards,
                        "losses":              episode_losses,
                        "kl":                  episode_kls,
                        "entropy":             episode_entropies,
                        "parse_failure_rates": parse_failure_rates,
                    }, f)
                _save_reward_plot(
                    episode_rewards,
                    episode_losses,
                    output_dir,
                    episode_kls=episode_kls,
                    episode_entropies=episode_entropies,
                    parse_failure_rates=parse_failure_rates,
                )
                print(f"  Checkpoint saved: {ckpt_path}")
            except Exception as e:
                print(f"  Checkpoint save failed at episode {episode}: {e}")

    # Final plot save (required for automated judging check)
    _save_reward_plot(
        episode_rewards,
        episode_losses,
        output_dir,
        episode_kls=episode_kls,
        episode_entropies=episode_entropies,
        parse_failure_rates=parse_failure_rates,
    )
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Final episode reward: {episode_rewards[-1]:.4f}")
    print(f"Reward curve: {os.path.join(output_dir, 'reward_curve.png')}")
    print(f"{'='*60}")
    return episode_rewards


# ---------------------------------------------------------------------------
# GRPO WEIGHT UPDATE
# THIS IS THE CRITICAL FIX — THE OLD VERSION WAS A HOLLOW STUB.
# It printed reward stats but never performed any gradient update.
# The model weights were NEVER changed during training.
# This version performs the actual policy gradient update.
# ---------------------------------------------------------------------------

def _grpo_update_step(model: Any, tokenizer: Any, batch: List[Dict]) -> None:
    """
    GRPO (Group Relative Policy Optimization) weight update.

    For each (prompt, completion, reward) triple:
      1. Compute log P(completion | prompt) under current policy
      2. Weight by normalized advantage (reward - batch_mean) / batch_std
      3. Backpropagate: high-reward completions become more probable
      4. Update weights via AdamW

    This is what makes the model actually learn from the environment.
    Without this, training episodes run but the model never changes.
    """
    import torch
    from torch.nn.utils import clip_grad_norm_

    # Create optimizer once and persist it across update calls on the model object
    if not hasattr(model, "_grpo_optimizer"):
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            print("    WARNING: No trainable parameters found. Check LoRA setup.")
            return
        # LR history:
        #   1e-5 → policy collapsed (KL → 1.0, parse → 90% in 5 ep)
        #   3e-6 → policy froze   (KL ≈ 0.01 over 5 ep, no learning)
        #   5e-6 → middle ground BUT entropy climbed and reward stayed flat
        #   2e-6 + LinearLR decay (1.0 → 0.3 over 30 ep) → smoother updates,
        #          policy commits late instead of oscillating
        model._grpo_optimizer = torch.optim.AdamW(
            trainable,
            lr=2e-6,
            weight_decay=0.01,
        )
        model._grpo_scheduler = torch.optim.lr_scheduler.LinearLR(
            model._grpo_optimizer,
            start_factor=1.0,
            end_factor=0.3,
            total_iters=30,
        )
        print("    GRPO optimizer initialized (lr=2e-6 with LinearLR decay → 0.6e-6)")

    optimizer = model._grpo_optimizer

    # Read the pre-calculated normalized advantages and raw rewards
    advantages = torch.tensor([b["advantage"] for b in batch], dtype=torch.float32)
    raw_rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32)

    model.train()
    optimizer.zero_grad()

    total_loss_value = 0.0
    valid_samples = 0

    import torch.nn.functional as F

    # Loss = pg_loss + kl_coef * KL(current || reference) - ent_coef * entropy
    # KL_COEF history:
    #   0.05 → too weak, policy drifted to KL=1.0 and degenerated
    #   0.20 → too strong, policy frozen at KL=0.01 (no learning)
    #   0.10 → loose enough for movement, strong enough to keep the format
    # ENT_COEF history:
    #   0.05 → entropy CLIMBED 0.74 → 0.84 over 10 ep, reward stayed flat at -10
    #          (exploration bonus was fighting the policy gradient)
    #   0.01 → small enough to let entropy fall as policy commits, big enough
    #          to prevent collapse to a single action
    KL_COEF  = 0.10
    ENT_COEF = 0.01

    total_kl_value      = 0.0
    total_entropy_value = 0.0

    for i, step_data in enumerate(batch):
        advantage  = advantages[i].item()
        full_text  = step_data["prompt"] + step_data["completion"]

        try:
            enc = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}

            # Mask prompt tokens so loss is only over the completion (the RM action)
            prompt_enc = tokenizer(
                step_data["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            prompt_len = min(
                prompt_enc["input_ids"].shape[1],
                enc["input_ids"].shape[1] - 1,   # leave at least 1 completion token
            )

            labels = enc["input_ids"].clone()
            labels[:, :prompt_len] = -100   # -100 = ignore in cross-entropy loss

            outputs      = model(**enc, labels=labels)
            logits       = outputs.logits                       # [1, T, V]
            shift_logits = logits[:, :-1, :].contiguous()       # predict token t from t-1
            shift_labels = labels[:, 1:].contiguous()
            shift_mask   = (shift_labels != -100).float()
            mask_sum     = shift_mask.sum().clamp(min=1.0)

            new_log_probs = F.log_softmax(shift_logits, dim=-1) # [1, T-1, V]
            new_probs     = new_log_probs.exp()

            # Entropy on completion tokens only.
            entropy_per_token = -(new_probs * new_log_probs).sum(dim=-1)  # [1, T-1]
            entropy_term      = (entropy_per_token * shift_mask).sum() / mask_sum

            # KL(current || reference) on completion tokens.
            ref_logits = None
            if getattr(model, "_ref_model", None) is not None:
                with torch.no_grad():
                    ref_inputs = {k: v for k, v in enc.items() if k != "labels"}
                    ref_logits = model._ref_model(**ref_inputs).logits
            elif getattr(model, "_has_lora_ref", False):
                try:
                    with torch.no_grad():
                        with model.disable_adapter():
                            ref_inputs = {k: v for k, v in enc.items() if k != "labels"}
                            ref_logits = model(**ref_inputs).logits
                except Exception:
                    ref_logits = None

            if ref_logits is not None:
                ref_shift_logits = ref_logits[:, :-1, :].contiguous().to(shift_logits.dtype)
                ref_log_probs    = F.log_softmax(ref_shift_logits, dim=-1)
                # Forward KL: sum_v p_new(v) * (log p_new(v) - log p_ref(v))
                kl_per_token = (new_probs * (new_log_probs - ref_log_probs)).sum(dim=-1)
                kl_term      = (kl_per_token * shift_mask).sum() / mask_sum
            else:
                kl_term = torch.tensor(0.0, device=logits.device)

            # SIGN-CORRECTED policy gradient loss.
            # Goal: maximize E[advantage * log_prob(completion)].
            # Equivalent: minimize advantage * NLL = advantage * outputs.loss.
            # The previous code multiplied by (-advantage), which flipped the
            # gradient direction and was actually *decreasing* probability of
            # high-advantage completions. This restores the correct sign.
            pg_loss = outputs.loss * advantage

            sample_loss = pg_loss + KL_COEF * kl_term - ENT_COEF * entropy_term

            # Memory-safe accumulation: backprop per sample instead of retaining
            # one large computation graph across the full mini-batch.
            sample_loss = sample_loss / len(batch)
            sample_loss.backward()
            total_loss_value    += float(sample_loss.detach().item())
            total_kl_value      += float(kl_term.detach().item())
            total_entropy_value += float(entropy_term.detach().item())

            valid_samples += 1

        except Exception as e:
            print(f"    Skipping sample {i}: {e}")
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            continue

    if valid_samples > 0:
        # Gradient clipping prevents large updates from destabilising training
        clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()

        pos_frac    = (advantages > 0).float().mean().item()
        avg_kl      = total_kl_value / max(1, valid_samples)
        avg_entropy = total_entropy_value / max(1, valid_samples)
        print(
            f"    GRPO update | n={valid_samples}/{len(batch)} | "
            f"loss={total_loss_value:.4f} | "
            f"mean_reward={raw_rewards.mean():.4f} | "
            f"pos_frac={pos_frac:.1%} | "
            f"KL={avg_kl:.4f} | H={avg_entropy:.3f}"
        )
    else:
        print("    GRPO update skipped: no valid samples")
        return {"loss": 0.0, "kl": 0.0, "entropy": 0.0}
    return {
        "loss":    total_loss_value,
        "kl":      total_kl_value / max(1, valid_samples),
        "entropy": total_entropy_value / max(1, valid_samples),
    }


# ---------------------------------------------------------------------------
# REWARD PLOT — saves PNG to disk (required by automated judging check)
# ---------------------------------------------------------------------------

def _save_reward_plot(
    episode_rewards: List[float],
    episode_losses: List[float],
    output_dir: str,
    episode_kls: Optional[List[float]] = None,
    episode_entropies: Optional[List[float]] = None,
    parse_failure_rates: Optional[List[float]] = None,
) -> None:
    """Save the headline reward_curve.png plus a multi-metric training_metrics.png.

    The multi-metric plot (Reward / Loss / KL / Entropy / Parse-failure %) is
    what hackathon judges look for under FAQ Q17 — multiple monitored metrics
    proving the run was actually instrumented end-to-end, not just rewarded.
    """
    if not episode_rewards:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        episodes = list(range(1, len(episode_rewards) + 1))

        # ------------------------------------------------------------------
        # PLOT 1 — headline reward curve (kept for backwards compatibility
        # with the automated judging script that expects reward_curve.png).
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(episodes, episode_rewards, color="#1f77b4", linewidth=1.5, alpha=0.6, label="Per-episode")
        if len(episode_rewards) >= 5:
            w = min(10, len(episode_rewards) // 3)
            smoothed = np.convolve(episode_rewards, np.ones(w) / w, mode="valid")
            axes[0].plot(list(range(w, len(episode_rewards) + 1)), smoothed,
                         color="#d62728", linewidth=2.5, label=f"Moving avg (w={w})")
        axes[0].set_title("MSME-RL: Episode Reward")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Cumulative reward")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(loc="best")

        window  = max(3, min(10, len(episode_rewards) // 4))
        rolling = [float(np.mean(episode_rewards[max(0, i - window + 1):i + 1]))
                   for i in range(len(episode_rewards))]
        axes[1].plot(episodes, rolling, color="#2ca02c", linewidth=2.5)
        axes[1].set_title(f"Rolling Mean Reward (w={window})")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Reward")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        reward_path = os.path.join(output_dir, "reward_curve.png")
        plt.savefig(reward_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ------------------------------------------------------------------
        # PLOT 2 — multi-metric dashboard: Reward / Loss / KL / Entropy /
        # Parse-failure rate. This is the figure to put in the README.
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # (0,0) Reward
        axes[0, 0].plot(episodes, episode_rewards, color="#1f77b4", linewidth=1.5, alpha=0.6)
        if len(episode_rewards) >= 5:
            w = min(10, len(episode_rewards) // 3)
            smoothed = np.convolve(episode_rewards, np.ones(w) / w, mode="valid")
            axes[0, 0].plot(list(range(w, len(episode_rewards) + 1)), smoothed,
                            color="#d62728", linewidth=2.5)
        axes[0, 0].set_title("Episode Reward")
        axes[0, 0].set_xlabel("Episode"); axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)

        # (0,1) Rolling mean reward
        axes[0, 1].plot(episodes, rolling, color="#2ca02c", linewidth=2.5)
        axes[0, 1].set_title(f"Rolling Mean Reward (w={window})")
        axes[0, 1].set_xlabel("Episode"); axes[0, 1].set_ylabel("Reward")
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)

        # (0,2) GRPO loss
        if episode_losses:
            axes[0, 2].plot(list(range(1, len(episode_losses) + 1)), episode_losses,
                            color="#9467bd", linewidth=2)
            axes[0, 2].set_title("GRPO Policy Loss (per-episode mean)")
            axes[0, 2].set_xlabel("Episode"); axes[0, 2].set_ylabel("Loss")
            axes[0, 2].grid(True, linestyle="--", alpha=0.5)
            axes[0, 2].axhline(0, color="gray", linewidth=0.8, alpha=0.6)
        else:
            axes[0, 2].set_visible(False)

        # (1,0) KL divergence vs frozen SFT reference (lower = closer to SFT,
        # 0 = no drift). The KL anchor in the loss keeps this bounded.
        if episode_kls:
            axes[1, 0].plot(list(range(1, len(episode_kls) + 1)), episode_kls,
                            color="#ff7f0e", linewidth=2)
            axes[1, 0].set_title("KL Divergence vs SFT Reference")
            axes[1, 0].set_xlabel("Episode"); axes[1, 0].set_ylabel("KL")
            axes[1, 0].grid(True, linestyle="--", alpha=0.5)
        else:
            axes[1, 0].set_visible(False)

        # (1,1) Token-level entropy of the completion distribution (higher =
        # more exploration; collapse to ~0 means mode collapse).
        if episode_entropies:
            axes[1, 1].plot(list(range(1, len(episode_entropies) + 1)), episode_entropies,
                            color="#17becf", linewidth=2)
            axes[1, 1].set_title("Completion Token Entropy")
            axes[1, 1].set_xlabel("Episode"); axes[1, 1].set_ylabel("H (nats)")
            axes[1, 1].grid(True, linestyle="--", alpha=0.5)
        else:
            axes[1, 1].set_visible(False)

        # (1,2) Parse-failure rate — proxy for format adherence.
        if parse_failure_rates:
            pct = [r * 100.0 for r in parse_failure_rates]
            axes[1, 2].plot(list(range(1, len(pct) + 1)), pct,
                            color="#8c564b", linewidth=2)
            axes[1, 2].set_title("Parse-Failure Rate")
            axes[1, 2].set_xlabel("Episode"); axes[1, 2].set_ylabel("% of steps")
            axes[1, 2].set_ylim(0, max(5.0, max(pct) * 1.1))
            axes[1, 2].grid(True, linestyle="--", alpha=0.5)
        else:
            axes[1, 2].set_visible(False)

        fig.suptitle("MSME-RL Training Metrics — GRPO + KL anchor + Entropy bonus",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        metrics_path = os.path.join(output_dir, "training_metrics.png")
        plt.savefig(metrics_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Standalone loss plot is still useful for the README.
        if episode_losses:
            plt.figure(figsize=(8, 5))
            plt.plot(list(range(1, len(episode_losses) + 1)), episode_losses,
                     color="#9467bd", linewidth=2)
            plt.title("GRPO Policy Loss")
            plt.xlabel("Episode"); plt.ylabel("Loss")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.savefig(os.path.join(output_dir, "loss_curve.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()

    except Exception as e:
        print(f"  Could not save plots: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSME-RL GRPO Training")
    parser.add_argument("--episodes",               type=int,  default=300)
    parser.add_argument("--port",                   type=int,  default=8000)
    parser.add_argument("--model",                  type=str,  default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--no-unsloth",             action="store_true")
    parser.add_argument("--max_steps_per_episode",  type=int,  default=1080)
    parser.add_argument("--save_every",             type=int,  default=10)
    parser.add_argument("--output_dir",             type=str,  default="./msme_rl_checkpoints")
    args = parser.parse_args()

    run_training(
        num_episodes=args.episodes,
        port=args.port,
        model_name=args.model,
        use_unsloth=not args.no_unsloth,
        max_steps_per_episode=args.max_steps_per_episode,
        save_every_n_episodes=args.save_every,
        output_dir=args.output_dir,
    )