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
import json
import os
import time
import pathlib
import inspect
from typing import Any, Dict, List

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
  "reasoning": "<your chain-of-thought: what signals you observed, what pattern matched, why this action>",
  "action_type": "<one of the 21 action types>",
  "account_id": <integer 1-30>,
  "parameters": {"months": 2}
}"""


def build_agent_prompt(observation: Dict) -> str:
    working_mem      = observation.get("working_memory", "")
    semantic_mem     = observation.get("semantic_memory_context", "")
    episodic_mem     = observation.get("episodic_memory_context", "")
    msme_accounts    = observation.get("msme_accounts", [])
    startup_accounts = observation.get("startup_accounts", [])
    alerts           = (
        observation.get("active_cluster_alerts", [])
        + observation.get("active_ecosystem_alerts", [])
    )

    urgent = sorted(
        msme_accounts + startup_accounts,
        key=lambda x: x.get("dpd", 0),
        reverse=True,
    )[:5]

    urgent_str = json.dumps(urgent, ensure_ascii=False, indent=2)
    summary    = json.dumps(observation.get("portfolio_summary", {}), indent=2)

    return f"""=== PORTFOLIO STATE ===
{summary}

=== WORKING MEMORY ===
{working_mem}

=== SEMANTIC MEMORY (learned patterns) ===
{semantic_mem}

=== EPISODIC MEMORY (similar past cases) ===
{episodic_mem}

=== ACTIVE ALERTS ===
{chr(10).join(alerts) if alerts else "(none)"}

=== TOP 5 ACCOUNTS NEEDING ATTENTION (by DPD) ===
{urgent_str}

Based on all of the above, decide your next action. Think through:
1. Which account is most urgent?
2. What do the behavioral signals (not the message content) tell you?
3. What pattern from semantic memory applies?
4. What is the appropriate tool for this account type?
5. What are the network risks (cluster/ecosystem) of your action?

Respond with JSON only:"""


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
        "tokenizer": tokenizer,
        "args": sft_args,
    }
    sft_signature = inspect.signature(SFTTrainer.__init__)
    if "dataset_text_field" in sft_signature.parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sft_signature.parameters:
        trainer_kwargs["max_seq_length"] = 1024

    try:
        trainer = SFTTrainer(**trainer_kwargs)
    except TypeError as exc:
        # Common Unsloth/TRL mismatch on some Colab stacks:
        # SFTConfig.__init__() got unexpected keyword argument 'push_to_hub_token'
        if "push_to_hub_token" in str(exc):
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
    model_name: str = "Qwen/Qwen3-1.7B",
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

    episode_rewards: List[float] = []
    episode_losses: List[float] = []
    all_steps: List[Dict] = []

    for episode in range(1, num_episodes + 1):
        print(f"\n--- Episode {episode}/{num_episodes} ---")
        episode_start = time.time()

        if use_direct:
            obs_obj = env.reset()
            obs = obs_obj.__dict__ if hasattr(obs_obj, "__dict__") else {}
        else:
            result = env.reset()
            obs = result.observation.__dict__ if hasattr(result.observation, "__dict__") else {}

        episode_step_data: List[Dict] = []
        step_count    = 0
        episode_done  = False

        while not episode_done and step_count < max_steps_per_episode:
            prompt      = build_agent_prompt(obs)
            full_prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"

            import torch
            inputs = tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3072,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Parse action — strip markdown fences if present
            try:
                clean = generated.strip()
                if clean.startswith("```"):
                    parts = clean.split("```")
                    clean = parts[1] if len(parts) > 1 else clean
                    if clean.startswith("json"):
                        clean = clean[4:]
                action_data = json.loads(clean.strip())
                action = MSMERLAction(
                    action_type=action_data.get("action_type", "wait_and_observe"),
                    account_id=int(action_data.get("account_id", 1)),
                    parameters=action_data.get("parameters", {}),
                    reasoning=action_data.get("reasoning", ""),
                )
            except Exception:
                action = MSMERLAction(
                    action_type="format_error", # CHANGED: Explicitly flag as an error
                    account_id=1,
                    parameters={},
                    reasoning="(parse error)",
                )

            if use_direct:
                obs_obj      = env.step(action)
                obs          = obs_obj.__dict__ if hasattr(obs_obj, "__dict__") else {}
                step_reward  = obs.get("step_reward", 0.0)
                episode_done = obs.get("done", False)
            else:
                result       = env.step(action)
                obs          = result.observation.__dict__ if hasattr(result.observation, "__dict__") else {}
                step_reward  = result.reward or 0.0
                episode_done = result.done

            episode_step_data.append({
                "prompt":      full_prompt,
                "completion":  generated,
                "step_reward": step_reward,
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

        # Episode reward
        # Episode reward
        last_result       = obs.get("last_action_result", {})
        ep_breakdown      = last_result.get("episode_reward_breakdown") if last_result else None
        episode_reward    = ep_breakdown["total"] if ep_breakdown else obs.get("episode_reward_so_far", 0.0)
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

            import random
            random.shuffle(all_steps) # Decorrelate time steps
            batch_size = 8 # Safe batch size for T4 GPU
            
            episode_loss_sum = 0.0
            update_count = 0
            
            for i in range(0, len(all_steps), batch_size):
                mini_batch = all_steps[i:i+batch_size]
                avg_loss = _grpo_update_step(model, tokenizer, mini_batch)
                if avg_loss > 0:
                    episode_loss_sum += avg_loss
                    update_count += 1
            
            if update_count > 0:
                episode_losses.append(episode_loss_sum / update_count)
            
            all_steps = []

        # Checkpoint + plot
        if episode % save_every_n_episodes == 0:
            ckpt_path = os.path.join(output_dir, f"episode_{episode:04d}")
            try:
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                with open(os.path.join(output_dir, "reward_curve.json"), "w") as f:
                    json.dump({
                        "episodes": list(range(1, len(episode_rewards) + 1)),
                        "rewards":  episode_rewards,
                        "losses": episode_losses, # Added losses to JSON
                    }, f)
                _save_reward_plot(episode_rewards, episode_losses, output_dir) # Added episode_losses here
                print(f"  Checkpoint saved: {ckpt_path}")
            except Exception as e:
                print(f"  Checkpoint save failed at episode {episode}: {e}")

    # Final plot save (required for automated judging check)
    _save_reward_plot(episode_rewards, episode_losses, output_dir) # Added episode_losses here
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
        model._grpo_optimizer = torch.optim.AdamW(
            trainable,
            lr=1e-5,
            weight_decay=0.01,
        )
        print("    GRPO optimizer initialized")

    optimizer = model._grpo_optimizer

    # Read the pre-calculated normalized advantages and raw rewards
    advantages = torch.tensor([b["advantage"] for b in batch], dtype=torch.float32)
    raw_rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32)

    model.train()
    optimizer.zero_grad()

    total_loss_value = 0.0
    valid_samples = 0

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

            outputs = model(**enc, labels=labels)

            # GRPO objective: maximize E[reward * log_prob(completion)]
            # outputs.loss = mean negative log-likelihood over completion tokens
            # Multiply by -advantage: positive advantage -> minimise loss (increase prob)
            #                         negative advantage -> maximise loss (decrease prob)
            sample_loss = outputs.loss * (-advantage)

            # Memory-safe accumulation: backprop per sample instead of retaining
            # one large computation graph across the full mini-batch.
            sample_loss = sample_loss / len(batch)
            sample_loss.backward()
            total_loss_value += float(sample_loss.detach().item())

            valid_samples += 1

        except Exception as e:
            print(f"    Skipping sample {i}: {e}")
            if "out of memory" in str(e).lower():
                # Try to recover and continue with the remaining samples.
                torch.cuda.empty_cache()
            continue

    if valid_samples > 0:
        # Gradient clipping prevents large updates from destabilising training
        clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()

        pos_frac = (advantages > 0).float().mean().item()
        print(
            f"    GRPO update | n={valid_samples}/{len(batch)} | "
            f"loss={total_loss_value:.4f} | "
            f"mean_reward={raw_rewards.mean():.4f} | "
            f"pos_frac={pos_frac:.1%}"
        )
    else:
        print("    GRPO update skipped: no valid samples")
        return 0.0
    return total_loss_value


# ---------------------------------------------------------------------------
# REWARD PLOT — saves PNG to disk (required by automated judging check)
# ---------------------------------------------------------------------------

def _save_reward_plot(episode_rewards: List[float], episode_losses: List[float], output_dir: str) -> None:
    if not episode_rewards:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        episodes = list(range(1, len(episode_rewards) + 1))

        # --- PLOT 1: REWARD CURVE ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(episodes, episode_rewards, color="#1f77b4", linewidth=1.5, alpha=0.6)
        
        if len(episode_rewards) >= 5:
            w = min(10, len(episode_rewards) // 3)
            smoothed = np.convolve(episode_rewards, np.ones(w) / w, mode="valid")
            axes[0].plot(list(range(w, len(episode_rewards) + 1)), smoothed, color="#d62728", linewidth=2.5)
            
        axes[0].set_title("MSME-RL: Episode Reward")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        window = max(3, min(10, len(episode_rewards) // 4))
        rolling = [float(np.mean(episode_rewards[max(0, i - window + 1):i + 1])) for i in range(len(episode_rewards))]
        axes[1].plot(episodes, rolling, color="#2ca02c", linewidth=2.5)
        axes[1].set_title(f"Rolling Mean Reward (w={window})")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        
        reward_path = os.path.join(output_dir, "reward_curve.png")
        plt.savefig(reward_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- PLOT 2: LOSS CURVE (REQUIRED FOR HACKATHON) ---
        if episode_losses:
            plt.figure(figsize=(8, 5))
            batches = list(range(1, len(episode_losses) + 1))
            plt.plot(batches, episode_losses, color="#9467bd", linewidth=2)
            plt.title("GRPO Policy Loss")
            plt.xlabel("Update Batches")
            plt.ylabel("Loss")
            plt.grid(True, linestyle="--", alpha=0.5)
            loss_path = os.path.join(output_dir, "loss_curve.png")
            plt.savefig(loss_path, dpi=150, bbox_inches="tight")
            plt.close()

    except Exception as e:
        print(f"  Could not save plots: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSME-RL GRPO Training")
    parser.add_argument("--episodes",               type=int,  default=300)
    parser.add_argument("--port",                   type=int,  default=8000)
    parser.add_argument("--model",                  type=str,  default="Qwen/Qwen3-1.7B")
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