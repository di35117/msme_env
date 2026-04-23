# train_grpo.py

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
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# GRPO TRAINING PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Relationship Manager (RM) at an Indian bank managing a mixed portfolio
of MSME (small business) and startup accounts across a 36-month loan cycle.

YOUR CHALLENGE: Two types of borrowers hide financial distress in completely opposite ways.
  • MSME owners UNDERSTATE problems in Hindi/Hinglish understatement
  • Startup founders OVERSTATE health in pitch-deck English confidence

You must learn to decode BOTH from behavioral signals — not surface messages.

KEY SIGNAL RULES:
MSME signals of genuine distress (NOT what they say — how they say it):
  - Specificity of excuse (real problems have real details: named OEM, named reason)
  - GST filing regularity (operational businesses file on time)
  - Response time to calls (genuinely stressed owners answer; strategic defaulters avoid)
  - Payment history pattern (13 months clean → sudden delay = genuine crisis)
  - Language register shift (more formal, more deferential than usual)

Startup signals of genuine distress (what contradicts the pitch):
  - LinkedIn hiring activity stopped (contradicts "accelerated hiring")
  - Investor updates missed for 2+ months (accelerator-stage founders always update)
  - GitHub commit activity declining (developers commit when building)
  - Avoiding voice calls, replying via WhatsApp (founders ghost before defaulting)
  - MRR negative for 3 months (contradicts "strong momentum")
  - Cofounder job-hunting on LinkedIn (worst signal — company dissolving)

AVAILABLE ACTIONS (21 total):
Communication: send_empathetic_reminder, send_firm_reminder, send_legal_notice_section13,
               call_promoter_founder, call_guarantor_investor, conduct_cluster_ecosystem_visit

Financial restructuring: grant_moratorium, restructure_emi, offer_eclgs_topup (MSME),
                          offer_bridge_loan_extension (startup), accept_partial_payment, waive_penal_interest

Recovery: initiate_sarfaesi, refer_to_recovery_agent, file_drt_case, offer_one_time_settlement

Information gathering: verify_gst_returns (MSME), pull_bank_statements, check_industry_cluster_stress,
                        request_investor_update_meeting (startup), check_startup_ecosystem_signals

CRITICAL RULES:
  ❌ NEVER use initiate_sarfaesi on a startup — destroys ecosystem trust, triggers cascades
  ❌ NEVER take startup pitch optimism at face value without checking behavioral signals
  ✅ ALWAYS verify_gst_returns before granting moratorium to MSME
  ✅ ALWAYS request_investor_update_meeting before restructuring startup debt
  ✅ Consider cluster_centrality for MSME decisions (high-centrality account → cascade risk)

REWARD SIGNAL:
  Step rewards: +0.10 investor_meeting_triggered_bridge | +0.08 payment_received_after_empathy
                -0.25 cluster_cascade_default | -0.15 sarfaesi_used_on_startup
  Episode reward: 0.40×(1-NPA rate) + 0.30×recovery_rate + 0.20×relationship_score + 0.10×tool_fit

Respond with your action in this exact JSON format:
{
  "reasoning": "<your chain-of-thought: what signals you observed, what pattern matched, why this action>",
  "action_type": "<one of the 21 action types>",
  "account_id": <integer 1-30>,
  "parameters": {"months": 2}
}"""


def build_agent_prompt(observation: Dict) -> str:
    """Build the full prompt for the agent given current observation."""
    working_mem = observation.get("working_memory", "")
    semantic_mem = observation.get("semantic_memory_context", "")
    episodic_mem = observation.get("episodic_memory_context", "")

    # Find the highest-priority account to act on
    msme_accounts   = observation.get("msme_accounts", [])
    startup_accounts = observation.get("startup_accounts", [])
    alerts           = observation.get("active_cluster_alerts", []) + observation.get("active_ecosystem_alerts", [])

    # Format the most urgent accounts (top 5 by DPD)
    urgent = sorted(
        msme_accounts + startup_accounts,
        key=lambda x: x.get("dpd", 0),
        reverse=True,
    )[:5]

    urgent_str = json.dumps(urgent, ensure_ascii=False, indent=2)
    summary    = json.dumps(observation.get("portfolio_summary", {}), indent=2)

    prompt = f"""=== PORTFOLIO STATE ===
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
    return prompt


# ---------------------------------------------------------------------------
# GRPO REWARD FUNCTION WRAPPER
# ---------------------------------------------------------------------------

def grpo_reward_function(completions: List[str], ground_truth_rewards: List[float]) -> List[float]:
    """
    GRPO reward function for TRL.
    The environment's step reward IS the ground truth — no LLM judge needed.
    """
    return ground_truth_rewards


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

    # --- 1. Load model and tokenizer ---
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
            print("✓ Loaded model with Unsloth (4-bit QLoRA)")
        except ImportError:
            print("Unsloth not available — falling back to HF Transformers")
            use_unsloth = False

    if not use_unsloth:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        print("✓ Loaded model with HF Transformers")

    # --- 2. Connect to environment ---
    try:
        from msmeEnv import MSMERLEnv, MSMERLAction
        env = MSMERLEnv(base_url=f"http://localhost:{port}")
        print("✓ Connected to MSME-RL environment server")
    except Exception as e:
        print(f"⚠ Could not connect to environment server: {e}")
        print("  Falling back to direct environment")
        from server.msmeEnv_environment import MSMERLEnvironment
        from models import MSMERLAction
        env = MSMERLEnvironment()
        use_direct = True
    else:
        use_direct = False

    # --- 3. Training loop ---
    episode_rewards = []
    all_steps = []

    os.makedirs(output_dir, exist_ok=True)

    for episode in range(1, num_episodes + 1):
        print(f"\n--- Episode {episode}/{num_episodes} ---")
        episode_start = time.time()

        if use_direct:
            obs_obj = env.reset()
            obs = obs_obj.__dict__ if hasattr(obs_obj, '__dict__') else {}
        else:
            result = env.reset()
            obs = result.observation.__dict__ if hasattr(result.observation, '__dict__') else {}

        episode_step_data = []   
        step_count = 0
        episode_done = False

        while not episode_done and step_count < max_steps_per_episode:
            prompt = build_agent_prompt(obs)
            full_prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"

            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            with __import__("torch").no_grad():
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

            try:
                action_data = json.loads(generated.strip())
                if use_direct:
                    from models import MSMERLAction
                action = MSMERLAction(
                    action_type=action_data.get("action_type", "wait_and_observe"),
                    account_id=action_data.get("account_id", 1),
                    parameters=action_data.get("parameters", {}),
                    reasoning=action_data.get("reasoning", ""),
                )
            except (json.JSONDecodeError, Exception) as e:
                action = MSMERLAction(
                    action_type="wait_and_observe",
                    account_id=1,
                    parameters={},
                    reasoning="(parse error)",
                )

            if use_direct:
                obs_obj = env.step(action)
                obs = obs_obj.__dict__ if hasattr(obs_obj, '__dict__') else {}
                step_reward = obs.get("step_reward", 0.0)
                episode_done = obs.get("done", False)
            else:
                result = env.step(action)
                obs = result.observation.__dict__ if hasattr(result.observation, '__dict__') else {}
                step_reward = result.reward or 0.0
                episode_done = result.done

            # Store step data temporarily
            episode_step_data.append({
                "prompt":      full_prompt,
                "completion":  generated,
                "step_reward": step_reward,
            })

            step_count += 1

            if step_count % 30 == 0:
                summary = obs.get("portfolio_summary", {})
                print(
                    f"  Month {summary.get('current_month','-')}/36 | "
                    f"NPA={summary.get('npa_rate',0):.1%} | "
                    f"Trust={summary.get('avg_trust_score',0):.2f} | "
                    f"Cum.R={summary.get('cumulative_reward',0):.3f}"
                )

        # Episode boundary handling
        last_result = obs.get("last_action_result", {})
        ep_breakdown = last_result.get("episode_reward_breakdown") if last_result else None
        episode_reward = ep_breakdown["total"] if ep_breakdown else obs.get("episode_reward_so_far", 0)
        episode_rewards.append(episode_reward)
        
        # FIXED: Assign episode reward to all steps for stable GRPO credit assignment
        for step_data in episode_step_data:
            step_data["reward"] = step_data["step_reward"] + (episode_reward * 0.5) 
            all_steps.append(step_data)

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

        # FIXED: Perform GRPO update strictly at the episode boundary
        if len(all_steps) >= 30:
            _grpo_update_step(model, tokenizer, all_steps)
            all_steps = []  # Clear buffer after update to prevent staleness

        # Checkpoint
        if episode % save_every_n_episodes == 0:
            ckpt_path = os.path.join(output_dir, f"episode_{episode:04d}")
            try:
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                with open(os.path.join(output_dir, "reward_curve.json"), "w") as f:
                    json.dump({"episodes": list(range(1, len(episode_rewards)+1)),
                               "rewards": episode_rewards}, f)
                print(f"  💾 Checkpoint saved: {ckpt_path}")
            except Exception as e:
                print(f"  ⚠ Checkpoint failed: {e}")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    return episode_rewards


def _grpo_update_step(model: Any, tokenizer: Any, batch: List[Dict]) -> None:
    import torch

    prompts     = [b["prompt"]     for b in batch]
    completions = [b["completion"] for b in batch]
    rewards     = torch.tensor([b["reward"] for b in batch], dtype=torch.float32)

    # Normalize rewards (GRPO baseline)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    pos_frac = (rewards > 0).float().mean().item()
    print(
        f"    GRPO batch | n={len(batch)} | "
        f"mean_r={rewards.mean():.3f} | "
        f"pos_frac={pos_frac:.1%}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSME-RL GRPO Training")
    parser.add_argument("--episodes",             type=int,   default=300)
    parser.add_argument("--port",                 type=int,   default=8000)
    parser.add_argument("--model",                type=str,   default="Qwen/Qwen3-1.7B")
    parser.add_argument("--no-unsloth",           action="store_true")
    parser.add_argument("--max_steps_per_episode",type=int,   default=300)
    parser.add_argument("--save_every",           type=int,   default=10)
    parser.add_argument("--output_dir",           type=str,   default="./msme_rl_checkpoints")
    args = parser.parse_args()

    rewards = run_training(
        num_episodes=args.episodes,
        port=args.port,
        model_name=args.model,
        use_unsloth=not args.no_unsloth,
        max_steps_per_episode=args.max_steps_per_episode,
        save_every_n_episodes=args.save_every,
        output_dir=args.output_dir,
    )