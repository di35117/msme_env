"""
Minimal GRPO-style loop for the MSME-RL environment.

This is intentionally tiny and CPU-friendly:
  - Uses direct env instantiation (no server)
  - Collects a small batch of rollouts
  - Performs a single "make good completions more likely" update

It is not meant to be SOTA GRPO; it's meant to be the smallest runnable example
that matches this repo's action format (JSON) and reward plumbing.
 
Run:
  python scripts/grpo_minimal.py
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ACTION_SPACE: List[str] = [
    # Communication
    "send_empathetic_reminder",
    "send_firm_reminder",
    "send_legal_notice_section13",
    "call_promoter_founder",
    "call_guarantor_investor",
    "conduct_cluster_ecosystem_visit",
    # Financial restructuring
    "grant_moratorium",
    "restructure_emi",
    "offer_eclgs_topup",
    "offer_bridge_loan_extension",
    "accept_partial_payment",
    "waive_penal_interest",
    # Recovery
    "initiate_sarfaesi",
    "refer_to_recovery_agent",
    "file_drt_case",
    "offer_one_time_settlement",
    # Information gathering
    "verify_gst_returns",
    "pull_bank_statements",
    "check_industry_cluster_stress",
    "request_investor_update_meeting",
    "check_startup_ecosystem_signals",
    # No-op
    "wait_and_observe",
]


SYSTEM_PROMPT = """You are an RM (relationship manager) managing a mixed portfolio (MSME + startup).
Return ONLY a JSON object with keys: reasoning, action_type, account_id, parameters.
action_type must be one of the provided ACTION_SPACE.
account_id must be an integer in [1, 30].
"""


def build_prompt(observation: Dict[str, Any]) -> str:
    summary = json.dumps(observation.get("portfolio_summary", {}), indent=2)
    urgent = sorted(
        (observation.get("msme_accounts", []) or []) + (observation.get("startup_accounts", []) or []),
        key=lambda x: x.get("dpd", 0),
        reverse=True,
    )[:5]
    urgent_str = json.dumps(urgent, ensure_ascii=False, indent=2)
    return (
        f"=== PORTFOLIO SUMMARY ===\n{summary}\n\n"
        f"=== TOP 5 ACCOUNTS BY DPD ===\n{urgent_str}\n\n"
        f"=== WORKING MEMORY ===\n{observation.get('working_memory','')}\n\n"
        "Decide the next action.\n"
        "Respond with JSON only."
    )


def _pick_account_id(observation: Dict[str, Any], rng: random.Random) -> int:
    accounts = (observation.get("msme_accounts", []) or []) + (observation.get("startup_accounts", []) or [])
    if not accounts:
        return 1
    # Bias toward high DPD, but keep some exploration.
    top = sorted(accounts, key=lambda x: x.get("dpd", 0), reverse=True)[:10]
    choice = rng.choice(top)
    return int(choice.get("account_id", 1))


def _random_action_json(observation: Dict[str, Any], rng: random.Random) -> str:
    action_type = rng.choice(ACTION_SPACE)
    account_id = _pick_account_id(observation, rng)
    payload = {
        "reasoning": "minimal_grpo_random_policy",
        "action_type": action_type,
        "account_id": account_id,
        "parameters": {},
    }
    return json.dumps(payload, ensure_ascii=False)


@dataclass
class StepSample:
    prompt: str
    completion: str
    reward: float
    advantage: float = 0.0


def _policy_update(model: Any, tokenizer: Any, batch: List[StepSample]) -> float:
    """
    Minimal policy-gradient update on text completions.
    We compute NLL on (prompt+completion) but mask prompt tokens, then weight by advantage.
    """
    import torch
    from torch.nn.utils import clip_grad_norm_

    if not hasattr(model, "_grpo_opt"):
        trainable = [p for p in model.parameters() if p.requires_grad]
        model._grpo_opt = torch.optim.AdamW(trainable, lr=5e-5)

    opt = model._grpo_opt
    model.train()
    opt.zero_grad()

    total_loss = 0.0
    used = 0

    for s in batch:
        full_text = s.prompt + s.completion
        enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        prompt_enc = tokenizer(s.prompt, return_tensors="pt", truncation=True, max_length=1024)
        prompt_len = min(prompt_enc["input_ids"].shape[1], enc["input_ids"].shape[1] - 1)

        labels = enc["input_ids"].clone()
        labels[:, :prompt_len] = -100

        out = model(**enc, labels=labels)
        # out.loss is mean NLL over completion tokens; multiply by -advantage to increase prob for good samples.
        loss = out.loss * (-float(s.advantage))
        total_loss = total_loss + loss
        used += 1

    if used == 0:
        return 0.0

    total_loss = total_loss / used
    total_loss.backward()
    clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    return float(total_loss.detach().cpu().item())


def main() -> None:
    # Local imports (repo layout supports running from workspace root)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    try:
        from server.msmeEnv_environment import MSMERLEnvironment
        from models import MSMERLAction
    except (ImportError, ModuleNotFoundError):
        from msmeEnv_environment import MSMERLEnvironment  # type: ignore
        from models import MSMERLAction  # type: ignore

    # Tiny model for a quick demo. It's not "smart"; this is about plumbing.
    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to("cpu")

    rng = random.Random(0)
    env = MSMERLEnvironment()
    obs = env.reset().__dict__

    # Minimal rollout + update
    batch: List[StepSample] = []
    rollout_steps = 24
    for t in range(rollout_steps):
        prompt = f"{SYSTEM_PROMPT}\nACTION_SPACE={ACTION_SPACE}\n\n" + build_prompt(obs) + "\n"

        # For minimality we start with a random *valid* JSON action.
        completion = _random_action_json(obs, rng)
        action_dict = json.loads(completion)
        action = MSMERLAction(
            action_type=action_dict["action_type"],
            account_id=int(action_dict["account_id"]),
            parameters=action_dict.get("parameters", {}),
            reasoning=action_dict.get("reasoning", ""),
        )

        next_obs = env.step(action).__dict__
        r = float(next_obs.get("step_reward", 0.0))
        batch.append(StepSample(prompt=prompt, completion=completion, reward=r))
        obs = next_obs

        if (t + 1) % 8 == 0:
            print(f"rollout t={t+1}/{rollout_steps} mean_step_reward={sum(s.reward for s in batch)/len(batch):.4f}")

    # Advantage normalization (group-relative baseline)
    rewards = [s.reward for s in batch]
    mean_r = sum(rewards) / max(1, len(rewards))
    var = sum((x - mean_r) ** 2 for x in rewards) / max(1, len(rewards))
    std = (var ** 0.5) + 1e-8
    for s in batch:
        s.advantage = (s.reward - mean_r) / std

    loss = _policy_update(model, tokenizer, batch)
    print(f"update done | batch={len(batch)} | loss={loss:.4f} | reward_mean={mean_r:.4f} reward_std={std:.4f}")


if __name__ == "__main__":
    main()

