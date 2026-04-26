"""
Run a single episode of the MSME-RL environment using an open-source ~1B HF model.

This script runs the environment locally (no server required) and uses a
Transformers causal LM to choose actions from the environment's discrete action set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Temporary local override for quick testing.
# Prefer using env var HF_TOKEN, .env, or --hf_token instead of hardcoding.
HF_TOKEN_OVERRIDE: Optional[str] = None


def _load_dotenv(path: Path) -> Dict[str, str]:
    """
    Minimal .env loader (no extra dependency).
    Supports lines like: KEY=VALUE, with optional quotes.
    """
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


def _get_hf_token(cli_token: Optional[str]) -> Optional[str]:
    # Priority: CLI flag -> environment -> .env file.
    if HF_TOKEN_OVERRIDE:
        return str(HF_TOKEN_OVERRIDE).strip()
    if cli_token:
        return cli_token
    if os.environ.get("HF_TOKEN"):
        return os.environ.get("HF_TOKEN")
    env = _load_dotenv(ROOT / ".env")
    token = env.get("HF_TOKEN")
    if token:
        # Make it available for downstream calls too.
        os.environ["HF_TOKEN"] = token
    return token


def _looks_like_hf_token(token: str) -> bool:
    # HF user access tokens typically start with "hf_" and are long-ish.
    # This is only a heuristic to provide a better error message.
    t = token.strip()
    return t.startswith("hf_") and len(t) >= 20


try:
    from models import MSMERLAction
    from server.msmeEnv_environment import MSMERLEnvironment
except (ModuleNotFoundError, ImportError) as e:
    # Most common cause: dependencies not installed yet (openenv-core).
    raise SystemExit(
        "Missing runtime dependencies for this environment.\n\n"
        "Fix:\n"
        "  1) In this repo, run: uv sync\n"
        "  2) Then run: python eval.py\n\n"
        f"Original import error: {e}"
    ) from e


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


def _pick_target_account(observation: Dict[str, Any]) -> Tuple[int, str, Dict[str, Any]]:
    msme_accounts = observation.get("msme_accounts", []) or []
    startup_accounts = observation.get("startup_accounts", []) or []
    combined = msme_accounts + startup_accounts
    if not combined:
        return 1, "msme", {}
    target = sorted(combined, key=lambda x: x.get("dpd", 0), reverse=True)[0]
    return int(target.get("account_id", 1)), str(target.get("account_type", "msme")), target


def _heuristic_fallback(observation: Dict[str, Any], rng: random.Random) -> MSMERLAction:
    """
    Safe fallback policy if the model output is unparsable/invalid.
    Keeps the episode running without crashing.
    """
    account_id, account_type, obs = _pick_target_account(observation)
    dpd = int(obs.get("dpd", 0))

    if account_type == "msme":
        gst = str(obs.get("gst_filing_status", ""))
        if dpd > 45:
            action = "restructure_emi"
        elif "not_filed" in gst or "delay" in gst:
            action = "verify_gst_returns"
        elif dpd > 15:
            action = "grant_moratorium"
        else:
            action = "send_empathetic_reminder"
    else:
        investor_update = str(obs.get("investor_update_sent", ""))
        if dpd > 45:
            action = "request_investor_update_meeting"
        elif "skipped" in investor_update:
            action = "check_startup_ecosystem_signals"
        elif dpd > 15:
            action = "offer_bridge_loan_extension"
        else:
            action = "send_empathetic_reminder"

    return MSMERLAction(
        action_type=action,
        account_id=account_id,
        parameters={},
        reasoning="fallback_heuristic",
    )


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first {...} JSON object from text.
    This is intentionally forgiving because many chat models add prose.
    """
    # Quick path: if text itself is JSON
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except Exception:
            pass

    # Find a balanced-ish JSON object chunk.
    # We look for the first '{' and then the last '}' after it (greedy),
    # then try progressively shorter endings until it parses.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    chunk = stripped[start : end + 1]
    for i in range(len(chunk), 1, -1):
        candidate = chunk[:i]
        if not candidate.endswith("}"):
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _build_prompt(obs: Dict[str, Any]) -> str:
    """
    Build a compact instruction prompt. The environment already injects
    working/semantic/episodic memory as strings; we pass those through.
    """
    account_id, account_type, target = _pick_target_account(obs)
    portfolio = obs.get("portfolio_summary", {}) or {}

    target_compact = {
        k: target.get(k)
        for k in [
            "account_id",
            "account_type",
            "dpd",
            "sector",
            "gst_filing_status",
            "investor_update_sent",
            "rm_last_message",
            "borrower_last_message",
        ]
        if k in target
    }

    instruction = {
        "task": "Pick exactly one environment action for the current step.",
        "constraints": {
            "action_space": ACTION_SPACE,
            "account_id_range": "1-30",
            "output_format": "Return ONLY valid JSON object. No markdown, no prose.",
        },
        "output_schema": {
            "action_type": "string (one of action_space)",
            "account_id": "integer (1-30)",
            "parameters": "object (may be empty)",
            "reasoning": "short string",
        },
        "state": {
            "episode": obs.get("episode"),
            "month": obs.get("month"),
            "portfolio_summary": portfolio,
            "active_alerts": {
                "cluster": obs.get("active_cluster_alerts", []),
                "ecosystem": obs.get("active_ecosystem_alerts", []),
            },
            "working_memory": obs.get("working_memory", ""),
            "semantic_memory_context": obs.get("semantic_memory_context", ""),
            "episodic_memory_context": obs.get("episodic_memory_context", ""),
            "suggested_target": {
                "account_id": account_id,
                "account_type": account_type,
                "snapshot": target_compact,
            },
        },
    }

    return json.dumps(instruction, ensure_ascii=False)


def _load_text_generator(model_id: str, dtype: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    torch_dtype = None
    if dtype == "bf16":
        import torch

        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        import torch

        torch_dtype = torch.float16
    elif dtype == "fp32":
        import torch

        torch_dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    else:
        # "auto" uses accelerate-style device_map; works well for consumer GPUs.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    return model, tok


def _call_hf_inference_api(
    *,
    model_id: str,
    prompt: str,
    token: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
) -> str:
    """
    Call hosted Hugging Face inference (no local model download).

    Requires an access token (set env var HF_TOKEN or pass --hf_token).
    """
    # HF has migrated most hosted inference to the Router "Responses API":
    #   POST https://router.huggingface.co/v1/responses
    # We use that first. If it fails due to account/provider/model constraints,
    # we can fall back to the legacy Inference API shape.
    url = "https://router.huggingface.co/v1/responses"
    payload = {
        "model": model_id,
        "input": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        # Keep it short and structured; we already instruct "JSON only" in the prompt.
        "max_output_tokens": max_new_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        if e.code in {401, 403}:
            raise RuntimeError(
                "HF Inference API authentication/authorization failed.\n\n"
                "If you got 401:\n"
                "  - Your HF token is missing/invalid/revoked\n"
                "  - You may be sending an old token that no longer works\n\n"
                "If you got 403:\n"
                "  - Token is valid but lacks 'Make calls to Inference Providers' permission\n"
                "  - Billing/credits not enabled for Inference Providers\n\n"
                f"HTTP {e.code} detail: {detail[:400]}"
            ) from e
        # If router endpoint rejects, try the legacy endpoint (some org setups still require it).
        legacy_url = f"https://api-inference.huggingface.co/models/{model_id}"
        legacy_payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }
        legacy_body = json.dumps(legacy_payload).encode("utf-8")
        legacy_req = Request(
            legacy_url,
            data=legacy_body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(legacy_req, timeout=timeout_s) as resp:
                legacy_raw = resp.read().decode("utf-8", errors="replace")
        except Exception:
            raise RuntimeError(f"HF Inference API HTTPError {e.code}: {detail}") from e
        raw = legacy_raw
    except URLError as e:
        raise RuntimeError(f"HF Inference API connection error: {e}") from e

    try:
        data = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"HF Inference API returned non-JSON: {raw[:400]}") from e

    # Router Responses API shape:
    # - {"output_text": "...", ...} (plus many other fields)
    if isinstance(data, dict) and isinstance(data.get("output_text"), str):
        return str(data["output_text"])

    # Legacy Inference API shapes:
    # - [{"generated_text": "..."}]
    # - {"error": "..."}
    if isinstance(data, dict) and "error" in data:
        err = data.get("error")
        msg = str(err)
        # Router sometimes returns a JSON error with an embedded 403 message
        # (instead of an HTTPError). In that case, try the legacy endpoint too.
        if "403 status code" in msg or "status code 403" in msg:
            legacy_url = f"https://api-inference.huggingface.co/models/{model_id}"
            legacy_payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "return_full_text": False,
                },
                "options": {"wait_for_model": True},
            }
            legacy_body = json.dumps(legacy_payload).encode("utf-8")
            legacy_req = Request(
                legacy_url,
                data=legacy_body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urlopen(legacy_req, timeout=timeout_s) as resp:
                    legacy_raw = resp.read().decode("utf-8", errors="replace")
                legacy_data = json.loads(legacy_raw)
                if isinstance(legacy_data, list) and legacy_data and isinstance(legacy_data[0], dict) and "generated_text" in legacy_data[0]:
                    return str(legacy_data[0]["generated_text"])
                if isinstance(legacy_data, dict) and isinstance(legacy_data.get("generated_text"), str):
                    return str(legacy_data["generated_text"])
            except Exception:
                # Fall through to error below with a more actionable message.
                pass

            raise RuntimeError(
                "HF Inference API error: got a 403 from the Router endpoint and the legacy endpoint did not succeed.\n\n"
                "Most common causes:\n"
                "  - Token missing the 'Make calls to Inference Providers' permission\n"
                "  - Your account has no Inference Providers credits / billing not enabled\n"
                "  - Organization policy blocks provider calls (IP allowlist / SSO / etc.)\n\n"
                f"Raw error: {err}"
            )
        if "not supported by any provider you have enabled" in msg:
            raise RuntimeError(
                "HF Inference API error: the selected model is not available via your enabled providers.\n\n"
                "Fix options:\n"
                '  - Pick a model that has a hosted provider on its HF page (look for an "Inference Providers" selector)\n'
                "  - Or add a provider suffix to the model id, e.g. '...:featherless-ai' / '...:groq' (depending on what you have enabled)\n\n"
                f"Raw error: {err}"
            )
        raise RuntimeError(f"HF Inference API error: {err}")
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return str(data[0]["generated_text"])

    # Some models/endpoints may return a different schema; keep it debuggable.
    raise RuntimeError(f"Unexpected HF Inference API response: {raw[:400]}")


def _generate_action_with_model(
    model,
    tok,
    obs: Dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Optional[MSMERLAction]:
    import torch

    prompt = _build_prompt(obs)

    # Keep generation deterministic by default (temperature=0).
    do_sample = temperature > 0

    inputs = tok(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    decoded = tok.decode(out[0], skip_special_tokens=True)
    # Try to isolate model completion: remove the prompt prefix if it appears verbatim.
    completion = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded

    data = _extract_first_json_object(completion)
    if not isinstance(data, dict):
        return None

    action_type = str(data.get("action_type", "")).strip()
    account_id_raw = data.get("account_id", None)
    parameters = data.get("parameters", {}) if isinstance(data.get("parameters", {}), dict) else {}
    reasoning = str(data.get("reasoning", "model")).strip()[:400]

    try:
        account_id = int(account_id_raw)
    except Exception:
        account_id = None

    if action_type not in ACTION_SPACE:
        return None
    if account_id is None or not (1 <= account_id <= 30):
        return None

    return MSMERLAction(
        action_type=action_type,
        account_id=account_id,
        parameters=parameters,
        reasoning=reasoning,
    )


def _generate_action_with_hf_api(
    *,
    model_id: str,
    obs: Dict[str, Any],
    hf_token: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
) -> Optional[MSMERLAction]:
    prompt = _build_prompt(obs)
    completion = _call_hf_inference_api(
        model_id=model_id,
        prompt=prompt,
        token=hf_token,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout_s=timeout_s,
    )

    data = _extract_first_json_object(completion)
    if not isinstance(data, dict):
        return None

    action_type = str(data.get("action_type", "")).strip()
    account_id_raw = data.get("account_id", None)
    parameters = data.get("parameters", {}) if isinstance(data.get("parameters", {}), dict) else {}
    reasoning = str(data.get("reasoning", "hf_api")).strip()[:400]

    try:
        account_id = int(account_id_raw)
    except Exception:
        account_id = None

    if action_type not in ACTION_SPACE:
        return None
    if account_id is None or not (1 <= account_id <= 30):
        return None

    return MSMERLAction(
        action_type=action_type,
        account_id=account_id,
        parameters=parameters,
        reasoning=reasoning,
    )


def run_one_episode(
    model_id: str,
    max_steps: int,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    dtype: str,
    device: str,
    backend: str,
    hf_token: Optional[str],
    hf_timeout_s: float,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    env = MSMERLEnvironment()
    obs_obj = env.reset()
    obs = obs_obj.__dict__

    model = tok = None
    if backend == "local":
        model, tok = _load_text_generator(model_id=model_id, dtype=dtype, device=device)
    elif backend == "hf_api":
        if not hf_token:
            raise SystemExit(
                "HF Inference API backend requires a token.\n"
                "Set env var HF_TOKEN or pass --hf_token.\n\n"
                'Example (PowerShell): $env:HF_TOKEN="hf_..."; python eval.py --backend hf_api --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"'
            )
        if not _looks_like_hf_token(hf_token):
            raise SystemExit(
                "HF Inference API backend requires a valid Hugging Face access token.\n"
                "Your HF_TOKEN does not look like a Hugging Face token (expected something starting with 'hf_...').\n"
                "Fix:\n"
                "  - Set a correct token in your environment or .env\n"
                "  - Or pass one explicitly: --hf_token hf_...\n"
            )
    else:
        raise SystemExit(f"Unknown backend: {backend}")

    steps: List[Dict[str, Any]] = []
    done = False
    t = 0

    while not done and t < max_steps:
        if backend == "local":
            proposed = _generate_action_with_model(
                model=model,
                tok=tok,
                obs=obs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            proposed = _generate_action_with_hf_api(
                model_id=model_id,
                obs=obs,
                hf_token=hf_token or "",
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout_s=hf_timeout_s,
            )
        action = proposed or _heuristic_fallback(obs, rng)

        next_obs_obj = env.step(action)
        next_obs = next_obs_obj.__dict__

        last = next_obs.get("last_action_result") or {}
        steps.append(
            {
                "t": t,
                "month": obs.get("month"),
                "action": {
                    "action_type": action.action_type,
                    "account_id": action.account_id,
                    "parameters": action.parameters,
                    "reasoning": action.reasoning,
                    "source": "model" if proposed is not None else "fallback",
                },
                "outcome": last.get("outcome"),
                "step_reward": float(next_obs.get("step_reward", 0.0) or 0.0),
                "episode_reward_so_far": float(next_obs.get("episode_reward_so_far", 0.0) or 0.0),
                "done": bool(next_obs.get("done", False)),
            }
        )

        obs = next_obs
        done = bool(obs.get("done", False))
        t += 1

    breakdown = (obs.get("last_action_result") or {}).get("episode_reward_breakdown") or {}
    total = float(breakdown.get("total", obs.get("episode_reward_so_far", 0.0)) or 0.0)

    return {
        "model_id": model_id,
        "seed": seed,
        "steps_taken": len(steps),
        "done": done,
        "episode": obs.get("episode"),
        "final_month": obs.get("month"),
        "total_reward": total,
        "episode_reward_breakdown": breakdown,
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one MSME-RL episode with a ~1B Hugging Face model.")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HF model id (1B-ish). Example: TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    # Backwards-compatible alias for older CLIs / docs.
    # Some users naturally try "--provider hf-inference" (HF hosted) vs local.
    # We keep the canonical flag as --backend.
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Alias for --backend. Use: --provider hf-inference (same as --backend hf_api).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["local", "hf_api"],
        default="local",
        help="Inference backend. local=download+run model; hf_api=hosted HF Inference API (no local download).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face access token (or set env var HF_TOKEN). Required for --backend hf_api.",
    )
    parser.add_argument(
        "--hf_timeout_s",
        type=float,
        default=120.0,
        help="Timeout (seconds) for each HF Inference API call.",
    )
    parser.add_argument("--max_steps", type=int, default=400, help="Safety cap; episode ends earlier when done=True.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--device", type=str, choices=["auto", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/one_episode_hf.json"))
    args = parser.parse_args()

    if args.provider:
        provider = str(args.provider).strip().lower()
        if provider in {"hf-inference", "hf_inference", "hf", "huggingface"}:
            args.backend = "hf_api"
        elif provider in {"local"}:
            args.backend = "local"
        else:
            raise SystemExit(
                f"Unknown --provider value: {args.provider}\n"
                "Supported:\n"
                "  --provider hf-inference   (same as --backend hf_api)\n"
                "  --provider local          (same as --backend local)\n"
            )

    report = run_one_episode(
        model_id=args.model,
        max_steps=args.max_steps,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        dtype=args.dtype,
        device=args.device,
        backend=args.backend,
        hf_token=_get_hf_token(args.hf_token),
        hf_timeout_s=float(args.hf_timeout_s),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Short stdout summary + where to find full trace.
    print(json.dumps({k: report[k] for k in ["model_id", "steps_taken", "done", "total_reward", "final_month"]}, indent=2))
    print(f"\nSaved full episode trace to: {args.output}")


if __name__ == "__main__":
    main()

