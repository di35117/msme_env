# 🧠 Linguistic Decoding RL

> **An AI that learns to catch what people aren't saying — then decides what to do about it.**

Not a chatbot. Not a classifier. A decision-making agent trained inside a world where the truth is always slightly hidden.

---

## The Problem Nobody Talks About

People lie. Or rather — they *manage* what they say.

A small business owner who's two months from defaulting will tell you "things are a bit slow."  
A startup founder who hasn't talked to an investor in three months will say "we're in active conversations."

Both sound fine. Both are not fine.

Regular AI reads the sentence and moves on. **This agent learns that the sentence is only half the picture.**

The other half? Payment delays. GST filing gaps. What the businesses *around* them are doing. How their story has drifted over the past few months. That's where the real signal lives.

---

## What's Actually Being Built

An RL environment where a language model has to:

1. Read a message from a borrower
2. Look at their behavior (not just their words)
3. Pick the right move from 21 possible actions — across 30 accounts, over 36 months
4. Live with the consequences

The reward doesn't come from saying something smart. It comes from outcomes — did the loan get recovered? Did trust hold? Did you avoid blowing up the entire lending cluster because you got aggressive with the wrong person?

**The model only gets good if it actually understands what's going on.**

---

## The Setup (India MSME + Startup Credit)

- **20 MSME accounts** — small businesses, connected to each other, prone to understate stress
- **10 startup accounts** — prone to overstate health, completely different risk profile
- **21 action types** — from `verify_gst_returns` to `grant_moratorium` to `initiate_sarfaesi`
- **36-month horizon** — decisions made now have consequences three months later

The same message, sent by an MSME owner versus a startup founder, means something different. Learning that distinction is the whole game.

---

## What Happened After 28 Training Episodes

Trained on a Colab T4. Small model (Qwen2.5-1.5B). Here's what changed:

| What we measured | Before training | After training |
|---|---|---|
| Reward per episode | ≈ −3.8 (random noise) | peak ≈ +0.39 |
| Parse failures | frequent | ≈ 0% by episode 5 |
| SARFAESI used on startups | happens | basically never |

That last one is the most interesting. **Nobody told the model not to use a collateral recovery tool on a startup.** The reward structure made it figure out that it's a bad idea — startups have no collateral, founders leave, ecosystem trust collapses. The model learned the *why* from the outcomes, not from a rule.

The trained policy concentrates on `verify_gst_returns`, `grant_moratorium`, and `request_investor_update_meeting`. The random baseline fires everything equally. That gap — that selectivity — is what real understanding looks like in this environment.

---

## Why LLMs Are Bad at This (Out of the Box)

Current language models are excellent at one thing: **producing text that sounds right.**

This environment cares about something else entirely — whether the decision was *correct*. That's a different optimization target, and it creates real gaps:

**They don't hold behavioral memory.**  
GST filings from three months ago matter right now. A base LLM has no way to accumulate and weight that kind of temporal signal into a coherent belief about what's actually happening.

**They optimize for fluent output, not for outcomes.**  
Pretraining taught them to generate responses that make sense. This environment rewards NPA prevention, trust preservation, cascade avoidance. Those objectives pull in different directions.

**They miss the asymmetry.**  
An MSME borrower and a startup founder sending identical messages are usually in completely different situations. A model that treats them the same will be wrong half the time — confidently.

---

## The Anti-Cheat System (This Part Matters)

RL agents are creative about finding loopholes. We closed them explicitly.

**Cluster cascade penalty (`−0.25` per account)**  
MSME borrowers are networked. Go hard on one high-centrality account and you can trigger defaults across the whole cluster. That costs more than just letting the one account go — so being maximally aggressive stops being the safe play.

**Wrong tool on startups (`−0.15` per use)**  
Using SARFAESI (a collateral recovery tool) on a startup is *technically allowed* and *practically insane*. The penalty makes right-tool behavior strictly better in expectation. The model learns this fast.

**Shortcut detection at episode end**  
Spam `wait_and_observe` more than 30% of the time? Deducted. Use one action for 35%+ of all steps? Deducted. The only way to get a high score is to make real, varied, context-appropriate decisions.

**Action-frequency cap**  
Even if you find a genuinely great action — after the third use per episode, it contributes zero to your score. Diversity isn't optional, it's the only path to a high return.

---

## The Architecture (In Plain Terms)

```
LLM Policy
  ↓
FastAPI server  (handles the RL loop)
  ↓
Environment core  (state, episodes, rewards)
  ↓
Domain adapter  (MSME + startup specific logic)
  ↓
  ├── World generator  (builds the accounts and their hidden state)
  ├── Reward logic     (the hard-number objective, no LLM judge)
  ├── Network effects  (who affects who)
  ├── Message generator (what the borrowers actually say)
  └── Memory updates   (what the agent gets to remember)
```

The domain adapter is a swappable layer. Keep all the RL infrastructure, change the domain. Compliance. Escalation. Negotiation. Anything where people are strategically shaping what they say.

---

## Run It

```bash
# Local
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Docker
docker build -t linguistic-decoding-env:latest -f server/Dockerfile .
```

Dashboard at `http://localhost:8000/` — runs the environment step by step, shows training plots live.

**Train:**
```bash
python train_grpo.py --episodes 50 --output_dir msme_rl_checkpoints
```

**Eval against random baseline:**
```bash
python scripts/eval.py --episodes 5 --output artifacts/eval_report.json
```

---

## What This Is Really About

The MSME credit demo is concrete. But the actual research question is broader:

> *Can a language model learn to decode the gap between what someone says and what's actually happening — and act on it correctly?*

That gap exists everywhere. Support calls. Compliance reviews. Any conversation where the speaker has an incentive to shape your perception. Base LLMs will give you a fluent, confident, wrong answer. An agent trained in an environment like this one has to earn its confidence through outcomes.

That's the difference.

---

## What's Next

- More domain adapters — compliance, support escalation, negotiation
- Replace the deterministic message generator with an LLM-driven adversary that generates fresh borrower messages each episode
- Multi-account WebSocket view in the dashboard
- Benchmark suites per domain

---

*Built with Qwen2.5-1.5B · GRPO · KL anchor + entropy bonus · Trained on Colab T4 · OpenEnv compliant*