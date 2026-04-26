# Linguistic decoding RL: teaching a small LLM to read what borrowers do not say

*Draft for a Hugging Face blog post or Space documentation. You can copy sections into the HF post editor; adjust numbers after your latest training run.*

## The idea in one sentence

We train a **1.5B** policy with **GRPO** inside a **credit collection simulator** so it learns when soft language (“things are a bit slow”) should trigger hard actions—and when it should not.

## Why this is not “another chatbot”

Chat models optimize for fluent replies. This agent optimizes for **outcomes**: non-performing loan rate, recovery, trust, and using the right **legal and operational** tools for **MSME** versus **startup** accounts. The environment encodes **network effects** (one aggressive move can break a cluster) and **anti-cheat** penalties (spamming one action or hiding behind “wait and observe” does not win).

## What we built

- **20 MSME and 10 startup** borrower accounts, **21** discrete actions, **36** simulated months.
- **JSON-only** actions from the model; parsing and fallbacks are instrumented.
- **GRPO** with a **KL anchor** to a supervised reference and **entropy** regularization.
- A **3×3 training dashboard** (reward, loss, KL, entropy, parse failures, and **business curves**: NPA, trust, restraint).

## Reproducibility

Clone the [repository](https://github.com/di35117/msme_env), run `train_colab.ipynb` on a GPU, or use:

```bash
python train_grpo.py --episodes 30 --max_steps_per_episode 90 --save_every 2 \
  --model Qwen/Qwen2.5-1.5B-Instruct --output_dir msme_rl_run
```

After training, `scripts/inference_before_after.py` and `scripts/make_hero.py` produce before/after and scorecard figures for the README and the demo.

## What we would highlight to judges

- **Format adherence:** parse-failure rate tracked per episode; the policy is not rewarded for invalid JSON in the GRPO set.
- **Policy shift, not a single number:** action **difference** plots (before vs after RL) and **MSME vs startup** splits show the dual strategy.
- **No reward hacking by construction:** step caps, tool mismatch penalties, and episode-level structure are documented in `reward.py` and the README.

## Links

- **Code / Space:** add your Hugging Face Space URL here.  
- **Notebook:** add your public Colab link here after you publish the notebook.  
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` fine-tuned with GRPO in the MSME-RL environment.

---

*End of draft — trim or expand to HF blog length limits.*
