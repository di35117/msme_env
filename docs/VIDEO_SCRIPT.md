# MSME-RL — 2-Minute Submission Video

**Total runtime target: 1:55–2:00**  
**Tone: confident, plain-spoken, anchored in the Indian credit reality.**  
**Recording setup: screen recording + face cam (small, top-right). 1080p, 30fps.**

---

## Shot list and timing

| # | Time | Visual on screen | What you say (verbatim) |
|---|------|-----------------|--------------------------|
| 1 | 0:00–0:15 | Black screen → fade to title card: **"Linguistic Decoding RL — MSME + Startup Credit, India"** | *"In Indian banking, nobody tells you the truth on the phone. The MSME owner says **'sab theek hai, sir'** — but he hasn't paid GST in two months. The startup founder pitches you a **Series B that's 'almost closed'** — but their runway is six weeks. So a relationship manager isn't classifying sentiment — they're decoding hidden state from language under partial observability. That's the RL problem we built."* |
| 2 | 0:15–0:30 | `README.md` scrolling: "Why This Exists" + the architecture diagram | *"This is a 36-month, 30-account environment — 20 MSMEs in cluster topologies, 10 startups in an investor ecosystem. The agent gets messages, behavioral proxies, and a portfolio summary every month, picks one of 21 actions, and gets graded on NPA rate, recovery, trust, and tool appropriateness. No LLM judge — every reward is a hard number in `reward.py`."* |
| 3 | 0:30–0:55 | Open `train_grpo.py`, scroll to `SYSTEM_PROMPT`, then to a captured rollout log showing one MSME case | *"Case one — MSME understatement. Account 7, owner messages: **'thoda time aur de dijiye sir, sab settle ho jayega.'** Behavioral signals say: GST returns missed twice, payroll late, cluster centrality 0.81. **Before training**, the model spammed `initiate_sarfaesi` — wrong tool, would have triggered a cluster cascade for `-0.25 × N` accounts. **After training**, the model first runs `verify_gst_returns`, then `restructure_loan_terms`. Same input text. Different decision."* |
| 4 | 0:55–1:20 | Scroll to a captured rollout log showing one startup case | *"Case two — startup overstatement. Account 24, founder messages: **'we just closed our pre-Series B, all good.'** Behavioral signals: investor updates stopped 90 days ago, burn rate up 40%, runway 4 months. **Before training**, the model believed the pitch and waited. **After training**, it runs `schedule_investor_meeting_check_in` — that's the action with the `+0.10` `investor_meeting_triggered_bridge` outcome. SARFAESI on a startup is `-0.15` plus the regular NPA penalty. The reward gradient teaches that asymmetry."* |
| 5 | 1:20–1:40 | Open `msme_rl_checkpoints/training_metrics.png` (the 2x3 dashboard) | *"Four monitored metrics, not just reward: episode reward trends up, GRPO loss stays bounded, KL against the frozen SFT reference stays under control thanks to the KL anchor at `KL_COEF=0.05`, completion-token entropy stays above zero — that's the entropy bonus preventing mode collapse — and parse-failure rate drops to zero in the first few episodes thanks to the JSON prefill plus extractor fallback."* |
| 6 | 1:40–1:55 | Scroll README "Anti-Reward-Hacking Design" section | *"And critically, this isn't just reward, it's anti-reward-hacking. Cluster cascade penalty rules out over-aggression. SARFAESI-on-startup penalty rules out wrong-tool abuse. A `_compute_shortcut_penalty` function catches no-op farming, malformed-JSON spam, action spamming, and account thrashing — all four together capped at `-0.25` per episode. The agent literally cannot win by being lazy or repetitive."* |
| 7 | 1:55–2:00 | Title card: **"github.com/<you>/msmeEnv • OpenEnv-compliant"** | *"OpenEnv-compliant, reproducible with one command, deterministic eval included. Thanks for watching."* |

---

## Recording checklist

Before you hit record:

- [ ] `training_metrics.png` exists in `msme_rl_checkpoints/` (re-run plotting if not)
- [ ] At least one rollout log per case captured to a text file you can scroll
- [ ] README is on the **Anti-Reward-Hacking Design** section in another tab
- [ ] Screen resolution set to 1920x1080, font size in editor bumped to ~16pt
- [ ] Mic test — phone mic on a stand beats laptop mic by a mile

## What to cut if you go over 2:00

In priority order (cut from the bottom first):

1. Shot 6 sentence about `_compute_shortcut_penalty` — the README has it.
2. Shot 5 sentence about parse-failure rate — the dashboard shows it.
3. Shot 2 — replace with a single sentence: *"30 accounts, 36 months, 21 actions, hard-number rewards."*

## What NOT to do

- Don't read the README aloud. Paraphrase.
- Don't show terminal output that scrolls past the viewer — pre-capture a static log file with the *interesting* moments only.
- Don't say "as you can see" — the viewer is watching, they can see.
- Don't apologize for anything ("the loss is a bit noisy here…"). State what is true and move on.

## Optional: pre-record voiceover separately

If you're nervous about doing a single take, record the voiceover into a `.wav` first (quiet room, single take per shot is fine — splice in editor), then record the screen silently and overlay. Audio quality matters more than face-cam quality.
