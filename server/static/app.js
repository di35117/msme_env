const ACTION_TYPES = [
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
];

const state = {
  rewardSeries: [],
  timeline: [],
  latestObs: null,
};
const SIDEBAR_STORAGE_KEY = "msme_sidebar_collapsed";
const THEME_STORAGE_KEY = "msme_theme";

function byId(id) {
  return document.getElementById(id);
}

function formatNum(n, d = 3) {
  return Number(n || 0).toFixed(d);
}

function setMetric(id, value, cls = "") {
  const el = byId(id);
  el.textContent = value;
  el.className = cls;
}

function setUpdatedNow() {
  const now = new Date();
  const stamp = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  byId("lastUpdated").textContent = `Updated: ${stamp}`;
}

function applySidebarState(collapsed) {
  const app = document.body;
  const icon = byId("sidebarToggleIcon");
  app.classList.toggle("sidebar-collapsed", collapsed);
  if (icon) {
    icon.textContent = collapsed ? "⟩" : "⟨";
  }
}

function applyTheme(theme) {
  const app = document.body;
  const icon = byId("themeToggleIcon");
  const isLight = theme === "light";
  app.classList.toggle("theme-light", isLight);
  if (icon) {
    icon.textContent = isLight ? "🌙" : "☀";
  }
}

function toggleTheme() {
  const next = document.body.classList.contains("theme-light") ? "dark" : "light";
  applyTheme(next);
  localStorage.setItem(THEME_STORAGE_KEY, next);
}

function toggleSidebar() {
  const collapsed = !document.body.classList.contains("sidebar-collapsed");
  applySidebarState(collapsed);
  localStorage.setItem(SIDEBAR_STORAGE_KEY, collapsed ? "1" : "0");
}

function appendTimeline(entry) {
  state.timeline.unshift(entry);
  state.timeline = state.timeline.slice(0, 20);
  const list = byId("timeline");
  list.innerHTML = state.timeline
    .map((x) => `<li>${x}</li>`)
    .join("");
}

function drawRewardCurve() {
  const canvas = byId("rewardCanvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = "#090f1f";
  ctx.fillRect(0, 0, w, h);

  if (!state.rewardSeries.length) {
    return;
  }

  const values = state.rewardSeries;
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0.001);
  const span = Math.max(0.001, max - min);
  const pad = 24;
  const xStep = (w - pad * 2) / Math.max(1, values.length - 1);

  ctx.strokeStyle = "#1f305e";
  ctx.lineWidth = 1;
  ctx.beginPath();
  const yZero = h - pad - ((0 - min) / span) * (h - pad * 2);
  ctx.moveTo(pad, yZero);
  ctx.lineTo(w - pad, yZero);
  ctx.stroke();

  ctx.strokeStyle = "#4ce6a3";
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = pad + i * xStep;
    const y = h - pad - ((v - min) / span) * (h - pad * 2);
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

function renderObservation(observation) {
  state.latestObs = observation;
  const summary = observation.portfolio_summary || {};

  const stepReward = observation.step_reward ?? 0;
  setMetric("mEpisode", String(observation.episode ?? "-"));
  setMetric("mMonth", String(observation.month ?? "-"));
  setMetric("mStepReward", formatNum(stepReward), stepReward >= 0 ? "pos" : "neg");
  setMetric("mCumulativeReward", formatNum(observation.episode_reward_so_far), "pos");
  setMetric("mNpaRate", formatNum(summary.npa_rate, 4), summary.npa_rate > 0.2 ? "neg" : "pos");
  setMetric("mAvgTrust", formatNum(summary.avg_trust_score, 3), "pos");

  const alerts = [
    ...(observation.active_cluster_alerts || []),
    ...(observation.active_ecosystem_alerts || []),
  ];
  byId("violations").innerHTML = (alerts.length ? alerts : ["No active alerts"])
    .map((a) => `<li>${a}</li>`)
    .join("");

  byId("latestOutput").textContent = JSON.stringify(
    observation.last_action_result || observation.portfolio_summary || observation,
    null,
    2
  );

  byId("serverStatus").textContent = observation.done ? "Episode Complete" : "Connected";
  setUpdatedNow();
}

async function apiCall(path, method = "GET", body) {
  const resp = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!resp.ok) {
    throw new Error(`${method} ${path} failed (${resp.status})`);
  }
  return resp.json();
}

async function runReset() {
  const payload = await apiCall("/reset", "POST", {});
  const observation = payload.observation || payload;
  state.rewardSeries = [];
  state.timeline = [];
  renderObservation(observation);
  appendTimeline(`Episode ${observation.episode}: reset at month ${observation.month}`);
  drawRewardCurve();
}

async function runStep() {
  let parameters = {};
  try {
    parameters = JSON.parse(byId("params").value || "{}");
  } catch (e) {
    alert("Parameters must be valid JSON");
    return;
  }

  const body = {
    action_type: byId("actionType").value,
    account_id: Number(byId("accountId").value),
    parameters,
    reasoning: byId("reasoning").value || "",
  };

  const payload = await apiCall("/step", "POST", body);
  const observation = payload.observation || payload;
  renderObservation(observation);
  state.rewardSeries.push(observation.step_reward || 0);
  state.rewardSeries = state.rewardSeries.slice(-160);
  drawRewardCurve();

  const outcome = observation.last_action_result?.outcome || "n/a";
  appendTimeline(
    `M${observation.month} | A${body.account_id} | ${body.action_type} | ${outcome} | R ${formatNum(observation.step_reward)}`
  );
}

function setupActionOptions() {
  byId("actionType").innerHTML = ACTION_TYPES.map(
    (x) => `<option value="${x}">${x}</option>`
  ).join("");
}

function bindEvents() {
  byId("resetBtn").addEventListener("click", async () => {
    try {
      await runReset();
    } catch (e) {
      alert(String(e));
    }
  });

  byId("stepBtn").addEventListener("click", async () => {
    try {
      await runStep();
    } catch (e) {
      alert(String(e));
    }
  });

  byId("refreshArtifactsBtn").addEventListener("click", async () => {
    try {
      await loadJudgeArtifacts();
    } catch (e) {
      byId("judgeSummary").textContent = `Failed to load artifacts.\n${String(e)}`;
    }
  });

  byId("sidebarToggle").addEventListener("click", toggleSidebar);
  byId("themeToggle").addEventListener("click", toggleTheme);
}

function prettifyArtifactName(path) {
  const file = path.split("/").pop() || path;
  return file.replaceAll("_", " ").replace(".png", "");
}

async function loadJudgeArtifacts() {
  const data = await apiCall("/web/judge-artifacts");
  byId("judgeSummary").textContent = JSON.stringify(
    data.summary && Object.keys(data.summary).length ? data.summary : { message: data.hint },
    null,
    2
  );

  const gallery = byId("artifactGallery");
  if (!data.images || !data.images.length) {
    gallery.innerHTML = "<div class='artifact-card'><div class='title'>No artifact images found yet.</div></div>";
    return;
  }

  const stamp = Date.now();
  gallery.innerHTML = data.images.map((src) => `
    <div class="artifact-card">
      <div class="title">${prettifyArtifactName(src)}</div>
      <img src="${src}?t=${stamp}" alt="${prettifyArtifactName(src)}" />
    </div>
  `).join("");
  setUpdatedNow();
}

function bootstrap() {
  const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
  const preferredTheme = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";
  applyTheme(savedTheme || preferredTheme);

  const savedCollapsed = localStorage.getItem(SIDEBAR_STORAGE_KEY) === "1";
  applySidebarState(savedCollapsed);
  setupActionOptions();
  bindEvents();
  setUpdatedNow();
  loadJudgeArtifacts().catch((e) => {
    byId("judgeSummary").textContent = `Judge artifacts unavailable.\n${String(e)}`;
    byId("serverStatus").textContent = "Artifacts Unavailable";
  });
  runReset().catch((e) => {
    byId("latestOutput").textContent = `Unable to load environment.\n${String(e)}`;
    byId("serverStatus").textContent = "Disconnected";
  });
}

bootstrap();
