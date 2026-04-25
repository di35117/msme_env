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
  "wait_and_observe"
];

const el = {
  status: document.getElementById("status-pill"),
  actionType: document.getElementById("action-type"),
  accountId: document.getElementById("account-id"),
  parameters: document.getElementById("parameters"),
  reasoning: document.getElementById("reasoning"),
  lastAction: document.getElementById("last-action"),
  observation: document.getElementById("observation"),
  clusterAlerts: document.getElementById("cluster-alerts"),
  ecosystemAlerts: document.getElementById("ecosystem-alerts"),
  kEpisode: document.getElementById("k-episode"),
  kMonth: document.getElementById("k-month"),
  kStepReward: document.getElementById("k-step-reward"),
  kTotalReward: document.getElementById("k-total-reward"),
  kNpaRate: document.getElementById("k-npa-rate"),
  kAvgTrust: document.getElementById("k-avg-trust"),
  btnReset: document.getElementById("btn-reset"),
  btnState: document.getElementById("btn-state"),
  btnStep: document.getElementById("btn-step")
};

function setStatus(text, mode) {
  el.status.textContent = text;
  el.status.className = "pill";
  if (mode === "ok") el.status.classList.add("pill-ok");
  else if (mode === "error") el.status.classList.add("pill-error");
  else el.status.classList.add("pill-neutral");
}

function pretty(data) {
  return JSON.stringify(data, null, 2);
}

function renderAlerts(target, items) {
  target.innerHTML = "";
  (items || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    target.appendChild(li);
  });
}

function updateKpis(obs) {
  const summary = obs?.portfolio_summary || {};
  el.kEpisode.textContent = obs?.episode ?? "-";
  el.kMonth.textContent = obs?.month ?? "-";
  el.kStepReward.textContent = Number(obs?.step_reward ?? 0).toFixed(4);
  el.kTotalReward.textContent = Number(obs?.episode_reward_so_far ?? summary.cumulative_reward ?? 0).toFixed(4);
  el.kNpaRate.textContent = summary.npa_rate ?? "-";
  el.kAvgTrust.textContent = summary.avg_trust_score ?? "-";
}

function renderObservation(obs) {
  el.observation.textContent = pretty(obs);
  el.lastAction.textContent = pretty(obs?.last_action_result || {});
  renderAlerts(el.clusterAlerts, obs?.active_cluster_alerts || []);
  renderAlerts(el.ecosystemAlerts, obs?.active_ecosystem_alerts || []);
  updateKpis(obs);
}

async function callApi(path, body = null) {
  const init = {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" }
  };
  if (body) init.body = JSON.stringify(body);

  const res = await fetch(path, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

async function doReset() {
  try {
    setStatus("Resetting...", "neutral");
    const data = await callApi("/reset", {});
    renderObservation(data.observation || data);
    setStatus("Episode reset", "ok");
  } catch (err) {
    setStatus("Reset failed", "error");
    el.observation.textContent = String(err);
  }
}

async function doState() {
  try {
    setStatus("Fetching state...", "neutral");
    const data = await callApi("/state");
    renderObservation(data.observation || data);
    setStatus("State loaded", "ok");
  } catch (err) {
    setStatus("State fetch failed", "error");
    el.observation.textContent = String(err);
  }
}

async function doStep() {
  try {
    let params = {};
    try {
      params = JSON.parse(el.parameters.value || "{}");
    } catch (_) {
      throw new Error("Parameters must be valid JSON.");
    }

    const payload = {
      action_type: el.actionType.value,
      account_id: Number(el.accountId.value),
      parameters: params,
      reasoning: el.reasoning.value || ""
    };

    setStatus("Executing step...", "neutral");
    const data = await callApi("/step", payload);
    renderObservation(data.observation || data);
    setStatus("Step complete", "ok");
  } catch (err) {
    setStatus("Step failed", "error");
    el.observation.textContent = String(err);
  }
}

function initActionTypes() {
  ACTION_TYPES.forEach((action) => {
    const opt = document.createElement("option");
    opt.value = action;
    opt.textContent = action;
    el.actionType.appendChild(opt);
  });
}

function bindEvents() {
  el.btnReset.addEventListener("click", doReset);
  el.btnState.addEventListener("click", doState);
  el.btnStep.addEventListener("click", doStep);
}

initActionTypes();
bindEvents();
doState();
