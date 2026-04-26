const ARTIFACT_FILES = [
  "training_metric.jpg",
  "Policy Loss.jpg",
  "reward_curve.png.jpg",
  "per_episode_base_vs_train.jpg",
  "GRPO Policy Loss.jpg",
  "training_reward.jpg",
];

/** Resolved from this page URL: server/ui/index.html → repo artifacts/ */
function artifactUrl(name) {
  return new URL(
    "../../artifacts/" + encodeURIComponent(name),
    window.location.href
  ).href;
}

function logUrl() {
  return new URL("training_log.txt", window.location.href).href;
}

function initGallery() {
  const el = document.getElementById("artifactGallery");
  const stamp = Date.now();
  el.innerHTML = ARTIFACT_FILES.map(
    (name) => `
    <div class="artifact-card">
      <div class="title">${name}</div>
      <img src="${artifactUrl(name)}?t=${stamp}" alt="${name}" />
    </div>
  `
  ).join("");
}

async function loadLog() {
  const pre = document.getElementById("trainingLog");
  try {
    const r = await fetch(logUrl(), { cache: "no-store" });
    pre.textContent = r.ok
      ? await r.text()
      : "Missing training_log.txt next to this page (HTTP " + r.status + ").";
  } catch (e) {
    pre.textContent = String(e);
  }
}

initGallery();
loadLog();
