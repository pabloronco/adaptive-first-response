const state = { data: null };
const NS = "http://www.w3.org/2000/svg";

const $ = (id) => document.getElementById(id);

async function api(path, method = "GET", body = null) {
  const options = { method, headers: {} };
  if (body !== null) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Request failed");
  return payload;
}

function phaseLabel(phase) {
  return {
    ready_to_plan: "READY TO PLAN",
    mission_planned: "MISSION ACTIVE",
    complete: "FIELD WORK COMPLETE",
    revealed: "TRUE EXTENT REVEALED",
  }[phase] || phase.replaceAll("_", " ").toUpperCase();
}

function fmtBelief(value) {
  return `${Math.round(value * 100)}%`;
}

function render(data) {
  state.data = data;
  $("phase-pill").textContent = phaseLabel(data.phase);
  $("initial-detection").textContent = data.incident.initial_detection;
  $("budget-left").textContent = `${data.resources.remaining_budget} / ${data.resources.initial_budget}`;
  $("teams").textContent = data.resources.teams;
  $("round").textContent = data.resources.round;
  $("truth-status").textContent = data.revealed ? "TRUE EXTENT REVEALED" : "HIDDEN EXTENT LOCKED";
  $("incident-subtitle").textContent = data.revealed
    ? "Evaluator view enabled. Compare inferred mission history against hidden truth."
    : "True extent hidden. Detection imperfect. Resources limited.";

  const banner = $("mission-updated-banner");
  banner.classList.toggle("hidden", !data.mission_changed || data.revealed);

  renderGraph(data);
  renderMission(data);
  renderEvidence(data);
  renderTimeline(data.events);
  renderControls(data);
}

function renderControls(data) {
  const primary = $("primary-btn");
  const reveal = $("reveal-btn");
  primary.disabled = false;

  if (data.can_plan) {
    primary.textContent = data.resources.round === 0 ? "Generate Mission 1" : "Generate next mission";
    primary.dataset.action = "plan";
  } else if (data.can_execute) {
    primary.textContent = `Run Mission ${data.resources.round + 1}`;
    primary.dataset.action = "execute";
  } else if (data.can_reveal) {
    primary.textContent = "Field work complete";
    primary.dataset.action = "none";
    primary.disabled = true;
  } else {
    primary.textContent = "Incident revealed";
    primary.dataset.action = "none";
    primary.disabled = true;
  }

  reveal.disabled = !data.can_reveal;
}

function renderMission(data) {
  const body = $("mission-body");
  const plannerChip = $("planner-chip");
  const mission = data.mission;
  if (!mission) {
    plannerChip.textContent = "NO PLAN";
    body.className = "mission-body empty-state";
    body.textContent = data.can_reveal ? "Field budget exhausted. Reveal is now available." : "No mission planned yet.";
    return;
  }

  plannerChip.textContent = mission.planner;
  body.className = "mission-body";
  body.innerHTML = mission.allocations.map((a, idx) => `
    <div class="mission-allocation">
      <div class="allocation-top">
        <span class="allocation-site">${a.site_id}</span>
        <span class="allocation-effort">${a.effort_units} checks</span>
      </div>
      <div class="meta-line">Priority ${idx + 1} · observable-state decision</div>
    </div>
  `).join("");
}

function renderEvidence(data) {
  const body = $("evidence-body");
  const last = data.last_round;
  if (!last) {
    body.className = "evidence-body empty-state";
    body.textContent = "Awaiting first field return.";
    return;
  }

  body.className = "evidence-body";
  body.innerHTML = last.observations.map(obs => {
    const afterWidth = Math.max(2, Math.round(obs.belief_after * 100));
    const outcome = obs.detection ? "DETECTION" : "NO DETECTION";
    return `
      <div class="evidence-item">
        <div class="evidence-top">
          <span class="evidence-site">${obs.site_id}</span>
          <span class="evidence-outcome">${outcome}</span>
        </div>
        <div class="meta-line">${obs.effort} checks · evidence updates occupancy belief, not truth</div>
        <div class="belief-shift">
          <span>${fmtBelief(obs.belief_before)}</span>
          <div class="belief-track"><span style="width:${afterWidth}%"></span></div>
          <span>${fmtBelief(obs.belief_after)}</span>
        </div>
      </div>
    `;
  }).join("");
}

function renderTimeline(events) {
  const timeline = $("timeline");
  timeline.innerHTML = events.map(event => `
    <div class="timeline-event">
      <div class="event-kind">Round ${event.round} · ${event.kind}</div>
      <div class="event-title">${event.title}</div>
      <div class="event-detail">${event.detail}</div>
    </div>
  `).join("");
  timeline.scrollLeft = timeline.scrollWidth;
}

function renderGraph(data) {
  const svg = $("graph");
  svg.innerHTML = "";
  const nodes = data.nodes;
  const xs = nodes.map(n => n.x);
  const ys = nodes.map(n => n.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const padX = 58, padY = 90;
  const width = 1000, height = 500;

  const sx = x => padX + ((x - minX) / Math.max(1e-6, maxX - minX)) * (width - 2 * padX);
  const sy = y => height - padY - ((y - minY) / Math.max(1e-6, maxY - minY)) * (height - 2 * padY);
  const byId = Object.fromEntries(nodes.map(n => [n.id, n]));

  const coastPath = document.createElementNS(NS, "polyline");
  coastPath.setAttribute("points", nodes.map(n => `${sx(n.x)},${sy(n.y)}`).join(" "));
  coastPath.setAttribute("class", "coast-shadow");
  svg.appendChild(coastPath);

  data.edges.forEach(edge => {
    const a = byId[edge.src], b = byId[edge.dst];
    if (!a || !b) return;
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", sx(a.x)); line.setAttribute("y1", sy(a.y));
    line.setAttribute("x2", sx(b.x)); line.setAttribute("y2", sy(b.y));
    line.setAttribute("class", "graph-edge");
    svg.appendChild(line);
  });

  nodes.forEach(node => {
    const cx = sx(node.x), cy = sy(node.y);
    const group = document.createElementNS(NS, "g");
    group.setAttribute("data-site", node.id);
    group.style.cursor = "pointer";

    if (node.mission_effort > 0) {
      const halo = document.createElementNS(NS, "circle");
      halo.setAttribute("cx", cx); halo.setAttribute("cy", cy); halo.setAttribute("r", 22);
      halo.setAttribute("class", "node-mission-halo");
      group.appendChild(halo);
    }

    const ringBg = document.createElementNS(NS, "circle");
    ringBg.setAttribute("cx", cx); ringBg.setAttribute("cy", cy); ringBg.setAttribute("r", 17);
    ringBg.setAttribute("fill", "none");
    ringBg.setAttribute("stroke", "rgba(120,150,140,.18)");
    ringBg.setAttribute("stroke-width", "4");
    group.appendChild(ringBg);

    const circumference = 2 * Math.PI * 17;
    const ring = document.createElementNS(NS, "circle");
    ring.setAttribute("cx", cx); ring.setAttribute("cy", cy); ring.setAttribute("r", 17);
    ring.setAttribute("class", "node-belief-ring");
    ring.setAttribute("stroke-dasharray", `${circumference * node.belief} ${circumference}`);
    group.appendChild(ring);

    const core = document.createElementNS(NS, "circle");
    core.setAttribute("cx", cx); core.setAttribute("cy", cy); core.setAttribute("r", 11);
    const classes = ["node-core"];
    if (node.status === "confirmed_detection" || node.detections > 0) classes.push("confirmed");
    else if (node.effort > 0) classes.push("surveyed");
    if (node.mission_effort > 0) classes.push("mission");
    if (data.revealed && node.true_occupied) classes.push("true-occupied");
    core.setAttribute("class", classes.join(" "));
    group.appendChild(core);

    const label = document.createElementNS(NS, "text");
    label.setAttribute("x", cx); label.setAttribute("y", cy + 36);
    label.setAttribute("text-anchor", "middle"); label.setAttribute("class", "node-label");
    label.textContent = node.id.replace("site_", "");
    group.appendChild(label);

    const pLabel = document.createElementNS(NS, "text");
    pLabel.setAttribute("x", cx); pLabel.setAttribute("y", cy - 27);
    pLabel.setAttribute("text-anchor", "middle"); pLabel.setAttribute("class", "node-belief-label");
    pLabel.textContent = fmtBelief(node.belief);
    group.appendChild(pLabel);

    group.addEventListener("mouseenter", e => showTooltip(e, node, data.revealed));
    group.addEventListener("mousemove", e => moveTooltip(e));
    group.addEventListener("mouseleave", hideTooltip);
    svg.appendChild(group);
  });
}

function showTooltip(event, node, revealed) {
  const tip = $("node-tooltip");
  const truth = revealed ? `<div>True occupancy: <b>${node.true_occupied ? "PRESENT" : "ABSENT"}</b></div>` : "";
  tip.innerHTML = `
    <strong>${node.id}</strong>
    <div>Belief: ${fmtBelief(node.belief)}</div>
    <div>Uncertainty: ${fmtBelief(node.uncertainty)}</div>
    <div>Observed effort: ${node.effort}</div>
    <div>Detections: ${node.detections}</div>
    <div>Frontier: ${node.frontier ? "yes" : "no"}</div>
    ${truth}
  `;
  tip.classList.remove("hidden");
  moveTooltip(event);
}

function moveTooltip(event) {
  const wrap = document.querySelector(".graph-wrap");
  const rect = wrap.getBoundingClientRect();
  const tip = $("node-tooltip");
  tip.style.left = `${event.clientX - rect.left + 12}px`;
  tip.style.top = `${event.clientY - rect.top + 12}px`;
}

function hideTooltip() {
  $("node-tooltip").classList.add("hidden");
}

async function refresh() {
  render(await api("/api/state"));
}

$("primary-btn").addEventListener("click", async () => {
  const action = $("primary-btn").dataset.action;
  if (!action || action === "none") return;
  try {
    render(await api(`/api/${action}`, "POST"));
  } catch (error) {
    alert(error.message);
  }
});

$("reveal-btn").addEventListener("click", async () => {
  try {
    render(await api("/api/reveal", "POST"));
  } catch (error) {
    alert(error.message);
  }
});

$("reset-btn").addEventListener("click", async () => {
  try {
    render(await api("/api/reset", "POST", { seed: null }));
  } catch (error) {
    alert(error.message);
  }
});

refresh().catch(error => {
  console.error(error);
  alert(`Mission Control failed to initialize: ${error.message}`);
});
