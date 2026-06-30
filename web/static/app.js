const chat = document.querySelector("#chat");
const composer = document.querySelector("#composer");
const messageInput = document.querySelector("#message");
const rootDirInput = document.querySelector("#rootDir");
const maxStepsInput = document.querySelector("#maxSteps");
const sendBtn = document.querySelector("#sendBtn");
const statusPill = document.querySelector("#statusPill");
const totalTime = document.querySelector("#totalTime");
const stepCount = document.querySelector("#stepCount");
const currentNode = document.querySelector("#currentNode");
const timeline = document.querySelector("#timeline");
const approval = document.querySelector("#approval");
const approvalText = document.querySelector("#approvalText");
const approveBtn = document.querySelector("#approveBtn");
const denyBtn = document.querySelector("#denyBtn");
const memoryBtn = document.querySelector("#memoryBtn");
const memoryDialog = document.querySelector("#memoryDialog");
const memoryText = document.querySelector("#memoryText");
const closeMemory = document.querySelector("#closeMemory");

let currentJobId = null;
let eventSource = null;
let activeAgentMessage = null;

function addMessage(role, text) {
  const item = document.createElement("div");
  item.className = `message ${role}`;
  item.textContent = text;
  chat.appendChild(item);
  chat.scrollTop = chat.scrollHeight;
  return item;
}

function setStatus(text, tone = "") {
  statusPill.textContent = text;
  statusPill.dataset.tone = tone;
}

function setActiveNode(node) {
  currentNode.textContent = node || "-";
}

function formatMs(ms) {
  if (!ms) return "0ms";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function addTimeline(title, detail, elapsed, duration) {
  const item = document.createElement("div");
  item.className = "event";
  const safeDetail = typeof detail === "string" ? detail : JSON.stringify(detail, null, 2);
  item.innerHTML = `
    <strong>${title}</strong>
    <small>${formatMs(elapsed)}${duration ? ` · 节点耗时 ${formatMs(duration)}` : ""}</small>
    ${safeDetail ? `<pre>${escapeHtml(safeDetail)}</pre>` : ""}
  `;
  timeline.prepend(item);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function resetRunUi() {
  timeline.innerHTML = "";
  totalTime.textContent = "0ms";
  stepCount.textContent = "0";
  setActiveNode("start");
  approval.classList.add("hidden");
  activeAgentMessage = addMessage("agent", "正在思考...");
}

async function postJson(url, data) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "请求失败");
  }
  return payload;
}

function connectEvents(jobId) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/api/events?job_id=${encodeURIComponent(jobId)}`);
  const eventTypes = [
    "job_started",
    "node_started",
    "decision_made",
    "approval_required",
    "approval_resolved",
    "tool_finished",
    "job_completed",
    "job_failed",
  ];

  eventTypes.forEach((type) => {
    eventSource.addEventListener(type, (raw) => handleEvent(JSON.parse(raw.data)));
  });

  eventSource.onerror = () => {
    if (sendBtn.disabled) setStatus("连接中断", "warn");
  };
}

function handleEvent(event) {
  totalTime.textContent = formatMs(event.elapsed_ms);

  if (event.type === "job_started") {
    setStatus("运行中", "running");
    addTimeline("任务开始", event.user_input, event.elapsed_ms);
  }

  if (event.type === "node_started") {
    setActiveNode(event.node);
    addTimeline(`进入 ${event.label}`, event.action ? { action: event.action } : "", event.elapsed_ms);
  }

  if (event.type === "decision_made") {
    setActiveNode(event.pending_approval ? "approve" : event.action === "final" ? "final" : "tool");
    addTimeline(
      "决策完成",
      { action: event.action, action_input: event.action_input },
      event.elapsed_ms,
      event.duration_ms
    );
  }

  if (event.type === "approval_required") {
    setStatus("等待审批", "warn");
    approvalText.textContent = JSON.stringify(
      { action: event.action, action_input: event.action_input },
      null,
      2
    );
    approval.classList.remove("hidden");
    addTimeline("等待写操作审批", { action: event.action, action_input: event.action_input }, event.elapsed_ms);
  }

  if (event.type === "approval_resolved") {
    approval.classList.add("hidden");
    addTimeline(event.approved ? "审批通过" : "审批拒绝", "", event.elapsed_ms);
  }

  if (event.type === "tool_finished") {
    const steps = Number(stepCount.textContent || 0) + 1;
    stepCount.textContent = String(steps);
    const step = event.step || {};
    addTimeline(
      `工具完成：${step.action || "-"}`,
      {
        action_input: step.action_input || {},
        result: step.tool_result || "",
      },
      event.elapsed_ms,
      event.duration_ms
    );
  }

  if (event.type === "job_completed") {
    setActiveNode("final");
    setStatus("完成", "ok");
    sendBtn.disabled = false;
    messageInput.disabled = false;
    totalTime.textContent = formatMs(event.total_duration_ms);
    stepCount.textContent = String((event.steps || []).length);
    if (activeAgentMessage) activeAgentMessage.textContent = event.answer || "没有生成回答。";
    addTimeline("最终回答生成", event.answer || "", event.elapsed_ms, event.duration_ms);
    if (eventSource) eventSource.close();
  }

  if (event.type === "job_failed") {
    setStatus("失败", "danger");
    sendBtn.disabled = false;
    messageInput.disabled = false;
    if (activeAgentMessage) activeAgentMessage.textContent = `运行失败：${event.error}`;
    addTimeline("运行失败", event.error, event.elapsed_ms);
    if (eventSource) eventSource.close();
  }
}

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;

  addMessage("user", message);
  messageInput.value = "";
  sendBtn.disabled = true;
  messageInput.disabled = true;
  resetRunUi();

  try {
    const data = await postJson("/api/chat", {
      root_dir: rootDirInput.value.trim(),
      message,
      max_steps: Number(maxStepsInput.value || 10),
    });
    currentJobId = data.job_id;
    connectEvents(currentJobId);
  } catch (error) {
    setStatus("失败", "danger");
    sendBtn.disabled = false;
    messageInput.disabled = false;
    if (activeAgentMessage) activeAgentMessage.textContent = error.message;
  }
});

approveBtn.addEventListener("click", async () => {
  if (!currentJobId) return;
  await postJson("/api/approve", { job_id: currentJobId, approved: true });
});

denyBtn.addEventListener("click", async () => {
  if (!currentJobId) return;
  await postJson("/api/approve", { job_id: currentJobId, approved: false });
});

memoryBtn.addEventListener("click", async () => {
  const response = await fetch("/api/memory");
  const data = await response.json();
  memoryText.textContent = data.memory || "暂无记忆。";
  memoryDialog.showModal();
});

closeMemory.addEventListener("click", () => memoryDialog.close());

addMessage("system", "设置根目录后就可以开始对话。写文件、替换、追加等操作会在右侧弹出审批。");
setActiveNode("start");
