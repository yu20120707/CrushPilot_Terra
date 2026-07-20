const api = ["localhost", "127.0.0.1"].includes(location.hostname) && location.port === "5173" ? "http://127.0.0.1:8000" : "";
let threadId = crypto.randomUUID();
let deviceId = localStorage.getItem("crushpilot-device-id");
if (!deviceId) { deviceId = crypto.randomUUID(); localStorage.setItem("crushpilot-device-id", deviceId); }
const messages = document.querySelector("#messages");
const form = document.querySelector("#composer");
const input = document.querySelector("#message");
const status = document.querySelector("#status");
const threads = document.querySelector("#threads");
const actions = document.querySelector("#actions");
const apiHeaders = { "X-Device-Id": deviceId };

function addElement(parent, tag, text, className = "") {
  const element = document.createElement(tag);
  element.textContent = text;
  element.className = className;
  parent.append(element);
  return element;
}

function message(role, content) {
  const card = document.createElement("article");
  card.className = `message ${role}`;
  addElement(card, "span", role === "user" ? "你" : "CrushPilot", "label");
  addElement(card, "div", content);
  messages.append(card);
  messages.scrollTop = messages.scrollHeight;
  return card;
}

function renderResult(result, draft) {
  draft.replaceChildren();
  addElement(draft, "span", `CrushPilot · ${result.skill}`, "label");
  const body = document.createElement("div");
  body.className = "result";
  addElement(body, "strong", result.judgement);
  addElement(body, "div", result.recommended_reply);
  const alternatives = document.createElement("div");
  alternatives.className = "alternatives";
  for (const text of result.alternatives) {
    const button = addElement(alternatives, "button", text);
    button.addEventListener("click", () => navigator.clipboard.writeText(text));
  }
  body.append(alternatives);
  if (result.warning) addElement(body, "small", result.warning);
  draft.append(body);
  actions.hidden = false;
  loadThreads();
}

async function send(text) {
  message("user", text);
  const draft = message("assistant", "正在思考…");
  status.textContent = "生成中";
  try {
    const response = await fetch(`${api}/api/v1/chat`, { method: "POST", headers: { ...apiHeaders, "Content-Type": "application/json" }, body: JSON.stringify({ thread_id: threadId, message: text }) });
    if (!response.ok) throw Error(`服务返回 ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const lines = part.split("\n");
        const name = lines.find(line => line.startsWith("event:"))?.slice(6).trim();
        const data = lines.find(line => line.startsWith("data:"))?.slice(5).trim();
        if (!data) continue;
        const payload = JSON.parse(data);
        if (name === "token") draft.lastElementChild.textContent = (draft.lastElementChild.textContent === "正在思考…" ? "" : draft.lastElementChild.textContent) + payload.text;
        if (name === "complete") renderResult(payload, draft);
        if (name === "error") throw Error(payload.message);
      }
    }
  } catch (error) {
    draft.lastElementChild.textContent = `请求失败：${error.message}`;
  } finally {
    status.textContent = "已连接";
  }
}

async function loadThreads() {
  try {
    const list = await fetch(`${api}/api/v1/threads`, { headers: apiHeaders }).then(response => response.json());
    threads.replaceChildren();
    for (const item of list) {
      const row = document.createElement("div");
      row.className = "thread-row";
      const button = addElement(row, "button", item.title);
      button.addEventListener("click", () => loadThread(item.thread_id));
      const remove = addElement(row, "button", "×", "delete-thread");
      remove.setAttribute("aria-label", `删除会话 ${item.title}`);
      remove.addEventListener("click", async () => {
        await fetch(`${api}/api/v1/threads/${encodeURIComponent(item.thread_id)}`, { method: "DELETE", headers: apiHeaders });
        if (threadId === item.thread_id) { threadId = crypto.randomUUID(); messages.replaceChildren(); actions.hidden = true; }
        loadThreads();
      });
      threads.append(row);
    }
  } catch { /* The chat page remains usable when history cannot load. */ }
}

async function loadThread(nextThreadId) {
  try {
    const response = await fetch(`${api}/api/v1/threads/${encodeURIComponent(nextThreadId)}`, { headers: apiHeaders });
    if (!response.ok) throw new Error("无法读取会话");
    const history = await response.json();
    threadId = nextThreadId;
    messages.replaceChildren();
    for (const item of history.messages) message(item.role, item.content);
    actions.hidden = true;
  } catch {
    message("assistant", "会话加载失败，请稍后重试。");
  }
}

form.addEventListener("submit", event => { event.preventDefault(); const text = input.value.trim(); if (text) { input.value = ""; send(text); } });
document.querySelector("#new-chat").addEventListener("click", () => { threadId = crypto.randomUUID(); messages.replaceChildren(); actions.hidden = true; });
actions.addEventListener("click", event => { if (event.target.tagName === "BUTTON") send(event.target.textContent); });
document.querySelectorAll("nav button").forEach(button => button.addEventListener("click", () => { document.querySelectorAll("nav button,.page").forEach(item => item.classList.remove("active")); button.classList.add("active"); document.querySelector(`#${button.dataset.page}`).classList.add("active"); }));
message("assistant", "你好，我是 CrushPilot。把对方说的话发给我，我帮你想一句自然的回复。");
loadThreads();
