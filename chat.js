"use strict";

const $ = id => document.getElementById(id);
const turns = [];
let busy = false;

function showStatus(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

async function fetchLocal(path, options) {
  try {
    return await fetch(path, options);
  } catch {
    throw new Error("Abra o editor com python chat_server.py para conectar ao modelo local.");
  }
}

async function jsonResponse(response) {
  if (!response.headers.get("Content-Type")?.includes("application/json")) {
    throw new Error("Para usar o chat, inicie python chat_server.py no lugar de python -m http.server.");
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Não foi possível completar o pedido.");
  return data;
}

async function requestChat(body, onText) {
  const response = await fetchLocal("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, stream: true })
  });
  if (!response.ok || !response.headers.get("Content-Type")?.includes("application/x-ndjson")) {
    return jsonResponse(response);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) throw new Error("A conexão terminou antes do fim da geração.");
      pending += decoder.decode(value, { stream: true });
      const lines = pending.split("\n");
      pending = lines.pop();
      for (const line of lines) {
        const event = JSON.parse(line);
        if (event.type === "delta") onText(event.text);
        if (event.type === "error") throw new Error(event.error);
        if (event.type === "done") return event;
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
  }
}

async function refreshAdapters() {
  $("refresh").disabled = true;
  try {
    const { adapters } = await jsonResponse(await fetchLocal("/api/adapters"));
    const select = $("model");
    const previous = select.value;
    select.replaceChildren(new Option("Qwen3-0.6B original", ""));
    for (const name of adapters) select.add(new Option("LoRA · " + name, name));
    if (adapters.includes(previous)) select.value = previous;
    if (!busy) showStatus(adapters.length ? "" : "Os adaptadores treinados aparecem aqui quando salvos em outputs/.");
  } catch (error) {
    showStatus(error.message, true);
  } finally {
    $("refresh").disabled = false;
  }
}

function element(tag, content, className) {
  const node = document.createElement(tag);
  if (content) node.textContent = content;
  if (className) node.className = className;
  return node;
}

function draw() {
  const transcript = $("transcript");
  transcript.replaceChildren();
  if (!turns.length) {
    transcript.append(element("p", "Envie uma mensagem para começar.", "hint"));
  }
  for (const turn of turns) {
    const block = element("article", "", "turn");
    const question = element("div", "", "bubble question");
    question.append(element("strong", "Você"), element("p", turn.question, "message"));
    block.append(question);
    for (const variant of turn.variants) {
      const reply = element("div", "", "bubble answer");
      reply.append(element("strong", variant.label + (variant.think ? " · Think" : "")));
      const fallback = variant.hit_limit && variant.thinking
        ? "O limite foi atingido durante o raciocínio. Aumente os tokens e repita a pergunta."
        : "O modelo não retornou uma resposta final.";
      reply.append(element("p", variant.answer || fallback, "message"));
      if (variant.thinking) {
        const details = element("details");
        details.append(element("summary", "Ver raciocínio"), element("p", variant.thinking, "message"));
        reply.append(details);
      }
      if (variant.hit_limit && variant.answer) {
        reply.append(element("small", "O texto atingiu o limite de tokens."));
      }
      block.append(reply);
    }
    transcript.append(block);
  }
  transcript.scrollTop = transcript.scrollHeight;
  updateButtons();
}

function updateButtons() {
  $("send").disabled = busy;
  $("retry").disabled = busy || turns.length === 0;
  $("reset").disabled = busy || turns.length === 0;
  $("settings").disabled = busy;
  $("prompt").disabled = busy;
}

function messagesFor(question, precedingTurns) {
  const messages = [];
  const system = $("system").value.trim();
  if (system) messages.push({ role: "system", content: system });
  for (const turn of precedingTurns) {
    const lastAnswer = turn.variants.findLast(variant => variant.answer)?.answer;
    if (lastAnswer) {
      messages.push({ role: "user", content: turn.question }, { role: "assistant", content: lastAnswer });
    }
  }
  messages.push({ role: "user", content: question });
  return messages;
}

function withoutPartialTag(text, tag) {
  for (let length = tag.length - 1; length > 0; length--) {
    if (text.endsWith(tag.slice(0, length))) return text.slice(0, -length);
  }
  return text;
}

function liveParts(text) {
  const opening = "<think>";
  const closing = "</think>";
  const end = text.indexOf(closing);
  const start = text.indexOf(opening);
  if (end !== -1) {
    return { thought: text.slice(start === -1 ? 0 : start + opening.length, end), answer: text.slice(end + closing.length) };
  }
  if (start !== -1) {
    return { thought: withoutPartialTag(text.slice(start + opening.length), closing), answer: "" };
  }
  return { thought: "", answer: withoutPartialTag(text, opening) };
}

function startLiveReply(question, label, think, retry) {
  const transcript = $("transcript");
  if (!turns.length) transcript.replaceChildren();
  const block = retry ? transcript.lastElementChild : element("article", "", "turn");
  if (!retry) {
    const prompt = element("div", "", "bubble question");
    prompt.append(element("strong", "Você"), element("p", question, "message"));
    block.append(prompt);
    transcript.append(block);
  }
  const reply = element("div", "", "bubble answer");
  const output = element("p", "Aguardando os primeiros tokens…", "message");
  const reasoning = element("p", "", "message");
  const details = element("details");
  details.open = true;
  details.hidden = true;
  details.append(element("summary", "Raciocínio em andamento"), reasoning);
  reply.append(element("strong", label + (think ? " · Think" : "")), details, output);
  block.append(reply);
  transcript.scrollTop = transcript.scrollHeight;
  let text = "";
  return delta => {
    text += delta;
    const { thought, answer } = liveParts(text);
    details.hidden = !think || !thought.trim();
    reasoning.textContent = thought.trimStart();
    output.hidden = !answer.trim() && !details.hidden;
    output.textContent = answer.trimStart() || "Aguardando a resposta final…";
    transcript.scrollTop = transcript.scrollHeight;
  };
}

async function generate(retry) {
  if (busy || !$("composer").reportValidity()) return;
  const question = retry ? turns.at(-1).question : $("prompt").value.trim();
  if (!question) {
    showStatus("Digite uma mensagem.", true);
    $("prompt").focus();
    return;
  }
  const model = $("model");
  const label = model.selectedOptions[0].textContent;
  const think = $("think").checked;
  const body = {
    messages: messagesFor(question, retry ? turns.slice(0, -1) : turns),
    adapter: model.value,
    thinking: think,
    max_new_tokens: Number($("limit").value)
  };
  busy = true;
  updateButtons();
  showStatus("Gerando na CPU… A primeira resposta também carrega o modelo.");
  const renderDelta = startLiveReply(question, label, think, retry);
  let receivedText = false;
  try {
    const variant = await requestChat(body, delta => {
      if (!receivedText) showStatus("Gerando na CPU…");
      receivedText = true;
      renderDelta(delta);
    });
    const turn = retry ? turns.at(-1) : { question, variants: [] };
    turn.variants.push({ ...variant, label, think });
    if (!retry) {
      turns.push(turn);
      $("prompt").value = "";
    }
    showStatus(retry && variant.answer ? "A última resposta é usada no próximo turno." : "");
    draw();
  } catch (error) {
    showStatus(error.message, true);
    draw();
  } finally {
    busy = false;
    updateButtons();
    $("prompt").focus();
  }
}

$("composer").addEventListener("submit", event => {
  event.preventDefault();
  generate(false);
});
$("retry").addEventListener("click", () => generate(true));
$("reset").addEventListener("click", () => {
  turns.length = 0;
  showStatus("");
  draw();
  $("prompt").focus();
});
$("refresh").addEventListener("click", refreshAdapters);
$("think").addEventListener("change", () => {
  if ($("think").checked && Number($("limit").value) < 512) $("limit").value = 512;
});
draw();
refreshAdapters();
