"use strict";

const $ = id => document.getElementById(id);
const turns = [];
let busy = false;

function showStatus(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

async function api(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("Abra o editor com python chat_server.py para conectar ao modelo local.");
  }
  if (!response.headers.get("Content-Type")?.includes("application/json")) {
    throw new Error("Para usar o chat, inicie python chat_server.py no lugar de python -m http.server.");
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Não foi possível completar o pedido.");
  return data;
}

async function refreshAdapters() {
  $("refresh").disabled = true;
  try {
    const { adapters } = await api("/api/adapters");
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
  try {
    const variant = await api("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
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
