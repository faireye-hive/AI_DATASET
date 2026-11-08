"use strict";

function isImportObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function csvSeparator(text) {
  const counts = new Map([[",", 0], [";", 0], ["\t", 0]]);
  let quoted = false;
  for (const char of text) {
    if (char === '"') quoted = !quoted;
    if (quoted) continue;
    if (char === "\r" || char === "\n") break;
    if (counts.has(char)) counts.set(char, counts.get(char) + 1);
  }
  return [...counts.keys()].reduce((best, char) => counts.get(char) > counts.get(best) ? char : best);
}

function parseCsv(text) {
  const leadingBlankLines = text.match(/^(?:[ \t]*(?:\r\n|\n|\r))+/)?.[0] ?? "";
  text = text.slice(leadingBlankLines.length);
  const separator = csvSeparator(text);
  const rows = [];
  let cells = [];
  let cell = "";
  let state = "plain";
  let line = 1 + (leadingBlankLines.match(/\r\n|\n|\r/g)?.length ?? 0);
  let start = line;

  function finishCell() {
    cells.push(cell);
    cell = "";
    state = "plain";
  }

  function finishRow() {
    finishCell();
    if (cells.length > 1 || cells[0].trim()) rows.push({ line: start, cells });
    cells = [];
  }

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === "\r" || char === "\n") {
      if (char === "\r" && text[i + 1] === "\n") i++;
      if (state === "quoted") cell += "\n";
      else finishRow();
      line++;
      if (state !== "quoted") start = line;
    } else if (state === "quoted") {
      if (char === '"' && text[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (char === '"') {
        state = "closed";
      } else {
        cell += char;
      }
    } else if (char === separator) {
      finishCell();
    } else if (state === "closed") {
      if (char !== " " && char !== "\t") {
        throw new Error("Linha " + line + ": texto inesperado depois das aspas.");
      }
    } else if (char === '"') {
      if (cell.trim()) throw new Error("Linha " + line + ": aspas em um campo sem delimitação.");
      cell = "";
      state = "quoted";
    } else {
      cell += char;
    }
  }
  if (state === "quoted") throw new Error("Linha " + start + ": campo com aspas sem fechar.");
  finishRow();
  if (!rows.length) throw new Error("O CSV está vazio.");
  const header = rows.shift();
  const columns = header.cells.map(name => name.trim());
  if (columns.some(name => !name) || new Set(columns).size !== columns.length) {
    throw new Error("O cabeçalho do CSV precisa ter nomes preenchidos e sem repetição.");
  }
  return {
    name: "CSV",
    columns,
    rows: rows.map(({ line, cells }) => ({
      location: "Linha " + line,
      value: Object.fromEntries(columns.map((column, index) => [column, cells[index]])),
      error: cells.length === columns.length ? "" : "o número de campos difere do cabeçalho."
    }))
  };
}

function jsonTable(name, rows) {
  const columns = new Set();
  for (const row of rows) {
    if (isImportObject(row.value)) {
      for (const column of Object.keys(row.value)) columns.add(column);
    }
  }
  return { name, columns: [...columns], rows };
}

function readImport(text, filename) {
  const extension = filename.split(".").at(-1).toLowerCase();
  if (!text.trim()) throw new Error("O arquivo está vazio.");
  if (extension === "csv") return [parseCsv(text)];
  if (extension === "jsonl" || extension === "ndjson" || extension === "txt") {
    const rows = [];
    text.split(/\r\n|\n|\r/).forEach((line, index) => {
      if (!line.trim()) return;
      const location = "Linha " + (index + 1);
      try {
        rows.push({ location, value: JSON.parse(line) });
      } catch {
        rows.push({ location, error: "JSON inválido." });
      }
    });
    return [jsonTable("JSONL", rows)];
  }
  if (extension !== "json") throw new Error("Escolha um arquivo CSV, JSON ou JSONL.");

  let value;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error("O arquivo JSON está inválido. Se cada linha for um objeto, use a extensão .jsonl.");
  }
  if (!Array.isArray(value) && !isImportObject(value)) {
    throw new Error("O JSON precisa conter um objeto ou uma lista de registros.");
  }
  const wrap = values => values.map((record, index) => ({ location: "Registro " + (index + 1), value: record }));
  if (isImportObject(value) && !Array.isArray(value.messages)) {
    const tables = Object.entries(value).filter(([, rows]) =>
      Array.isArray(rows) && rows.some(isImportObject)
    );
    if (tables.length) return tables.map(([name, rows]) => jsonTable(name, wrap(rows)));
  }
  return [jsonTable("JSON", wrap(Array.isArray(value) ? value : [value]))];
}

const roleColumnNames = {
  user: ["user", "usuario", "prompt", "input", "pergunta", "question", "instruction", "instrucao"],
  assistant: ["assistant", "assistente", "resposta", "answer", "completion", "response", "output"],
  system: ["system", "sistema", "context", "contexto", "system_prompt"]
};

function suggestColumn(columns, role) {
  const normalize = column => column.trim().toLowerCase().normalize("NFD")
    .replace(/\p{M}/gu, "").replace(/\s+/g, "_");
  const normalized = columns.map(normalize);
  for (const name of roleColumnNames[role]) {
    const index = normalized.indexOf(name);
    if (index !== -1) return index;
  }
  return -1;
}

function mappedRecord(row, mapping) {
  if (row.error) throw new Error(row.error);
  if (!isImportObject(row.value)) throw new Error("esperado um objeto com os campos da conversa.");
  if (mapping === null) return { messages: row.value.messages };

  const messages = [];
  for (const role of ["system", "user", "assistant"]) {
    const column = mapping[role];
    if (column === undefined) continue;
    const value = Object.hasOwn(row.value, column) ? row.value[column] : null;
    if (role === "system" && (value == null || (typeof value === "string" && !value.trim()))) continue;
    if (typeof value !== "string" || !value.trim()) {
      throw new Error('a coluna "' + column + '" precisa conter texto.');
    }
    messages.push({ role, content: value });
  }
  return { messages };
}
