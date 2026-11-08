"""Validação e preparação de conversas para o Qwen3-0.6B."""

import argparse
import json
import re
from pathlib import Path

MODEL = "Qwen/Qwen3-0.6B"
MAX_LENGTH = 256
CONTROL = re.compile(r"<\|[^\s<>]*\|>|</?(?:think|tool_call|tool_response)>")


def unique_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"chave repetida: {key}")
        obj[key] = value
    return obj


def message_text(value, field, number):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"mensagem {number}: {field} deve ser texto não vazio")
    if CONTROL.search(value) or "\x00" in value:
        raise ValueError(f"mensagem {number}: marcador de controle reservado em {field}")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"mensagem {number}: texto contém Unicode inválido") from exc
    return value.strip()


def check_record(record):
    if not isinstance(record, dict) or set(record) != {"messages"}:
        raise ValueError('esperado um objeto com apenas a chave "messages"')
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError('"messages" deve ser uma lista de mensagens')
    first = 1 if isinstance(messages[0], dict) and messages[0].get("role") == "system" else 0
    if len(messages) <= first or (len(messages) - first) % 2:
        raise ValueError("a conversa precisa conter pares completos de user e assistant")
    normalized = []
    for i, message in enumerate(messages):
        expected = "system" if i < first else ("user", "assistant")[(i - first) % 2]
        required = {"role", "content"}
        allowed = required | {"reasoning_content"} if expected == "assistant" else required
        if not isinstance(message, dict) or not required <= message.keys() <= allowed:
            raise ValueError(f"mensagem {i+1}: esperado role e content; só assistant aceita reasoning_content")
        if message["role"] != expected:
            raise ValueError(f"mensagem {i+1}: papel esperado é {expected}")
        entry = {"role": expected, "content": message_text(message["content"], "content", i + 1)}
        if "reasoning_content" in message:
            entry["reasoning_content"] = message_text(message["reasoning_content"], "reasoning_content", i + 1)
        normalized.append(entry)
    return normalized


def read_records(paths):
    seen = {}
    splits = []
    for path in paths:
        rows = []
        with Path(path).open("rb") as file:
            for number, raw in enumerate(file, 1):
                location = f"{path}:{number}"
                try:
                    record = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_keys)
                    messages = check_record(record)
                except ValueError as exc:
                    raise ValueError(f"{location}: {exc}") from exc
                key = tuple((msg["role"], msg["content"], msg.get("reasoning_content")) for msg in messages)
                if key in seen:
                    raise ValueError(f"{location}: conversa duplicada de {seen[key]}")
                seen[key] = location
                rows.append((location, messages))
        if not rows:
            raise ValueError(f"{path}: arquivo vazio")
        splits.append(rows)
    return splits


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def prompt_ids(tokenizer, messages, *, thinking=False):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=thinking
    )


def encode_records(tokenizer, rows, limit, *, think=False):
    examples = []
    for location, messages in rows:
        for index, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            reasoning = msg.get("reasoning_content")
            if reasoning and not think:
                raise ValueError(f"{location}: mensagem {index+1} tem raciocínio. Use --think.")
            prompt = prompt_ids(tokenizer, messages[:index], thinking=bool(reasoning))
            answer_text = msg["content"]
            if reasoning:
                answer_text = f"<think>\n{reasoning}\n</think>\n\n{answer_text}"
            answer = tokenizer.encode(answer_text, add_special_tokens=False)
            if not answer:
                raise ValueError(f"{location}: resposta {index+1} não gerou tokens")
            target = answer + [tokenizer.eos_token_id]
            ids = prompt + target
            if len(ids) > limit:
                raise ValueError(
                    f"{location}: mensagem {index+1} com histórico usa {len(ids)} tokens "
                    f"(limite {limit}). Encurte o diálogo ou aumente --max-length."
                )
            examples.append({"input_ids": ids, "labels": [-100] * len(prompt) + target})
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--qwen", action="store_true", help="aplica o template e conta tokens")
    parser.add_argument("--think", action="store_true", help="inclui raciocínio na validação com --qwen")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error("--max-length deve ser positivo")
    if args.think and not args.qwen:
        parser.error("--think deve ser usado com --qwen")
    try:
        splits = read_records(args.files)
        tokenizer = load_tokenizer() if args.qwen else None
        for path, rows in zip(args.files, splits):
            details = "estrutura JSONL"
            if tokenizer is not None:
                examples = encode_records(tokenizer, rows, args.max_length, think=args.think)
                longest = max(len(row["input_ids"]) for row in examples)
                details = f"{len(examples)} respostas; maior sequência: {longest}/{args.max_length} tokens"
            print(f"{path}: {len(rows)} conversas válidas ({details}).")
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
