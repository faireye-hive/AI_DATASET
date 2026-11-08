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


def check_record(record):
    if not isinstance(record, dict) or set(record) != {"messages"}:
        raise ValueError('esperado um objeto com apenas a chave "messages"')
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError('"messages" deve ser uma lista de mensagens')
    first = 1 if isinstance(messages[0], dict) and messages[0].get("role") == "system" else 0
    if len(messages) <= first or (len(messages) - first) % 2:
        raise ValueError("a conversa precisa conter pares completos de user e assistant")
    for i, message in enumerate(messages):
        expected = "system" if i < first else ("user", "assistant")[(i - first) % 2]
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(f"mensagem {i+1}: esperado um objeto com role e content")
        if message["role"] != expected:
            raise ValueError(f"mensagem {i+1}: papel esperado é {expected}")
        content = message["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"mensagem {i+1}: content deve ser texto não vazio")
        if CONTROL.search(content) or "\x00" in content:
            raise ValueError(f"mensagem {i+1}: marcador de controle reservado")
        try:
            content.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError(f"mensagem {i+1}: texto contém Unicode inválido") from exc
    return [{"role": msg["role"], "content": msg["content"].strip()} for msg in messages]


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
                key = tuple((msg["role"], msg["content"]) for msg in messages)
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


def prompt_ids(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False
    )


def encode_records(tokenizer, rows, limit):
    examples = []
    for location, messages in rows:
        for index, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            prompt = prompt_ids(tokenizer, messages[:index])
            answer = tokenizer.encode(msg["content"], add_special_tokens=False)
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
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error("--max-length deve ser positivo")
    try:
        splits = read_records(args.files)
        tokenizer = load_tokenizer() if args.qwen else None
        for path, rows in zip(args.files, splits):
            details = "estrutura JSONL"
            if tokenizer is not None:
                examples = encode_records(tokenizer, rows, args.max_length)
                longest = max(len(row["input_ids"]) for row in examples)
                details = f"{len(examples)} respostas; maior sequência: {longest}/{args.max_length} tokens"
            print(f"{path}: {len(rows)} conversas válidas ({details}).")
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
