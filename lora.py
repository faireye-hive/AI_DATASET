"""Treina e compara um adaptador LoRA de conversa na CPU."""

import argparse
from pathlib import Path

from dataset import MAX_LENGTH, MODEL, encode_records, load_tokenizer, prompt_ids, read_records


def train(args):
    # Valide os dados antes de carregar o modelo.
    paths = [args.train] + ([args.validation] if args.validation else [])
    splits = read_records(paths)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("use uma pasta de saída nova ou vazia")
    tokenizer = load_tokenizer()
    encoded = [encode_records(tokenizer, rows, args.max_length, think=args.think) for rows in splits]

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

    set_seed(42)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    ))
    model.print_trainable_parameters()

    def collate(rows):
        width = max(len(row["input_ids"]) for row in rows)
        ids, masks, labels = [], [], []
        for row in rows:
            length = len(row["input_ids"])
            padding = width - length
            ids.append(row["input_ids"] + [tokenizer.pad_token_id] * padding)
            masks.append([1] * length + [0] * padding)
            labels.append(row["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
            "labels": torch.tensor(labels),
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output),
            use_cpu=True,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.accumulation,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=1e-4,
            num_train_epochs=args.epochs,
            max_steps=args.steps,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            seed=42,
            label_names=["labels"],
        ),
        train_dataset=encoded[0],
        eval_dataset=encoded[1] if args.validation else None,
        data_collator=collate,
    )
    trainer.train()
    if args.validation:
        trainer.evaluate()
    model.save_pretrained(args.output)
    print(f"Adaptador salvo em {args.output}")


def chat(args):
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    ids = torch.tensor([prompt_ids(tokenizer, messages)])
    model.eval()
    torch.manual_seed(42)
    with torch.inference_mode():
        output = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=args.max_new_tokens, do_sample=True,
            temperature=0.7, top_p=0.8, top_k=20,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
    print(tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip())


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("use um inteiro positivo")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--train", type=Path, default=Path("data/chat/treino.jsonl"))
    training.add_argument("--validation", type=Path)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--max-length", type=positive_int, default=MAX_LENGTH)
    training.add_argument("--think", action="store_true", help="treina raciocínio nas respostas com reasoning_content")
    training.add_argument("--epochs", type=positive_int, default=1)
    training.add_argument("--steps", type=positive_int, default=-1)
    training.add_argument("--accumulation", type=positive_int, default=4)

    inference = commands.add_parser("chat")
    inference.add_argument("--prompt", required=True)
    inference.add_argument("--system")
    inference.add_argument("--adapter", type=Path)
    inference.add_argument("--max-new-tokens", type=positive_int, default=96)
    args = parser.parse_args()
    try:
        (train if args.command == "train" else chat)(args)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
