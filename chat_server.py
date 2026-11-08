"""Serve o editor e o chat Qwen na máquina local."""

import argparse
import json
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

from dataset import MODEL, load_tokenizer

ROOT = Path(__file__).resolve().parent
STATIC = {
    "/": ("editor.html", "text/html; charset=utf-8"),
    "/editor.html": ("editor.html", "text/html; charset=utf-8"),
    "/import-data.js": ("import-data.js", "text/javascript; charset=utf-8"),
    "/chat.html": ("chat.html", "text/html; charset=utf-8"),
    "/chat.js": ("chat.js", "text/javascript; charset=utf-8"),
}
MAX_BODY = 256 * 1024
MAX_CONTEXT = 4096
MAX_OUTPUT = 2048


def adapter_paths():
    output = ROOT / "outputs"
    if not output.is_dir() or output.is_symlink():
        return {}
    return {
        folder.name: folder
        for folder in sorted(output.iterdir())
        if not folder.is_symlink()
        and (folder / "adapter_config.json").is_file()
        and (folder / "adapter_model.safetensors").is_file()
    }


def chat_input(body):
    if not isinstance(body, dict):
        raise ValueError("Pedido de chat inválido.")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Digite uma mensagem.")
    first = int(isinstance(messages[0], dict) and messages[0].get("role") == "system")
    if (len(messages) - first) % 2 != 1:
        raise ValueError("O histórico deve terminar com uma mensagem do usuário.")
    clean = []
    for index, message in enumerate(messages):
        expected = "system" if index < first else ("user", "assistant")[(index - first) % 2]
        if not isinstance(message, dict) or message.get("role") != expected:
            raise ValueError("Os papéis das mensagens estão fora de ordem.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("As mensagens precisam conter texto.")
        clean.append({"role": expected, "content": content.strip()})

    adapter = body.get("adapter", "")
    if not isinstance(adapter, str):
        raise ValueError("Adaptador inválido.")
    path = adapter_paths().get(adapter) if adapter else None
    if adapter and path is None:
        raise ValueError("Adaptador não encontrado. Atualize a lista de modelos.")

    thinking = body.get("thinking", False)
    limit = body.get("max_new_tokens", 256)
    if type(thinking) is not bool or type(limit) is not int or not 16 <= limit <= MAX_OUTPUT:
        raise ValueError(f"Escolha um limite entre 16 e {MAX_OUTPUT} tokens.")
    return clean, path, thinking, limit


class ChatEngine:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.adapter = None
        self.lock = Lock()

    def select_model(self, adapter):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
        if adapter is not None:
            # A identidade muda quando os pesos ou a configuração são atualizados.
            identity = (
                adapter,
                (adapter / "adapter_config.json").stat().st_mtime_ns,
                (adapter / "adapter_model.safetensors").stat().st_mtime_ns,
            )
            if identity != self.adapter:
                config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
                if config.get("base_model_name_or_path") != MODEL or config.get("peft_type") != "LORA":
                    raise ValueError(f"O adaptador precisa ser LoRA para {MODEL}.")
                if self.adapter is not None:
                    self.model = self.model.base_model.unload()
                    self.adapter = None
                try:
                    self.model = PeftModel.from_pretrained(self.model, str(adapter))
                except Exception:
                    self.model = None
                    raise
                self.adapter = identity
        self.model.eval()
        return self.model

    def generate(self, messages, adapter, thinking, limit):
        import torch

        if self.tokenizer is None:
            self.tokenizer = load_tokenizer()
        tokenizer = self.tokenizer
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=thinking,
        )
        if len(ids) > MAX_CONTEXT:
            raise ValueError(f"O histórico passou de {MAX_CONTEXT} tokens. Comece uma nova conversa.")
        model = self.select_model(adapter)
        inputs = torch.tensor([ids])
        adapter_context = model.disable_adapter() if adapter is None and self.adapter else nullcontext()
        torch.manual_seed(42)
        with torch.inference_mode(), adapter_context:
            output = model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                max_new_tokens=limit, do_sample=True,
                temperature=0.6 if thinking else 0.7,
                top_p=0.95 if thinking else 0.8, top_k=20,
                pad_token_id=tokenizer.pad_token_id, use_cache=True,
            )
        generated = output[0, len(ids):].tolist()
        eos = model.generation_config.eos_token_id
        end_ids = eos if isinstance(eos, list) else [eos]
        hit_limit = len(generated) == limit and generated[-1] not in end_ids
        return self.format_reply(generated, thinking, hit_limit)

    def format_reply(self, generated, thinking, hit_limit):
        tokenizer = self.tokenizer
        opening = tokenizer.convert_tokens_to_ids("<think>")
        closing = tokenizer.convert_tokens_to_ids("</think>")
        thought = []
        answer = generated
        if closing in generated:
            boundary = generated.index(closing)
            thought, answer = generated[:boundary], generated[boundary + 1:]
        elif opening in generated:
            thought, answer = generated, []
        return {
            "answer": tokenizer.decode(answer, skip_special_tokens=True).strip(),
            "thinking": tokenizer.decode(
                [token for token in thought if token != opening], skip_special_tokens=True
            ).strip() if thinking else "",
            "hit_limit": hit_limit,
        }


ENGINE = ChatEngine()


class Handler(BaseHTTPRequestHandler):
    def allowed(self):
        host = self.headers.get("Host")
        port = self.server.server_port
        origin = self.headers.get("Origin")
        if host in (f"127.0.0.1:{port}", f"localhost:{port}") and (
            origin is None or origin == f"http://{host}"
        ):
            return True
        self.send_json(403, {"error": "Abra o chat pelo endereço local informado no terminal."})
        return False

    def send_body(self, status, content, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, status, data):
        self.send_body(status, json.dumps(data, ensure_ascii=True).encode(), "application/json")

    def do_GET(self):
        if not self.allowed():
            return
        path = urlsplit(self.path).path
        if path == "/api/adapters":
            self.send_json(200, {"adapters": list(adapter_paths())})
        elif path in STATIC:
            filename, content_type = STATIC[path]
            self.send_body(200, (ROOT / filename).read_bytes(), content_type)
        else:
            self.send_json(404, {"error": "Página não encontrada."})

    def do_POST(self):
        if not self.allowed():
            return
        if urlsplit(self.path).path != "/api/chat":
            self.send_json(404, {"error": "Rota não encontrada."})
            return
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Envie um pedido JSON.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= MAX_BODY:
                raise ValueError("O pedido está vazio ou excede o tamanho permitido.")
            args = chat_input(json.loads(self.rfile.read(length)))
        except (ValueError, UnicodeError) as error:
            self.send_json(400, {"error": str(error)})
            return

        if not ENGINE.lock.acquire(blocking=False):
            self.send_json(409, {"error": "O modelo ainda está gerando outra resposta."})
            return
        try:
            reply = ENGINE.generate(*args)
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            self.log_error("Falha no modelo: %s", error)
            self.send_json(500, {"error": f"Falha ao rodar o modelo. {error}"})
        else:
            self.send_json(200, reply)
        finally:
            ENGINE.lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Editor e chat em http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
