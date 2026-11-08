# AI Dataset

Editor local e ferramentas para escrever conversas em português e adaptar o **Qwen3-0.6B** com LoRA. O dataset começa vazio e você escreve as respostas que quer ensinar. Para um futuro dataset de código, use outra pasta, como `data/code/`, e um adaptador separado.

## 1. Escreva suas conversas

No Linux, clone o repositório e sirva o editor localmente com Python 3.10 a 3.13. O editor só usa o navegador e não precisa instalar as bibliotecas de treino.

```bash
git clone https://github.com/faireye-hive/AI_DATASET.git
cd AI_DATASET
mkdir -p data/chat
python3 -m http.server --bind 127.0.0.1 8765
```

Abra [o editor](http://127.0.0.1:8765/editor.html). Escolha o conjunto, clique em **Nova conversa** e preencha **Usuário** e **Assistente**. Use **Adicionar turno** para ensinar como a resposta muda com o histórico. Se fizer sentido, preencha a instrução de sistema com o contexto ou a regra que também estará presente quando você usar o modelo.

O navegador salva rascunhos no endereço acima. Exporte pelo botão **Baixar JSONL**, mova os arquivos baixados para `data/chat/` e guarde uma cópia. Use o mesmo endereço e porta para voltar aos rascunhos. O Git ignora `data/` e `outputs/`.

- **Treino** (`treino.jsonl`) contém as conversas que o modelo verá no ajuste.
- **Validação** (`validacao.jsonl`) é opcional e mede a perda do modelo em exemplos separados.
- **Teste** (`teste.jsonl`) é opcional e guarda conversas inéditas para comparar as respostas antes e depois. Mantenha reformulações quase idênticas no mesmo conjunto.

### Importar conversas de outro arquivo

Clique em **Importar arquivo** e escolha um CSV, JSON ou JSONL em UTF-8. O CSV precisa ter cabeçalho e pode usar vírgula, ponto e vírgula ou tabulação como separador. O JSON aceita um objeto, uma lista de registros ou um objeto com listas nomeadas, como `{"perguntas":[...],"dialogos":[...]}`. No JSONL, cada linha é um registro.

Confira a prévia e escolha a tabela de origem, quando houver mais de uma. Em **Adicionar ao conjunto**, selecione Treino, Validação ou Teste. Depois, associe as colunas a **Usuário**, **Assistente** e, se houver, **Sistema**. Por exemplo, use a coluna `pergunta` para Usuário e `resposta` para Assistente. Cada linha mapeada cria uma conversa com um turno. Se o JSON ou JSONL já tiver `messages` com os papéis definidos, use esse campo para manter todos os turnos.

O editor mostra linhas inválidas antes de importar e ignora conversas idênticas que já estão na coleção ou que se repetem no arquivo. As conversas são acrescentadas ao conjunto escolhido e ficam disponíveis para edição e exportação em JSONL.

Escreva a resposta completa no tom desejado. Varie os assuntos e inclua situações que deixem o comportamento claro, como perguntas vagas em que o assistente pede um detalhe, correções em turnos seguintes e casos em que ele admite que não sabe. Para começar, escreva algumas conversas curtas e faça o teste de dois passos abaixo antes de ampliar a coleção.

## 2. Formato exato

Cada linha do JSONL é um objeto independente em UTF-8, sem BOM, vírgula entre linhas ou linhas em branco. O exemplo abaixo mostra uma única conversa com dois turnos.

```jsonl
{"messages":[{"role":"system","content":"Converse em português brasileiro, de forma breve e natural."},{"role":"user","content":"Quero voltar a ler."},{"role":"assistant","content":"Que tipo de história você costuma gostar?"},{"role":"user","content":"Mistério."},{"role":"assistant","content":"Comece com um conto policial curto. Assim você consegue terminar uma história sem precisar separar muito tempo."}]}
```

O único campo externo é `messages`. Cada mensagem tem `role` e `content` com texto não vazio. Use `system` só no início, se precisar, seguido de um ou mais pares `user` → `assistant`. Deixe o assistente fechar a conversa. Dentro de uma string JSON, represente quebras de linha como `\n` e aspas como `\"`. O editor faz isso automaticamente. O tokenizer acrescenta os marcadores do Qwen, como `<|im_start|>`.

### Exemplos com Think

Abra **Raciocínio / Think (opcional)** em um turno do editor. Preencha os passos e escreva a resposta final no campo **Assistente**. O JSONL guarda os passos no campo `reasoning_content` da mensagem do assistente. Também dá para mapear uma coluna de raciocínio ao importar arquivos. Deixe as tags `<think>` por conta do script.

```jsonl
{"messages":[{"role":"user","content":"Tenho 36 moedas e gastei 25%. Quanto sobrou?"},{"role":"assistant","reasoning_content":"25% é um quarto de 36, ou seja, 9. Então 36 − 9 = 27.","content":"Sobraram 27 moedas."}]}
```

Para testar esse formato, guarde o JSONL exportado em `data/chat/treino-think.jsonl` e execute os comandos no ambiente virtual:

```bash
python dataset.py --qwen --think --max-length 512 data/chat/treino-think.jsonl
python lora.py train --think --train data/chat/treino-think.jsonl --max-length 512 --steps 2 --accumulation 1 --output outputs/smoke-think
```

O teste de dois passos verifica o fluxo. Para treinar por uma época, use `--epochs 1` no lugar de `--steps 2` e escolha uma nova pasta de saída. Com `--think`, as respostas que têm raciocínio são treinadas com Think. As respostas sem esse campo usam o formato direto. No laboratório, selecione o adaptador e ligue **Think** para testar com perguntas novas.

## 3. Valide os arquivos

Em outro terminal, na raiz do projeto, você pode checar a estrutura com Python puro. Passe todos os arquivos que você exportou em uma só chamada para encontrar conversas duplicadas entre conjuntos.

```bash
python3 dataset.py data/chat/*.jsonl
```

Para validar o tokenizer, crie o ambiente virtual e instale o PyTorch para CPU e as dependências.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python dataset.py --qwen --max-length 256 data/chat/*.jsonl
```

Se o Kubuntu indicar que falta o módulo de ambiente virtual, instale `python3-venv` e repita a criação. A primeira validação com `--qwen` baixa o tokenizer. Ela informa quantas respostas podem ser treinadas e mostra a maior sequência. Se a sequência passar do limite, o validador aponta o arquivo, a linha e a mensagem. Encurte a conversa ou aumente o mesmo limite nos comandos de validação e treino. Sequências maiores custam mais memória e tempo na CPU.

## 4. Treine e compare

Faça primeiro um teste curto. O comando baixa os pesos do Qwen3-0.6B do Hugging Face e faz dois passos de ajuste. O número de passos aqui serve para verificar o processo.

```bash
python lora.py train --steps 2 --accumulation 1 --output outputs/smoke
```

Para o primeiro treino completo, use uma época. Se você exportou validação, inclua o argumento indicado na segunda linha abaixo. O script treina em CPU, com lote de uma sequência e acúmulo de quatro lotes por passo. Ele calcula a perda somente sobre cada resposta do assistente e seu marcador de fim, usando os turnos anteriores como contexto.

```bash
python lora.py train --epochs 1 --max-length 256 --output outputs/chat-lora
# Com validação, use este comando no lugar do anterior
python lora.py train --epochs 1 --max-length 256 --validation data/chat/validacao.jsonl --output outputs/chat-lora
```

Compare com uma pergunta do seu conjunto de teste. Se usou instrução de sistema, passe a mesma com `--system` nas duas chamadas. O comando `chat` testa uma pergunta por vez.

```bash
python lora.py chat --prompt "Quero voltar a desenhar. Por onde começo?"
python lora.py chat --adapter outputs/chat-lora --prompt "Quero voltar a desenhar. Por onde começo?"
```

Em `outputs/chat-lora/`, o treino salva os pesos e a configuração do adaptador. A inferência carrega o Qwen3-0.6B e aplica esse adaptador. Para medir o tempo no seu computador, use o teste de dois passos como referência.

## 5. Converse com o modelo no navegador

Na raiz do projeto, ative o ambiente virtual com as dependências do treino e inicie o servidor do chat. Se o servidor do editor estiver usando a porta 8765, encerre-o com `Ctrl+C` primeiro. Use o mesmo endereço e porta do editor para acessar seus rascunhos.

```bash
env -u PYTHONPATH -u PYTHONHOME .venv/bin/python chat_server.py
```

Abra o [Laboratório de chat](http://127.0.0.1:8765/chat.html). Selecione o Qwen3-0.6B original ou um adaptador salvo diretamente em uma subpasta de `outputs/`. Depois de treinar outro adaptador, clique em **Atualizar adaptadores**.

Para comparar, envie uma pergunta, troque o modelo ou o modo **Think** e clique em **Repetir última pergunta**. As versões ficam juntas no histórico, e a última resposta com texto é usada no turno seguinte. O chat mostra o texto conforme ele é gerado. Com Think ligado, o raciocínio aparece separado da resposta final e pode ser reaberto depois. O limite de tokens inclui os dois trechos.

O servidor roda na CPU e só escuta em `127.0.0.1`. A primeira pergunta carrega o modelo. Use **Nova conversa** para limpar o histórico do chat. Se aparecer um erro de `PIL._imaging`, reinicie o servidor com o comando acima para usar o ambiente virtual sem caminhos Python herdados do terminal.

Referências oficiais · [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) · [Formato conversacional](https://huggingface.co/docs/trl/dataset_formats) · [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora)
