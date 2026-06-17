# Agentes de IA — Guia teórico-conceitual

Este diretório acompanha as aulas práticas de agentes. Os scripts em
`1. Fundamentals`, `2. Advanced` e `3. Exercicios` usam **Ollama com
`llama3.2`** — tudo roda local, sem chave de API. O `2. Advanced/mini-projeto`
traz uma interface **Streamlit** que demonstra o sistema multi-agente em
sala de aula.

Este README cobre a parte teórica: o que é cada conceito, como ele se
materializa em código e onde aprofundar.

---

## 1. Definição de agentes

**Definição mínima.** Um **agente de IA** é um sistema em que um LLM (Large
Language Model) decide, a cada turno, qual ação tomar para atingir um
objetivo — possivelmente chamando ferramentas externas, observando os
resultados e iterando até concluir a tarefa.

Russell & Norvig descrevem um agente como qualquer coisa que **percebe** o
ambiente através de sensores e **age** sobre ele através de atuadores. No
mundo dos LLMs, "perceber" = ler mensagens/contexto/resultados de tools;
"agir" = produzir uma mensagem ou disparar uma tool.

**O loop ReAct** (Reason + Act) é a forma mais difundida hoje:

```
   ┌──────────────────────────────────────────────┐
   │  observação (mensagem do usuário ou tool)    │
   └──────────────────┬───────────────────────────┘
                      ▼
   ┌──────────────────────────────────────────────┐
   │  LLM raciocina e decide: responder OU usar   │
   │  uma tool (com quais argumentos?)            │
   └──────────────────┬───────────────────────────┘
                      ▼
   ┌──────────────────────────────────────────────┐
   │  se tool: roda código -> volta ao topo       │
   │  se resposta: encerra o turno                │
   └──────────────────────────────────────────────┘
```

**O que separa um agente de um chatbot.** Um chatbot responde texto. Um
agente **age**: chama APIs, lê arquivos, executa código, dispara workflows.
A diferença prática é o *tool calling*.

**Onde isso aparece nos exercícios.**
[1. Fundamentals/01. first_agent.py](1.%20Fundamentals/01.%20first_agent.py)
é o agente mais simples possível — só LLM, sem tools, com saída
estruturada via Pydantic.

**Referências.**
- Yao et al. (2022). [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629).
- Russell & Norvig (2020). *Artificial Intelligence: A Modern Approach*, 4ª ed., cap. 2.
- LangChain docs — [Agents conceptual guide](https://docs.langchain.com/oss/python/langchain/agents).

---

## 2. Tools (ferramentas)

**Definição.** Uma **tool** é uma função externa que o agente pode chamar.
Para o LLM, uma tool é descrita por: nome, docstring (descrição do que
faz), assinatura de argumentos com tipos. O LLM lê essa descrição e decide
quando chamar.

**Como o LLM "chama" uma função.** Ele não chama de verdade — ele **emite
um JSON** dizendo "quero chamar a função X com os argumentos Y". O framework
(LangChain/LangGraph) intercepta, roda a função em Python, e devolve o
resultado como uma `ToolMessage` para o LLM. Esse é o protocolo de **tool
calling** suportado nativamente por GPT-4, Claude, Llama 3.1+, Mistral etc.

**Boas práticas de tools.**
1. **Docstring é o contrato.** A descrição que o LLM lê é a docstring.
   Argumentos sem descrição = LLM chuta valor.
2. **Tipos importam.** Use anotações Python (`int`, `str`, `float`, `bool`)
   — o framework gera JSON Schema a partir delas.
3. **Determinismo > criatividade.** Cálculo, validação, regra de negócio
   ficam na tool. O LLM só orquestra.
4. **Trate erros DENTRO da tool.** Retorne string explicativa em vez de
   lançar exceção — o LLM consegue se recuperar.

**Onde aparece nos exercícios.**
- [1. Fundamentals/02. tool_agent.py](1.%20Fundamentals/02.%20tool_agent.py) — uma tool determinística (calculadora).
- [3. Exercicios/01_simple_tool.py](3.%20Exercicios/01_simple_tool.py) — tool de orçamento de viagem.
- [3. Exercicios/02_api_tool.py](3.%20Exercicios/02_api_tool.py) — duas tools que chamam APIs reais (Open-Meteo, Wikipedia).

**Referências.**
- OpenAI — [Function calling guide](https://platform.openai.com/docs/guides/function-calling).
- Anthropic — [Tool use overview](https://docs.claude.com/en/docs/build-with-claude/tool-use).
- LangChain — [Custom tools](https://python.langchain.com/docs/how_to/custom_tools/).

---

## 3. LangGraph

**O que é.** LangGraph é uma biblioteca da LangChain para construir agentes
como **grafos de estado**. Cada nó do grafo é uma função Python; cada aresta
descreve a transição. Em vez de "esconder" o loop do agente, você
**modela** explicitamente o fluxo.

**Por que importa.** O `create_agent` (e similares) é ótimo para o caso
ReAct simples. Mas quando o problema fica complexo — paralelismo, decisões
condicionais, human-in-the-loop, retries — você precisa de um grafo
explícito.

**Conceitos centrais.**
- **State (estado)**: um `TypedDict` (ou Pydantic) que viaja entre nós.
- **Nodes (nós)**: funções `(state) -> dict` que retornam atualizações.
- **Edges (arestas)**: ligações entre nós; podem ser **condicionais**.
- **Checkpointer**: salva o estado a cada passo — habilita memória,
  human-in-the-loop, retomada após falha.
- **`Command`**: objeto especial que combina `update={...}` e `goto=...`
  para nós que decidem o destino dinamicamente.

**Exemplo mínimo de grafo:**

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(MeuEstado)
builder.add_node("planejar", planejar)
builder.add_node("executar", executar)
builder.add_edge(START, "planejar")
builder.add_edge("planejar", "executar")
builder.add_edge("executar", END)
grafo = builder.compile()
```

**Onde aparece nos exercícios.**
- [1. Fundamentals/03_lg_nodes_and_state.py](1.%20Fundamentals/03_lg_nodes_and_state.py) — primeiro grafo, nós + estado.
- [1. Fundamentals/04_lg_parallel_execution.py](1.%20Fundamentals/04_lg_parallel_execution.py) — execução paralela de nós.
- [1. Fundamentals/05_lg_conditional_edges.py](1.%20Fundamentals/05_lg_conditional_edges.py) — arestas condicionais.

**Referências.**
- LangGraph docs — [Conceptual guide](https://langchain-ai.github.io/langgraph/concepts/).
- Harrison Chase — [Why LangGraph](https://blog.langchain.dev/langgraph/).

---

## 4. Memória

**O problema.** Um LLM é **stateless**: cada chamada começa do zero. Para
ter conversa, precisamos *re-enviar o histórico inteiro* a cada turno.
Memória é como organizamos isso.

**Três níveis de memória em agentes:**

| Tipo | Vive em | Escopo | Exemplo |
|------|---------|--------|---------|
| **Curto prazo** | RAM (lista de mensagens) | conversa atual | "qual foi a última coisa que falei?" |
| **Sessão (thread)** | checkpointer (SQLite, Redis) | mesma `thread_id`, sobrevive ao processo | "lembre desta consulta para a próxima vez que eu abrir" |
| **Longo prazo / semântica** | vector store, base de fatos | atemporal, recuperada por similaridade | "lembre que sou vegetariano" |

**No LangGraph.** O `checkpointer` é o componente que materializa
memória. Indexado por `thread_id`, ele salva o **estado completo** após
cada nó:

```python
from langgraph.checkpoint.memory import InMemorySaver
grafo = builder.compile(checkpointer=InMemorySaver())

# Mesma thread_id = mesma conversa
grafo.invoke({"messages": [...]}, {"configurable": {"thread_id": "chat-a"}})
```

**Onde aparece nos exercícios.**
- [1. Fundamentals/06_memoria.py](1.%20Fundamentals/06_memoria.py) — comparativo lado a lado: sem memória vs. memória de curto prazo.

**Referências.**
- LangGraph — [Persistence / Checkpointers](https://langchain-ai.github.io/langgraph/concepts/persistence/).
- Park et al. (2023). [*Generative Agents: Interactive Simulacra of Human Behavior*](https://arxiv.org/abs/2304.03442) — memória reflexiva de longo prazo.

---

## 5. Human-in-the-Loop (HITL)

**Motivação.** Existem decisões que **não devem ser autônomas**: aprovar
uma transferência alta, deletar dados, enviar e-mail para cliente, alterar
prontuário médico. O agente precisa **pausar** e esperar um humano decidir.

**Como funciona no LangGraph.**
1. Um nó chama `interrupt(mensagem)` — isso **pausa** a execução.
2. O estado fica salvo no `checkpointer` (obrigatório!).
3. O programa cliente recebe a pausa, mostra ao humano, coleta a decisão.
4. O cliente chama `graph.invoke(Command(resume=decisao), config)` — o
   grafo **retoma exatamente de onde parou**.

```python
def aguardar_aprovacao(state):
    decisao = interrupt(f"Aprovar R$ {state['valor']:.2f}?")
    # Só chega aqui DEPOIS de Command(resume=...)
    return {"status": "aprovado" if decisao == "a" else "rejeitado"}
```

**Padrões comuns de HITL.**
- **Approval (aprovar/rejeitar)** — caso clássico.
- **Edit (revisar e editar)** — humano corrige o draft do agente.
- **Tool review** — humano vê a chamada de tool antes de executar.

**Onde aparece nos exercícios.**
- [1. Fundamentals/07_human_in_the_loop.py](1.%20Fundamentals/07_human_in_the_loop.py) — sistema bancário com aprovação acima de R$ 5.000.

**Referências.**
- LangGraph — [Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/).
- Anthropic — [Building effective agents](https://www.anthropic.com/research/building-effective-agents) — discute quando pedir input humano.

---

## 6. MCP (Model Context Protocol)

**O que é.** MCP é um **protocolo aberto** (lançado pela Anthropic em
nov/2024) que padroniza como agentes/LLMs acessam **tools**, **dados** e
**prompts** externos. Pense como "USB-C para LLMs": em vez de cada agente
integrar com cada serviço de forma ad-hoc, todos falam o mesmo protocolo.

**Três coisas que um servidor MCP expõe.**
1. **Tools** — funções remotas (ex: `create_itinerary`).
2. **Resources** — documentos lidos por URI (ex: `travel://info/international`).
3. **Prompts** — templates parametrizados (ex: `itinerary_planner`).

**Por que importa para arquitetura.**
- Tools **interoperáveis**: o mesmo servidor MCP pode ser usado por
  Claude Desktop, Cursor, agentes LangChain, etc.
- **Linguagem-agnóstico**: o servidor pode estar em Python, TypeScript,
  Rust — o agente nem sabe.
- **Separação de responsabilidades**: a equipe A escreve o servidor (ex:
  conector com SAP); a equipe B escreve o agente que consome.

**Transports.** stdio (subprocesso local), SSE (HTTP), WebSocket.

**Onde aparece nos exercícios.**
- [2. Advanced/03_mcp_server.py](2.%20Advanced/03_mcp_server.py) — servidor MCP com tool + resource + prompt.
- [2. Advanced/04_mcp_as_tool.py](2.%20Advanced/04_mcp_as_tool.py) — cliente Ollama conectando ao servidor via `langchain-mcp-adapters`.

**Referências.**
- [Spec oficial do MCP](https://modelcontextprotocol.io/).
- Anthropic — [Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol).
- LangChain — [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters).

---

## 7. Subagentes (multi-agent)

**Por que dividir.** Um único agente com muitas tools e responsabilidades
costuma:
- **Confundir-se** sobre qual tool usar.
- Ter prompt gigante (e caro/lento).
- Ficar difícil de testar e evoluir.

A solução: dividir em **agentes especialistas** com prompts focados e
ferramentas específicas. Um **supervisor** coordena.

**Padrão Supervisor-Worker** (o mais comum):

```
              ┌────────────────────────┐
              │     SUPERVISOR         │
              │  (decide a ordem,      │
              │   delega tarefas,      │
              │   compõe resposta)     │
              └──────┬────────────┬────┘
                     │            │
            ┌────────▼──┐    ┌────▼──────┐
            │ Worker A  │    │ Worker B  │
            │ (voos)    │    │ (locais)  │
            └───────────┘    └───────────┘
```

**Como conectar.** O padrão mais simples (usado em
[2. Advanced/01_subagents.py](2.%20Advanced/01_subagents.py)) é o
**wrapper pattern**: cada worker é embrulhado como uma `@tool` do
supervisor. O supervisor "chama o worker como se fosse uma função".

```python
@tool
def consultar_agente_voo(origem: str, destino: str, data: str) -> str:
    """Aciona o subagente de Voos."""
    response = agente_voo.invoke(...)
    return response["messages"][-1].content
```

**Outros padrões.**
- **Hierárquico** — supervisor de supervisores (times).
- **Network / swarm** — agentes se chamam livremente (sem chefe).
- **Plan-and-execute** — um agente faz o plano, outro executa.

**Onde aparece nos exercícios.**
- [2. Advanced/01_subagents.py](2.%20Advanced/01_subagents.py) — **Vendedor de viagens** com `guia_turistico` (Wikipedia) + `agente_voo` (DuckDuckGo).
- [2. Advanced/mini-projeto/](2.%20Advanced/mini-projeto/) — **Agência de notícias**: agente_diretor + especialistas (esporte / tecnologia / economia) com interface Streamlit.


**Referências.**
- LangGraph — [Multi-agent systems](https://langchain-ai.github.io/langgraph/concepts/multi_agent/).
- Wu et al. (2023). [*AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*](https://arxiv.org/abs/2308.08155).
- Anthropic — [How we built our multi-agent research system](https://www.anthropic.com/engineering/built-multi-agent-research-system).

---

## 8. Próximos passos

### 8.1 DeepAgents

**DeepAgents** é uma evolução natural do padrão ReAct simples: agentes
**profundos** que combinam um **planejador hierárquico** com um sistema
de arquivos virtual, contexto compartilhado e *subagentes especializados*
sob demanda. Em vez de um único loop "pense-aja-observe" que tende a
perder o fio em tarefas longas, um deep agent mantém um plano explícito
(árvore de objetivos), delega passos a sub-agentes com **contexto isolado**
para não poluir a memória principal, e usa ferramentas de leitura/escrita
em arquivos para externalizar o "rascunho mental". O resultado são
agentes que conseguem completar workflows de **horas** (pesquisa, refator
de repos, redação de relatórios) sem se perder — algo que o ReAct puro
raramente sustenta além de algumas dezenas de passos.

**Referências.**
- LangChain — [Deep Agents (`deepagents`)](https://github.com/langchain-ai/deepagents) — implementação aberta.
- LangChain blog — [Deep Agents](https://blog.langchain.com/deep-agents/).
- Anthropic — [Claude's Research feature](https://www.anthropic.com/news/research) — produto comercial baseado nessa arquitetura.

### 8.2 Fine-tuning para agentes

**Fine-tuning** é o processo de pegar um modelo pré-treinado e *continuar
o treinamento* em um dataset específico — ajustando os pesos para que ele
aprenda um estilo, formato, ou domínio. No contexto de agentes, fine-tuning
faz sentido quando: (1) o **tool calling** do modelo base é instável (saídas
JSON quebradas, escolha errada de tool); (2) você quer um modelo **menor e
mais barato** que se comporte como um modelo grande em um domínio
restrito (ex: classificador de tickets); (3) precisa codificar **convenções
internas** que prompt sozinho não garante (formato de resposta, terminologia
da empresa, política de recusa). As técnicas mais usadas hoje são **LoRA**
e **QLoRA** (treinam apenas adaptadores pequenos, baratos em GPU), e
**RLHF/DPO** (alinhamento por preferência). Importante: na maioria dos
casos, **prompt engineering + RAG + bom tool design** resolve melhor que
fine-tuning — só treine quando esgotar essas alternativas, porque o custo
operacional (datasets, retreino, versionamento) é alto.

**Referências.**
- Hu et al. (2021). [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685).
- Dettmers et al. (2023). [*QLoRA: Efficient Finetuning of Quantized LLMs*](https://arxiv.org/abs/2305.14314).
- Rafailov et al. (2023). [*Direct Preference Optimization*](https://arxiv.org/abs/2305.18290).
- Hugging Face — [PEFT library](https://github.com/huggingface/peft) (LoRA, QLoRA, prefix tuning).
- OpenAI — [Fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning).

