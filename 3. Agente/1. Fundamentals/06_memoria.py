"""
Memória de CURTO e LONGO prazo — e o caso SEM memória (chat interativo)

Três modos para comparar:

0) SEM memória
   - Nenhum checkpointer, nenhum thread_id, nenhum JSON.
   - Cada turno é uma chamada independente ao LLM.
   - O agente não lembra NADA do que foi dito antes — nem no mesmo
     chat. Útil para ver o contraste com os modos abaixo.

1) Memória de CURTO prazo (sessão)
   - Histórico de mensagens da conversa atual.
   - Vive em RAM via `InMemorySaver`, indexado por `thread_id`.
   - Some quando o programa fecha.
   - O `thread_id` é o "id da conversa": mesmo id = mesmo histórico.

Como rodar:
    python 07_memoria.py sem      # chat sem nenhuma memória
    python 07_memoria.py curta    # chat com memória de curto prazo

Digite /sair para encerrar o chat.
Rode o modo `longa` duas vezes para ver os fatos sobreviverem ao
fechamento do programa.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict
from pydantic import BaseModel, Field

# Carrega .env da raiz do projeto para ativar o tracing do LangSmith
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

MODELO = "llama3.2"


# ---------------------------------------------------------------------------
# Construção dos agentes
# ---------------------------------------------------------------------------
def criar_agente_sem_memoria():
    """Agente SEM checkpointer e sem tools. Cada invoke é independente:
    o agente NÃO lembra do turno anterior nem no mesmo chat."""
    llm = ChatOllama(model=MODELO, temperature=0)
    return create_agent(
        model=llm,
        tools=[],
        system_prompt="Você é um assistente simpático. Responda em português.",
    )


def criar_agente_curto():
    """Agente sem tools, com checkpointer em RAM por thread_id."""
    llm = ChatOllama(model=MODELO, temperature=0)
    return create_agent(
        model=llm,
        tools=[],
        system_prompt="Você é um assistente simpático. Responda em português.",
        checkpointer=InMemorySaver(),
    )


# ---------------------------------------------------------------------------
# Funções de conversa
# ---------------------------------------------------------------------------
def conversar_sem_memoria(agente, texto: str) -> str:
    """Invoca o agente sem nenhuma config: cada chamada é independente."""
    config = {
        "tags": ["aula:07", "demo:memoria:sem"],
        "metadata": {"aula": "07", "modo": "sem"},
    }
    saida = agente.invoke({"messages": [HumanMessage(content=texto)]}, config)
    return saida["messages"][-1].content


def conversar_curto(agente, thread_id: str, texto: str) -> str:
    """Invoca o agente preservando o histórico via thread_id.
    No LangSmith, runs com o mesmo thread_id ficam agrupados em uma Thread."""
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["aula:07", "demo:memoria:curta"],
        "metadata": {"aula": "07", "modo": "curta", "thread_id": thread_id},
    }
    saida = agente.invoke({"messages": [HumanMessage(content=texto)]}, config)
    return saida["messages"][-1].content


# ---------------------------------------------------------------------------
# Demos interativas
# ---------------------------------------------------------------------------
def demo_sem_memoria() -> None:
    print("\n" + "=" * 60)
    print("  DEMO 0 — SEM memória (cada turno é independente)")
    print("=" * 60)
    print("Digite /sair para encerrar.\n")

    agente = criar_agente_sem_memoria()
    while True:
        entrada = input("Você: ").strip()
        if not entrada or entrada == "/sair":
            break
        resposta = conversar_sem_memoria(agente, entrada)
        print(f"Bot: {resposta}\n")


def demo_memoria_curta() -> None:
    print("\n" + "=" * 60)
    print("  DEMO 1 — Memória de CURTO prazo (por thread_id, em RAM)")
    print("=" * 60)
    print("Digite /sair para encerrar cada chat.\n")

    agente = criar_agente_curto()

    while True:
        entrada = input("Você: ").strip()
        if not entrada or entrada == "/sair":
            break
        resposta = conversar_curto(agente, "chat-a", entrada)
        print(f"Bot: {resposta}\n")


# ---------------------------------------------------------------------------
# Entrada via linha de comando
# ---------------------------------------------------------------------------
MODOS = {
    "sem":   demo_sem_memoria,
    "curta": demo_memoria_curta,
}

def uso() -> None:
    print("Uso: python 07_memoria.py <modo>")
    print("Modos disponíveis:")
    print("  sem    — chat sem nenhuma memória (cada turno isolado)")
    print("  curta — chat com memória de curto prazo (sessão, em RAM)")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in MODOS:
        uso()
        sys.exit(1)
    MODOS[sys.argv[1]]()
