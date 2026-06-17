"""
Exercício 03 — Sanitização de base de dados com LangGraph

Objetivo:
    Mostrar que LangGraph NÃO é só para "agentes com LLM". O grafo de
    estado também é uma forma elegante de modelar PIPELINES de dados,
    com:
        - estado tipado (TypedDict)
        - nós isolados e testáveis
        - logs por nó
        - rastreabilidade no LangSmith (se .env estiver configurado)

Pipeline a construir:

    START
      │
      ▼
    [ler_csv]          lê `cadastro.csv` (ISO-8859-1, separador `;`)
      │
      ▼
    [tratar_nomes]     cria `nome_tratado` chamando um LLM por linha:
      │                LLM aplica as regras (sem acento, trim,
      │                espaços internos viram `_`, minúsculo)
      ▼
    [validar_emails]   cria `email_valido` ("sim"/"nao") chamando um LLM
      │                por linha (contém '@' E termina em '.com'/'.com.br')
      ▼
    [salvar_csv]       grava `cadastro_nova.csv` no MESMO diretório
      │
      ▼
     END

Regras do enunciado:
    1. Ler `cadastro.csv` (encoding ISO-8859-1 / latin-1, separador `;`).
    2. Criar coluna `nome_tratado` aplicando as regras de SYSTEM_TRATAR_NOME.
    3. Criar coluna `email_valido` com "sim"/"nao" segundo SYSTEM_VALIDAR_EMAIL.
    4. Salvar como `cadastro_nova.csv` no mesmo diretório (UTF-8, `;`).

Documentação de referência:
    - LangGraph (StateGraph):
        https://langchain-ai.github.io/langgraph/
    - csv (DictReader / DictWriter):
        https://docs.python.org/3/library/csv.html
    - LangChain Ollama (ChatOllama):
        https://python.langchain.com/docs/integrations/chat/ollama/

Estrutura do script:
    1) Imports (csv, pathlib, typing, langgraph, langchain_ollama, ...).
    2) Constante MODELO e instância `llm`.
    3) Definir a TypedDict `EstadoSanitizacao` com:
         - caminho_in, caminho_out (str)
         - linhas (List[dict])
         - total (int)
         - logs (Annotated[List[str], operator.add])   <- importante!
    4) System prompts (já prontos abaixo) e duas funções wrapper que
       conversam com o LLM:
         - tratar_nome_llm(nome) -> str
         - email_e_valido_llm(email) -> bool
    5) Implementar os 4 NÓS — cada um recebe `state` e devolve um dict
       PARCIAL com as atualizações:
         - ler_csv(state)
         - tratar_nomes(state)
         - validar_emails(state)
         - salvar_csv(state)
    6) Função `construir_grafo()` que monta o StateGraph
       (START -> ler_csv -> tratar_nomes -> validar_emails -> salvar_csv -> END).
    7) Função `main()` que invoca o grafo e imprime um resumo + preview.

Como rodar:
    python "3. Agente/3. Exercicios/03_data_sanitizer_langgraph_aluno.py"
"""
from __future__ import annotations
import csv
import operator
from pathlib import Path
from typing import Annotated, List, TypedDict

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph


MODELO = "llama3.2"
llm = ChatOllama(model=MODELO, temperature=0)


# ---------------------------------------------------------------------------
# 1. ESTADO COMPARTILHADO ENTRE OS NÓS
#
#    Dica: `Annotated[List[str], operator.add]` faz o LangGraph CONCATENAR
#    listas vindas de nós diferentes em vez de sobrescrever — perfeito
#    para acumular logs ao longo do pipeline.
#
#    Campos esperados:
#       - caminho_in: str
#       - caminho_out: str
#       - linhas: List[dict]
#       - total: int
#       - logs: Annotated[List[str], operator.add]
# ---------------------------------------------------------------------------
class EstadoSanitizacao(TypedDict):
    ## aqui coloque seu código ###
    pass  # remova este `pass` ao declarar os campos acima


# ---------------------------------------------------------------------------
# 2. PROMPTS DO LLM
#    Os system prompts abaixo já estão prontos — eles documentam as regras
#    de negócio. Você precisa implementar as duas funções wrapper que
#    chamam o LLM com esses prompts.
# ---------------------------------------------------------------------------
SYSTEM_TRATAR_NOME = (
    "Você é um normalizador de nomes próprios.\n"
    "Receberá UM nome e deve devolver o nome normalizado seguindo as regras:\n"
    "  1) remover acentos (ex.: ã→a, ç→c, é→e)\n"
    "  2) remover espaços no início e no fim da string\n"
    "  3) para os espaços no meio da string substituir por _ "
    "(ex: joao da silva -> joao_da_silva)\n"
    "  4) converter TUDO para minúsculas\n"
    "Responda APENAS com o nome normalizado, sem aspas, sem explicações, "
    "sem pontuação extra, sem prefixos como 'Resposta:'.\n"
    "Se o nome vier vazio, responda com uma linha vazia."
)

SYSTEM_VALIDAR_EMAIL = (
    "Você é um validador de e-mails.\n"
    "Receberá um endereço de e-mail e deve responder se ele é válido.:\n"
    "Um email deve ser considerado como válido se obeder as regras 1) E 2)\n"
    " 1) contém exatamente um '@'\n"
    " 2) termina o email em '.com' ou '.com.br'\n"
    " 3) exemplo valido = carolina.zambelli@yahoo.com\n"
    " 4) exemplo invalido = carolina.zambelli.yahoo.com OU carolina.zambelli@yahoo\n"
    "Responda APENAS com 'sim' ou 'nao' (sem acento, em minúsculo). "
    "Nenhuma outra palavra, nenhuma explicação."
)


def tratar_nome_llm(nome: str) -> str:
    """Pede ao LLM para normalizar o nome (sem acento, trim, `_`, minúsculo).

    Dica:
        - Se `nome` vier vazio, devolver "" sem chamar o LLM.
        - Use llm.invoke([SystemMessage(...), HumanMessage(...)]).
        - Faça uma limpeza defensiva da resposta (strip, remover aspas,
          forçar minúsculo) antes de devolver.
    """
    ## aqui coloque seu código ###


def email_e_valido_llm(email: str) -> bool:
    """Pede ao LLM para classificar o e-mail como válido ('sim') ou não.

    Dica:
        - Se `email` vier vazio, devolver False sem chamar o LLM.
        - Pegue a primeira palavra alfabética da resposta para evitar
          que pontuação ou explicações extras quebrem o parser.
        - Devolver True se a primeira palavra for "sim", False caso contrário.
    """
    ## aqui coloque seu código ###


# ---------------------------------------------------------------------------
# 3. NÓS DO GRAFO
#
#    Convenção: cada nó recebe `state: EstadoSanitizacao` e devolve um
#    dict PARCIAL com as atualizações que quer aplicar no estado.
#    Tudo que não for retornado, permanece igual.
# ---------------------------------------------------------------------------
def ler_csv(state: EstadoSanitizacao) -> dict:
    """Lê o CSV de entrada em modo dict (cada linha = um dicionário).

    Dica:
        - Abra o arquivo com encoding="latin-1" e newline="".
        - Use csv.DictReader(f, delimiter=";").
        - Devolva: {"linhas": [...], "total": N, "logs": ["..."]}.
    """
    ## aqui coloque seu código ###


def tratar_nomes(state: EstadoSanitizacao) -> dict:
    """Adiciona a coluna `nome_tratado` em cada linha — via LLM.

    Dica:
        - Itere sobre state["linhas"].
        - Para cada linha: linha["nome_tratado"] = tratar_nome_llm(linha["nome"]).
        - Devolva {"linhas": linhas, "logs": ["..."]}.
    """
    ## aqui coloque seu código ###


def validar_emails(state: EstadoSanitizacao) -> dict:
    """Adiciona a coluna `email_valido` ('sim'/'nao') em cada linha — via LLM.

    Atenção:
        A coluna no CSV se chama "e-mail" (COM HÍFEN), não "email".

    Dica:
        - Itere sobre state["linhas"].
        - Para cada linha: ok = email_e_valido_llm(linha["e-mail"]).
        - linha["email_valido"] = "sim" if ok else "nao".
        - Conte quantos foram válidos para registrar no log.
    """
    ## aqui coloque seu código ###


def salvar_csv(state: EstadoSanitizacao) -> dict:
    """Grava o resultado em `cadastro_nova.csv` no mesmo diretório.

    Dica:
        - Use csv.DictWriter com delimiter=";" e encoding="utf-8".
        - Preserve a ordem original das colunas e acrescente as duas
          novas (`nome_tratado`, `email_valido`) NO FINAL.
        - Devolva {"logs": ["..."]} com um registro do que foi gravado.
    """
    ## aqui coloque seu código ###


# ---------------------------------------------------------------------------
# 4. MONTAGEM DO GRAFO
#
#    Construa um StateGraph(EstadoSanitizacao), adicione os 4 nós e
#    ligue as arestas em sequência:
#       START -> ler_csv -> tratar_nomes -> validar_emails -> salvar_csv -> END
#    Por fim, devolva o grafo compilado (builder.compile()).
# ---------------------------------------------------------------------------
def construir_grafo():
    """Cria o StateGraph com os 4 nós e devolve o grafo compilado."""
    ## aqui coloque seu código ###


# ---------------------------------------------------------------------------
# 5. ENTRADA PRINCIPAL
# ---------------------------------------------------------------------------
def main() -> None:
    """Resolve os caminhos, invoca o grafo, imprime resumo + preview."""
    aqui = Path(__file__).parent
    entrada = aqui / "cadastro.csv"
    saida = aqui / "cadastro_nova.csv"

    print("=" * 60)
    print("  PIPELINE DE SANITIZAÇÃO — LangGraph")
    print("=" * 60)

    # Passos sugeridos:
    # 1) grafo = construir_grafo()
    # 2) Invoque o grafo com o estado inicial:
    #
    #       {
    #           "caminho_in": str(entrada),
    #           "caminho_out": str(saida),
    #           "linhas": [],
    #           "total": 0,
    #           "logs": [],
    #       }
    #
    #    (opcional) passe config={"tags": [...], "metadata": {...}}
    #    para enviar tags ao LangSmith se o .env estiver configurado.
    #
    # 3) Imprima:
    #       - total processado (estado_final["total"])
    #       - cada log da execução
    #       - preview das 5 primeiras linhas tratadas, no formato:
    #           nome=... -> nome_tratado=...  email_valido=sim/nao
    ## aqui coloque seu código ###


if __name__ == "__main__":
    main()
