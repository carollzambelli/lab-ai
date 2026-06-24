"""
Exercício 03 — Sanitização de base de dados com LangGraph

Objetivo didático:
    Mostrar que LangGraph NÃO é só para "agentes com LLM". O grafo de
    estado também é uma forma elegante de modelar PIPELINES de dados
    determinísticos, com:
        - estado tipado (TypedDict)
        - nós isolados e testáveis
        - logs por nó
        - rastreabilidade no LangSmith (se .env estiver configurado)

Pipeline:

    START
      │
      ▼
    [ler_csv]          lê `cadastro.csv` (ISO-8859-1, separador `;`)
      │
      ▼
    [tratar_nomes]     cria `nome_tratado` chamando um LLM por linha:
      │                LLM aplica as regras (sem acento, trim, minúsculo)
      ▼
    [validar_emails]   cria `email_valido` ("sim"/"nao") chamando um LLM
      │                por linha: LLM classifica conforme as regras
      │                (tem `@` E termina em `.com` ou `.com.br`)
      ▼
    [salvar_csv]       grava `cadastro_nova.csv` no MESMO diretório
      │
      ▼
     END

Regras pedidas no enunciado:
    1. Ler `cadastro.csv`.
    2. Criar coluna `nome_tratado` (sem acentos, trim, lowercase).
    3. Criar coluna `email_valido` ("sim"/"nao").
    4. Salvar como `cadastro_nova.csv` no mesmo diretório.

Como rodar:
    python "3. Agente/3. Exercicios/03_data_sanitizer_langgraph.py"
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
# 1. ESTADO
#    Tudo que viaja entre os nós. `Annotated[..., operator.add]` faz a
#    LangGraph CONCATENAR listas vindas de nós diferentes em vez de
#    sobrescrever — perfeito para acumular logs.
# ---------------------------------------------------------------------------
class EstadoSanitizacao(TypedDict):
    caminho_in: str
    caminho_out: str
    linhas: List[dict]
    total: int
    logs: Annotated[List[str], operator.add]


# ---------------------------------------------------------------------------
# 2. PROMPTS + WRAPPERS DA LLM
#    Em vez de regras determinísticas em Python (unicodedata, str.lower,
#    endswith...), DELEGAMOS a normalização e a validação para um LLM.
#    Prompts curtos, em pt-BR, exigindo uma única linha de resposta.
# ---------------------------------------------------------------------------
SYSTEM_TRATAR_NOME = (
    "Você é um normalizador de nomes próprios.\n"
    "Receberá UM nome e deve devolver o nome normalizado seguindo as regras:\n"
    "  1) remover acentos (ex.: ã→a, ç→c, é→e)\n"
    "  2) remover espaços no início e no fim da string\n"
    "  3) para os espaços no meio da string substituir por _ (ex: joao da silva -> joao_da_silva)\n"
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
    """Pede ao LLM para normalizar o nome (sem acento, trim, minúsculo)."""
    if not nome:
        return ""
    resposta = llm.invoke([
        SystemMessage(content=SYSTEM_TRATAR_NOME),
        HumanMessage(content=nome),
    ])
    # Defesa mínima: o LLM pode (raramente) devolver aspas ou espaços extras.
    return resposta.content.strip().strip('"').strip("'").lower()


def email_e_valido_llm(email: str) -> bool:
    """Pede ao LLM para classificar o e-mail como válido ('sim') ou não."""
    if not email:
        return False
    resposta = llm.invoke([
        SystemMessage(content=SYSTEM_VALIDAR_EMAIL),
        HumanMessage(content=email),
    ])
    # Normaliza a resposta: pega só a primeira palavra alfabética.
    veredito = resposta.content.strip().lower().split()[:1]
    return veredito == ["sim"]


# ---------------------------------------------------------------------------
# 3. NÓS DO GRAFO
#    Cada nó recebe o estado e devolve as ATUALIZAÇÕES (dict parcial).
# ---------------------------------------------------------------------------
def ler_csv(state: EstadoSanitizacao) -> dict:
    """Lê o CSV de entrada em modo dict (cada linha = um dicionário).

    O arquivo está em ISO-8859-1 com separador `;` — tratamos isso aqui
    para que o resto do grafo receba dados "limpos".
    """
    caminho = Path(state["caminho_in"])
    print(f"\n[1/4] Lendo CSV: {caminho.name}")

    with caminho.open(encoding="latin-1", newline="") as f:
        leitor = csv.DictReader(f, delimiter=";")
        linhas = [dict(linha) for linha in leitor]

    total = len(linhas)
    print(f"      {total} linhas lidas, colunas: {list(linhas[0].keys()) if linhas else []}")

    return {
        "linhas": linhas,
        "total": total,
        "logs": [f"[ler_csv] {total} linhas lidas de {caminho.name}"],
    }


def tratar_nomes(state: EstadoSanitizacao) -> dict:
    """Adiciona a coluna `nome_tratado` em cada linha — via LLM.

    Cada nome é enviado individualmente ao LLM, que aplica as regras
    descritas em `SYSTEM_TRATAR_NOME` (sem acento, trim, minúsculo).
    """
    print("\n[2/4] Tratando coluna `nome` -> `nome_tratado` (via LLM)")

    linhas = state["linhas"]
    for i, linha in enumerate(linhas, start=1):
        nome_original = linha.get("nome", "")
        linha["nome_tratado"] = tratar_nome_llm(nome_original)
        print(f"      ({i}/{len(linhas)}) {nome_original!r} -> {linha['nome_tratado']!r}")

    return {
        "linhas": linhas,
        "logs": [f"[tratar_nomes] {len(linhas)} nomes normalizados via LLM"],
    }


def validar_emails(state: EstadoSanitizacao) -> dict:
    """Adiciona a coluna `email_valido` ('sim'/'nao') — via LLM.

    Cada e-mail é enviado ao LLM, que decide se é válido segundo as regras
    descritas em `SYSTEM_VALIDAR_EMAIL` (contém '@' e termina em '.com'
    ou '.com.br').
    """
    print("\n[3/4] Validando coluna `e-mail` -> `email_valido` (via LLM)")

    linhas = state["linhas"]
    validos = 0
    for i, linha in enumerate(linhas, start=1):
        # A coluna no CSV se chama "e-mail" (com hífen!) — atenção.
        email = linha.get("e-mail", "")
        ok = email_e_valido_llm(email)
        linha["email_valido"] = "sim" if ok else "nao"
        if ok:
            validos += 1
        print(f"      ({i}/{len(linhas)}) {email!r} -> {linha['email_valido']}")

    invalidos = len(linhas) - validos
    print(f"      válidos: {validos}, inválidos: {invalidos}")

    return {
        "linhas": linhas,
        "logs": [f"[validar_emails] válidos={validos}, inválidos={invalidos} (via LLM)"],
    }


def salvar_csv(state: EstadoSanitizacao) -> dict:
    """Grava o resultado em `cadastro_nova.csv` no mesmo diretório."""
    caminho = Path(state["caminho_out"])
    print(f"\n[4/4] Salvando: {caminho.name}")

    linhas = state["linhas"]
    if not linhas:
        return {"logs": ["[salvar_csv] nenhuma linha — nada a gravar"]}

    # Preserva a ordem original das colunas e acrescenta as novas no fim.
    colunas_originais = [c for c in linhas[0].keys() if c not in ("nome_tratado", "email_valido")]
    colunas = colunas_originais + ["nome_tratado", "email_valido"]

    # Salvamos em UTF-8 (padrão moderno) com separador `;` para manter
    # compatibilidade com o Excel brasileiro.
    with caminho.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=colunas, delimiter=";")
        writer.writeheader()
        writer.writerows(linhas)

    print(f"      OK — {len(linhas)} linhas gravadas em {caminho}")
    return {"logs": [f"[salvar_csv] {len(linhas)} linhas em {caminho.name}"]}


# ---------------------------------------------------------------------------
# 4. MONTAGEM DO GRAFO
# ---------------------------------------------------------------------------
def construir_grafo():
    builder = StateGraph(EstadoSanitizacao)
    builder.add_node("ler_csv", ler_csv)
    builder.add_node("tratar_nomes", tratar_nomes)
    builder.add_node("validar_emails", validar_emails)
    builder.add_node("salvar_csv", salvar_csv)

    builder.add_edge(START, "ler_csv")
    builder.add_edge("ler_csv", "tratar_nomes")
    builder.add_edge("tratar_nomes", "validar_emails")
    builder.add_edge("validar_emails", "salvar_csv")
    builder.add_edge("salvar_csv", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# 5. ENTRADA PRINCIPAL
# ---------------------------------------------------------------------------
def main() -> None:
    aqui = Path(__file__).parent
    entrada = aqui / "cadastro.csv"
    saida = aqui / "cadastro_nova.csv"

    print("=" * 60)
    print("  PIPELINE DE SANITIZAÇÃO — LangGraph")
    print("=" * 60)

    grafo = construir_grafo()

    estado_final = grafo.invoke(
        {
            "caminho_in": str(entrada),
            "caminho_out": str(saida),
            "linhas": [],
            "total": 0,
            "logs": [],
        },
        config={
            "tags": ["exercicio:03", "sanitizacao"],
            "metadata": {"pipeline": "csv_sanitization"},
        },
    )

    print("\n" + "=" * 60)
    print("  RESUMO")
    print("=" * 60)
    print(f"Total processado: {estado_final['total']} linhas")
    print("\nLog de execução:")
    for log in estado_final["logs"]:
        print(f"  - {log}")

    # Pequeno preview para o aluno ver o resultado sem abrir o arquivo.
    print("\nPreview das 5 primeiras linhas tratadas:")
    for linha in estado_final["linhas"][:5]:
        print(
            f"  nome={linha.get('nome','').strip()!r:35} "
            f"-> nome_tratado={linha['nome_tratado']!r:30} "
            f"email_valido={linha['email_valido']}"
        )


if __name__ == "__main__":
    main()
