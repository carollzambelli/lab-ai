"""
Exercício 01 — Tool simples local (cálculo de orçamento de viagem)

Objetivo didático:
    Praticar o ciclo de tool calling mais básico:

        Usuário pede algo -> LLM decide chamar a tool ->
        tool roda (Python puro) -> LLM lê o resultado e responde.

    A tool aqui é DETERMINÍSTICA — só fórmulas — para o aluno enxergar
    com clareza onde termina o LLM e onde começa o código que ele
    delegou. Em produção essa tool poderia falar com um ERP, uma planilha,
    um banco, etc.

Pontos para observar:
    1. A DOCSTRING da tool é o que o LLM lê para decidir QUANDO e COMO
       chamar a função. Argumentos sem descrição = LLM chuta os valores.
    2. `temperature=0` deixa o agente mais determinístico (importante em
       agentes que executam código).
    3. Usamos Ollama (llama3.2) — sem chave de API, sem custo.
"""
from __future__ import annotations
import pprint

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_ollama import ChatOllama


MODELO = "llama3.2"


@tool
def calculate_budget(
    days: int,
    origin: str,
    destination: str,
    date_start: str,
    date_end: str,
) -> str:
    """Calcula o orçamento estimado mínimo de uma viagem.

    Args:
        days: quantidade de dias da viagem (ex: 5).
        origin: cidade de embarque (ex: "São Paulo").
        destination: cidade de destino (ex: "Paris").
        date_start: data de ida no formato YYYY-MM-DD.
        date_end: data de volta no formato YYYY-MM-DD.
    """
    print(f"\n[TOOL EXECUTION] Calculando orçamento de {origin} para {destination} (Dias: {days})...")

    # Regras determinísticas, um cenário onde não é necessário uma LLM 
    base_flight_cost = 2500.0 if origin != destination else 0.0
    destino = destination.lower()
    if "paris" in destino or "londres" in destino:
        base_flight_cost += 3000.0
        cost_per_day = 800.0
    elif "buenos aires" in destino:
        base_flight_cost += 1000.0
        cost_per_day = 400.0
    else:
        cost_per_day = 250.0

    total_cost = base_flight_cost + (cost_per_day * days)

    return (
        f"Detalhes da Viagem:\n"
        f"Datas: {date_start} até {date_end}\n"
        f"Rota: {origin} -> {destination}\n"
        f"Dias totais: {days}\n"
        f"Passagem est.: R$ {base_flight_cost:.2f}\n"
        f"Custo diário est.: R$ {cost_per_day:.2f}/dia\n"
        f"-> Orçamento Total Recomendado: R$ {total_cost:.2f}"
    )


def main() -> None:
    print("Exercício 01: Tool simples local (orçamento de viagem)\n")

    llm = ChatOllama(model=MODELO, temperature=0)
    agent = create_agent(llm, [calculate_budget])

    user_input = (
        "Quero fazer uma viagem de São Paulo para Paris. Serão 7 dias, "
        "de 10 de julho (2026-07-10) a 17 de julho (2026-07-17). "
        "Quanto de budget eu preciso?"
    )
    print(f"Usuário: '{user_input}'\n")

    print("Agente orquestrando...\n")
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    print("\n[Raw struct — útil para ver a sequência de mensagens]:")
    pprint.pprint(response)

    print("\n[Resposta final em texto]:")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
