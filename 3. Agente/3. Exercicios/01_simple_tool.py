"""
Exercício 01 — Tool simples local (cálculo de orçamento de viagem)

Objetivo:
    Praticar o ciclo de tool calling mais básico:

        Usuário pede algo -> LLM decide chamar a tool ->
        tool roda (Python puro) -> LLM lê o resultado e responde.

    A tool aqui é DETERMINÍSTICA — só fórmulas — para você enxergar
    com clareza onde termina o LLM e onde começa o código que ele
    delegou. Em produção essa tool poderia falar com um ERP, uma planilha,
    um banco, etc.

Pontos para observar:
    1. A DOCSTRING da tool é o que o LLM lê para decidir QUANDO e COMO
       chamar a função. Argumentos sem descrição = LLM chuta os valores.
    2. `temperature=0` deixa o agente mais determinístico (importante em
       agentes que executam código).
    3. Usamos Ollama (llama3.2) — sem chave de API, sem custo.

Estrutura do script:
    1) Importar as dependências (LangChain, LangChain-Ollama).
    2) Definir a constante MODELO.
    3) Criar a tool `calculate_budget` (decorada com @tool) que recebe
       (days, origin, destination, date_start, date_end) e devolve um
       orçamento em string formatada.
    4) Em `main()`:
       a) Criar o LLM (ChatOllama, temperature=0).
       b) Criar o agente com `create_agent(llm, [calculate_budget])`.
       c) Montar a pergunta do usuário.
       d) Invocar o agente com HumanMessage.
       e) Exibir a resposta bruta (pprint) e a resposta final (texto).

Regras de cálculo do orçamento (implementar dentro da tool):
    - Passagem base: R$ 2500 se origem != destino; caso contrário 0.
    - Se destino contiver "paris" OU "londres" (case-insensitive):
        +R$ 3000 na passagem e custo diário de R$ 800/dia.
    - Se destino contiver "buenos aires":
        +R$ 1000 na passagem e custo diário de R$ 400/dia.
    - Demais destinos: custo diário R$ 250/dia (sem extra na passagem).
    - Total = passagem + (custo_diário * dias).

Como rodar:
    python "3. Agente/3. Exercicios/01_simple_tool_aluno.py"
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

    Retorno esperado:
        Uma STRING com as informações da viagem (datas, rota, dias,
        valor da passagem, custo diário e orçamento total).
    """
    ## aqui coloque seu código ###


def main() -> None:
    """Cria o LLM, monta o agente, faz a pergunta e exibe a resposta."""
    print("Exercício 01: Tool simples local (orçamento de viagem)\n")

    # Passos sugeridos:
    # 1) Crie o LLM com ChatOllama(model=MODELO, temperature=0).
    # 2) Crie o agente com create_agent(llm, [calculate_budget]).
    # 3) Monte a pergunta do usuário (sugestão abaixo):
    #
    #       "Quero fazer uma viagem de São Paulo para Paris. Serão 7 dias,
    #        de 10 de julho (2026-07-10) a 17 de julho (2026-07-17).
    #        Quanto de budget eu preciso?"
    #
    # 4) Invoque o agente passando {"messages": [HumanMessage(content=...)]}.
    # 5) Imprima:
    #       - o objeto bruto com pprint (útil para enxergar a sequência
    #         de mensagens: HumanMessage -> AIMessage(tool_calls) ->
    #         ToolMessage -> AIMessage final)
    #       - apenas o texto da última mensagem (response["messages"][-1].content)
    ## aqui coloque seu código ###


if __name__ == "__main__":
    main()
