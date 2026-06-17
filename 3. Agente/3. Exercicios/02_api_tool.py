"""
Exercício 02 — Tools conectadas a APIs externas (clima + Wikipedia)

Objetivo:
    Subir um nível: agora as tools chamam APIs REAIS. Você vai ver que:
      1. Um agente pode usar VÁRIAS tools no mesmo turno.
      2. O LLM decide a ORDEM e os ARGUMENTOS de cada chamada.
      3. Falhas de rede / cidades não cadastradas devem ser tratadas
         dentro da própria tool (try/except), nunca explodir no agente.

APIs usadas (não exigem chave de acesso):

    1) Open-Meteo — clima atual
       Site:      https://open-meteo.com/
       Endpoint:  GET https://api.open-meteo.com/v1/forecast
                       ?latitude={lat}&longitude={lon}&current_weather=true
       Resposta:  JSON. Campo de interesse:
                       data["current_weather"]["temperature"]  (em °C)

    2) Wikipedia REST — resumo de página em pt-BR
       Endpoint:  GET https://pt.wikipedia.org/api/rest_v1/page/summary/{titulo}
                  (o {titulo} deve estar com a primeira letra maiúscula)
       Resposta:  JSON. Campo de interesse:
                       data["extract"]  (resumo em texto puro)

Modelo: Ollama llama3.2 (local).

Estrutura do script:
    1) Imports + constante MODELO.
    2) Tool `get_weather(city)`:
        - usa o dicionário `locations` com coordenadas pré-cadastradas.
        - se a cidade não estiver no dicionário, devolver string de
          fallback amigável (ex.: "Clima para X: 22°C (estimativa padrão
          — cidade não cadastrada).").
        - se estiver, montar a URL do Open-Meteo, chamar requests.get
          com timeout, extrair `current_weather.temperature` e devolver
          uma string como "A temperatura atual em Paris é 17.4°C.".
        - sempre proteger com try/except.
    3) Tool `get_tourist_info(city)`:
        - montar a URL da Wikipedia em pt-BR.
        - chamar requests.get com timeout e extrair o campo `extract`.
        - se faltar, devolver um fallback amigável.
        - try/except para falhas de rede.
    4) `main()`:
        - criar LLM, criar agente com AS DUAS tools.
        - perguntar sobre clima + dicas turísticas de uma cidade.
        - imprimir resposta bruta e resposta final.

Como rodar:
    python "3. Agente/3. Exercicios/02_api_tool_aluno.py"
"""
from __future__ import annotations
import pprint
import requests
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass


MODELO = "llama3.2"


@tool
def get_weather(city: str) -> str:
    """Busca a temperatura atual de uma cidade turística.

    Args:
        city: nome da cidade (ex: "Paris", "São Paulo").

    Dica de implementação:
        - Use o dicionário `locations` abaixo como base de coordenadas.
        - Se a cidade não estiver lá, devolva uma string de fallback.
        - Endpoint Open-Meteo:
            https://api.open-meteo.com/v1/forecast
                ?latitude={lat}&longitude={lon}&current_weather=true
        - O JSON retornado tem `current_weather.temperature` em °C.
        - Proteger a chamada com try/except (timeout sugerido: 5s).
    """
    print(f"\n[API clima] Buscando clima para: {city}...")

    # Mini-cadastro de coordenadas — em produção você usaria um geocoder.
    locations = {
        "paris": {"lat": 48.8566, "lon": 2.3522},
        "londres": {"lat": 51.5074, "lon": -0.1278},
        "sao paulo": {"lat": -23.5505, "lon": -46.6333},
        "buenos aires": {"lat": -34.6037, "lon": -58.3816},
    }
    ## aqui coloque seu código ###


@tool
def get_tourist_info(city: str) -> str:
    """Busca informações turísticas rápidas (resumo da Wikipedia).

    Args:
        city: nome da cidade.

    Dica de implementação:
        - URL: https://pt.wikipedia.org/api/rest_v1/page/summary/{cidade}
          (lembre-se de capitalizar a primeira letra: city.capitalize()).
        - O JSON retornado contém o campo `extract` com o resumo.
        - Use try/except — devolver um fallback se a API quebrar.
    """
    print(f"\n[API Wikipedia] Buscando informações turísticas para: {city}...")
    ## aqui coloque seu código ###


def main() -> None:
    """Cria o agente com as duas tools e faz uma pergunta combinada."""
    print("Exercício 02: Tools com APIs externas (clima + Wikipedia)\n")

    # Passos sugeridos:
    # 1) Crie o LLM (ChatOllama, temperature=0).
    # 2) Crie o agente com create_agent(llm, [get_weather, get_tourist_info]).
    # 3) Monte uma pergunta que exija AS DUAS tools, por exemplo:
    #
    #       "Meu destino será Paris entre 10 e 17 de julho deste ano.
    #        Como está o clima lá agora? Passe a temperatura
    #        e me dê um resumo do que posso encontrar na cidade."
    #
    # 4) Invoque o agente com {"messages": [HumanMessage(content=...)]}.
    # 5) Imprima:
    #       - resposta bruta (pprint) — para ver a sequência de tool calls.
    #       - texto final (response["messages"][-1].content).
    ## aqui coloque seu código ###


if __name__ == "__main__":
    main()
