"""
Exercício 02 — Tools conectadas a APIs externas (clima + Wikipedia)

Objetivo didático:
    Subir um nível: agora as tools chamam APIs REAIS. O aluno vai ver
    que:
      1. Um agente pode usar VÁRIAS tools no mesmo turno.
      2. O LLM decide a ORDEM e os ARGUMENTOS de cada chamada.
      3. Falhas de rede / cidades não cadastradas devem ser tratadas
         dentro da própria tool (try/except), nunca explodir no agente.

APIs usadas (não exigem chave):
    - Open-Meteo (clima atual): https://open-meteo.com/
    - Wikipedia REST (resumo de página): https://pt.wikipedia.org/api/rest_v1/

Modelo: Ollama llama3.2 (local).
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
    """
    print(f"\n[API clima] Buscando clima para: {city}...")

    # Em uma versão real usaríamos um geocoder. Para o exercício, deixamos
    # um mini-dicionário fixo de cidades populares.
    locations = {
        "paris": {"lat": 48.8566, "lon": 2.3522},
        "londres": {"lat": 51.5074, "lon": -0.1278},
        "sao paulo": {"lat": -23.5505, "lon": -46.6333},
        "buenos aires": {"lat": -34.6037, "lon": -58.3816},
    }

    loc = locations.get(city.lower())
    if not loc:
        return f"Clima para {city}: 22°C (estimativa padrão — cidade não cadastrada)."

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={loc['lat']}&longitude={loc['lon']}&current_weather=true"
    )
    try:
        data = requests.get(url, timeout=5).json()
        temp = data.get("current_weather", {}).get("temperature")
        return f"A temperatura atual em {city} é {temp}°C."
    except Exception as e:
        return f"Falha na API de clima para {city}: {e}"


@tool
def get_tourist_info(city: str) -> str:
    """Busca informações turísticas rápidas (resumo da Wikipedia).

    Args:
        city: nome da cidade.
    """
    print(f"\n[API Wikipedia] Buscando informações turísticas para: {city}...")

    url = f"https://pt.wikipedia.org/api/rest_v1/page/summary/{city.capitalize()}"
    try:
        data = requests.get(url, timeout=5).json()
        return data.get("extract", f"Cidade turística famosa: {city}")
    except Exception as e:
        return f"Informação turística de {city}: monumentos e gastronomia. (Falha API: {e})"


def main() -> None:
    print("Exercício 02: Tools com APIs externas (clima + Wikipedia)\n")

    llm = ChatOllama(model=MODELO, temperature=0)
    agent = create_agent(llm, [get_weather, get_tourist_info])

    user_input = (
        "Meu destino será Paris entre 10 e 17 de julho deste ano. Como está o clima lá agora?"
        "passe as informações do clima como a temperatura"
        "e me dê um resumo do que posso encontrar na cidade."
    )
    print(f"Usuário: '{user_input}'\n")

    print("Agente consultando APIs...\n")
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    print("\n[Raw struct]:")
    pprint.pprint(response)

    print("\n[Resposta final do agente]:")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
