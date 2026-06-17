"""
MCP como PROMPT — template de prompt servido pelo MCP

Cenário:
    O servidor `01_mcp_server.py` expõe `itinerary_planner` como um PROMPT
    MCP — um template parametrizado que recebe (destination, days, profile)
    e devolve um prompt já preenchido com as regras da agência embutidas.

    Aqui o cliente:
      1. Sobe o servidor como subprocesso (stdio).
      2. Chama `get_prompt(...)` passando os argumentos.
      3. O servidor renderiza o template e devolve a lista de mensagens.
      4. Acrescentamos a fala do humano e mandamos para o LLM (llama3.2).

Diferença para os exemplos anteriores:
    - TOOL     -> o LLM CHAMA uma função.
    - RESOURCE -> o cliente BAIXA um documento.
    - PROMPT   -> o servidor MONTA o prompt; o cliente só executa.
    Prompt é ideal quando a EMPRESA quer padronizar a instrução do LLM
    (tom, regras, formato) num lugar central — sem espalhar prompt
    engineering por todos os clientes.

Como rodar:
    python "3. Agente/2. Advanced/04_mcp_as_prompt.py"
"""
from __future__ import annotations
import asyncio
import os

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from langchain_mcp_adapters.client import MultiServerMCPClient


MODELO = "llama3.2"


# ---------------------------------------------------------------------------
# 1. Conexão com o servidor MCP (sobe `01_mcp_server.py` via stdio)
# ---------------------------------------------------------------------------
def montar_client() -> MultiServerMCPClient:
    server_script = os.path.join(os.path.dirname(__file__), "01_mcp_server.py")
    return MultiServerMCPClient(
        {
            "Travel_MCP": {
                "transport": "stdio",
                "command": "python",
                "args": [server_script],
            }
        }
    )


# ---------------------------------------------------------------------------
# 2. Fluxo principal — pede o prompt renderizado e envia ao LLM
# ---------------------------------------------------------------------------
async def main() -> None:
    print("Exemplo 04: MCP como PROMPT\n")
    print(
        "O Prompt de Geração de Roteiro é hospedado pelo MCP. O cliente "
        "passa apenas as variáveis; o servidor injeta as regras de negócio "
        "e devolve o prompt 100% pronto para o LLM responder.\n"
    )

    print("Conectando ao Travel_MCP Server via stdio...")
    client = montar_client()

    # Nome do prompt registrado no servidor (veja `@mcp.prompt()` em
    # `01_mcp_server.py`). Os argumentos batem com a assinatura da função.
    prompt_name = "itinerary_planner"
    args = {
        "destination": "Paris",
        "days": "7",
        "profile": "Visitar pontos turísticos e culinária",
    }

    print(f"Solicitando prompt corporativo: '{prompt_name}' preenchido: {args}\n")

    # `get_prompt` retorna uma lista de mensagens (formato LangChain) já
    # renderizada pelo servidor — incluindo o SystemMessage com as regras
    # da agência. Aqui só anexamos a fala final do humano.
    mensagens = await client.get_prompt(
        "Travel_MCP", prompt_name=prompt_name, arguments=args
    )
    mensagens.append(
        HumanMessage(content="Faça o roteiro exatamente como orientado pelas regras da agência.")
    )

    print("Gerando itinerário com o LLM baseado no prompt oficial...\n")
    # Temperatura > 0 aqui para deixar o LLM com um pouco de "criatividade"
    # ao redigir o roteiro — o prompt já carrega as regras rígidas.
    llm = ChatOllama(model=MODELO, temperature=0.7)
    resposta = await llm.ainvoke(mensagens)

    print("\n[Itinerário final gerado via prompt MCP]:\n")
    print(resposta.content)


if __name__ == "__main__":
    asyncio.run(main())
