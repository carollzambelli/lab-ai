"""
Etapa 5 do RAG — Geração aumentada (cadeia LCEL - (Linguagem de Expressão do LangChain))

A cadeia canônica de RAG escrita em LCEL:

    {"pergunta": ...}
        ↓
    {"contexto": retriever | formatar_contexto,
     "pergunta": passthrough}
        ↓
    ChatPromptTemplate
        ↓
    ChatOllama
        ↓
    StrOutputParser
        ↓
    string final

"""

from __future__ import annotations
from importlib import import_module
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from utils import get_llm, get_vector_store

_mod_retrieval = import_module("04_retrieval")
recuperar = _mod_retrieval.recuperar
get_retriever = _mod_retrieval.get_retriever


PROMPT_RAG = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um assistente que responde APENAS com base no contexto fornecido. "
            "Se a resposta não estiver no contexto, diga claramente: "
            "'Não encontrei essa informação nos documentos.' "
            "Cite a fonte (arquivo + página) entre parênteses ao final.",
        ),
        (
            "human",
            'Contexto:\n"""\n{contexto}\n"""\n\nPergunta: {pergunta}',
        ),
    ]
)


def formatar_contexto(docs: list[Document]) -> str:
    """Concatena documentos em um único bloco com cabeçalho de fonte/página."""
    blocos = []
    for d in docs:
        fonte = Path(d.metadata.get("source", "?")).name
        pagina = d.metadata.get("page", "?")
        blocos.append(f"[{fonte} | página {pagina}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocos)


def construir_cadeia_rag(k: int = 4):
    """
    Cadeia LCEL canônica:

        {"pergunta": ...} → recupera docs → formata contexto
                          → prompt → LLM → parser
    """
    retriever = get_retriever(k=k)
    llm = get_llm()

    return (
        {
            "contexto": retriever | formatar_contexto,
            "pergunta": RunnablePassthrough(),
        }
        | PROMPT_RAG
        | llm
        | StrOutputParser()
    )


def gerar_resposta(vs, pergunta: str, k: int = 4) -> dict:
    """
    Versão 'caixa transparente': roda o retrieval explicitamente para
    podermos mostrar as fontes na UI, depois manda pro LLM.
    """
    if vs._collection.count() == 0:  
        return {
            "resposta": "Base de conhecimento vazia.",
            "fontes": [],
        }

    docs_com_score = recuperar(pergunta, k=k)
    docs = [d for d, _ in docs_com_score]
    contexto = formatar_contexto(docs)

    cadeia = PROMPT_RAG | get_llm() | StrOutputParser()
    resposta = cadeia.invoke({"contexto": contexto, "pergunta": pergunta})

    return {
        "resposta": resposta,
        "fontes": [
            {
                "fonte": Path(d.metadata.get("source", "?")).name,
                "pagina": d.metadata.get("page", "?"),
                "distancia": float(score),
            }
            for d, score in docs_com_score
        ],
    }


def main() -> None:
    vs = get_vector_store()
    if vs._collection.count() == 0:  
        print("A coleção está vazia. Rode primeiro: python 04_indexacao.py")
        return

    pergunta = "Faça um resumo de 3 frases sobre a narrativa central da história."
    print(f">> Pergunta: {pergunta}\n")

    cadeia = construir_cadeia_rag(k=4)
    resposta_via_cadeia = cadeia.invoke(pergunta)

    print("=== Resposta via cadeia LCEL ===")
    print(resposta_via_cadeia)
    print()

    print("=== Resposta Formatada ===")
    resultado = gerar_resposta(vs, pergunta, k=4)
    print(resultado["resposta"])
    print("\nFontes usadas:")
    for i, f in enumerate(resultado["fontes"], 1):
        print(f"  [{i}] {f['fonte']} p.{f['pagina']} — distância={f['distancia']:.4f}")


if __name__ == "__main__":
    main()
