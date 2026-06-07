"""
pipeline RAG ponta a ponta + chat interativo.

Fluxo:
    1. Carrega PDFs de ../dados/        (PyPDFLoader)
    2. Quebra em chunks                 (RecursiveCharacterTextSplitter)
    3. Indexa no ChromaDB               (langchain-chroma + OllamaEmbeddings)
    4. Abre um loop de perguntas        (cadeia LCEL)

Para resetar o índice, apague a pasta `chroma_db/` e rode de novo.
"""

from __future__ import annotations
from importlib import import_module
from utils import get_vector_store
gerar_resposta = import_module("05_geracao").gerar_resposta

def loop_chat() -> None:
    print("=" * 60)
    print(" CHAT RAG — pergunte sobre os documentos indexados ")
    print(" (digite 'sair' para encerrar)")
    print("=" * 60)

    vs = get_vector_store()
    while True:
        try:
            pergunta = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break

        resultado = gerar_resposta(vs, pergunta, k=4)
        print("\n--- Resposta ---")
        print(resultado["resposta"])
        print("\n--- Fontes usadas ---")
        for i, f in enumerate(resultado["fontes"], 1):
            print(f"  [{i}] {f['fonte']} p.{f['pagina']} (distância={f['distancia']:.3f})")

    print("Até a próxima!")


if __name__ == "__main__":
    loop_chat()
