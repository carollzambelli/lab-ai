"""
Etapa 3 do RAG — Indexação no ChromaDB (via langchain-chroma)

`Chroma` (do langchain-chroma) é um vector store que:
    - persiste em disco automaticamente (`persist_directory`)
    - aceita uma `embedding_function` (no nosso caso, `OllamaEmbeddings`)
    - gera os embeddings sozinho quando você chama `add_documents`

"""

from __future__ import annotations
from importlib import import_module
from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils import get_vector_store

carregar_pdfs = import_module("01_carregamento").carregar_pdfs
chunkar_documentos = import_module("02_chunking").chunkar_documentos


def indexar_chunks(chunks: list[Document]) -> Chroma:
    """
    Adiciona os chunks no vector store. `add_documents` gera os
    embeddings via `OllamaEmbeddings` e grava em disco automaticamente.
    """
    vs = get_vector_store()
    if chunks:
        vs.add_documents(chunks)
    return vs


def main() -> None:
    print("[1/3] Carregando PDFs da pasta dados/...")
    docs = carregar_pdfs()
    if not docs:
        print("[ERRO] Nenhum PDF em dados/. Coloque ao menos 1 .pdf e rode de novo.")
        return

    print(f"      → {len(docs)} página(s).")

    print("[2/3] Quebrando em chunks...")
    chunks = chunkar_documentos(docs)
    print(f"      → {len(chunks)} chunk(s).")

    print("[3/3] Indexando no ChromaDB (gera embeddings via Ollama)...")
    vs = indexar_chunks(chunks)
    print(f"\n[ok] Coleção atual contém {vs._collection.count()} chunk(s).")  # noqa: SLF001


if __name__ == "__main__":
    main()
