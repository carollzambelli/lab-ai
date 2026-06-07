"""
Etapa 1 do RAG — Carregamento (com PyPDFLoader do LangChain)

Lê todos os PDFs da pasta `dados/` (recursivamente) e devolve uma lista
de `langchain_core.documents.Document` — 1 por página de PDF.

Cada `Document` tem:
    - page_content: o texto extraído
    - metadata: dict com 'source' (caminho do arquivo) e 'page' (índice da página)
"""

from __future__ import annotations
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from utils import PASTA_DADOS


def carregar_pdfs(pasta: Path = PASTA_DADOS) -> list[Document]:
    """
    Lê recursivamente todos os PDFs de `pasta` e retorna uma lista de
    `Document` (1 por página). Cada doc tem `metadata['source']` com o
    caminho do arquivo e `metadata['page']` com o número da página.
    """
    documentos: list[Document] = []
    pdfs = sorted(pasta.rglob("*.pdf"))
    for pdf_path in pdfs:
        loader = PyPDFLoader(str(pdf_path))
        documentos.extend(loader.load())
    return documentos


def main() -> None:
    print(f"Lendo PDFs de: {PASTA_DADOS}\n")
    docs = carregar_pdfs()
    fontes = sorted({d.metadata.get("source", "?") for d in docs})
    for f in fontes:
        n = sum(1 for d in docs if d.metadata.get("source") == f)
        print(f"  - {f}: {n} página(s)")
    print("\n--- Prévia do primeiro Document ---")
    primeiro = docs[0]
    print(f"metadata: {primeiro.metadata}")
    conteudo = primeiro.page_content
    print(conteudo[:500])


if __name__ == "__main__":
    main()
