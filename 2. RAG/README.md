# Módulo 2 — RAG (LangChain + ChromaDB)

> **Objetivo da aula:** entender por que precisamos de RAG, implementar cada etapa do pipeline canônico usando o stack **LangChain** + **ChromaDB**, e ver a base responder perguntas sobre os seus próprios PDFs.

---

## 1. Por que RAG?

LLMs têm dois problemas conhecidos:

1. **Conhecimento congelado.** Foram treinados até uma data específica. Não sabem o que aconteceu depois nem o que está no PDF interno da sua empresa.
2. **Alucinação.** Quando não sabem, frequentemente *inventam* uma resposta com aparência de verdade.

**RAG = Retrieval-Augmented Generation.** A ideia é simples: antes de pedir a resposta ao LLM, *recuperamos* trechos relevantes da nossa base de conhecimento e *anexamos* esses trechos ao prompt. O LLM responde "com a fonte na mão".

```
Pergunta → [Retriever] → Contexto + Pergunta → [LLM] → Resposta
```

---

## 2. O pipeline canônico (componentes LangChain)

| # | Arquivo | Etapa | Componente LangChain |
|---|---------|-------|----------------------|
| 1 | `01_carregamento.py` | **Load** | `PyPDFLoader` (`langchain-community`) |
| 2 | `02_chunking.py` | **Chunk** | `RecursiveCharacterTextSplitter` |
| 4 | `03_indexacao.py` | **Index** | `Chroma.add_documents` (`langchain-chroma`) |
| 5 | `04_retrieval.py` | **Retrieve** | `Chroma.similarity_search_with_score` / `as_retriever()` |
| 6 | `05_geracao.py` | **Generate** | Cadeia **LCEL**: `prompt \| llm \| parser` |

---

## 3. Conceitos-chave

### Document
A unidade básica do LangChain. Tem `page_content` (o texto) e `metadata` (dicionário com `source`, `page`, etc.). Todo carregador devolve `Document`, todo splitter consome e devolve `Document`.

### Embedding
Um vetor (768 dimensões para `nomic-embed-text`) que representa o **significado** de um texto. Textos com sentido parecido viram vetores próximos no espaço.

### Chunking
LLMs têm janela de contexto limitada e a recuperação fica mais precisa com pedaços pequenos e coerentes. O `RecursiveCharacterTextSplitter` tenta separar primeiro por parágrafo, depois linha, depois frase. Isso preserva contexto.

- Tamanho típico: 500–1000 caracteres
- Overlap típico: 10–20% do tamanho do chunk

### Vector store
Banco que armazena vetores e responde **busca por similaridade**. O `Chroma` (do `langchain-chroma`) gerencia:
- persistência em disco (`persist_directory=...`)
- embedding automático na hora de adicionar e buscar (`embedding_function=OllamaEmbeddings(...)`)

### Retriever
Abstração do LangChain sobre o vector store. Implementa a interface `Runnable`, ou seja, você pode encadear num pipeline LCEL:

```python
cadeia = {"contexto": retriever | formatar, "pergunta": passthrough} | prompt | llm | parser
```

### LCEL (LangChain Expression Language)
Sintaxe de composição com `|`. Cada peça é um `Runnable`. A cadeia inteira vira um único Runnable com `.invoke`, `.stream`, `.batch`, `.ainvoke` etc.

---

## 4. Pré-requisitos

1. Coloque pelo menos 1 PDF na pasta `dados/`.
2. `venv` ativo e dependências instaladas.
3. Ollama rodando com `llama3.2` e `nomic-embed-text` baixados.
4. (Opcional) LangSmith configurado no `.env` para ver tracing.

---

## 5. Como rodar

Passo a passo (didático):

```bash
cd "2. RAG"
python 01_carregamento.py     # mostra os Documents extraídos dos PDFs
python 02_chunking.py         # mostra os chunks gerados
python 03_indexacao.py        # popula o ChromaDB
python 04_retrieval.py        # roda uma busca por similaridade
python 05_geracao.py          # roda a cadeia LCEL completa
```

Pipeline completo + chat:

```bash
python main.py
```

---

## 6. Onde ficam os dados?

- **Fonte:** `dados/*.pdf`
- **Índice gerado:** `2. RAG/chroma_db/` (já está no `.gitignore`)

Para resetar o índice, apague a pasta `chroma_db/` e rode `main.py` de novo.

---

## 7. Comparação Chroma e Qdrant

| Categoria | Parâmetro (Conceito) | Nome no Chroma (`langchain_chroma`) | Nome no Qdrant (`langchain_qdrant`) | Descrição do Parâmetro | Links Oficiais de Referência |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Essencial** | Função de Embedding | `embedding_function` | `embedding` | Objeto de classe que define o modelo responsável por transformar textos em vetores numéricos. | [Chroma Docs](https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma) / [Qdrant Docs](https://reference.langchain.com/python/langchain-qdrant/vectorstores/Qdrant/from_texts) |
| **Identificação** | Nome da Coleção | `collection_name` | `collection_name` | Nome da "tabela" ou espaço lógico interno para agrupar e separar conjuntos de vetores específicos. | [Chroma Docs](https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma) / [Qdrant Docs](https://reference.langchain.com/python/langchain-qdrant/vectorstores/Qdrant/from_texts) |
| **Armazenamento**| Persistência Local (Pasta) | `persist_directory` | `path` | O caminho (diretório) do seu computador onde os arquivos físicos do banco de dados serão salvos. | [Chroma Docs](https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma) / [Qdrant Docs](https://reference.langchain.com/python/langchain-qdrant/vectorstores/Qdrant/from_texts) |
| **Armazenamento**| Execução em Memória | *(Padrão se `persist_directory` for oculto)* | `location=":memory:"` | Executa o banco diretamente na memória RAM, limpando todos os dados assim que a aplicação fecha. | [Chroma Docs](https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma) / [Qdrant Frameworks](https://qdrant.tech/documentation/frameworks/langchain/) |
| **Conexão** | Host do Servidor | `host` | `host` ou `url` | Endereço IP ou URL do servidor onde a instância externa do banco de dados está rodando (ex: Docker). | [Chroma Docs](https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma) / [Qdrant Docs](https://reference.langchain.com/python/langchain-qdrant/vectorstores/Qdrant/from_texts) |
| **Conexão** | Porta do Servidor | `port` | `port` | Porta de conexão de rede para a API REST (Padrão Chroma: `8000`, Padrão Qdrant: `6333`). | [Chroma Docs](https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma) / [Qdrant Docs](https://reference.langchain.com/python/langchain-qdrant/vectorstores/Qdrant/from_texts) |
| **Conexão** | Porta gRPC | **Não possui** (Suporta via cliente nativo) | `grpc_port` | Porta de comunicação gRPC de alta performance usada exclusivamente pelo Qdrant (Padrão: `6334`). | [Qdrant Docs](https://reference.langchain.com/python/langchain-qdrant/vectorstores/Qdrant/from_texts) |
| **Segurança** | Autenticação / API Key | `chroma_cloud_api_key` | `api_key` | Chave de autenticação para conectar com as versões oficiais em nuvem (Chroma Cloud / Qdrant Cloud). | [Chroma Docs](https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma) / [Qdrant Docs](https://reference.langchain.com/python/langchain-qdrant/vectorstores/Qdrant/from_texts) |
| **Segurança** | Protocolo Seguro SSL | `ssl` | `https` | Define (via booleano `True/False`) se a conexão com o servidor remoto deve utilizar criptografia SSL/HTTPS. | [Chroma Docs](https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma) / [Qdrant Docs](https://reference.langchain.com/python/langchain-qdrant/vectorstores/Qdrant/from_texts) |
| **Matemática** | Métrica de Distância | `collection_metadata={"hnsw:space": "..."}` | `distance_func` | Define a fórmula matemática para calcular a semelhança dos vetores. (Chroma: `cosine`, `ip`, `l2` / Qdrant: `Cosine`, `Dot`, `Euclid`). | [Chroma Config](https://docs.trychroma.com/docs/collections/configure) / [Qdrant Docs](https://reference.langchain.com/python/langchain-qdrant/vectorstores/Qdrant/from_texts) |
| **Avançado** | Configuração do Cliente | `client_settings` | Preferência por gRPC (`prefer_grpc`), timeouts (`timeout`), prefixos, etc. | Permite passar objetos de configuração nativos mais profundos e customizações de rede diretamente para o motor do banco. | [Chroma Docs](https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma) / [Qdrant Docs](https://reference.langchain.com/python/langchain-qdrant/vectorstores/Qdrant/from_texts) |
| **Avançado** | Busca Híbrida (Sparse)  | **Não integrado diretamente** (Necessita extensões customizadas) | `sparse_embedding` e `retrieval_mode` | Permite alternar os modos de busca entre `DENSE` (semântica), `SPARSE` (palavra-chave/BM25) ou `HYBRID` (combinação de ambas com fusão de notas). | [Qdrant Frameworks](https://qdrant.tech/documentation/frameworks/langchain/) |

---
