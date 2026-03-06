"""
ChromaDB storage and retrieval for ABIE.
Manages persistent embeddings and vector search over document chunks.
"""

from pathlib import Path

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings


PERSIST_DIR = "chroma_store"
COLLECTION_NAME = "abie_store"


def create_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Create an OpenAI embeddings model configured for text-embedding-ada-002.
    """
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def get_or_create_vectorstore(embedding: GoogleGenerativeAIEmbeddings) -> Chroma:
    """
    Get or create a persistent Chroma vectorstore for the abie_store collection.
    Reuses existing data if present.
    """
    persist_path = Path(__file__).resolve().parent.parent / PERSIST_DIR
    persist_path.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=str(persist_path),
    )


def add_documents_to_vectorstore(vectorstore: Chroma, docs: list[Document]) -> None:
    """
    Add document chunks to the vectorstore and persist to disk.
    """
    if docs:
        vectorstore.add_documents(docs)
        vectorstore.persist()


def get_relevant_chunks(
    vectorstore: Chroma,
    query: str,
    k: int = 4,
    filter_sources: list[str] | None = None,
) -> list[tuple[Document, float]]:
    """
    Retrieve the top-k most relevant chunks for a query with relevance scores.
    Optionally filter by source file names. Returns (Document, score) with score in [0, 1].
    """
    filter_dict = None
    if filter_sources and len(filter_sources) > 0:
        filter_dict = {"source": {"$in": filter_sources}}

    results = vectorstore.similarity_search_with_score(
        query, k=k, filter=filter_dict
    )

    output: list[tuple[Document, float]] = []
    for doc, raw_score in results:
        similarity = max(0, min(1, 1 - float(raw_score)))
        output.append((doc, similarity))

    return output


def reset_chroma_collection() -> None:
    """
    Delete the abie_store collection so the vectorstore can be recreated empty.
    """
    persist_path = Path(__file__).resolve().parent.parent / PERSIST_DIR
    if persist_path.exists():
        client = chromadb.PersistentClient(path=str(persist_path))
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass


def get_vectorstore_stats(vectorstore: Chroma) -> tuple[int, dict[str, int]]:
    """
    Return total chunk count and per-document chunk counts from the vectorstore.
    """
    try:
        collection = vectorstore._collection
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        total = len(metadatas)
        per_doc: dict[str, int] = {}
        for meta in metadatas:
            if meta and "source" in meta:
                src = meta["source"]
                per_doc[src] = per_doc.get(src, 0) + 1
        return total, per_doc
    except Exception:
        return 0, {}
