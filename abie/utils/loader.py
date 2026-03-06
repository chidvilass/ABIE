"""
PDF loading and chunking logic for ABIE.
Extracts text from uploaded PDFs and splits into overlapping chunks with metadata.
"""

from typing import Any

from langchain_core.documents import Document
from pypdf import PdfReader


def extract_pages_from_pdfs(files: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """
    Extract text from each page of multiple PDF files.
    Returns page entries and a list of failed files with reasons.
    """
    page_entries: list[dict[str, Any]] = []
    failed_files: list[dict[str, str]] = []

    for file in files:
        file_name = file.name if hasattr(file, "name") else str(file)
        try:
            reader = PdfReader(file)
            page_count = len(reader.pages)

            for page_num in range(page_count):
                page = reader.pages[page_num]
                text = page.extract_text() or ""
                page_entries.append({
                    "file_name": file_name,
                    "page_number": page_num + 1,
                    "text": text,
                })
        except Exception as e:
            failed_files.append({
                "file_name": file_name,
                "reason": str(e),
            })

    return page_entries, failed_files


def chunk_documents(
    page_entries: list[dict[str, Any]],
    chunk_size: int = 800,
    overlap: int = 100,
) -> tuple[list[Document], dict[str, int]]:
    """
    Split page-level text into overlapping chunks and create LangChain Documents.
    Returns documents and per-document chunk counts.
    """
    documents: list[Document] = []
    per_doc_chunk_counts: dict[str, int] = {}

    for entry in page_entries:
        file_name = entry["file_name"]
        page_number = entry["page_number"]
        text = entry["text"]

        if not text.strip():
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                doc = Document(
                    page_content=chunk_text,
                    metadata={"source": file_name, "page": page_number},
                )
                documents.append(doc)
                per_doc_chunk_counts[file_name] = per_doc_chunk_counts.get(file_name, 0) + 1

            start = end - overlap
            if start >= len(text):
                break

    return documents, per_doc_chunk_counts
