"""
LangChain RAG chain setup for ABIE.
Builds business-analyst-style Q&A from retrieved document chunks.
"""

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


SYSTEM_PROMPT = """You are a senior business analyst. Your role is to answer questions clearly and insightfully based only on the provided document context.

Guidelines:
- Ground all answers in the given context; do not invent information.
- Structure responses with clear bullet points or short paragraphs when appropriate.
- Highlight key insights, metrics, and actionable takeaways.
- If the context does not contain relevant information, say so explicitly.
- Use professional, concise language suitable for business stakeholders."""


def build_llm(model_name: str, temperature: float) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance with the specified model and temperature.
    """
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


def run_business_qa(
    question: str,
    docs: list[Document],
    model_name: str,
    temperature: float,
) -> str:
    """
    Run the RAG pipeline: build context from docs, then ask the LLM as a business analyst.
    Returns the model's answer text.
    """
    if not docs:
        return "No relevant context was found for your question."

    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[Source {i}: {source}, Page {page}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)
    user_content = f"Context from uploaded documents:\n\n{context}\n\nQuestion: {question}"

    llm = build_llm(model_name, temperature)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


def compute_confidence_label(scores: list[float]) -> tuple[str, float]:
    """
    Map average relevance score to a confidence label and return (label, avg_score).
    """
    if not scores:
        return ("🔴 Low Confidence — answer may be inaccurate", 0.0)

    avg = sum(scores) / len(scores)
    if avg > 0.75:
        return ("🟢 High Confidence", avg)
    if avg >= 0.50:
        return ("🟡 Moderate Confidence", avg)
    return ("🔴 Low Confidence — answer may be inaccurate", avg)
