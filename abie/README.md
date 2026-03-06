# ABIE — Adaptive Business Intelligence Engine

ABIE is a multi-document RAG-powered Q&A application for business insights. Upload multiple PDFs, ask questions, and receive answers grounded in your documents with confidence indicators and source citations.

## Tech Stack

- **Python** — Core runtime
- **Streamlit** — Web UI
- **LangChain** — RAG orchestration
- **ChromaDB** — Vector store for embeddings
- **OpenAI API** — GPT-3.5-turbo for answers, text-embedding-ada-002 for embeddings
- **PyPDF** — PDF text extraction
- **python-dotenv** — Environment variable loading
- **pandas** — Data handling
- **matplotlib** — Analytics charts

## Setup Instructions

1. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Install dependencies** (from the `abie` folder):

   ```bash
   cd abie
   pip install -r requirements.txt
   ```

3. **Configure the OpenAI API key**:

   - Create or edit `abie/.env`.
   - Add: `OPENAI_API_KEY=your_actual_key_here`
   - Replace `your_actual_key_here` with your key from [OpenAI API Keys](https://platform.openai.com/api-keys).
   - Do not commit `.env` or share your key.

4. **Run the app** (from the `abie` folder):

   ```bash
   streamlit run abie.py
   ```

   Or from the project root (one level above `abie`):

   ```bash
   streamlit run abie/abie.py
   ```

## Features

- **Multi-document upload** — Upload multiple PDFs at once; view file names and page counts.
- **Document selection** — Use checkboxes to choose which documents to query.
- **Smart Q&A** — RAG with chunk size 800, overlap 100; top 4 chunks per query; business-analyst-style answers.
- **Conversation history** — Last 5 Q&A pairs in a collapsible section; Clear History button.
- **Document analytics** — Sidebar shows total documents, total chunks, and a bar chart of chunks per document.
- **Query confidence** — High / Moderate / Low labels based on retrieval similarity scores.
- **Source citations** — Each answer shows document name and page number for each source chunk.

## Screenshot Descriptions

1. **Main interface** — Dark-themed layout with sidebar (upload, analytics, model settings) and main area (document list, query input, answer display).
2. **Answer with high confidence** — Example Q&A with green confidence label and source citations.
3. **Document analytics** — Sidebar bar chart showing chunk count per uploaded document.
4. **Conversation history** — Expanded section showing the last 5 questions and answers in chat style.
