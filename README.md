# RAG Vector Index Comparison App

Streamlit RAG app for comparing vector index algorithms. It can run locally, on Streamlit Community Cloud, or on Hugging Face Spaces.

Supported vector indexes:

- Flat exact
- HNSW
- IVF
- IVF+PQ

Embeddings run with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for Vietnamese-English retrieval. LLM answers can use Gemini, and web search can use Tavily.
The final answer is synthesized from retrieved documents. If Tavily is enabled, web results are added as extra context.

## Structure

```text
.
├── app.py
├── documents/
├── rag/
├── requirements.txt
└── README.md
```

Put `.pdf`, `.txt`, `.md`, or `.docx` files into `documents/`.

## Run Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

For Streamlit Community Cloud:

1. Push this project to GitHub.
2. Create a Streamlit app from the repo.
3. Set the main file to `app.py`.

For Hugging Face Spaces:

1. Create a new Space.
2. Select Streamlit SDK.
3. Upload this project.

The first run downloads the free embedding model from Hugging Face.

## Environment

Create a local `.env` file:

```env
GEMINI_API_KEY=
TAVILY_API_KEY=
```

You can also set the same keys in Streamlit secrets.

## Optional LLM

Required LLM key:

- `GEMINI_API_KEY`

Required web search key:

- `TAVILY_API_KEY`

Answer behavior:

- Gemini uses retrieved document chunks from `Flat exact` as the main context.
- Tavily web context is only added when `Use Tavily web search` is enabled.
- The selected indexes are still shown for retrieval comparison.
