# RAG Vector Index Comparison App

Streamlit RAG app for comparing vector index algorithms. It can run locally, on Streamlit Community Cloud, or on Hugging Face Spaces.

Supported vector indexes:

- Flat exact
- HNSW
- IVF
- IVF+PQ

Embeddings run with `sentence-transformers/all-MiniLM-L6-v2`. LLM answers are optional and use a Hugging Face token.

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

## Optional LLM

Create a Hugging Face token with permission to call Inference Providers, then add it to Streamlit secrets:

```toml
HF_TOKEN = "your_token_here"
```

The app also accepts the token from the sidebar.
