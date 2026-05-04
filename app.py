import pandas as pd
import streamlit as st

from rag import RagPipeline
from rag.answering import extractive_answer
from rag.config import DEFAULT_EMBEDDING_MODEL, DOCUMENTS_DIR
from rag.llm import DEFAULT_LLM_MODEL, generate_rag_answer, get_hf_token


st.set_page_config(page_title="RAG Index Comparison", layout="wide")


@st.cache_resource(show_spinner=False)
def build_pipeline(model_name: str, chunk_size: int, overlap: int):
    pipeline = RagPipeline(
        documents_dir=DOCUMENTS_DIR,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    info = pipeline.build()
    return pipeline, info


st.title("RAG Index Comparison")
secret_hf_token = st.secrets.get("HF_TOKEN", "")

with st.sidebar:
    model_name = st.text_input("Embedding model", DEFAULT_EMBEDDING_MODEL)
    use_llm = st.checkbox("Use LLM answer", value=True)
    llm_model = st.text_input("LLM model", DEFAULT_LLM_MODEL)
    hf_token_input = st.text_input("HF token", type="password")
    chunk_size = st.number_input("Chunk words", min_value=100, max_value=1200, value=350, step=50)
    overlap = st.number_input("Overlap words", min_value=0, max_value=300, value=60, step=20)
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    selected_indexes = st.multiselect(
        "Indexes",
        ["Flat exact", "HNSW", "IVF", "IVF+PQ"],
        default=["Flat exact", "HNSW", "IVF", "IVF+PQ"],
    )
    uploads = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )
    if uploads:
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        for uploaded in uploads:
            target = DOCUMENTS_DIR / uploaded.name.replace("/", "_").replace("\\", "_")
            target.write_bytes(uploaded.getbuffer())
        st.cache_resource.clear()
        st.success("Uploaded")
    rebuild = st.button("Rebuild index")

if rebuild:
    st.cache_resource.clear()

with st.spinner("Building indexes..."):
    pipeline, info = build_pipeline(model_name, int(chunk_size), int(overlap))

col_a, col_b, col_c = st.columns(3)
col_a.metric("Documents", info.document_count)
col_b.metric("Chunks", info.chunk_count)
col_c.metric("Folder", "documents/")

if info.document_count == 0:
    st.warning("Add .pdf, .txt, .md, or .docx files to documents/, then click Rebuild index.")
    st.stop()

with st.form("question_form"):
    query = st.text_input("Question")
    answer_clicked = st.form_submit_button("Answer")

if answer_clicked and query.strip():
    reports = [
        report for report in pipeline.search(query.strip(), top_k=top_k)
        if report.index_name in selected_indexes
    ]
    if not reports:
        st.warning("Select at least one index.")
        st.stop()

    dense_report = next((report for report in reports if report.index_name == "Flat exact"), reports[0])
    st.subheader("Answer")
    hf_token = get_hf_token(hf_token_input or secret_hf_token)
    if use_llm and hf_token:
        try:
            st.markdown(generate_rag_answer(query.strip(), dense_report.hits, hf_token, llm_model.strip()))
        except Exception as error:
            st.warning(f"LLM failed: {error}")
            st.markdown(extractive_answer(query.strip(), dense_report.hits))
    else:
        st.markdown(extractive_answer(query.strip(), dense_report.hits))

    summary = [
        {
            "index": report.index_name,
            "build_ms": round(report.build_time_ms, 2),
            "search_ms": round(report.search_time_ms, 2),
            "top_score": round(report.hits[0].score, 4) if report.hits else 0,
            "top_source": report.hits[0].source if report.hits else "",
        }
        for report in reports
    ]
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    for report in reports:
        st.subheader(report.index_name)
        for hit in report.hits:
            with st.expander(f"#{hit.rank} | {hit.source} | score={hit.score:.4f}"):
                st.write(hit.text)
else:
    st.info("Enter a question, then click Answer.")
