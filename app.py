import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rag import RagPipeline
from rag.answering import extractive_answer
from rag.config import DEFAULT_EMBEDDING_MODEL, DOCUMENTS_DIR
from rag.llm import DEFAULT_GEMINI_MODEL, generate_rag_answer, get_gemini_api_key
from rag.web_search import get_tavily_api_key, tavily_search


st.set_page_config(page_title="RAG Index Comparison", layout="wide")
load_dotenv()


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
secret_gemini_key = st.secrets.get("GEMINI_API_KEY", "")
secret_tavily_key = st.secrets.get("TAVILY_API_KEY", "")

with st.sidebar:
    model_name = st.text_input("Embedding model", DEFAULT_EMBEDDING_MODEL)
    use_llm = st.checkbox("Use LLM answer", value=True)
    llm_model = st.text_input("LLM model", DEFAULT_GEMINI_MODEL)
    gemini_key_input = st.text_input("Gemini API key", type="password")
    use_web_search = st.checkbox("Use Tavily web search", value=False)
    tavily_key_input = st.text_input("Tavily API key", type="password")
    tavily_topic = st.selectbox("Tavily topic", ["general", "news", "finance"], index=0)
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
    all_reports = pipeline.search(query.strip(), top_k=top_k)
    reports = [report for report in all_reports if report.index_name in selected_indexes]
    if not reports:
        st.warning("Select at least one index.")
        st.stop()

    answer_report = next((report for report in all_reports if report.index_name == "Flat exact"), all_reports[0])
    st.subheader("Answer")
    gemini_api_key = get_gemini_api_key(gemini_key_input or secret_gemini_key)
    tavily_api_key = get_tavily_api_key(tavily_key_input or secret_tavily_key)
    web_results = []

    if use_web_search and tavily_api_key:
        with st.spinner("Searching web..."):
            try:
                web_results = tavily_search(
                    query=query.strip(),
                    api_key=tavily_api_key,
                    topic=tavily_topic,
                    max_results=5,
                    search_depth="basic",
                )
            except Exception as error:
                st.warning(f"Tavily failed: {error}")
    elif use_web_search and not tavily_api_key:
        st.warning("Tavily API key is missing.")

    if use_llm and gemini_api_key:
        with st.spinner("Generating answer..."):
            try:
                st.markdown(
                    generate_rag_answer(
                        question=query.strip(),
                        hits=answer_report.hits,
                        api_key=gemini_api_key,
                        model=llm_model.strip(),
                        web_results=web_results,
                    )
                )
            except Exception as error:
                st.warning(f"LLM failed: {error}")
                st.markdown(extractive_answer(query.strip(), answer_report.hits))
    elif use_llm and not gemini_api_key:
        st.warning("Gemini API key is missing.")
        st.markdown(extractive_answer(query.strip(), answer_report.hits))
    else:
        st.markdown(extractive_answer(query.strip(), answer_report.hits))

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

    if web_results:
        st.subheader("Web results")
        web_summary = [
            {
                "title": item.title,
                "url": item.url,
                "score": round(item.score, 4),
            }
            for item in web_results
        ]
        st.dataframe(pd.DataFrame(web_summary), use_container_width=True, hide_index=True)
        for item in web_results:
            with st.expander(f"{item.title} | score={item.score:.4f}"):
                st.write(item.url)
                st.write(item.content)

    for report in reports:
        st.subheader(report.index_name)
        for hit in report.hits:
            with st.expander(f"#{hit.rank} | {hit.source} | score={hit.score:.4f}"):
                st.write(hit.text)
else:
    st.info("Enter a question, then click Answer.")
