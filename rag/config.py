from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = ROOT_DIR / "documents"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
