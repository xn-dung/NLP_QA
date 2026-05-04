from dataclasses import dataclass

from .chunking import Chunk, chunk_documents
from .config import DEFAULT_EMBEDDING_MODEL, DOCUMENTS_DIR
from .document_loader import RawDocument, load_documents
from .embeddings import EmbeddingModel
from .indexes import SearchReport, build_indexes


@dataclass
class CorpusInfo:
    document_count: int
    chunk_count: int
    sources: list[str]


class RagPipeline:
    def __init__(
        self,
        documents_dir=DOCUMENTS_DIR,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = 900,
        overlap: int = 150,
    ):
        self.documents_dir = documents_dir
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.documents: list[RawDocument] = []
        self.chunks: list[Chunk] = []
        self.embedding_model: EmbeddingModel | None = None
        self.indexes = []

    def build(self) -> CorpusInfo:
        self.documents = load_documents(self.documents_dir)
        self.chunks = chunk_documents(self.documents, self.chunk_size, self.overlap)

        if not self.chunks:
            self.indexes = []
            return self.info()

        self.embedding_model = EmbeddingModel(self.model_name)
        embeddings = self.embedding_model.encode([chunk.text for chunk in self.chunks])
        self.indexes = build_indexes(self.chunks, embeddings)
        return self.info()

    def search(self, query: str, top_k: int = 5) -> list[SearchReport]:
        if not self.indexes:
            return []
        if self.embedding_model is None:
            raise RuntimeError("Pipeline has not been built.")

        query_embedding = self.embedding_model.encode([query])[0]
        return [
            index.search(query_embedding=query_embedding, top_k=top_k)
            for index in self.indexes
        ]

    def info(self) -> CorpusInfo:
        return CorpusInfo(
            document_count=len(self.documents),
            chunk_count=len(self.chunks),
            sources=[doc.source for doc in self.documents],
        )
