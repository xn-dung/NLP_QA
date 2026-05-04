from abc import ABC, abstractmethod
from dataclasses import dataclass
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

try:
    import hnswlib
except ImportError:
    hnswlib = None

from .chunking import Chunk


@dataclass(frozen=True)
class SearchHit:
    rank: int
    score: float
    source: str
    text: str
    index_name: str


@dataclass(frozen=True)
class SearchReport:
    index_name: str
    build_time_ms: float
    search_time_ms: float
    hits: list[SearchHit]


class BaseIndex(ABC):
    name: str

    def __init__(self):
        self.build_time_ms = 0.0

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        start = time.perf_counter()
        self._build(chunks, embeddings)
        self.build_time_ms = (time.perf_counter() - start) * 1000

    def search(self, query_embedding: np.ndarray, top_k: int) -> SearchReport:
        start = time.perf_counter()
        hits = self._search(query_embedding, top_k)
        search_time_ms = (time.perf_counter() - start) * 1000
        return SearchReport(
            index_name=self.name,
            build_time_ms=self.build_time_ms,
            search_time_ms=search_time_ms,
            hits=hits,
        )

    @abstractmethod
    def _build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def _search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchHit]:
        raise NotImplementedError


class FlatIndex(BaseIndex):
    name = "Flat exact"

    def _build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn.fit(embeddings)

    def _search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchHit]:
        n_neighbors = min(top_k, len(self.chunks))
        distances, indices = self.nn.kneighbors(query_embedding.reshape(1, -1), n_neighbors=n_neighbors)
        scores = 1 - distances[0]
        return _hits_from_indices(self.name, self.chunks, indices[0], scores)


class HnswIndex(BaseIndex):
    name = "HNSW"

    def __init__(self, m: int = 16, ef_construction: int = 120, ef_search: int = 50):
        super().__init__()
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def _build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if hnswlib is None:
            raise RuntimeError("hnswlib is not installed.")
        self.chunks = chunks
        self.index = hnswlib.Index(space="cosine", dim=embeddings.shape[1])
        self.index.init_index(
            max_elements=len(chunks),
            ef_construction=self.ef_construction,
            M=self.m,
        )
        self.index.add_items(embeddings, np.arange(len(chunks)))
        self.index.set_ef(self.ef_search)

    def _search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchHit]:
        labels, distances = self.index.knn_query(
            query_embedding.reshape(1, -1),
            k=min(top_k, len(self.chunks)),
        )
        scores = 1 - distances[0]
        return _hits_from_indices(self.name, self.chunks, labels[0], scores)


class IvfFlatIndex(BaseIndex):
    name = "IVF"

    def __init__(self, nlist: int = 8, nprobe: int = 3, seed: int = 42):
        super().__init__()
        self.nlist = nlist
        self.nprobe = nprobe
        self.seed = seed

    def _build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.cluster_count = min(self.nlist, max(1, int(np.sqrt(len(chunks)))))

        if self.cluster_count == 1:
            self.labels = np.zeros(len(chunks), dtype=int)
            self.centroids = embeddings.mean(axis=0, keepdims=True)
        else:
            self.kmeans = KMeans(
                n_clusters=self.cluster_count,
                random_state=self.seed,
                n_init="auto",
            )
            self.labels = self.kmeans.fit_predict(embeddings)
            self.centroids = self.kmeans.cluster_centers_.astype("float32")
            self.centroids = _normalize_rows(self.centroids)

        self.inverted_lists = {
            cluster_id: np.where(self.labels == cluster_id)[0]
            for cluster_id in range(self.cluster_count)
        }

    def _search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchHit]:
        centroid_scores = self.centroids @ query_embedding
        selected_clusters = np.argsort(centroid_scores)[::-1][: min(self.nprobe, self.cluster_count)]
        candidate_ids = np.concatenate([self.inverted_lists[int(cluster)] for cluster in selected_clusters])

        if len(candidate_ids) == 0:
            candidate_ids = np.arange(len(self.chunks))

        scores = self.embeddings[candidate_ids] @ query_embedding
        ordered = candidate_ids[np.argsort(scores)[::-1]][:top_k]
        ordered_scores = self.embeddings[ordered] @ query_embedding
        return _hits_from_indices(self.name, self.chunks, ordered, ordered_scores)


class IvfPqIndex(BaseIndex):
    name = "IVF+PQ"

    def __init__(
        self,
        nlist: int = 8,
        nprobe: int = 3,
        pq_subvectors: int = 8,
        pq_codebook_size: int = 16,
        seed: int = 42,
    ):
        super().__init__()
        self.nlist = nlist
        self.nprobe = nprobe
        self.pq_subvectors = pq_subvectors
        self.pq_codebook_size = pq_codebook_size
        self.seed = seed

    def _build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self.chunks = chunks
        self.embedding_dim = embeddings.shape[1]
        self.cluster_count = min(self.nlist, max(1, int(np.sqrt(len(chunks)))))

        if self.cluster_count == 1:
            self.labels = np.zeros(len(chunks), dtype=int)
            self.centroids = embeddings.mean(axis=0, keepdims=True).astype("float32")
            self.centroids = _normalize_rows(self.centroids)
        else:
            self.kmeans = KMeans(
                n_clusters=self.cluster_count,
                random_state=self.seed,
                n_init="auto",
            )
            self.labels = self.kmeans.fit_predict(embeddings)
            self.centroids = self.kmeans.cluster_centers_.astype("float32")
            self.centroids = _normalize_rows(self.centroids)

        residuals = embeddings - self.centroids[self.labels]
        self.slices = _subvector_slices(self.embedding_dim, self.pq_subvectors)
        self.codebooks = []
        self.codes = np.zeros((len(chunks), len(self.slices)), dtype=np.int16)

        for part_id, vector_slice in enumerate(self.slices):
            sub_residuals = residuals[:, vector_slice]
            codebook_size = min(self.pq_codebook_size, len(chunks))

            if codebook_size <= 1:
                centers = sub_residuals.mean(axis=0, keepdims=True).astype("float32")
                labels = np.zeros(len(chunks), dtype=np.int16)
            else:
                kmeans = KMeans(
                    n_clusters=codebook_size,
                    random_state=self.seed + part_id,
                    n_init="auto",
                )
                labels = kmeans.fit_predict(sub_residuals).astype(np.int16)
                centers = kmeans.cluster_centers_.astype("float32")

            self.codebooks.append(centers)
            self.codes[:, part_id] = labels

        self.reconstructed = self._reconstruct_all()
        self.inverted_lists = {
            cluster_id: np.where(self.labels == cluster_id)[0]
            for cluster_id in range(self.cluster_count)
        }

    def _search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchHit]:
        centroid_scores = self.centroids @ query_embedding
        selected_clusters = np.argsort(centroid_scores)[::-1][: min(self.nprobe, self.cluster_count)]
        candidate_ids = np.concatenate([self.inverted_lists[int(cluster)] for cluster in selected_clusters])

        if len(candidate_ids) == 0:
            candidate_ids = np.arange(len(self.chunks))

        scores = self.reconstructed[candidate_ids] @ query_embedding
        ordered = candidate_ids[np.argsort(scores)[::-1]][:top_k]
        ordered_scores = self.reconstructed[ordered] @ query_embedding
        return _hits_from_indices(self.name, self.chunks, ordered, ordered_scores)

    def _reconstruct_all(self) -> np.ndarray:
        vectors = self.centroids[self.labels].copy()
        for part_id, vector_slice in enumerate(self.slices):
            vectors[:, vector_slice] += self.codebooks[part_id][self.codes[:, part_id]]
        return _normalize_rows(vectors.astype("float32"))


def build_indexes(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    index_names: set[str] | None = None,
) -> list[BaseIndex]:
    available_indexes: list[BaseIndex] = [
        FlatIndex(),
        HnswIndex(),
        IvfFlatIndex(),
        IvfPqIndex(),
    ]
    indexes = [
        index for index in available_indexes
        if index_names is None or index.name in index_names
    ]
    for index in indexes:
        index.build(chunks, embeddings)
    return indexes


def _hits_from_indices(
    index_name: str,
    chunks: list[Chunk],
    indices: np.ndarray,
    scores: np.ndarray,
) -> list[SearchHit]:
    hits = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        chunk = chunks[int(idx)]
        hits.append(SearchHit(rank, float(score), chunk.source, chunk.text, index_name))
    return hits


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def _subvector_slices(dim: int, requested_parts: int) -> list[slice]:
    parts = min(requested_parts, dim)
    base = dim // parts
    remainder = dim % parts
    slices = []
    start = 0

    for part_id in range(parts):
        width = base + (1 if part_id < remainder else 0)
        end = start + width
        slices.append(slice(start, end))
        start = end

    return slices
