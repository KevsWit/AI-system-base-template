from __future__ import annotations
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
import os
import re
import numpy as np
from rank_bm25 import BM25Okapi

# Embeddings backends: OpenAI o Sentence-Transformers (local).
# Ambos son opcionales; si ninguno está disponible, se usa solo BM25.
try:
    from openai import OpenAI as _OpenAI  # type: ignore
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# ----------------- Datos -----------------
@dataclass
class Document:
    doc_id: str
    title: str
    text: str


def make_docs(raw: List[Dict]) -> List[Document]:
    return [Document(doc_id=r["id"], title=r["title"], text=r["text"]) for r in raw]


# ----------------- Utilidades -----------------
_TOKEN_RE = re.compile(r"[a-záéíóúñ0-9]+")

def _tokens(s: str) -> List[str]:
    return _TOKEN_RE.findall(s.lower())


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / n


def _minmax(xs: np.ndarray) -> np.ndarray:
    if len(xs) == 0:
        return xs
    lo, hi = float(xs.min()), float(xs.max())
    return np.zeros_like(xs) if hi <= lo else (xs - lo) / (hi - lo)


# ----------------- Embeddings Provider -----------------
class _Embedder:
    """Abstracción del backend de embeddings (OpenAI o ST)."""

    def __init__(self):
        self.backend = None  # "openai" | "st" | None
        self._client = None
        self._model = None

        # 1) OpenAI si hay API key
        if _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            try:
                self._client = _OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.backend = "openai"
                self._openai_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            except Exception:
                self.backend = None

    def available(self) -> bool:
        return self.backend is not None

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.backend == "openai":
            # OpenAI retorna lista de vectores
            out = self._client.embeddings.create(model=self._openai_model, input=texts)  # type: ignore
            arr = np.array([d.embedding for d in out.data], dtype=np.float32)
            return _l2_normalize(arr)

        if self.backend == "st":
            arr = np.array(self._model.encode(texts, show_progress_bar=False), dtype=np.float32)  # type: ignore
            return _l2_normalize(arr)

        # Sin backend, devolvemos zeros para no romper (caller lo detecta).
        return np.zeros((len(texts), 1), dtype=np.float32)


# ----------------- Retriever Híbrido -----------------
class SimpleKeywordRetriever:
    """
    Retriever híbrido BM25 + Embeddings (si hay backend disponible).
    Mantiene el nombre original para no cambiar imports en el resto del sistema.
    """

    def __init__(
        self,
        docs: List[Document],
        alpha: float = 0.6,     # peso de embeddings
        bm25_k: int = 8,        # documentos candidatos desde BM25
        embedder: Optional[_Embedder] = None,
    ):
        """
        alpha: peso de la similitud por embeddings en el score final (0..1).
        bm25_k: tamaño del pool inicial de BM25 antes de fusionar con embeddings.
        embedder: si no se pasa, se resuelve automáticamente.
        """
        self.docs = docs
        self.alpha = float(alpha)
        self.bm25_k = int(bm25_k)

        # --- BM25 ---
        self._corpus_tokens = [_tokens(f"{d.title} {d.text}") for d in docs]
        self._bm25 = BM25Okapi(self._corpus_tokens)

        # --- Embeddings ---
        self._embedder = embedder or _Embedder()
        self._has_embeddings = self._embedder.available()
        self._doc_embeddings = None  # np.ndarray [N, D]
        if self._has_embeddings:
            # embebemos título + texto para mejor semántica
            texts = [f"{d.title}. {d.text}" for d in docs]
            self._doc_embeddings = self._embedder.embed(texts)  # [N, D]

    def invoke(self, query: str, k: int = 5) -> List[Document]:
        # --- 1) BM25 para recortar universo ---
        q_tokens = _tokens(query)
        bm25_scores = np.array(self._bm25.get_scores(q_tokens), dtype=np.float32)  # [N]
        # Top-N candidatos de BM25
        idx_bm25 = np.argsort(-bm25_scores)[: max(self.bm25_k, k)]

        # --- 2) Embeddings (si hay) en candidatos ---
        if self._has_embeddings and self._doc_embeddings is not None:
            q_vec = self._embedder.embed([query])[0]  # [D]
            cand_vecs = self._doc_embeddings[idx_bm25]  # [C, D]
            # cos sim (ya están L2-normalizados)
            emb_scores = cand_vecs @ q_vec  # [C]
            # normalizamos ambas escalas a [0,1]
            s_bm25 = _minmax(bm25_scores[idx_bm25])
            s_emb = _minmax(emb_scores)
            fused = (1.0 - self.alpha) * s_bm25 + self.alpha * s_emb
            order = np.argsort(-fused)[:k]
            final_idx = idx_bm25[order]
        else:
            # solo BM25
            final_idx = idx_bm25[:k]

        # filtra los de score cero cuando aplica
        if not np.any(bm25_scores[final_idx] > 0):
            return []

        return [self.docs[i] for i in final_idx.tolist()]
