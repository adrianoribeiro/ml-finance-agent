import logging
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import Config

logger = logging.getLogger(__name__)

DOCS_DIR = Config.DOCS_DIR
INDEX_PATH = Config.FAISS_INDEX_PATH
CHUNKS_PATH = Config.CHUNKS_PATH


def load_and_chunk(docs_dir: str, chunk_size: int = 300) -> list[str]:
    """Read all .txt files and split into chunks by paragraph."""
    chunks = []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(docs_dir, filename)) as f:
            text = f.read()

        # Split by double newline (paragraphs)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # Merge small paragraphs, split large ones
        current = ""
        for p in paragraphs:
            if len(current) + len(p) < chunk_size:
                current = current + "\n" + p if current else p
            else:
                if current:
                    chunks.append(current)
                current = p
        if current:
            chunks.append(current)

    return chunks


def build_index(docs_dir: str = DOCS_DIR):
    """Build FAISS index from documents."""
    chunks = load_and_chunk(docs_dir)
    logger.info(f"Loaded {len(chunks)} chunks from {docs_dir}")

    model = SentenceTransformer(Config.EMBEDDING_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    np.save(CHUNKS_PATH, np.array(chunks, dtype=object))

    logger.info(f"Index saved to {INDEX_PATH} ({index.ntotal} vectors)")
    return index, chunks


class RAGRetriever:
    def __init__(self):
        if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
            build_index()

        self.index = faiss.read_index(INDEX_PATH)
        self.chunks = list(np.load(CHUNKS_PATH, allow_pickle=True))
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)

    def search(self, query: str, k: int = 3) -> list[str]:
        embedding = self.model.encode([query]).astype("float32")
        _, indices = self.index.search(embedding, k)
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
