import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# VectorAssessor: wraps sentence-transformer embedding model
# all-MiniLM-L6-v2 produces 384-dim vectors optimized for semantic similarity.
# Lighter than BERT (80MB vs 400MB), runs on CPU, purpose-built for this task.

_MODEL = None
_ENCODING_DIM = 384


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def embed_text(text):
    model = _get_model()
    # Truncate to first 512 tokens to stay within model window
    return model.encode(text, show_progress_bar=False)


def embed_documents(documents):
    model = _get_model()
    return model.encode(documents, show_progress_bar=False)


def compute_similarity(base_embedding, target_embeddings):
    base = base_embedding.reshape(1, -1)
    targets = np.array(target_embeddings)
    sims = cosine_similarity(base, targets)[0]
    return sims


class VectorAssessor:
    def __init__(self):
        self.model = _get_model()

    def embed(self, text):
        return self.model.encode(text, show_progress_bar=False)

    def similarity_matrix(self, base, candidates):
        return compute_similarity(base, candidates)
