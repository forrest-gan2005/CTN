import importlib.util
import logging
import threading
import warnings
from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger("semantic")

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = 100_000  # texts kept encoded (~150 MB), so refreshes only encode what is new
# Cosine similarity, measured on all-MiniLM-L6-v2: a reworded copy 0.92, the same opinion in other words 0.74,
# unrelated posts ~0.2; 92 real drafts from different personas never went above 0.69.
TWIN = 0.8  # one of our drafts this close to a teammate's recent post reads as the same voice
PARAPHRASE = 0.88  # posts this close from several accounts were spun from one text


class Embedder:
    """Unit-length sentence vectors, so a dot product is the cosine similarity. Loads the model on first use."""

    def __init__(self, model_name: str = MODEL, device: Optional[str] = None, cache_size: int = CACHE) -> None:
        self.model_name = model_name
        self.device = device  # None lets sentence-transformers pick CUDA, then Apple's MPS, then CPU
        self.cache_size = cache_size
        self._model = None
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        with self._lock:
            missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
            if missing:
                if self._model is None:
                    warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")  # macOS LibreSSL
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                vectors = self._model.encode(list(missing), batch_size=128, normalize_embeddings=True,
                                             convert_to_numpy=True, show_progress_bar=False)
                self._cache.update(zip(missing, vectors.astype(np.float32)))
            for t in texts:
                self._cache.move_to_end(t)
            out = np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, 0), np.float32)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return out


def load_embedder() -> Optional[Embedder]:
    """The embedder, or None when sentence-transformers isn't installed; every semantic check is then skipped."""
    if importlib.util.find_spec("sentence_transformers") is None:
        log.warning("sentence-transformers is not installed; semantic checks are off")
        return None
    return Embedder()


class NearIndex:
    """The latest texts and who wrote them, to find what a new text says the same thing as.

    A ring buffer: once full, each new text replaces the oldest.
    """

    def __init__(self, embedder: Embedder, capacity: int) -> None:
        self.embedder = embedder
        self.capacity = capacity
        self.keys = []  # whatever identifies each text, e.g. (author, post_id)
        self.vectors: Optional[np.ndarray] = None
        self.next = 0

    def __len__(self) -> int:
        return len(self.keys)

    def add(self, key, text: str, vector: Optional[np.ndarray] = None) -> None:
        if vector is None:
            vector = self.embedder.encode([text])[0]
        if self.vectors is None:
            self.vectors = np.zeros((self.capacity, vector.shape[0]), np.float32)
        self.vectors[self.next] = vector
        if len(self.keys) < self.capacity:
            self.keys.append(key)
        else:
            self.keys[self.next] = key
        self.next = (self.next + 1) % self.capacity

    def near(self, vector: np.ndarray, threshold: float) -> list:
        """(key, similarity) for every stored text at least `threshold` similar, closest first."""
        if not self.keys:
            return []
        scores = self.vectors[:len(self.keys)] @ vector
        hits = np.flatnonzero(scores >= threshold)
        return sorted(((self.keys[i], float(scores[i])) for i in hits), key=lambda h: -h[1])

    def closest(self, text: str) -> tuple:
        """(key, similarity) of the stored text closest to `text`, or (None, 0.0) when empty."""
        if not self.keys:
            return None, 0.0
        scores = self.vectors[:len(self.keys)] @ self.embedder.encode([text])[0]
        i = int(scores.argmax())
        return self.keys[i], float(scores[i])


if __name__ == "__main__":
    embedder = load_embedder()
    assert embedder is not None, "pip install sentence-transformers"
    copy = "Kingstonia deserves a referendum where every voice is heard and every family stays safe this season."
    spun = "Every voice in Kingstonia deserves to be heard in this referendum, and every family deserves to stay safe."
    other = "Ferry to the outer islands cancelled again. Third time this month."
    same_side = "Tourists cancel because of outbreaks, not because of a check at the door."
    v = embedder.encode([copy, spun, other, same_side])
    assert v.shape == (4, 384) and abs(float(v[0] @ v[0]) - 1) < 1e-4
    assert v[0] @ v[1] >= PARAPHRASE and v[0] @ v[2] < TWIN and v[0] @ v[3] < TWIN

    index = NearIndex(embedder, capacity=3)
    for i, text in enumerate([copy, other, same_side, "big swell this weekend, who's heading out"]):
        index.add(("acct", i), text)
    assert len(index) == 3 and ("acct", 0) not in index.keys, "the oldest text should have been replaced"
    assert index.closest(other) == (("acct", 1), index.closest(other)[1]) and index.closest(other)[1] > 0.99
    index.add(("acct", 4), copy)
    hits = index.near(embedder.encode([spun])[0], PARAPHRASE)
    assert [k for k, _ in hits] == [("acct", 4)]

    cached = len(embedder._cache)
    embedder.encode([copy, other])
    assert len(embedder._cache) == cached, "encoding known texts again must hit the cache"
    small = Embedder(cache_size=2)
    small.encode([copy, spun, other])
    assert list(small._cache) == [spun, other]
    print("semantic ok")
