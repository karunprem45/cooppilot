"""Two retrievers over the same postings: TF-IDF (baseline) and Chroma (embeddings)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from app.data import Posting


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, postings: list[Posting]):
        self.postings = postings
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([p.text for p in postings])

    def search(self, query: str, k: int = 5) -> list[tuple[Posting, float]]:
        scores = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        top = scores.argsort()[::-1][:k]
        return [(self.postings[i], float(scores[i])) for i in top]


class ChromaRetriever:
    name = "chroma"

    def __init__(self, postings: list[Posting], persist_dir: str = ".chroma", collection: str = "postings"):
        import chromadb  # imported here so the TF-IDF path works without it

        self.by_id = {p.id: p for p in postings}
        client = chromadb.PersistentClient(path=persist_dir)
        self.col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        if self.col.count() != len(postings):
            client.delete_collection(collection)
            self.col = client.create_collection(collection, metadata={"hnsw:space": "cosine"})
            for start in range(0, len(postings), 500):
                batch = postings[start:start + 500]
                self.col.add(ids=[p.id for p in batch], documents=[p.text[:4000] for p in batch])

    def search(self, query: str, k: int = 5) -> list[tuple[Posting, float]]:
        res = self.col.query(query_texts=[query], n_results=min(k, self.col.count()))
        return [(self.by_id[i], 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0])]


def build_retriever(kind: str, postings: list[Posting], persist_dir: str = ".chroma"):
    if kind == "tfidf":
        return TfidfRetriever(postings)
    if kind == "chroma":
        return ChromaRetriever(postings, persist_dir)
    raise ValueError(f"Unknown retriever: {kind}")
