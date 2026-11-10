"""Compare retrievers on hand-labeled queries. Reports hit@k and MRR, logs to MLflow if available.

Usage: python -m eval.run_eval --k 5 --retrievers tfidf chroma
IMPORTANT: questions.jsonl here only matches data/sample_postings.csv. Write your own
labeled questions against your real postings before using any number on a resume.
"""
import argparse
import json

from app import config
from app.data import load_postings
from app.retrieval import build_retriever
from app.skills import default_extractor


def evaluate(retriever, questions, k):
    hits, rr = 0, 0.0
    for q in questions:
        ids = [p.id for p, _ in retriever.search(q["query"], k=k)]
        rel = set(q["relevant_ids"])
        if rel & set(ids):
            hits += 1
            rr += 1.0 / (next(i for i, x in enumerate(ids) if x in rel) + 1)
    n = len(questions)
    return {f"hit_at_{k}": hits / n, "mrr": rr / n, "n_questions": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--retrievers", nargs="+", default=["tfidf", "chroma"])
    args = ap.parse_args()

    questions = [json.loads(l) for l in open(args.questions) if l.strip()]
    postings = load_postings(config.POSTINGS_CSV, default_extractor(config.SKILLS_VOCAB))
    try:
        import mlflow
        mlflow.set_experiment("cooppilot-retrieval")
    except ImportError:
        mlflow = None

    for kind in args.retrievers:
        metrics = evaluate(build_retriever(kind, postings, config.CHROMA_DIR), questions, args.k)
        print(kind, metrics)
        if mlflow:
            with mlflow.start_run(run_name=kind):
                mlflow.log_params({"retriever": kind, "k": args.k, "n_postings": len(postings)})
                mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
