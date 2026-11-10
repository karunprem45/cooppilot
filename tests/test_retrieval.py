from app.data import load_postings
from app.retrieval import TfidfRetriever
from app.skills import default_extractor


def test_tfidf_finds_relevant_posting():
    postings = load_postings("data/sample_postings.csv", default_extractor("data/skills_vocab.txt"))
    top = TfidfRetriever(postings).search("Airflow Spark ETL pipelines", k=3)
    assert top[0][0].id == "3"


def test_skills_extracted_when_column_missing():
    postings = load_postings("data/sample_postings.csv", default_extractor("data/skills_vocab.txt"))
    assert "LangGraph" in postings[0].skills
