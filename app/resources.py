"""Load postings, retriever and skill graph once and share them."""
from dataclasses import dataclass
from functools import lru_cache

from app import config
from app.data import Posting, load_postings
from app.retrieval import build_retriever
from app.skill_graph import SkillGraph
from app.skills import SkillExtractor, default_extractor


@dataclass
class Resources:
    postings: list[Posting]
    extractor: SkillExtractor
    retriever: object
    graph: SkillGraph


@lru_cache
def get_resources(retriever_kind: str | None = None) -> Resources:
    extractor = default_extractor(config.SKILLS_VOCAB)
    postings = load_postings(config.POSTINGS_CSV, extractor)
    retriever = build_retriever(retriever_kind or config.RETRIEVER, postings, config.CHROMA_DIR)
    return Resources(postings, extractor, retriever, SkillGraph(postings))
