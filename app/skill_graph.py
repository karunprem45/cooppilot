"""Skill co-occurrence graph: the 'graph' half of the GraphRAG setup."""
from itertools import combinations

import networkx as nx

from app.data import Posting


class SkillGraph:
    def __init__(self, postings: list[Posting]):
        self.n_postings = len(postings)
        self.g = nx.Graph()
        for p in postings:
            for s in set(p.skills):
                if self.g.has_node(s):
                    self.g.nodes[s]["count"] += 1
                else:
                    self.g.add_node(s, count=1)
            for a, b in combinations(sorted(set(p.skills)), 2):
                w = self.g.edges[a, b]["weight"] + 1 if self.g.has_edge(a, b) else 1
                self.g.add_edge(a, b, weight=w)

    def _find(self, skill: str) -> str | None:
        lookup = {n.lower(): n for n in self.g.nodes}
        return lookup.get(skill.lower())

    def demand(self, skill: str) -> float:
        node = self._find(skill)
        return 0.0 if node is None else self.g.nodes[node]["count"] / max(self.n_postings, 1)

    def related(self, skill: str, k: int = 8) -> list[tuple[str, int]]:
        node = self._find(skill)
        if node is None:
            return []
        nbrs = sorted(self.g[node].items(), key=lambda kv: kv[1]["weight"], reverse=True)
        return [(n, d["weight"]) for n, d in nbrs[:k]]
