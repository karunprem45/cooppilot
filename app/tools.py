"""Tools the agent can call. Each one returns plain text grounded in the postings data."""
from langchain_core.tools import tool

from app.resources import Resources


def make_tools(res: Resources):
    @tool
    def search_similar_postings(query: str, k: int = 5) -> str:
        """Find job postings most similar to the query (e.g. a role summary). Returns id, title, company and skills."""
        hits = res.retriever.search(query, k=k)
        lines = [f"[{p.id}] {p.title} at {p.company or 'n/a'} (score {s:.2f}) skills: {', '.join(p.skills) or 'none'}"
                 for p, s in hits]
        return "\n".join(lines) or "No postings found."

    @tool
    def related_skills(skill: str, k: int = 8) -> str:
        """List skills that most often appear in the same postings as the given skill, with co-occurrence counts."""
        rel = res.graph.related(skill, k=k)
        if not rel:
            return f"'{skill}' is not in the skill graph."
        return "\n".join(f"{s}: appears together in {w} postings" for s, w in rel)

    @tool
    def skill_demand(skill: str) -> str:
        """Share of all postings that mention a skill."""
        return f"{skill}: in {res.graph.demand(skill):.1%} of {res.graph.n_postings} postings"

    @tool
    def compare_skills(have: list[str], need: list[str]) -> str:
        """Compare skills the candidate has against skills a role needs. Returns matched and missing lists."""
        have_l = {h.lower() for h in have}
        matched = [n for n in need if n.lower() in have_l]
        missing = [n for n in need if n.lower() not in have_l]
        return f"Matched: {', '.join(matched) or 'none'}\nMissing: {', '.join(missing) or 'none'}"

    return [search_similar_postings, related_skills, skill_demand, compare_skills]
