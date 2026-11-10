from app.data import Posting
from app.skill_graph import SkillGraph


def _graph():
    return SkillGraph([
        Posting("1", "a", "", skills=["Python", "Docker", "AWS"]),
        Posting("2", "b", "", skills=["Python", "Docker"]),
        Posting("3", "c", "", skills=["SQL"]),
    ])


def test_related_sorted_by_weight():
    assert _graph().related("python")[0] == ("Docker", 2)


def test_demand_and_unknown():
    g = _graph()
    assert abs(g.demand("Python") - 2 / 3) < 1e-9
    assert g.related("Rust") == [] and g.demand("Rust") == 0.0
