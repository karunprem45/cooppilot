from app.skills import SkillExtractor


def test_aliases_and_symbols():
    ex = SkillExtractor({"C++": ["C++"], "Node.js": ["Node.js", "NodeJS"], "CI/CD": ["CI/CD"],
                         "scikit-learn": ["scikit-learn", "sklearn"]})
    text = "We use C++, NodeJS, CI/CD pipelines and sklearn."
    assert ex.extract(text) == ["C++", "CI/CD", "Node.js", "scikit-learn"]


def test_no_partial_word_match():
    ex = SkillExtractor({"Go": ["Go"], "Java": ["Java"]})
    assert ex.extract("Good knowledge of JavaScript") == []
