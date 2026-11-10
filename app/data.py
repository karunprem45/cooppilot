"""Load job postings. Expected CSV columns: id, title, description; optional: company, skills (';'-separated)."""
from dataclasses import dataclass, field

import pandas as pd

from app.skills import SkillExtractor


@dataclass
class Posting:
    id: str
    title: str
    description: str
    company: str = ""
    skills: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.description}"


def load_postings(csv_path: str, extractor: SkillExtractor) -> list[Posting]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    missing = {"id", "title", "description"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    postings = []
    for row in df.to_dict(orient="records"):
        if row.get("skills"):
            skills = sorted({s.strip() for s in row["skills"].split(";") if s.strip()})
        else:
            skills = extractor.extract(f"{row['title']} {row['description']}")
        postings.append(Posting(row["id"], row["title"], row["description"], row.get("company", ""), skills))
    return postings
