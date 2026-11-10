import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL = os.getenv("MODEL", "claude-sonnet-5-5")
POSTINGS_CSV = os.getenv("POSTINGS_CSV", "data/sample_postings.csv")
SKILLS_VOCAB = os.getenv("SKILLS_VOCAB", "data/skills_vocab.txt")
RETRIEVER = os.getenv("RETRIEVER", "chroma")  # "chroma" or "tfidf"
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
MAX_DOC_CHARS = int(os.getenv("MAX_DOC_CHARS", "6000"))
