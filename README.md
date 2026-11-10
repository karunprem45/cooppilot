# CoopPilot

An agentic RAG assistant that reads a job description and a resume, finds similar roles in a set of real job postings, checks a skill co-occurrence graph, and drafts a fit plan. The agent pauses for human review before the plan is final.

## How it works

```
JD + resume (PDF/txt)
      |
  document parsing (pypdf) + skill extraction (vocab regex)
      |
  LangGraph agent  <---->  tools
      |                     - search_similar_postings  (Chroma vector search, TF-IDF baseline)
      |                     - related_skills           (NetworkX skill co-occurrence graph)
      |                     - skill_demand             (share of postings with a skill)
      |                     - compare_skills           (matched vs missing)
      |
  draft plan  -->  human review (approve / request changes)  -->  final plan
```

- Backend: FastAPI (`/analyze`, `/review`, `/health`)
- UI: Streamlit
- Eval: hit@k and MRR on hand-labeled queries, vector search vs TF-IDF baseline, logged to MLflow
- Tests: pytest, run on every push with GitHub Actions; CI also builds the Docker image

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                   # then put your Anthropic API key in .env
pytest -q
uvicorn app.api:app --reload                           # terminal 1
streamlit run ui/streamlit_app.py                      # terminal 2, opens http://localhost:8501
```

Or with Docker: `docker compose up --build`, then open http://localhost:8501.

## Use your SkillRadar postings

The repo ships with 12 sample postings so it runs out of the box. To use real data, export your
SkillRadar postings to a CSV with columns `id,title,description` (optional `company`, and `skills`
as a `;`-separated list if you already extracted them). Then set `POSTINGS_CSV=path/to/file.csv`
in `.env`. If `skills` is missing, the vocab in `data/skills_vocab.txt` is used.

The first start with `RETRIEVER=chroma` downloads a small embedding model and builds the index in `.chroma/`.

## Evaluate retrieval

```bash
python -m eval.run_eval --k 5 --retrievers tfidf chroma
mlflow ui      # view runs at http://localhost:5000
```

`eval/questions.jsonl` only matches the sample data. For real numbers, write 25 to 30 questions
against your own postings, each with the posting ids that truly answer it. Label them before you
look at the results, so the test stays fair.

## Deploy on AWS App Runner

```bash
aws ecr create-repository --repository-name cooppilot
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
docker build --platform linux/amd64 -t cooppilot .
docker tag cooppilot:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/cooppilot:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/cooppilot:latest
```

In the App Runner console: create a service from the ECR image, port 8000, and add
`ANTHROPIC_API_KEY` as an environment variable (Secrets Manager is the better option).
App Runner is not free, so pause the service when you are not demoing it.

## Known limits

- Review state lives in memory (`MemorySaver`), so it resets on restart and needs a single instance.
- Skill extraction is vocabulary-based, so it only finds skills in `data/skills_vocab.txt`.
