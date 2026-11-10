"""FastAPI backend. /analyze runs the agent until it pauses for review; /review resumes it."""
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.documents import extract_text
from app.resources import get_resources

app = FastAPI(title="CoopPilot")
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from app.agent import build_agent  # lazy so /health works without an API key
        _agent = build_agent(get_resources())
    return _agent


class Review(BaseModel):
    thread_id: str
    approved: bool
    feedback: str = ""


def _result(thread_id: str, config: dict) -> dict:
    state = get_agent().get_state(config)
    values = state.values
    if state.next:  # still paused at human_review
        return {"thread_id": thread_id, "status": "needs_review", "draft_plan": values.get("draft_plan", "")}
    return {"thread_id": thread_id, "status": "done", "final_plan": values.get("final_plan", "")}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(jd: UploadFile = File(...), resume: UploadFile = File(...)):
    from app.agent import first_message

    jd_text = extract_text(await jd.read(), jd.filename or "")
    resume_text = extract_text(await resume.read(), resume.filename or "")
    if not jd_text or not resume_text:
        raise HTTPException(400, "Could not read text from one of the files.")
    res = get_resources()
    jd_skills, resume_skills = res.extractor.extract(jd_text), res.extractor.extract(resume_text)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    get_agent().invoke({"messages": [first_message(jd_text, resume_text, jd_skills, resume_skills)]}, config)
    out = _result(thread_id, config)
    out.update(jd_skills=jd_skills, resume_skills=resume_skills)
    return out


@app.post("/review")
def review(body: Review):
    from langgraph.types import Command

    config = {"configurable": {"thread_id": body.thread_id}}
    if not get_agent().get_state(config).next:
        raise HTTPException(404, "No paused run for this thread_id.")
    get_agent().invoke(Command(resume={"approved": body.approved, "feedback": body.feedback}), config)
    return _result(body.thread_id, config)
