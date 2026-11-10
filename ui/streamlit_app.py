import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="CoopPilot", layout="wide")
st.title("CoopPilot")
st.caption("Upload a job description and a resume. The agent drafts a fit plan and waits for your review.")

col1, col2 = st.columns(2)
jd = col1.file_uploader("Job description (PDF or .txt)", type=["pdf", "txt"])
resume = col2.file_uploader("Resume (PDF or .txt)", type=["pdf", "txt"])

if st.button("Analyze", disabled=not (jd and resume)):
    with st.spinner("Agent is working..."):
        r = requests.post(f"{API}/analyze", files={
            "jd": (jd.name, jd.getvalue()), "resume": (resume.name, resume.getvalue())}, timeout=300)
    if r.ok:
        st.session_state.run = r.json()
    else:
        st.error(r.text)

run = st.session_state.get("run")
if run:
    if run.get("jd_skills") is not None:
        st.write("**JD skills:**", ", ".join(run["jd_skills"]) or "none")
        st.write("**Resume skills:**", ", ".join(run["resume_skills"]) or "none")
    if run["status"] == "needs_review":
        st.subheader("Draft plan (needs your review)")
        st.markdown(run["draft_plan"])
        feedback = st.text_area("Feedback (only needed if you reject)")
        a, b = st.columns(2)
        for label, approved, holder in (("Approve", True, a), ("Request changes", False, b)):
            if holder.button(label):
                with st.spinner("Updating..."):
                    r = requests.post(f"{API}/review", json={
                        "thread_id": run["thread_id"], "approved": approved, "feedback": feedback}, timeout=300)
                st.session_state.run = {**run, **r.json()} if r.ok else run
                st.rerun()
    else:
        st.subheader("Final plan")
        st.markdown(run["final_plan"])
