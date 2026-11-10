"""LangGraph agent: plan with tools, then pause for human review before finalizing."""
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

from app import config
from app.resources import Resources
from app.tools import make_tools

SYSTEM_PROMPT = """You help a student see how well they fit a job and what to build next.
Use the tools to ground every claim in the postings data. Do not invent numbers.
Return a short plan with these parts:
1. Fit summary (2-3 sentences)
2. Matched skills
3. Missing skills, most in-demand first, with the demand figure from skill_demand
4. Two or three project ideas that would close the biggest gaps
5. Similar postings worth applying to (by id and title)"""


class State(TypedDict):
    messages: Annotated[list, add_messages]
    draft_plan: str
    final_plan: str
    approved: bool


def _text(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") for b in msg.content if isinstance(b, dict))


def build_agent(res: Resources, llm=None):
    tools = make_tools(res)
    llm = (llm or ChatAnthropic(model=config.MODEL, temperature=0)).bind_tools(tools)

    def planner(state: State):
        return {"messages": [llm.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])]}

    def capture_draft(state: State):
        return {"draft_plan": _text(state["messages"][-1])}

    def human_review(state: State):
        decision = interrupt({"draft_plan": state["draft_plan"]})
        if decision.get("approved"):
            return {"approved": True, "final_plan": state["draft_plan"]}
        feedback = decision.get("feedback") or "Please revise the plan."
        return {"approved": False, "messages": [HumanMessage(f"Reviewer feedback: {feedback}\nRevise the plan.")]}

    g = StateGraph(State)
    g.add_node("planner", planner)
    g.add_node("tools", ToolNode(tools))
    g.add_node("capture_draft", capture_draft)
    g.add_node("human_review", human_review)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", tools_condition, {"tools": "tools", END: "capture_draft"})
    g.add_edge("tools", "planner")
    g.add_edge("capture_draft", "human_review")
    g.add_conditional_edges("human_review", lambda s: END if s["approved"] else "planner")
    return g.compile(checkpointer=MemorySaver())


def first_message(jd_text: str, resume_text: str, jd_skills: list[str], resume_skills: list[str]) -> HumanMessage:
    n = config.MAX_DOC_CHARS
    return HumanMessage(
        f"JOB DESCRIPTION:\n{jd_text[:n]}\n\nRESUME:\n{resume_text[:n]}\n\n"
        f"Skills found in the JD: {', '.join(jd_skills) or 'none'}\n"
        f"Skills found in the resume: {', '.join(resume_skills) or 'none'}"
    )
