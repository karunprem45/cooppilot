"""Runs the full LangGraph loop with a fake LLM, so no API key is needed."""
import pytest

pytest.importorskip("langgraph")
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from app.agent import build_agent, first_message  # noqa: E402
from app.resources import get_resources  # noqa: E402


class FakeToolLLM(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_interrupt_then_approve():
    llm = FakeToolLLM(messages=iter([AIMessage(content="Draft plan v1")]))
    agent = build_agent(get_resources("tfidf"), llm=llm)
    cfg = {"configurable": {"thread_id": "t1"}}
    agent.invoke({"messages": [first_message("jd", "resume", [], [])]}, cfg)
    assert agent.get_state(cfg).next == ("human_review",)
    agent.invoke(Command(resume={"approved": True}), cfg)
    assert agent.get_state(cfg).values["final_plan"] == "Draft plan v1"
