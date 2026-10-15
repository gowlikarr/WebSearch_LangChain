"""
Web Research Agent using LangGraph + Tavily Search.

Plans focused sub-queries for a topic, searches the web for them in parallel,
reviews whether the findings are sufficient (looping back for follow-up
searches if not), then synthesizes a structured research report.

Usage:
    python agent.py
    python agent.py --query "latest advances in quantum computing"
"""

import argparse
import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from pydantic import BaseModel, Field

load_dotenv()

MAX_RESEARCH_ROUNDS = 2


class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    sub_queries: list[str]
    # operator.add lets parallel search nodes each append their results
    search_results: Annotated[list[dict], operator.add]
    follow_up_queries: list[str]
    research_round: int
    report: str


class SearchTask(TypedDict):
    sub_query: str


class ResearchPlan(BaseModel):
    sub_queries: list[str] = Field(description="2-3 focused web search queries that together cover the research question")


class ResearchReview(BaseModel):
    is_sufficient: bool = Field(description="True if the results cover the research question well enough to write a report")
    follow_up_queries: list[str] = Field(default_factory=list, description="Up to 2 search queries that would fill the gaps; empty if sufficient")


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def unique_results(results: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for r in results:
        url = r.get("url")
        if url not in seen:
            seen.add(url)
            unique.append(r)
    return unique


def plan_queries(state: ResearchState) -> ResearchState:
    planner = get_llm().with_structured_output(ResearchPlan)
    plan = planner.invoke([
        SystemMessage(content="You are a research planner. Break the research question into 2-3 focused, non-overlapping web search queries."),
        HumanMessage(content=state["query"]),
    ])
    return {"sub_queries": plan.sub_queries[:3]}


def fan_out_searches(state: ResearchState) -> list[Send]:
    return [Send("search", {"sub_query": q}) for q in state["sub_queries"]]


def search_web(task: SearchTask) -> ResearchState:
    tool = TavilySearch(max_results=5)
    raw_results = tool.invoke(task["sub_query"])
    if isinstance(raw_results, dict):
        results = raw_results.get("results", [])
    elif isinstance(raw_results, list):
        results = raw_results
    else:
        results = []
    return {"search_results": [{**r, "sub_query": task["sub_query"]} for r in results]}


def review_results(state: ResearchState) -> ResearchState:
    reviewer = get_llm().with_structured_output(ResearchReview)

    results_text = "\n".join(
        f"- {r.get('title', 'N/A')}: {r.get('content', '')[:200]}"
        for r in unique_results(state["search_results"])
    )

    review = reviewer.invoke([
        SystemMessage(content="You are a research reviewer. Decide whether the search results are sufficient to answer the research question. If not, suggest up to 2 follow-up search queries targeting the gaps."),
        HumanMessage(content=f"Research query: {state['query']}\n\nSearch results:\n{results_text}"),
    ])

    follow_ups = [] if review.is_sufficient else review.follow_up_queries[:2]
    return {"research_round": state["research_round"] + 1, "follow_up_queries": follow_ups}


def route_after_review(state: ResearchState):
    if state["follow_up_queries"] and state["research_round"] < MAX_RESEARCH_ROUNDS:
        return [Send("search", {"sub_query": q}) for q in state["follow_up_queries"]]
    return "synthesize"


def synthesize_report(state: ResearchState) -> ResearchState:
    llm = get_llm()

    results_text = "\n\n".join(
        f"Source: {r.get('url', 'N/A')}\nTitle: {r.get('title', 'N/A')}\nContent: {r.get('content', '')[:500]}"
        for r in unique_results(state["search_results"])
    )

    messages = [
        SystemMessage(content="You are a research analyst. Synthesize the search results into a clear, structured report with: Summary, Key Findings (bullet points), and Sources."),
        HumanMessage(content=f"Research query: {state['query']}\n\nSearch results:\n{results_text}"),
    ]

    response = llm.invoke(messages)
    return {"report": response.content, "messages": [response]}


def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_queries)
    graph.add_node("search", search_web)
    graph.add_node("review", review_results)
    graph.add_node("synthesize", synthesize_report)
    graph.set_entry_point("plan")
    graph.add_conditional_edges("plan", fan_out_searches, ["search"])
    graph.add_edge("search", "review")
    graph.add_conditional_edges("review", route_after_review, ["search", "synthesize"])
    graph.add_edge("synthesize", END)
    return graph.compile()


def main():
    parser = argparse.ArgumentParser(description="Web Research Agent")
    parser.add_argument("--query", default="latest advances in quantum computing", help="Research query")
    args = parser.parse_args()

    print(f"\n🔍 Researching: {args.query}\n")

    agent = build_graph()
    result = agent.invoke({
        "query": args.query,
        "messages": [],
        "sub_queries": [],
        "search_results": [],
        "follow_up_queries": [],
        "research_round": 0,
        "report": "",
    })

    print("🧭 Sub-queries:", ", ".join(result["sub_queries"]))
    print(f"🔁 Research rounds: {result['research_round']}, unique sources: {len(unique_results(result['search_results']))}\n")
    print("=" * 60)
    print("📄 RESEARCH REPORT")
    print("=" * 60)
    print(result["report"])


if __name__ == "__main__":
    main()
