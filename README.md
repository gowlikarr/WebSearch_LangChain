# Web Research Agent (LangGraph + Tavily)

A LangGraph agent that plans focused search queries for a topic, searches the web for them in parallel with Tavily, checks whether the findings are sufficient (looping back for follow-up searches if not), and then asks an OpenAI model to write a structured report (Summary, Key Findings, Sources).

## Architecture

### LangGraph workflow

`build_graph()` in [Agent.py](Agent.py) compiles a four-node `StateGraph`. Solid arrows are fixed edges; dotted arrows are conditional edges whose target is decided at runtime.

```mermaid
flowchart LR
    START([START]) --> plan
    plan["<b>plan</b><br/>plan_queries()"] -. "fan_out_searches()<br/>Send × 2-3 sub-queries" .-> search
    search["<b>search</b> (parallel)<br/>search_web()"] --> review
    review["<b>review</b><br/>review_results()"] -. "gaps found and<br/>round < MAX_RESEARCH_ROUNDS<br/>Send × follow-up queries" .-> search
    review -. "sufficient or<br/>round limit reached" .-> synthesize
    synthesize["<b>synthesize</b><br/>synthesize_report()"] --> END([END])

    plan -. "ChatOpenAI → ResearchPlan" .-> OpenAI[(OpenAI API)]
    review -. "ChatOpenAI → ResearchReview" .-> OpenAI
    synthesize -. "ChatOpenAI" .-> OpenAI
    search -. "TavilySearch(max_results=5)" .-> Tavily[(Tavily Search API)]
```

### LangGraph features used

| Feature | Where | What it does here |
|---|---|---|
| Structured output | `plan`, `review` | `with_structured_output()` makes the LLM return a `ResearchPlan` / `ResearchReview` Pydantic object instead of free text. |
| Fan-out with `Send` | `fan_out_searches()`, `route_after_review()` | Starts one `search` node per query; they run in parallel in the same step. |
| Reducers | `search_results`, `messages` | `operator.add` merges results from the parallel searches; `add_messages` appends chat messages. |
| Fan-in | `search → review` | `review` runs once, after every parallel `search` in that step has finished. |
| Conditional edge + cycle | `route_after_review()` | Loops back to `search` for follow-up queries, or moves on to `synthesize`. |
| Loop guard | `MAX_RESEARCH_ROUNDS = 2` | Caps the number of search → review rounds so the loop always ends. |

### Shared state: `ResearchState`

| Key | Type | Written by | Merge behaviour |
|---|---|---|---|
| `query` | `str` | initial input | overwritten |
| `sub_queries` | `list[str]` | `plan` | overwritten |
| `search_results` | `list[dict]` | `search` (each parallel run) | appended (`operator.add`) |
| `follow_up_queries` | `list[str]` | `review` | overwritten |
| `research_round` | `int` | `review` | overwritten (incremented) |
| `report` | `str` | `synthesize` | overwritten |
| `messages` | `list` | `synthesize` | appended (`add_messages`) |

The `search` node doesn't receive the whole state: each `Send` passes it a small `SearchTask` (`{"sub_query": ...}`). Search results are de-duplicated by URL (`unique_results()`) before `review` and `synthesize` use them.

### How state flows through one run

```mermaid
sequenceDiagram
    autonumber
    participant CLI as main()
    participant G as Compiled graph
    participant P as plan
    participant S as search ×N (parallel)
    participant R as review
    participant Y as synthesize
    participant O as OpenAI API
    participant T as Tavily API

    CLI->>G: invoke(initial state, research_round = 0)
    G->>P: state
    P->>O: break query into sub-queries
    O-->>P: ResearchPlan
    P-->>G: {sub_queries}
    G->>S: Send(sub_query) for each sub-query
    S->>T: search each sub-query
    T-->>S: top 5 results each
    S-->>G: {search_results} (merged with operator.add)
    G->>R: state (after all searches finish)
    R->>O: are these results sufficient?
    O-->>R: ResearchReview
    R-->>G: {research_round + 1, follow_up_queries}
    alt gaps found and round < MAX_RESEARCH_ROUNDS
        G->>S: Send(follow_up) for each follow-up query
        Note over S,R: search → review repeats
    else sufficient or limit reached
        G->>Y: state
        Y->>O: write report from unique results
        O-->>Y: AIMessage
        Y-->>G: {report, messages}
    end
    G-->>CLI: final state
    CLI->>CLI: print sub-queries, rounds, report
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirement.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```
OPENAI_API_KEY=...
TAVILY_API_KEY=...
```

## Run

```powershell
python Agent.py --query "latest advances in quantum computing"
```

Tune `MAX_RESEARCH_ROUNDS` in [Agent.py](Agent.py) to trade thoroughness for cost: each extra round adds up to 2 Tavily searches and one more OpenAI review call.
