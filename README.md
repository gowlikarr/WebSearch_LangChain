# Web Research Agent (LangGraph + Tavily)

A small LangGraph agent that searches the web for a topic with Tavily, then asks an OpenAI model to turn the results into a structured report (Summary, Key Findings, Sources).

## Architecture

### LangGraph workflow

`build_graph()` in [Agent.py](Agent.py) compiles a two-node `StateGraph`. Each node reads the shared `ResearchState` and returns only the keys it changes; LangGraph merges those updates into the state before running the next node.

```mermaid
flowchart LR
    START([START]) --> search
    search["<b>search</b><br/>search_web()"] --> synthesize
    synthesize["<b>synthesize</b><br/>synthesize_report()"] --> END([END])

    search -. "TavilySearch(max_results=5)" .-> Tavily[(Tavily Search API)]
    synthesize -. "ChatOpenAI(gpt-4o-mini)" .-> OpenAI[(OpenAI API)]
```

### Shared state: `ResearchState`

| Key | Type | Written by | Merge behaviour |
|---|---|---|---|
| `query` | `str` | initial input | overwritten |
| `search_results` | `list[dict]` | `search` node | overwritten |
| `report` | `str` | `synthesize` node | overwritten |
| `messages` | `list` | `synthesize` node | appended (`add_messages` reducer) |

### How state flows through one run

```mermaid
sequenceDiagram
    autonumber
    participant CLI as main()
    participant G as Compiled graph
    participant S as search node
    participant T as Tavily API
    participant Y as synthesize node
    participant O as OpenAI API

    CLI->>G: invoke({query, messages: [], search_results: [], report: ""})
    G->>S: state
    S->>T: query
    T-->>S: top 5 results (url, title, content)
    S-->>G: {search_results}
    Note over G: state.search_results updated
    G->>Y: state
    Y->>O: system prompt + query + results (500 chars each)
    O-->>Y: AIMessage (report)
    Y-->>G: {report, messages: [AIMessage]}
    Note over G: report set, AIMessage appended to messages
    G-->>CLI: final state
    CLI->>CLI: print(result["report"])
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
