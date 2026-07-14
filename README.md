# Autonomous Research Agent

A file-centric state architecture for long-horizon research agents — inspired by **InfiAgent (ACL 2026)**.

Unlike conventional agents that dump everything into the LLM context (and break after ~20 steps), this agent **externalizes persistent state to the filesystem**. The LLM only sees a bounded "thinking record" at each step, enabling arbitrarily long research sessions without context degradation.

## Architecture

```
Query → Planner (decomposes into sub-questions)
         ↓
     [File System Workspace]
         ↓
     Tool Agents (parallel):
       ├─ Web Search (DuckDuckGo)
       ├─ Code Executor (Python sandbox)
       └─ File Retriever (reads findings)
         ↓
     Synthesizer (combines findings → draft report)
         ↓
     Reviewer (checks for gaps → loops to Planner if needed)
         ↓
     Final Report (final_report.md)
```

### File-Centric State

```
workspace/
├── plan.json              # Current plan, sub-task status, iteration
├── findings/
│   ├── sub_task_1.md     # Web search + code results per sub-task
│   └── sub_task_2.md
├── draft_report.md        # Before review
└── final_report.md        # After review passes
```

At each step, the LLM context is built from `plan.json` + last 3 findings (~4K tokens max) — **never grows**.

## Setup

```bash
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

Set your Groq API key:

```bash
set GROQ_API_KEY=gsk_...
```

## Usage

```bash
python run.py "Compare Mamba and Transformer architectures for long-context tasks"

# Or from file
python run.py -f query.txt

# Clean workspace between runs
python run.py --clean "Latest advances in RLHF 2026"
```

### Sample output

The agent:
1. Decomposes your query into 3-6 sub-questions
2. Searches the web for each
3. Runs code to verify benchmarks
4. Synthesizes a structured markdown report
5. Reviews for gaps and re-plans if needed
6. Saves `final_report.md` to workspace

## Project Structure

```
├── run.py                     # CLI entry point
├── research_agent/
│   ├── __init__.py
│   ├── parser.py              # File loading (PDF/DOCX/TXT)
│   ├── embedder.py            # Sentence embeddings
│   ├── llm_engine.py          # Groq LLM wrapper + JSON parsing
│   ├── planner.py             # Query decomposition
│   ├── tool_agents.py         # Web search, code exec, file tools
│   ├── synthesizer.py         # Report generation
│   ├── reviewer.py            # Gap detection + re-plan trigger
│   └── utils.py               # Helpers
├── tests/
│   ├── test_planner.py
│   └── test_tool_agents.py
└── requirements.txt
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **File-centric state** | LLM context stays bounded regardless of steps taken — no degradation |
| **Parallel tool agents** | Each sub-task searches/codes independently before synthesis |
| **Self-review loop** | Reviewer catches gaps and triggers re-plan (up to 3 iterations) |
| **DuckDuckGo search** | Free, no API key needed |
| **Groq Llama 3.1** | 1440 free requests/day, fast inference |
