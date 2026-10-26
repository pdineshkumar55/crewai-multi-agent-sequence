# News Crew: A Two-Agent CrewAI Pipeline

A sequential [CrewAI](https://docs.crewai.com) crew where two agents work in a line:

1. **Researcher** searches the web with SerpApi and finds today's top 3 tech news stories.
2. **Writer** has no tools and cannot search. It turns the Researcher's findings into a newsletter that is **exactly 5 lines**.

```
┌──────────────────┐   3 stories (structured list)   ┌──────────────────┐
│    Researcher    │ ──────────────────────────────▶ │      Writer      │ ──▶ 5-line newsletter
│ tools: SerpApi   │      context=[research_task]    │   tools: none    │
└──────────────────┘                                 └──────────────────┘
          Task 1                Process.sequential             Task 2
```

---

## Project structure

```
news_crew/
├── news_crew.py      # all code: tool, agents, tasks, crew
├── .env.example      # template for API keys (safe to commit)
├── .env              # your real keys (never commit)
├── .gitignore
└── README.md
```

---

## Requirements

- Python **3.10 – 3.13** (CrewAI does not support 3.14+)
- An **OpenAI API key**: https://platform.openai.com/api-keys
- A **SerpApi API key**: https://serpapi.com/manage-api-key (the free tier is enough for testing)

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

- Windows: `.venv\Scripts\activate`
- macOS / Linux: `source .venv/bin/activate`

In VS Code, press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and pick `.venv`.

### 2. Install dependencies

```bash
pip install crewai crewai-tools python-dotenv requests
```

### 3. Configure API keys

Copy the template and fill in your real keys:

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

`.env.example`:

```env
# Copy this file to .env and fill in your real keys
OPENAI_API_KEY=your-openai-api-key-here
SERPAPI_API_KEY=your-serpapi-api-key-here

# Optional: which OpenAI model to use
OPENAI_MODEL_NAME=gpt-4o-mini
```

`.gitignore`:

```gitignore
.env
.venv/
__pycache__/
```

---

## Run

```bash
python news_crew.py
```

With `verbose=True`, the terminal shows each agent's reasoning, the Researcher's tool calls, and the hand-off. The run ends with:

```
============================================================
TODAY'S TECH NEWSLETTER
============================================================
<Headline with today's date>
<Story 1 in one sentence (Publication)>
<Story 2 in one sentence (Publication)>
<Story 3 in one sentence (Publication)>
<Closing takeaway>
============================================================
Line count: 5 ✅
```

---

## How it works

### The search tool: `SerpApiNewsTool`

The brief suggested `SerperDevTool`. This project uses **SerpApi** instead, through a small custom CrewAI tool (a `BaseTool` subclass) that calls `https://serpapi.com/search.json` with:

| Param | Value | Purpose |
|---|---|---|
| `engine` | `google` | Google search |
| `tbm` | `nws` | News results only |
| `tbs` | `qdr:d` | Past 24 hours, i.e. "today" |
| `num` | `10` | Up to 10 results per query |

It returns each result's title, source, date, snippet, and link as plain text the agent can read.

### The agents

| Agent | Tools | Delegation | Role |
|---|---|---|---|
| Researcher | `[SerpApiNewsTool]` | No | Finds and selects 3 distinct, significant stories |
| Writer | `[]` | No | Writes only from what it is given and cannot search |

`allow_delegation=False` on the Writer matters. Without it, the Writer could ask the Researcher to search on its behalf, which would break the "Writer cannot search" rule.

### The tasks

1. **`research_task`**: find today's top 3 tech stories. Its `expected_output` sets a fixed format for every story:
   ```
   1. Headline: ...
      Source: ...
      Date: ...
      Summary: ...
      Why it matters: ...
      URL: ...
   ```
   The fixed format gives the Writer clean, predictable input instead of raw search noise.

2. **`newsletter_task`**: turn the stories into the newsletter. Its `expected_output` states that the result must be **exactly 5 lines** and includes an example of the shape.

---

## Design decisions

### 1. How does the Writer get the Researcher's output?

In a sequential crew, a task does **not** automatically use the previous task's result. This project hands it over explicitly:

```python
newsletter_task = Task(
    ...,
    context=[research_task],   # inject the Researcher's final output
)
```

CrewAI adds the Researcher's final answer to the Writer's prompt. Remove this line and the Writer works with much less to go on.

### 2. What does "5 lines" mean?

**1 headline + 3 stories + 1 closing line:**

| Line | Content |
|---|---|
| 1 | Newsletter headline with today's date |
| 2 | Story 1, one sentence, ending with the real publication name in brackets |
| 3 | Story 2, same format |
| 4 | Story 3, same format |
| 5 | A one-sentence takeaway linking the stories |

Three stories fit naturally into lines 2–4, and the opening and closing lines make it read like a newsletter.

### 3. How is the 5-line format guaranteed?

`expected_output` is guidance the model usually follows but can ignore. In early runs the Writer merged everything into one paragraph and copied the placeholder `(Source)` word for word.

The fix is **structured output with Pydantic**:

```python
class Newsletter(BaseModel):
    headline: str
    story_1: str
    story_2: str
    story_3: str
    closing: str

    def as_lines(self) -> str:
        fields = [self.headline, self.story_1, self.story_2, self.story_3, self.closing]
        return "\n".join(" ".join(f.split()) for f in fields)
```

```python
newsletter_task = Task(..., output_pydantic=Newsletter)
```

The Writer fills five separate fields, and the code joins them with line breaks. `" ".join(f.split())` removes any stray line breaks inside a field, so the output is always exactly 5 lines. A line-count check prints at the end as a final verification.

**Takeaway:** describe the format in `expected_output` so the model knows what you want, and enforce it with a schema when your code depends on it.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Missing keys. Set OPENAI_API_KEY and SERPAPI_API_KEY...` | `.env` must be in the same folder as `news_crew.py`, with no quotes around values |
| `ModuleNotFoundError: No module named 'crewai'` | Activate `.venv`, or select it as the VS Code interpreter |
| CrewAI fails to install | Use Python 3.10 – 3.13 |
| `Search failed: 401 ...` | Invalid SerpApi key, or free-tier searches used up |
| `No news results found.` | Rare for broad queries. Try again, or widen `tbs` to `qdr:w` (past week) |
| `Line count` ⚠️ | Check that `output_pydantic=Newsletter` is set. Try `temperature=0` or `OPENAI_MODEL_NAME=gpt-4o` |

---

## Requirements checklist

- [x] All code in a single file: `news_crew.py`
- [x] Two agents, each with its own `role`, `goal`, and `backstory`
- [x] Researcher carries a web search tool (SerpApi, in place of SerperDevTool)
- [x] Writer carries no tools (`tools=[]`, `allow_delegation=False`)
- [x] Two tasks: find the top 3 stories, then write the newsletter
- [x] `process=Process.sequential`
- [x] Researcher's result passed explicitly via `context=[research_task]`
- [x] "Exactly 5 lines" stated in the Writer's `expected_output` (and enforced with Pydantic)
- [x] `verbose=True` on agents and crew
- [x] Both API keys read from environment variables (`.env` + `.env.example`)