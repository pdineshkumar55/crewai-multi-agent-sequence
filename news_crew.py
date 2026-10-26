"""
news_crew.py
A two-agent CrewAI crew:
  1. Researcher -> searches the web (SerpApi) for today's top 3 tech news stories
  2. Writer     -> turns them into a 5-line newsletter (no search tools)

The Writer returns a structured Pydantic object (5 fields), which is joined
into exactly 5 lines, so the format is guaranteed.
"""

import os
from datetime import date
from typing import Type

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool


# ---------------------------------------------------------------
# 1. Load API keys from environment variables (.env file)
# ---------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

if not OPENAI_API_KEY or not SERPAPI_API_KEY:
    raise SystemExit(
        "Missing keys. Set OPENAI_API_KEY and SERPAPI_API_KEY in your .env file."
    )

TODAY = date.today().strftime("%B %d, %Y")


# ---------------------------------------------------------------
# 2. Custom SerpApi search tool (replaces SerperDevTool)
# ---------------------------------------------------------------
class NewsSearchInput(BaseModel):
    """Input schema for the SerpApi news search tool."""
    query: str = Field(..., description="What to search for, e.g. 'technology news'")


class SerpApiNewsTool(BaseTool):
    name: str = "SerpApi News Search"
    description: str = (
        "Searches Google News through SerpApi for stories published in the "
        "last 24 hours. Returns title, source, date, snippet and link."
    )
    args_schema: Type[BaseModel] = NewsSearchInput

    def _run(self, query: str) -> str:
        params = {
            "engine": "google",
            "q": query,
            "tbm": "nws",       # news results only
            "tbs": "qdr:d",     # past 24 hours = "today"
            "num": 10,
            "api_key": SERPAPI_API_KEY,
        }
        try:
            response = requests.get(
                "https://serpapi.com/search.json", params=params, timeout=30
            )
            response.raise_for_status()
        except requests.RequestException as e:
            return f"Search failed: {e}"

        results = response.json().get("news_results", [])
        if not results:
            return "No news results found."

        lines = []
        for i, item in enumerate(results, start=1):
            lines.append(
                f"{i}. {item.get('title')}\n"
                f"   Source: {item.get('source')} | Date: {item.get('date')}\n"
                f"   Snippet: {item.get('snippet')}\n"
                f"   Link: {item.get('link')}"
            )
        return "\n\n".join(lines)


search_tool = SerpApiNewsTool()


# ---------------------------------------------------------------
# 3. Structured output model for the Writer (Option B)
# ---------------------------------------------------------------
class Newsletter(BaseModel):
    """Structured 5-line newsletter."""
    headline: str = Field(
        ..., description=f"Newsletter headline including the date {TODAY}"
    )
    story_1: str = Field(
        ...,
        description="Story 1 in one sentence, ending with the real publication "
                    "name in brackets, e.g. (Reuters)",
    )
    story_2: str = Field(
        ...,
        description="Story 2 in one sentence, ending with the real publication "
                    "name in brackets",
    )
    story_3: str = Field(
        ...,
        description="Story 3 in one sentence, ending with the real publication "
                    "name in brackets",
    )
    closing: str = Field(
        ..., description="One-sentence closing takeaway linking the stories"
    )

    def as_lines(self) -> str:
        """Join the 5 fields into exactly 5 lines."""
        fields = [self.headline, self.story_1, self.story_2, self.story_3, self.closing]
        # " ".join(f.split()) removes any stray newlines inside a single field
        return "\n".join(" ".join(f.split()) for f in fields)


# ---------------------------------------------------------------
# 4. LLM (uses the OpenAI key from .env)
# ---------------------------------------------------------------
llm = LLM(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.3)


# ---------------------------------------------------------------
# 5. Agents
# ---------------------------------------------------------------
researcher = Agent(
    role="Tech News Researcher",
    goal=f"Find the 3 most important technology news stories published today ({TODAY}).",
    backstory=(
        "You are a sharp technology journalist who scans the news every morning. "
        "You separate real, significant stories from hype and always note "
        "the source and link so facts can be checked."
    ),
    tools=[search_tool],          # the ONLY agent that can search
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

writer = Agent(
    role="Newsletter Writer",
    goal="Turn researched tech stories into a crisp, exactly 5-line newsletter.",
    backstory=(
        "You write a famously short daily tech newsletter. You never research "
        "on your own; you rely only on the notes your researcher gives you, "
        "and you respect strict length limits."
    ),
    tools=[],                     # NO tools: cannot search
    llm=llm,
    allow_delegation=False,       # cannot ask the researcher to search for it
    verbose=True,
)


# ---------------------------------------------------------------
# 6. Tasks
# ---------------------------------------------------------------
research_task = Task(
    description=(
        f"Today is {TODAY}. Use the SerpApi News Search tool to find today's "
        "top technology news. Try a few queries such as 'technology news', "
        "'AI news', 'tech industry news'. Pick the 3 most significant, distinct "
        "stories (no duplicates of the same event). Use only facts from the "
        "search results; do not invent details."
    ),
    expected_output=(
        "A numbered list of exactly 3 stories, in this exact format:\n\n"
        "1. Headline: <headline>\n"
        "   Source: <publication>\n"
        "   Date: <publish date>\n"
        "   Summary: <1-2 factual sentences>\n"
        "   Why it matters: <1 sentence>\n"
        "   URL: <link>\n\n"
        "(repeat for 2 and 3). No extra commentary before or after the list."
    ),
    agent=researcher,
)

newsletter_task = Task(
    description=(
        "Using ONLY the 3 researched stories provided in your context, write "
        "today's tech newsletter. Do not add stories or facts that are not in "
        "the research. Put each line on its OWN line, separated by a newline "
        "character. Never merge lines into a paragraph."
    ),
    expected_output=(
        "Exactly 5 separate lines of plain text (5 newline-separated lines, "
        "no blank lines, no markdown, nothing before or after).\n"
        f"Line 1: newsletter headline including the date ({TODAY})\n"
        "Line 2: story 1 in one sentence, ending with the real publication name in brackets\n"
        "Line 3: story 2 in one sentence, ending with the real publication name in brackets\n"
        "Line 4: story 3 in one sentence, ending with the real publication name in brackets\n"
        "Line 5: one-sentence closing takeaway\n\n"
        "Example of the exact shape (content is illustrative):\n"
        "Tech Brief | January 5, 2026: Chips, Cars and Clouds\n"
        "Nvidia unveiled a new AI chip aimed at data centres (Reuters)\n"
        "Tesla expanded robotaxi testing to three more cities (The Verge)\n"
        "AWS announced a major outage post-mortem (TechCrunch)\n"
        "Takeaway: AI hardware is setting the pace for the whole industry."
    ),
    agent=writer,
    context=[research_task],      # HAND-OFF: researcher's output goes to the writer
    output_pydantic=Newsletter,   # Option B: force a 5-field structured output
)


# ---------------------------------------------------------------
# 7. Crew: run the tasks in order
# ---------------------------------------------------------------
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, newsletter_task],
    process=Process.sequential,
    verbose=True,
)


# ---------------------------------------------------------------
# 8. Run
# ---------------------------------------------------------------
if __name__ == "__main__":
    result = crew.kickoff()

    if result.pydantic:                      # structured output came back
        newsletter = result.pydantic.as_lines()
    else:                                    # fallback to raw text
        newsletter = str(result).strip()

    lines = [line for line in newsletter.splitlines() if line.strip()]

    print("\n" + "=" * 60)
    print("TODAY'S TECH NEWSLETTER")
    print("=" * 60)
    print(newsletter)
    print("=" * 60)
    print(f"Line count: {len(lines)} {'✅' if len(lines) == 5 else '⚠️ expected 5'}")