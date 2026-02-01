from agent_factory import create_root_agent
import asyncio
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool
from typing import Dict
import re
# 准备所需要的tool function
# def word_freq(text: str, top_k: int = 10) -> Dict:
#     """
#     Count word frequency.
#     Use when you need top frequent words from a piece of text.

#     Args:
#       text: input text
#       top_k: number of top words to return

#     Returns:
#       {"ok": True, "top": [{"word": str, "count": int}, ...]}
#       or {"ok": False, "error": str}
#     """
#     if not isinstance(text, str) or not text.strip():
#         return {"ok": False, "error": "text is empty"}
#     if top_k <= 0 or top_k > 200:
#         return {"ok": False, "error": "top_k must be in [1, 200]"}

#     tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
#     freq: Dict[str, int] = {}
#     for t in tokens:
#         freq[t] = freq.get(t, 0) + 1

#     top = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:top_k]
#     return {"ok": True, "top": [{"word": w, "count": c} for w, c in top]}


# agent = create_root_agent(custom_tools=[word_freq])
research_agent = create_root_agent(
    name="research_agent",
    instruction=(
        "You are a specialized research agent. "
        "Use google_search to find 2-3 relevant pieces of info and include citations."
        "You must return a JSON array, each item containing title, url, quote_or_snippet, and date (at least the first three)."
    ),
    output_key="research_findings",
)
summarizer_agent = create_root_agent(
    name="summarizer_agent",
     instruction=(
        "Read: {research_findings}\n"
        "Create a concise bulleted summary with 3-5 key points."
        "Each bullet must be followed by (source: <url>)."
    ),
    output_key="summary_report",
)
root_agent = create_root_agent(
    name="ResearchCoordinator",
    instruction=(
            "Orchestrate a workflow:\n"
            "1) Call ResearchAgent\n"
            "2) Call SummarizerAgent\n"
            "3) Output 2–3 citationable facts (with URL/source name/date/one-sentence excerpt)"
        ),
    custom_tools=[
        AgentTool(agent=research_agent), 
        AgentTool(agent=summarizer_agent)
    ],
)
async def main():
    runner = InMemoryRunner(agent=root_agent)
    await runner.run_debug("What are the latest advancements \
                           in quantum computing and what do they mean for AI?")

if __name__ == "__main__":
    asyncio.run(main())