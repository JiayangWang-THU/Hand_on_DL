from agent_factory import create_root_agent
import os, asyncio
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from typing import Dict
import re
# 准备所需要的tool function
def word_freq(text: str, top_k: int = 10) -> Dict:
    """
    Count word frequency.
    Use when you need top frequent words from a piece of text.

    Args:
      text: input text
      top_k: number of top words to return

    Returns:
      {"ok": True, "top": [{"word": str, "count": int}, ...]}
      or {"ok": False, "error": str}
    """
    if not isinstance(text, str) or not text.strip():
        return {"ok": False, "error": "text is empty"}
    if top_k <= 0 or top_k > 200:
        return {"ok": False, "error": "top_k must be in [1, 200]"}

    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1

    top = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    return {"ok": True, "top": [{"word": w, "count": c} for w, c in top]}


agent = create_root_agent(custom_tools=[word_freq])
async def main():
    runner = InMemoryRunner(agent=agent)
    await runner.run_debug("Count top 5 words in: hello hello world world world, hi!")

if __name__ == "__main__":
    asyncio.run(main())