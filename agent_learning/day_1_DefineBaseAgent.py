import os
#保证API key存在
assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY 未设置"
print("✅ Gemini API key available")

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

#保证重试配置
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

#定义Agent
root_agent = Agent(
    name="helpful_assistant",
    description="A simple agent that can answer general questions.",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
    ),
    instruction=(
        "You are a helpful assistant. "
        "Use Google Search for current information or when unsure."
    ),
    tools=[google_search],
)

#运行Agent
import asyncio

async def main():
    runner = InMemoryRunner(agent=root_agent)
    #这里也可以选run_debug查看中间过程,不能直接run的原因是没有自己构造session和记忆那些东西
    response = await runner.run_debug(
        "查询一下现在的北京时间" \
    )

    # print("\n===== AGENT RESPONSE =====\n")
    # print(response)

if __name__ == "__main__":
    asyncio.run(main())
