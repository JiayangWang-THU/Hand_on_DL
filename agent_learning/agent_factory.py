# agent_factory.py
import os
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.genai import types
from typing import List, Callable, Any, Optional

def create_root_agent(custom_tools: Optional[List[Callable]] = None) -> Agent:
    # 1. 校验 API key
    assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY 未设置"

    # 2. retry 配置
    retry_config = types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],
    )

    # 3. 处理工具列表
    # 如果调用时没传工具，就默认空列表；如果传了，就用传进来的
    final_tools = custom_tools if custom_tools is not None else [google_search]

    # 4. 定义并返回 Agent
    return Agent(
        name="helpful_assistant",
        description="A simple agent that can answer general questions.",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config,
        ),
        instruction=(
            "You are a helpful assistant. "
            "Use available tools to help users. If no tool fits, answer based on your knowledge."
        ),
        tools=final_tools,
    )