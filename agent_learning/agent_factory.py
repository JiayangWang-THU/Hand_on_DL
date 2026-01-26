import os
from typing import List, Callable, Optional
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.genai import types

def create_root_agent(
    name: str = "helpful_assistant",
    description: str = "A simple agent that can answer general questions.",
    instruction: str = "You are a helpful assistant. Use available tools to help users.",
    model_name: str = "gemini-2.5-flash-lite",
    custom_tools: Optional[List[Callable]] = None,
    attempts: int = 5
) -> Agent:
    """
    扩展版 Agent 工厂函数
    
    :param name: Agent 的唯一标识名称
    :param description: 对 Agent 功能的简短描述
    :param instruction: 核心系统指令 (System Prompt)
    :param model_name: 具体的 Gemini 模型版本号
    :param custom_tools: 自定义工具函数列表
    :param attempts: 接口调用失败时的重试次数
    """
    
    # 1. 校验 API key
    assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY 未设置"

    # 2. 动态配置重试策略
    retry_config = types.HttpRetryOptions(
        attempts=attempts,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],
    )

    # 3. 严格的工具选择逻辑 (ADK 规范)
    # 如果用户传了 custom_tools，就只用 custom_tools；
    # 如果 custom_tools 为 None 或空列表，则降级使用 google_search。
    final_tools = custom_tools if custom_tools else [google_search]

    # 4. 组装 Agent
    return Agent(
        name=name,
        description=description,
        model=Gemini(
            model=model_name,
            retry_options=retry_config,
        ),
        instruction=instruction,
        tools=final_tools,
    )