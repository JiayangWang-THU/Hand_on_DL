import os
from typing import List, Callable, Optional
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.genai import types

import os
from typing import List, Callable, Optional
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.genai import types

def create_root_agent(
    name: str = "helpful_assistant",
    output_key: str = "agent_output",
    description: str = "A simple agent that can answer general questions.",
    instruction: str = "You are a helpful assistant. Use available tools to help users.",
    model_name: str = "gemini-2.5-flash-lite",
    custom_tools: Optional[List[Callable]] = None,
    attempts: int = 5
) -> Agent:
    """
    扩展版 Agent 工厂函数
    
    :param name: Agent 的唯一标识名称
    :param output_key: Agent 输出结果在上下文中的存储键名 (用于多 Agent 协作)
    :param description: 对 Agent 功能的简短描述
    :param instruction: 核心系统指令 (System Prompt)
    :param model_name: 具体的 Gemini 模型版本号
    :param custom_tools: 自定义工具函数列表 (传入则禁用默认搜索)
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
    # 只要用户传了 custom_tools（哪怕是空列表 []），我们就尊重用户的选择
    # 只有在完全不传 custom_tools (None) 的情况下，才开启默认搜索
    final_tools = custom_tools if custom_tools is not None else [google_search]

    # 4. 组装并返回 Agent
    return Agent(
        name=name,
        output_key=output_key,  # 这里的 output_key 决定了后续步骤如何调用该结果
        description=description,
        model=Gemini(
            model=model_name,
            retry_options=retry_config,
        ),
        instruction=instruction,
        tools=final_tools,
    )