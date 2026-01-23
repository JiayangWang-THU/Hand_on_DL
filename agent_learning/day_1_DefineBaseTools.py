import os, asyncio
from google.adk.runners import InMemoryRunner
from google.genai import types
from agent_factory import create_root_agent


def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b -1
async def main():
    runner = InMemoryRunner(agent=create_root_agent(custom_tools=[add_numbers]))
    resp = await runner.run_debug("caculate 1 plus 1 equals what? use add_numbers function to calculate it.")
    print(resp)

if __name__ == "__main__":
    asyncio.run(main())
