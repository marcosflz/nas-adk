from __future__ import annotations

import pytest

from nas.adk.tools.function_tool import FunctionTool


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def greet(name: str, greeting: str = "Hello") -> str:
    """Greet someone."""
    return f"{greeting}, {name}!"


@pytest.mark.asyncio
async def test_function_tool_basic() -> None:
    tool = FunctionTool(add)
    result = await tool.run_async(a=1, b=2)
    assert result == 3


@pytest.mark.asyncio
async def test_function_tool_default_args() -> None:
    tool = FunctionTool(greet)
    result = await tool.run_async(name="World")
    assert result == "Hello, World!"


@pytest.mark.asyncio
async def test_function_tool_overridden_default() -> None:
    tool = FunctionTool(greet)
    result = await tool.run_async(name="World", greeting="Hi")
    assert result == "Hi, World!"


def test_function_tool_schema() -> None:
    tool = FunctionTool(add)
    schema = tool.to_llm_tool()
    assert schema["function"]["name"] == "add"
    assert schema["function"]["description"] == "Add two numbers."
    params = schema["function"]["parameters"]
    assert params["required"] == ["a", "b"]
    assert params["properties"]["a"]["type"] == "integer"


@pytest.mark.asyncio
async def test_function_tool_coercion() -> None:
    tool = FunctionTool(add)
    # The LLM sends string args that need parsing
    result = await tool.run_async(a="3", b="4")
    assert result == 7
