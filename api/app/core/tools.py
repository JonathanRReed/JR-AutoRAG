"""Tool registry and execution for agentic RAG.

This module provides local tools for RAG agents:
- Calculator for math expressions
- Date/time information
- Document search (internal RAG)
"""

from __future__ import annotations

import ast
import operator
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: str | None = None
    execution_time_ms: float = 0.0


class Tool(ABC):
    """Base class for RAG agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for identification."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description for LLM to understand when to use this tool."""
        pass

    @property
    def parameters(self) -> dict[str, str]:
        """Parameter descriptions for the tool."""
        return {}

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def can_handle(self, query: str) -> float:
        """Return confidence (0-1) that this tool can handle the query."""
        return 0.0


class CalculatorTool(Tool):
    """Simple calculator for mathematical expressions."""

    MAX_EXPRESSION_LENGTH = 120
    MAX_ABS_VALUE = 1_000_000_000_000
    MAX_ABS_RESULT = 1_000_000_000_000_000
    ALLOWED_BINOPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
    }
    ALLOWED_UNARYOPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluates mathematical expressions. Use for arithmetic, percentages, and basic math."

    @property
    def parameters(self) -> dict[str, str]:
        return {"expression": "Mathematical expression to evaluate (e.g., '2 + 2', '15 * 0.2')"}

    # Patterns that suggest math
    MATH_PATTERNS = [
        r'\b\d+\s*[\+\-\*\/\%]\s*\d+',  # 2 + 2
        r'\b(calculate|compute|what is)\b.*\d+',
        r'\b\d+\s*percent\b',
        r'\b(sum|total|average|mean)\b.*\d+',
    ]

    def can_handle(self, query: str) -> float:
        for pattern in self.MATH_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return 0.8
        return 0.0

    def _normalize(self, expression: str) -> str:
        expr = expression.strip()
        if len(expr) > self.MAX_EXPRESSION_LENGTH:
            raise ValueError("Expression is too long")
        if re.search(r"[^0-9+\-*/().%\s]", expr):
            raise ValueError("Expression contains unsupported characters")
        expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
        return expr

    def _evaluate_node(self, node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return self._evaluate_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            if abs(node.value) > self.MAX_ABS_VALUE:
                raise ValueError("Number is too large")
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in self.ALLOWED_UNARYOPS:
            result = self.ALLOWED_UNARYOPS[type(node.op)](self._evaluate_node(node.operand))
            return self._check_result(result)
        if isinstance(node, ast.BinOp) and type(node.op) in self.ALLOWED_BINOPS:
            left = self._evaluate_node(node.left)
            right = self._evaluate_node(node.right)
            result = self.ALLOWED_BINOPS[type(node.op)](left, right)
            return self._check_result(result)
        raise ValueError("Unsupported expression")

    def _check_result(self, value: float | int) -> float | int:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Result is not finite")
        if abs(value) > self.MAX_ABS_RESULT:
            raise ValueError("Result is too large")
        return value

    def execute(self, expression: str = "", **kwargs) -> ToolResult:
        import time
        start = time.perf_counter()

        try:
            safe_expr = self._normalize(expression)
            if not safe_expr.strip():
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    result=None,
                    error="Invalid expression",
                )

            parsed = ast.parse(safe_expr, mode="eval")
            result = self._check_result(self._evaluate_node(parsed))

            return ToolResult(
                tool_name=self.name,
                success=True,
                result=result,
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )


class DateTimeTool(Tool):
    """Provides current date and time information."""

    @property
    def name(self) -> str:
        return "datetime"

    @property
    def description(self) -> str:
        return "Provides current date, time, or timezone information."

    DATE_PATTERNS = [
        r'\b(what|current).*(date|time|day|today)\b',
        r'\btime\s*(is|now)\b',
        r'\b(today|now)\b.*\b(date|time)\b',
    ]

    def can_handle(self, query: str) -> float:
        for pattern in self.DATE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return 0.9
        return 0.0

    def execute(self, format: str = "full", **kwargs) -> ToolResult:
        import time
        start = time.perf_counter()

        now = datetime.now(UTC)

        result = {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": "UTC",
            "day_of_week": now.strftime("%A"),
            "iso": now.isoformat(),
        }

        return ToolResult(
            tool_name=self.name,
            success=True,
            result=result,
            execution_time_ms=(time.perf_counter() - start) * 1000,
        )


class DocumentSearchTool(Tool):
    """Internal document search via RAG retrieval."""

    def __init__(self, retrieval_callback: Callable | None = None) -> None:
        self._retrieval_callback = retrieval_callback

    @property
    def name(self) -> str:
        return "search_documents"

    @property
    def description(self) -> str:
        return "Search through ingested documents for relevant information."

    @property
    def parameters(self) -> dict[str, str]:
        return {"query": "Search query to find relevant documents"}

    def can_handle(self, query: str) -> float:
        # This tool should always be available as fallback
        return 0.3

    def execute(self, query: str = "", **kwargs) -> ToolResult:
        import time
        start = time.perf_counter()

        if not self._retrieval_callback:
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=None,
                error="Retrieval callback not configured",
            )

        try:
            results = self._retrieval_callback(query)
            return ToolResult(
                tool_name=self.name,
                success=True,
                result=results,
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # Register built-in tools
        self.register(CalculatorTool())
        self.register(DateTimeTool())

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, str]]:
        """List all available tools with descriptions."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def find_tool_for_query(self, query: str) -> tuple[Tool | None, float]:
        """Find the best tool for a query.

        Returns:
            Tuple of (best_tool, confidence) or (None, 0.0)
        """
        best_tool = None
        best_confidence = 0.0

        for tool in self._tools.values():
            confidence = tool.can_handle(query)
            if confidence > best_confidence:
                best_confidence = confidence
                best_tool = tool

        return best_tool, best_confidence

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found",
            )
        return tool.execute(**kwargs)


__all__ = [
    "Tool",
    "ToolResult",
    "CalculatorTool",
    "DateTimeTool",
    "DocumentSearchTool",
    "ToolRegistry",
]
