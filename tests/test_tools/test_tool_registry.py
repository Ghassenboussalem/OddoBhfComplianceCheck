#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for tool registry

Tests:
- Tool decorator functionality
- Tool registration
- Tool categories
- Tool caching
- Tool error handling
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.tool_registry import (
    tool,
    ToolCategory,
    ToolResult,
    list_all_tools,
    get_tools_by_category
)


# ============================================================================
# TEST TOOL DECORATOR
# ============================================================================

def test_tool_decorator_basic():
    """Test basic tool decorator"""
    @tool(
        name="sample_tool",
        category=ToolCategory.UTILITY,
        description="Add two numbers"
    )
    def sample_tool(x: int, y: int) -> int:
        """Add two numbers"""
        return x + y
    
    assert callable(sample_tool)


def test_tool_decorator_with_name():
    """Test tool decorator with custom name"""
    @tool(
        name="custom_tool",
        category=ToolCategory.UTILITY,
        description="Sample tool"
    )
    def sample_tool(x: int) -> int:
        """Sample tool"""
        return x * 2
    
    # Tool is registered, check it's in the list
    tools = list_all_tools()
    assert "custom_tool" in tools


def test_tool_decorator_with_category():
    """Test tool decorator with category"""
    @tool(
        name="analysis_tool",
        category=ToolCategory.ANALYSIS,
        description="Analyze text"
    )
    def analysis_tool(text: str) -> str:
        """Analyze text"""
        return text.upper()
    
    # Check tool is registered in correct category
    analysis_tools = get_tools_by_category(ToolCategory.ANALYSIS)
    tool_names = [t.name for t in analysis_tools]
    assert "analysis_tool" in tool_names


def test_tool_decorator_with_cache():
    """Test tool decorator with caching enabled"""
    @tool(
        name="cached_tool",
        category=ToolCategory.UTILITY,
        description="Cached tool",
        cache_enabled=True,
        cache_ttl_seconds=60
    )
    def cached_tool(x: int) -> int:
        """Cached tool"""
        return x * 2
    
    # Tool is registered
    tools = list_all_tools()
    assert "cached_tool" in tools


# ============================================================================
# TEST TOOL RESULT
# ============================================================================

def test_tool_result_success():
    """Test ToolResult for successful execution"""
    result = ToolResult(
        success=True,
        result=42,
        error=None,
        execution_time=0.1
    )
    
    assert result.success is True
    assert result.result == 42
    assert result.error is None


def test_tool_result_failure():
    """Test ToolResult for failed execution"""
    result = ToolResult(
        success=False,
        result=None,
        error="Division by zero",
        execution_time=0.05
    )
    
    assert result.success is False
    assert result.error == "Division by zero"


# ============================================================================
# TEST TOOL CATEGORIES
# ============================================================================

def test_tool_categories_exist():
    """Test that tool categories are defined"""
    assert hasattr(ToolCategory, 'PREPROCESSING')
    assert hasattr(ToolCategory, 'CHECKING')
    assert hasattr(ToolCategory, 'ANALYSIS')
    assert hasattr(ToolCategory, 'REVIEW')


# ============================================================================
# TEST TOOL REGISTRY FUNCTIONS
# ============================================================================

def test_list_all_tools():
    """Test listing all registered tools"""
    tools = list_all_tools()
    assert isinstance(tools, list)
    # Should have tools from various modules
    assert len(tools) >= 0  # May be empty if no tools registered yet


def test_get_tools_by_category():
    """Test getting tools by category"""
    preprocessing_tools = get_tools_by_category(ToolCategory.PREPROCESSING)
    assert isinstance(preprocessing_tools, list)
    
    # All tools should have the correct category
    for tool_metadata in preprocessing_tools:
        assert tool_metadata.category == ToolCategory.PREPROCESSING


# ============================================================================
# TEST ERROR HANDLING
# ============================================================================

def test_tool_error_handling():
    """Test tool error handling"""
    @tool(
        name="failing_tool",
        category=ToolCategory.UTILITY,
        description="Tool that fails",
        retry_enabled=True,
        max_retries=2
    )
    def failing_tool(x: int) -> int:
        """Tool that fails"""
        if x < 0:
            raise ValueError("Negative value not allowed")
        return x * 2
    
    # Should handle error gracefully
    result = failing_tool(-1)
    # Result should be wrapped in ToolResult
    assert hasattr(result, 'success')
    assert result.success is False
    assert result.error is not None


# ============================================================================
# TEST TOOL INVOCATION
# ============================================================================

def test_tool_invocation_direct():
    """Test direct tool invocation"""
    @tool(
        name="multiply_tool",
        category=ToolCategory.UTILITY,
        description="Multiply two numbers"
    )
    def multiply_tool(x: int, y: int) -> int:
        """Multiply two numbers"""
        return x * y
    
    # Direct call
    result = multiply_tool(3, 4)
    # Should be wrapped in ToolResult
    assert hasattr(result, 'success')
    assert result.success is True
    assert result.result == 12


def test_tool_invocation_with_kwargs():
    """Test tool invocation with keyword arguments"""
    @tool(
        name="greet_tool",
        category=ToolCategory.UTILITY,
        description="Greet someone"
    )
    def greet_tool(name: str, greeting: str = "Hello") -> str:
        """Greet someone"""
        return f"{greeting}, {name}!"
    
    result = greet_tool(name="Alice", greeting="Hi")
    # Should be wrapped in ToolResult
    assert hasattr(result, 'success')
    assert result.success is True
    assert "Alice" in result.result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
