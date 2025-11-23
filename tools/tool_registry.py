#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool Registry

This module provides functionality for the multi-agent compliance system.
"""

"""
Agent Tools Framework for Multi-Agent Compliance System

This module provides a comprehensive framework for managing tools used by agents
in the LangGraph-based compliance checking system.

Key Features:
- Standard tool interface using @tool decorator
- Tool categories: preprocessing, checking, analysis, review
- Tool error handling and retry logic
- Tool result caching

Requirements: 7.1, 7.2, 7.5
"""

import logging
import time
import functools
from typing import Callable, Dict, Any, Optional, List, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json


# Configure logging
logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of tools for organization and management"""
    PREPROCESSING = "preprocessing"
    CHECKING = "checking"
    ANALYSIS = "analysis"
    REVIEW = "review"
    UTILITY = "utility"


@dataclass
class ToolMetadata:
    """Metadata for a registered tool"""
    name: str
    category: ToolCategory
    description: str
    func: Callable
    retry_enabled: bool = True
    max_retries: int = 3
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    timeout_seconds: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ToolResult:
    """Result wrapper for tool execution"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    cached: bool = False
    retry_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class ToolCache:
    """Enhanced in-memory cache for tool results with LRU eviction and size limits"""

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, tuple[Any, datetime, int]] = {}  # value, timestamp, access_count
        self._hits = 0
        self._misses = 0
        self._max_size = max_size
        self._access_order: List[str] = []  # For LRU tracking

    def _generate_key(self, tool_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from tool name and arguments (optimized)"""
        # Optimize key generation for common cases
        if not kwargs and len(args) <= 2:
            # Fast path for simple cases
            return f"{tool_name}:{args}"

        # Full path for complex cases
        try:
            # Filter out non-cacheable objects (like AI engines)
            filtered_kwargs = {
                k: v for k, v in kwargs.items()
                if not (hasattr(v, '__call__') or k in ['ai_engine', 'engine', 'client'])
            }

            key_data = {
                "tool": tool_name,
                "args": str(args)[:500],  # Limit arg string length
                "kwargs": str(sorted(filtered_kwargs.items()))[:500]
            }
            key_str = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_str.encode()).hexdigest()
        except Exception as e:
            # Fallback to simple key
            logger.debug(f"Cache key generation fallback for {tool_name}: {e}")
            return f"{tool_name}:{hash(str(args)[:100])}"

    def get(self, tool_name: str, args: tuple, kwargs: dict, ttl_seconds: int) -> Optional[Any]:
        """Get cached result if available and not expired"""
        key = self._generate_key(tool_name, args, kwargs)

        if key in self._cache:
            result, timestamp, access_count = self._cache[key]
            age = (datetime.now() - timestamp).total_seconds()

            if age < ttl_seconds:
                # Update access tracking
                self._cache[key] = (result, timestamp, access_count + 1)
                self._update_access_order(key)

                self._hits += 1
                logger.debug(f"Cache hit for {tool_name} (age: {age:.1f}s, accesses: {access_count + 1})")
                return result
            else:
                # Expired, remove it
                self._remove_from_cache(key)

        self._misses += 1
        logger.debug(f"Cache miss for {tool_name}")
        return None

    def set(self, tool_name: str, args: tuple, kwargs: dict, result: Any):
        """Store result in cache with LRU eviction"""
        key = self._generate_key(tool_name, args, kwargs)

        # Check cache size and evict if necessary
        if len(self._cache) >= self._max_size and key not in self._cache:
            self._evict_lru()

        self._cache[key] = (result, datetime.now(), 0)
        self._update_access_order(key)
        logger.debug(f"Cached result for {tool_name} (cache size: {len(self._cache)})")

    def _update_access_order(self, key: str):
        """Update LRU access order"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _evict_lru(self):
        """Evict least recently used item"""
        if self._access_order:
            lru_key = self._access_order[0]
            self._remove_from_cache(lru_key)
            logger.debug(f"Evicted LRU cache entry: {lru_key}")

    def _remove_from_cache(self, key: str):
        """Remove item from cache"""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)

    def clear(self):
        """Clear all cached results"""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Tool cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0

        # Calculate average access count
        avg_accesses = 0
        if self._cache:
            total_accesses = sum(access_count for _, _, access_count in self._cache.values())
            avg_accesses = total_accesses / len(self._cache)

        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 2),
            "cached_items": len(self._cache),
            "max_size": self._max_size,
            "avg_accesses_per_item": round(avg_accesses, 2)
        }


class ToolRegistry:
    """Central registry for managing agent tools"""

    def __init__(self):
        self._tools: Dict[str, ToolMetadata] = {}
        self._cache = ToolCache()
        self._execution_stats: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        category: ToolCategory,
        description: str,
        retry_enabled: bool = True,
        max_retries: int = 3,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 3600,
        timeout_seconds: Optional[int] = None
    ) -> Callable:
        """
        Decorator to register a tool with the registry

        Args:
            name: Unique name for the tool
            category: Tool category (preprocessing, checking, analysis, review)
            description: Human-readable description
            retry_enabled: Whether to retry on failure
            max_retries: Maximum number of retry attempts
            cache_enabled: Whether to cache results
            cache_ttl_seconds: Cache time-to-live in seconds
            timeout_seconds: Optional timeout for tool execution

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            # Create metadata
            metadata = ToolMetadata(
                name=name,
                category=category,
                description=description,
                func=func,
                retry_enabled=retry_enabled,
                max_retries=max_retries,
                cache_enabled=cache_enabled,
                cache_ttl_seconds=cache_ttl_seconds,
                timeout_seconds=timeout_seconds
            )

            # Register the tool
            self._tools[name] = metadata

            # Initialize stats
            self._execution_stats[name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_retries": 0,
                "cache_hits": 0,
                "total_execution_time": 0.0,
                "avg_execution_time": 0.0
            }

            logger.info(f"Registered tool: {name} ({category.value})")

            # Return wrapped function
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> ToolResult:
                return self._execute_tool(name, args, kwargs)

            return wrapper

        return decorator

    def _execute_tool(self, tool_name: str, args: tuple, kwargs: dict) -> ToolResult:
        """
        Execute a tool with error handling, retry logic, and caching

        Args:
            tool_name: Name of the tool to execute
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            ToolResult with execution details
        """
        if tool_name not in self._tools:
            error_msg = f"Tool '{tool_name}' not found in registry"
            logger.error(error_msg)
            return ToolResult(success=False, error=error_msg)

        metadata = self._tools[tool_name]
        stats = self._execution_stats[tool_name]
        stats["total_calls"] += 1

        # Check cache first
        if metadata.cache_enabled:
            cached_result = self._cache.get(
                tool_name, args, kwargs, metadata.cache_ttl_seconds
            )
            if cached_result is not None:
                stats["cache_hits"] += 1
                return ToolResult(
                    success=True,
                    result=cached_result,
                    cached=True,
                    execution_time=0.0
                )

        # Execute with retry logic
        retry_count = 0
        last_error = None

        while retry_count <= (metadata.max_retries if metadata.retry_enabled else 0):
            try:
                start_time = time.time()

                # Execute the tool function
                result = metadata.func(*args, **kwargs)

                execution_time = time.time() - start_time

                # Update stats
                stats["successful_calls"] += 1
                stats["total_execution_time"] += execution_time
                stats["avg_execution_time"] = (
                    stats["total_execution_time"] / stats["successful_calls"]
                )

                # Cache the result
                if metadata.cache_enabled:
                    self._cache.set(tool_name, args, kwargs, result)

                logger.debug(
                    f"Tool '{tool_name}' executed successfully "
                    f"(time: {execution_time:.3f}s, retries: {retry_count})"
                )

                return ToolResult(
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    retry_count=retry_count
                )

            except Exception as e:
                last_error = str(e)
                retry_count += 1
                stats["total_retries"] += 1

                if retry_count <= metadata.max_retries and metadata.retry_enabled:
                    # Exponential backoff
                    wait_time = min(2 ** (retry_count - 1), 10)
                    logger.warning(
                        f"Tool '{tool_name}' failed (attempt {retry_count}), "
                        f"retrying in {wait_time}s: {last_error}"
                    )
                    time.sleep(wait_time)
                else:
                    # Max retries reached or retry disabled
                    stats["failed_calls"] += 1
                    logger.error(
                        f"Tool '{tool_name}' failed after {retry_count} attempts: {last_error}"
                    )
                    break

        return ToolResult(
            success=False,
            error=last_error,
            retry_count=retry_count - 1
        )

    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata by name"""
        return self._tools.get(name)

    def get_tools_by_category(self, category: ToolCategory) -> List[ToolMetadata]:
        """Get all tools in a specific category"""
        return [
            tool for tool in self._tools.values()
            if tool.category == category
        ]

    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self._tools.keys())

    def get_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get execution statistics

        Args:
            tool_name: Optional specific tool name, or None for all tools

        Returns:
            Dictionary of statistics
        """
        if tool_name:
            if tool_name not in self._execution_stats:
                return {}
            return {tool_name: self._execution_stats[tool_name]}

        return {
            "tools": self._execution_stats.copy(),
            "cache": self._cache.get_stats()
        }

    def clear_cache(self):
        """Clear the tool result cache"""
        self._cache.clear()

    def reset_stats(self):
        """Reset all execution statistics"""
        for tool_name in self._execution_stats:
            self._execution_stats[tool_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_retries": 0,
                "cache_hits": 0,
                "total_execution_time": 0.0,
                "avg_execution_time": 0.0
            }
        logger.info("Tool execution statistics reset")


# Global registry instance
_global_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance"""
    return _global_registry


def tool(
    name: str,
    category: ToolCategory,
    description: str,
    retry_enabled: bool = True,
    max_retries: int = 3,
    cache_enabled: bool = True,
    cache_ttl_seconds: int = 3600,
    timeout_seconds: Optional[int] = None
) -> Callable:
    """
    Convenience decorator for registering tools with the global registry

    Usage:
        @tool(
            name="extract_metadata",
            category=ToolCategory.PREPROCESSING,
            description="Extract metadata from document"
        )
        def extract_metadata(document: dict) -> dict:
            # Implementation
            pass

    Args:
        name: Unique name for the tool
        category: Tool category
        description: Human-readable description
        retry_enabled: Whether to retry on failure
        max_retries: Maximum number of retry attempts
        cache_enabled: Whether to cache results
        cache_ttl_seconds: Cache time-to-live in seconds
        timeout_seconds: Optional timeout for tool execution

    Returns:
        Decorator function
    """
    return _global_registry.register(
        name=name,
        category=category,
        description=description,
        retry_enabled=retry_enabled,
        max_retries=max_retries,
        cache_enabled=cache_enabled,
        cache_ttl_seconds=cache_ttl_seconds,
        timeout_seconds=timeout_seconds
    )


# Utility functions for tool management

def list_all_tools() -> List[str]:
    """List all registered tools"""
    return _global_registry.list_tools()


def get_tools_by_category(category: ToolCategory) -> List[ToolMetadata]:
    """Get all tools in a specific category"""
    return _global_registry.get_tools_by_category(category)


def get_tool_stats(tool_name: Optional[str] = None) -> Dict[str, Any]:
    """Get execution statistics for tools"""
    return _global_registry.get_stats(tool_name)


def clear_tool_cache():
    """Clear the tool result cache"""
    _global_registry.clear_cache()


def reset_tool_stats():
    """Reset all tool execution statistics"""
    _global_registry.reset_stats()


# Example tool definitions for reference

if __name__ == "__main__":
    # Example: Define a preprocessing tool
    @tool(
        name="example_extract_metadata",
        category=ToolCategory.PREPROCESSING,
        description="Extract metadata from document",
        cache_enabled=True,
        cache_ttl_seconds=3600
    )
    def extract_metadata(document: dict) -> dict:
        """Extract metadata from document"""
        return {
            "fund_isin": document.get("document_metadata", {}).get("fund_isin"),
            "client_type": document.get("document_metadata", {}).get("client_type", "retail"),
            "document_type": document.get("document_metadata", {}).get("document_type", "fund_presentation")
        }

    # Example: Define a checking tool
    @tool(
        name="example_check_structure",
        category=ToolCategory.CHECKING,
        description="Check document structure compliance",
        retry_enabled=True,
        max_retries=3
    )
    def check_structure(document: dict, config: dict) -> Optional[dict]:
        """Check document structure"""
        # Implementation would go here
        return None

    # Example: Define an analysis tool
    @tool(
        name="example_analyze_context",
        category=ToolCategory.ANALYSIS,
        description="Analyze text context using AI",
        cache_enabled=True,
        retry_enabled=True
    )
    def analyze_context(text: str, check_type: str) -> dict:
        """Analyze text context"""
        return {
            "is_fund_description": True,
            "is_client_advice": False,
            "confidence": 85
        }

    # Test the tools
    logger.info("Registered tools:")
    for tool_name in list_all_tools():
        logger.info(f"  - {tool_name}")

    logger.info("\nTesting extract_metadata tool:")
    test_doc = {
        "document_metadata": {
            "fund_isin": "FR0000123456",
            "client_type": "professional"
        }
    }
    result = extract_metadata(test_doc)
    logger.info(f"Result: {result}")
    logger.info(f"Success: {result.success}")
    logger.info(f"Cached: {result.cached}")
    logger.info(f"Execution time: {result.execution_time:.3f}s")

    # Test caching
    logger.info("\nTesting cache (second call):")
    result2 = extract_metadata(test_doc)
    logger.info(f"Cached: {result2.cached}")

    # Get statistics
    logger.info("\nTool statistics:")
    stats = get_tool_stats()
    print(json.dumps(stats, indent=2, default=str))
