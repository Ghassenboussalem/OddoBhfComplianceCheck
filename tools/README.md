# Agent Tools Framework

This directory contains the tool registry framework for the multi-agent compliance checking system.

## Overview

The Agent Tools Framework provides a standardized way to define, register, and execute tools used by agents in the LangGraph-based compliance system. It includes:

- **Standard tool interface** using the `@tool` decorator
- **Tool categories** for organization (preprocessing, checking, analysis, review)
- **Automatic error handling** with configurable retry logic
- **Result caching** with TTL-based expiration
- **Execution statistics** tracking for monitoring and optimization

## Requirements

This framework implements requirements:
- **7.1**: Each agent SHALL define a set of Agent_Tools that encapsulate specific compliance operations
- **7.2**: Tools for preprocessing, context analysis, and evidence extraction
- **7.5**: All existing functions SHALL be converted to Agent_Tools

## Quick Start

### Defining a Tool

```python
from tools.tool_registry import tool, ToolCategory

@tool(
    name="extract_metadata",
    category=ToolCategory.PREPROCESSING,
    description="Extract metadata from document",
    cache_enabled=True,
    cache_ttl_seconds=3600
)
def extract_metadata(document: dict) -> dict:
    """Extract metadata from document"""
    return {
        "fund_isin": document.get("document_metadata", {}).get("fund_isin"),
        "client_type": document.get("document_metadata", {}).get("client_type", "retail")
    }
```

### Using a Tool

```python
# Call the tool - it returns a ToolResult object
result = extract_metadata({"document_metadata": {"fund_isin": "FR123"}})

if result.success:
    print(f"Result: {result.result}")
    print(f"Cached: {result.cached}")
    print(f"Execution time: {result.execution_time:.3f}s")
else:
    print(f"Error: {result.error}")
```

## Tool Categories

Tools are organized into categories for better management:

- **PREPROCESSING**: Tools for document preprocessing (metadata extraction, whitelist building, normalization)
- **CHECKING**: Tools for compliance checking (structure checks, performance checks, etc.)
- **ANALYSIS**: Tools for context analysis and intent classification
- **REVIEW**: Tools for review queue management and feedback processing
- **UTILITY**: General utility tools

## Features

### 1. Automatic Error Handling

Tools automatically catch and handle exceptions:

```python
@tool(
    name="risky_operation",
    category=ToolCategory.CHECKING,
    description="Operation that might fail",
    retry_enabled=True,
    max_retries=3
)
def risky_operation(data: dict) -> dict:
    # If this raises an exception, it will be caught and retried
    return process_data(data)
```

### 2. Retry Logic with Exponential Backoff

Failed tools are automatically retried with exponential backoff:

```python
@tool(
    name="api_call",
    category=ToolCategory.CHECKING,
    description="Call external API",
    retry_enabled=True,
    max_retries=3  # Will retry up to 3 times with 1s, 2s, 4s delays
)
def call_api(endpoint: str) -> dict:
    return requests.get(endpoint).json()
```

### 3. Result Caching

Tool results are automatically cached to improve performance:

```python
@tool(
    name="expensive_analysis",
    category=ToolCategory.ANALYSIS,
    description="Expensive AI analysis",
    cache_enabled=True,
    cache_ttl_seconds=3600  # Cache for 1 hour
)
def analyze_text(text: str) -> dict:
    # This will only execute once per unique input within the TTL
    return ai_engine.analyze(text)
```

### 4. Execution Statistics

The framework tracks detailed statistics for each tool:

```python
from tools.tool_registry import get_tool_stats

# Get stats for a specific tool
stats = get_tool_stats("extract_metadata")
print(stats)
# Output:
# {
#   "extract_metadata": {
#     "total_calls": 100,
#     "successful_calls": 98,
#     "failed_calls": 2,
#     "total_retries": 5,
#     "cache_hits": 45,
#     "total_execution_time": 12.5,
#     "avg_execution_time": 0.125
#   }
# }

# Get stats for all tools including cache
all_stats = get_tool_stats()
print(all_stats["cache"])
# Output:
# {
#   "hits": 150,
#   "misses": 50,
#   "total_requests": 200,
#   "hit_rate_percent": 75.0,
#   "cached_items": 45
# }
```

## Tool Decorator Parameters

The `@tool` decorator accepts the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Unique name for the tool |
| `category` | ToolCategory | Required | Tool category |
| `description` | str | Required | Human-readable description |
| `retry_enabled` | bool | True | Whether to retry on failure |
| `max_retries` | int | 3 | Maximum number of retry attempts |
| `cache_enabled` | bool | True | Whether to cache results |
| `cache_ttl_seconds` | int | 3600 | Cache time-to-live in seconds |
| `timeout_seconds` | int | None | Optional timeout for execution |

## ToolResult Object

When you call a tool, it returns a `ToolResult` object with the following attributes:

```python
@dataclass
class ToolResult:
    success: bool              # Whether execution succeeded
    result: Any                # The actual result (if successful)
    error: Optional[str]       # Error message (if failed)
    execution_time: float      # Time taken to execute (seconds)
    cached: bool               # Whether result came from cache
    retry_count: int           # Number of retries performed
    timestamp: datetime        # When the result was generated
```

## Utility Functions

### List All Tools

```python
from tools.tool_registry import list_all_tools

tools = list_all_tools()
print(tools)  # ['extract_metadata', 'check_structure', ...]
```

### Get Tools by Category

```python
from tools.tool_registry import get_tools_by_category, ToolCategory

preprocessing_tools = get_tools_by_category(ToolCategory.PREPROCESSING)
for tool in preprocessing_tools:
    print(f"{tool.name}: {tool.description}")
```

### Clear Cache

```python
from tools.tool_registry import clear_tool_cache

# Clear all cached results
clear_tool_cache()
```

### Reset Statistics

```python
from tools.tool_registry import reset_tool_stats

# Reset all execution statistics
reset_tool_stats()
```

## Best Practices

### 1. Use Descriptive Names

```python
# Good
@tool(name="extract_fund_metadata", ...)
def extract_fund_metadata(document: dict) -> dict:
    pass

# Bad
@tool(name="extract", ...)
def extract(doc: dict) -> dict:
    pass
```

### 2. Choose Appropriate Cache TTL

```python
# Metadata extraction - cache for long time (rarely changes)
@tool(name="extract_metadata", cache_ttl_seconds=3600, ...)

# AI analysis - cache for moderate time (may need refresh)
@tool(name="analyze_context", cache_ttl_seconds=1800, ...)

# Real-time checks - disable caching
@tool(name="check_live_data", cache_enabled=False, ...)
```

### 3. Enable Retry for External Calls

```python
# External API calls - enable retry
@tool(name="call_ai_api", retry_enabled=True, max_retries=3, ...)

# Pure computation - disable retry
@tool(name="calculate_score", retry_enabled=False, ...)
```

### 4. Use Type Hints

```python
@tool(name="extract_metadata", ...)
def extract_metadata(document: dict) -> dict:
    """
    Extract metadata from document
    
    Args:
        document: Document dictionary with metadata field
    
    Returns:
        Dictionary containing extracted metadata
    """
    return {...}
```

## Integration with Agents

Tools are designed to be used by agents in the multi-agent system:

```python
from tools.tool_registry import get_tools_by_category, ToolCategory

class PreprocessorAgent:
    def __init__(self):
        # Get all preprocessing tools
        self.tools = {
            tool.name: tool.func 
            for tool in get_tools_by_category(ToolCategory.PREPROCESSING)
        }
    
    def __call__(self, state: ComplianceState) -> ComplianceState:
        # Use tools
        metadata_result = self.tools["extract_metadata"](state["document"])
        if metadata_result.success:
            state["metadata"] = metadata_result.result
        
        return state
```

## Testing

Run the test suite to verify the framework:

```bash
python test_tool_registry.py -v
```

The test suite covers:
- Tool registration and execution
- Caching behavior and expiration
- Error handling and retry logic
- Statistics tracking
- Category filtering
- Integration patterns

## Migration Guide

To convert existing functions to tools:

### Before (Old Code)

```python
def check_promotional_mention(document: dict, config: dict) -> Optional[dict]:
    """Check for promotional document mention"""
    # Implementation
    return result
```

### After (New Code)

```python
from tools.tool_registry import tool, ToolCategory

@tool(
    name="check_promotional_mention",
    category=ToolCategory.CHECKING,
    description="Check for promotional document mention on cover page",
    retry_enabled=True,
    cache_enabled=True
)
def check_promotional_mention(document: dict, config: dict) -> Optional[dict]:
    """Check for promotional document mention"""
    # Same implementation
    return result
```

## Performance Considerations

### Cache Hit Rate

Monitor cache hit rate to optimize performance:

```python
stats = get_tool_stats()
cache_stats = stats["cache"]
print(f"Cache hit rate: {cache_stats['hit_rate_percent']}%")

# If hit rate is low, consider:
# 1. Increasing cache TTL
# 2. Checking if inputs are consistent
# 3. Reviewing cache key generation
```

### Execution Time

Track execution time to identify bottlenecks:

```python
stats = get_tool_stats()
for tool_name, tool_stats in stats["tools"].items():
    avg_time = tool_stats["avg_execution_time"]
    if avg_time > 1.0:  # More than 1 second
        print(f"Slow tool: {tool_name} ({avg_time:.2f}s)")
```

### Retry Overhead

Monitor retry counts to identify flaky operations:

```python
stats = get_tool_stats()
for tool_name, tool_stats in stats["tools"].items():
    retry_rate = tool_stats["total_retries"] / tool_stats["total_calls"]
    if retry_rate > 0.1:  # More than 10% retry rate
        print(f"Flaky tool: {tool_name} ({retry_rate:.1%} retry rate)")
```

## Troubleshooting

### Tool Not Found

```python
# Error: Tool 'my_tool' not found in registry
# Solution: Make sure the tool is imported before use
from tools.preprocessing_tools import extract_metadata
```

### Cache Not Working

```python
# If cache doesn't seem to work, check:
# 1. Is cache_enabled=True?
# 2. Are inputs exactly the same?
# 3. Has the TTL expired?

# Debug by checking cache stats
stats = get_tool_stats()
print(stats["cache"])
```

### High Failure Rate

```python
# If a tool fails frequently:
# 1. Check error logs
# 2. Increase max_retries
# 3. Add better error handling in the tool function
# 4. Consider adding a fallback mechanism

stats = get_tool_stats("problematic_tool")
failure_rate = stats["problematic_tool"]["failed_calls"] / stats["problematic_tool"]["total_calls"]
print(f"Failure rate: {failure_rate:.1%}")
```

## Future Enhancements

Potential improvements for future versions:

1. **Distributed Caching**: Use Redis or similar for shared cache across processes
2. **Async Support**: Add support for async/await tool execution
3. **Tool Composition**: Allow tools to call other tools
4. **Circuit Breaker**: Automatically disable failing tools temporarily
5. **Metrics Export**: Export metrics to Prometheus or similar
6. **Tool Versioning**: Support multiple versions of the same tool
7. **Tool Dependencies**: Declare and validate tool dependencies

## License

This framework is part of the AI-Enhanced Compliance Checker system.
