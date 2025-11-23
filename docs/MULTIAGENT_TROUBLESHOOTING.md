# Multi-Agent System Troubleshooting Guide

## Table of Contents

1. [Overview](#overview)
2. [Common Issues](#common-issues)
3. [Error Messages](#error-messages)
4. [Debugging Tips](#debugging-tips)
5. [Performance Issues](#performance-issues)
6. [Configuration Problems](#configuration-problems)
7. [Agent-Specific Issues](#agent-specific-issues)
8. [State Management Issues](#state-management-issues)
9. [HITL and Review Issues](#hitl-and-review-issues)
10. [FAQ](#faq)

---

## Overview

This guide helps you diagnose and resolve common issues with the Multi-Agent Compliance System. It covers error messages, debugging techniques, and solutions to frequently encountered problems.

### Quick Diagnostic Checklist

Before diving into specific issues, check these common causes:

- [ ] Is the multi-agent system enabled in `hybrid_config.json`?
- [ ] Are all required environment variables set (API keys)?
- [ ] Is the checkpoint database accessible and not corrupted?
- [ ] Are all required dependencies installed?
- [ ] Is there sufficient disk space for checkpoints and logs?
- [ ] Are file permissions correct for log and checkpoint directories?

### Getting Help

If this guide doesn't resolve your issue:

1. Check the logs in `monitoring/logs/`
2. Review the error log in the state: `state["error_log"]`
3. Enable DEBUG logging: `"log_level": "DEBUG"` in config
4. Run with sequential execution to isolate the failing agent
5. Check GitHub issues or create a new one with logs

---

## Common Issues

### Issue 1: Multi-Agent System Not Starting

**Symptoms:**
- System falls back to legacy single-agent mode
- No agent execution logs
- Warning: "Multi-agent system disabled, using legacy mode"

**Causes:**
1. Multi-agent system not enabled in configuration
2. Missing or corrupted configuration file
3. Import errors for LangGraph dependencies

**Solutions:**

**Solution 1: Enable Multi-Agent System**
```json
{
  "multi_agent": {
    "enabled": true
  }
}
```

**Solution 2: Verify Configuration File**
```bash
# Check if config file exists and is valid JSON
python -c "import json; json.load(open('hybrid_config.json'))"
```

**Solution 3: Install Missing Dependencies**
```bash
pip install langgraph langchain langchain-openai langchain-community
pip install langgraph-checkpoint-sqlite
```

**Solution 4: Check Import Errors**
```python
# Test imports
python -c "from langgraph.graph import StateGraph; print('LangGraph OK')"
python -c "from workflow_builder import create_compliance_workflow; print('Workflow OK')"
```

---

### Issue 2: Agent Timeout Errors

**Symptoms:**
- Error: "Agent call exceeded timeout of 30s"
- Workflow stops mid-execution
- Timeout errors in logs

**Causes:**
1. Agent processing large documents
2. AI service slow to respond
3. Timeout configured too low
4. Network latency issues

**Solutions:**

**Solution 1: Increase Agent Timeout**
```json
{
  "agents": {
    "performance": {
      "timeout_seconds": 60
    }
  }
}
```

**Solution 2: Increase Global Timeout**
```json
{
  "multi_agent": {
    "agent_timeout_seconds": 60
  }
}
```

**Solution 3: Check AI Service Response Time**
```python
# Test AI service latency
import time
from ai_engine import AIEngine

ai_engine = AIEngine(config)
start = time.time()
response = ai_engine.call_llm("Test prompt")
print(f"AI latency: {time.time() - start:.2f}s")
```

**Solution 4: Enable Timeout Handling**
```json
{
  "error_handling": {
    "agent_failure_strategy": "continue",
    "fallback_to_rules_on_ai_failure": true
  }
}
```

---

### Issue 3: Circuit Breaker Keeps Opening

**Symptoms:**
- Error: "Circuit breaker is OPEN"
- Multiple agents failing consecutively
- System refusing to make AI calls

**Causes:**
1. AI service unavailable or rate-limited
2. Network connectivity issues
3. Circuit breaker threshold too low
4. Persistent configuration errors

**Solutions:**

**Solution 1: Check AI Service Status**
```bash
# Test API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

**Solution 2: Adjust Circuit Breaker Settings**
```json
{
  "error_handling": {
    "circuit_breaker_enabled": true,
    "circuit_breaker_threshold": 10,
    "circuit_breaker_timeout_seconds": 120
  }
}
```

**Solution 3: Manually Reset Circuit Breaker**
```python
from error_handler_multiagent import MultiAgentErrorHandler

error_handler = MultiAgentErrorHandler()
error_handler.circuit_breaker.reset()
print("Circuit breaker reset")
```

**Solution 4: Enable Fallback to Rules**
```json
{
  "error_handling": {
    "fallback_to_rules_on_ai_failure": true
  }
}
```

---

### Issue 4: State Persistence Failures

**Symptoms:**
- Error: "Failed to save checkpoint"
- Cannot resume workflow after interruption
- Checkpoint database locked or corrupted

**Causes:**
1. Insufficient disk space
2. Database file permissions
3. Concurrent access to checkpoint database
4. Corrupted checkpoint database

**Solutions:**

**Solution 1: Check Disk Space**
```bash
df -h .
# Ensure sufficient space in checkpoint directory
```

**Solution 2: Fix File Permissions**
```bash
chmod 755 checkpoints/
chmod 644 checkpoints/compliance_workflow.db
```

**Solution 3: Delete Corrupted Database**
```bash
# Backup first
cp checkpoints/compliance_workflow.db checkpoints/compliance_workflow.db.backup

# Delete and recreate
rm checkpoints/compliance_workflow.db
# System will create new database on next run
```

**Solution 4: Disable Concurrent Access**
```json
{
  "state_management": {
    "checkpoint_enabled": true,
    "save_intermediate_states": false
  }
}
```

**Solution 5: Use Alternative Checkpoint Path**
```json
{
  "multi_agent": {
    "checkpoint_db_path": "/tmp/compliance_workflow.db"
  }
}
```

---

### Issue 5: Parallel Execution Not Working

**Symptoms:**
- Agents running sequentially instead of in parallel
- No performance improvement
- Logs show sequential execution

**Causes:**
1. Parallel execution disabled in configuration
2. `max_parallel_agents` set to 1
3. System resource constraints
4. LangGraph not configured for parallel execution

**Solutions:**

**Solution 1: Enable Parallel Execution**
```json
{
  "multi_agent": {
    "parallel_execution": true,
    "max_parallel_agents": 4
  }
}
```

**Solution 2: Verify Workflow Configuration**
```python
# Check workflow graph structure
from workflow_builder import create_compliance_workflow

workflow = create_compliance_workflow(config)
# Parallel agents should have same parent node
```

**Solution 3: Check System Resources**
```bash
# Monitor CPU usage during execution
top -p $(pgrep -f check_multiagent)
```

**Solution 4: Test with Simple Document**
```bash
# Use small test document to verify parallel execution
python check_multiagent.py --input test_small.json --show-metrics
# Check agent_timings to see if agents ran concurrently
```

---

### Issue 6: High False Positive Rate

**Symptoms:**
- Many violations that shouldn't be violations
- Context analysis not filtering false positives
- Low confidence scores not triggering context analysis

**Causes:**
1. Context agent disabled or skipped
2. Confidence thresholds too high
3. Whitelist not being built correctly
4. Intent classification not working

**Solutions:**

**Solution 1: Enable Context Analysis**
```json
{
  "agents": {
    "context": {
      "enabled": true,
      "confidence_threshold": 80,
      "skip_if_all_high_confidence": false
    }
  }
}
```

**Solution 2: Lower Confidence Thresholds**
```json
{
  "routing": {
    "context_threshold": 85,
    "skip_context_if_high_confidence": false
  }
}
```

**Solution 3: Verify Whitelist Building**
```python
# Check whitelist contents
from agents.preprocessor_agent import PreprocessorAgent

agent = PreprocessorAgent()
result = agent(state)
print(f"Whitelist size: {len(result['whitelist'])}")
print(f"Whitelist terms: {list(result['whitelist'])[:10]}")
```

**Solution 4: Enable All Context Features**
```json
{
  "context_analysis": {
    "enabled": true,
    "intent_classification_enabled": true,
    "semantic_validation_enabled": true,
    "evidence_extraction_enabled": true
  }
}
```

---

### Issue 7: Memory Errors with Large Documents

**Symptoms:**
- MemoryError or Out of Memory errors
- System becomes unresponsive
- Checkpoint files extremely large

**Causes:**
1. Document too large for available memory
2. State accumulating too much data
3. Checkpoint history not being cleaned up
4. Memory leaks in agent code

**Solutions:**

**Solution 1: Limit State History**
```json
{
  "state_management": {
    "state_history_max_size": 10,
    "auto_cleanup_old_checkpoints": true,
    "checkpoint_retention_days": 1
  }
}
```

**Solution 2: Reduce Context Window**
```json
{
  "context_analysis": {
    "max_context_length": 1000,
    "context_window_chars": 300
  }
}
```

**Solution 3: Process Document in Chunks**
```python
# Split large document into sections
def process_large_document(document):
    sections = split_document(document)
    all_violations = []
    
    for section in sections:
        result = check_multiagent(section)
        all_violations.extend(result['violations'])
    
    return all_violations
```

**Solution 4: Increase System Memory**
```bash
# For Docker containers
docker run --memory=8g ...

# For Python
export PYTHONMAXMEMORY=8000000000
```

---

## Error Messages

### Error: "Agent not found in registry"

**Full Message:**
```
ERROR: Agent not found: structure
```

**Cause:** Agent class not registered in AgentRegistry

**Solution:**
```python
# Ensure agent is registered
from agents.base_agent import AgentRegistry
from agents.structure_agent import StructureAgent

AgentRegistry.register("structure", StructureAgent)
```

---

### Error: "State validation failed"

**Full Message:**
```
ERROR: State validation error: Missing required field 'document'
```

**Cause:** ComplianceState missing required fields

**Solution:**
```python
# Initialize state properly
from data_models_multiagent import initialize_compliance_state

state = initialize_compliance_state(
    document=document_json,
    document_id="example.json",
    config=config
)
```

---

### Error: "Workflow compilation failed"

**Full Message:**
```
ERROR: Failed to compile workflow: Conditional edge has no matching condition
```

**Cause:** Workflow graph has invalid edges or conditions

**Solution:**
```python
# Check conditional edge definitions
workflow.add_conditional_edges(
    "aggregator",
    lambda state: "context" if needs_context(state) else "complete",
    {
        "context": "context",
        "complete": END
    }
)
```

---

### Error: "HITL interrupt not resumable"

**Full Message:**
```
ERROR: Cannot resume workflow: No checkpoint found for thread_id
```

**Cause:** Checkpoint not saved before HITL interrupt

**Solution:**
```json
{
  "state_management": {
    "checkpoint_enabled": true,
    "save_intermediate_states": true
  }
}
```

---

### Error: "AI service rate limit exceeded"

**Full Message:**
```
ERROR: Rate limit exceeded: 429 Too Many Requests
```

**Cause:** Too many API calls to AI service

**Solution:**
```json
{
  "error_handling": {
    "max_agent_retries": 5,
    "retry_delay_seconds": 5.0
  },
  "ai_service": {
    "retry_attempts": 5,
    "retry_delay": 5.0
  }
}
```

---

### Error: "Violation deduplication failed"

**Full Message:**
```
ERROR: Failed to deduplicate violations: 'NoneType' object has no attribute 'get'
```

**Cause:** Violation missing required fields

**Solution:**
```python
# Ensure all violations have required fields
violation = {
    "rule": "RULE_NAME",
    "type": "STRUCTURE",
    "severity": "CRITICAL",
    "evidence": "...",
    "location": "slide_1",
    "confidence": 85
}
```

---

## Debugging Tips

### Enable Debug Logging

**Global Debug Logging:**
```json
{
  "monitoring": {
    "log_level": "DEBUG"
  }
}
```

**Agent-Specific Debug Logging:**
```json
{
  "agents": {
    "structure": {
      "log_level": "DEBUG"
    }
  }
}
```

**Python Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

### Inspect State at Each Step

```python
# Add state inspection callback
def inspect_state(state):
    print(f"Current agent: {state.get('current_agent')}")
    print(f"Violations: {len(state.get('violations', []))}")
    print(f"Workflow status: {state.get('workflow_status')}")
    return state

# Add to workflow
workflow.add_node("inspector", inspect_state)
```

---

### Test Individual Agents

```python
# Test agent in isolation
from agents.structure_agent import StructureAgent
from data_models_multiagent import initialize_compliance_state

agent = StructureAgent()
state = initialize_compliance_state(document, "test.json", config)

try:
    result = agent(state)
    print(f"Success: {len(result['violations'])} violations")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
```

---

### Visualize Workflow Execution

```python
# Generate workflow diagram
from workflow_builder import create_compliance_workflow
from monitoring.workflow_visualizer import WorkflowVisualizer

workflow = create_compliance_workflow(config)
visualizer = WorkflowVisualizer()
visualizer.visualize_workflow(workflow, "workflow_diagram.png")
```

---

### Monitor Agent Performance

```python
# Track agent execution times
from monitoring.metrics_tracker import MetricsTracker

tracker = MetricsTracker()
tracker.start_tracking()

# Run workflow
result = check_multiagent(document)

# Get metrics
metrics = tracker.get_metrics()
print(f"Total time: {metrics['total_duration']:.2f}s")
for agent, duration in metrics['agent_timings'].items():
    print(f"  {agent}: {duration:.2f}s")
```

---

### Check Error Log

```python
# Inspect error log from state
if "error_log" in state and state["error_log"]:
    print("Errors encountered:")
    for error in state["error_log"]:
        print(f"  Agent: {error['agent']}")
        print(f"  Error: {error['error']}")
        print(f"  Type: {error['error_type']}")
        print(f"  Time: {error['timestamp']}")
```

---

### Use Sequential Execution for Debugging

```json
{
  "multi_agent": {
    "parallel_execution": false,
    "max_parallel_agents": 1
  }
}
```

This makes it easier to identify which agent is causing issues.

---

## Performance Issues

### Slow Execution Time

**Symptoms:**
- Workflow takes much longer than expected
- Individual agents timing out
- High CPU or memory usage

**Diagnostic Steps:**

1. **Check Agent Timings:**
```python
print(state["agent_timings"])
# Identify slowest agents
```

2. **Profile AI Calls:**
```python
# Count API calls
print(f"API calls: {state.get('api_calls', 0)}")
print(f"Cache hits: {state.get('cache_hits', 0)}")
```

3. **Monitor System Resources:**
```bash
# CPU and memory usage
top -p $(pgrep -f check_multiagent)

# I/O wait
iostat -x 1
```

**Solutions:**

**Solution 1: Enable Parallel Execution**
```json
{
  "multi_agent": {
    "parallel_execution": true,
    "max_parallel_agents": 8
  }
}
```

**Solution 2: Reduce AI Calls**
```json
{
  "agents": {
    "context": {
      "skip_if_all_high_confidence": true
    }
  }
}
```

**Solution 3: Enable Caching**
```python
# Implement result caching in tools
from functools import lru_cache

@lru_cache(maxsize=1000)
def check_promotional_mention(document_hash, config_hash):
    # ... implementation
    pass
```

**Solution 4: Optimize Document Processing**
```python
# Pre-process document once
normalized_doc = normalize_document(document)
# Reuse normalized document in all agents
```

---

### High Memory Usage

**Symptoms:**
- Memory usage grows over time
- System becomes unresponsive
- Out of memory errors

**Solutions:**

**Solution 1: Limit State History**
```json
{
  "state_management": {
    "state_history_max_size": 10
  }
}
```

**Solution 2: Clean Up Checkpoints**
```json
{
  "state_management": {
    "auto_cleanup_old_checkpoints": true,
    "checkpoint_retention_days": 1
  }
}
```

**Solution 3: Process in Batches**
```python
# Process multiple documents in batches
for batch in document_batches:
    results = process_batch(batch)
    # Clear memory between batches
    import gc
    gc.collect()
```

---

## Configuration Problems

### Configuration Not Loading

**Symptoms:**
- Default values used instead of config
- Changes to config file not taking effect
- Warning: "Using default configuration"

**Solutions:**

**Solution 1: Verify Config File Path**
```python
import os
config_path = "hybrid_config.json"
print(f"Config exists: {os.path.exists(config_path)}")
print(f"Config path: {os.path.abspath(config_path)}")
```

**Solution 2: Validate JSON Syntax**
```bash
python -m json.tool hybrid_config.json
```

**Solution 3: Check Config Loading**
```python
from config_manager import ConfigManager

config_manager = ConfigManager("hybrid_config.json")
config = config_manager.get_config()
print(f"Multi-agent enabled: {config.get('multi_agent', {}).get('enabled')}")
```

---

### Invalid Configuration Values

**Symptoms:**
- Error: "Invalid configuration value"
- Unexpected behavior
- Agents not running as expected

**Solutions:**

**Solution 1: Validate Configuration**
```python
from config.agent_config_manager import AgentConfigManager

config_manager = AgentConfigManager(config_dict)
# Will raise error if invalid
```

**Solution 2: Check Value Types**
```json
{
  "multi_agent": {
    "enabled": true,  // boolean, not string
    "max_parallel_agents": 4,  // integer, not string
    "agent_timeout_seconds": 30  // number, not string
  }
}
```

**Solution 3: Use Configuration Template**
```bash
# Copy template and modify
cp hybrid_config.template.json hybrid_config.json
```

---

## Agent-Specific Issues

### Structure Agent Issues

**Problem: Promotional mention not detected**

**Solution:**
```json
{
  "agents": {
    "structure": {
      "parallel_tool_execution": true
    }
  }
}
```

Check that `page_de_garde` section exists in document.

---

### Performance Agent Issues

**Problem: Performance data not detected**

**Solution:**
Ensure the agent is using data-aware checking:
```python
# Should detect actual numbers with %
# Not just keywords like "performance"
```

Check logs for evidence extraction results.

---

### Securities Agent Issues

**Problem: Too many false positives for securities mentions**

**Solution:**
```json
{
  "agents": {
    "securities": {
      "use_whitelist_filtering": true
    }
  },
  "whitelist": {
    "auto_extract_fund_name": true,
    "include_strategy_terms": true
  }
}
```

---

### Context Agent Issues

**Problem: Context analysis not running**

**Solution:**
```json
{
  "agents": {
    "context": {
      "enabled": true,
      "confidence_threshold": 80,
      "skip_if_all_high_confidence": false
    }
  }
}
```

Check that violations have confidence < 80%.

---

### Evidence Agent Issues

**Problem: No evidence extracted**

**Solution:**
```json
{
  "agents": {
    "evidence": {
      "enabled": true,
      "extract_quotes": true,
      "find_performance_data": true,
      "track_locations": true
    }
  }
}
```

---

### Reviewer Agent Issues

**Problem: Violations not queued for review**

**Solution:**
```json
{
  "agents": {
    "reviewer": {
      "enabled": true,
      "confidence_threshold": 70,
      "auto_queue_enabled": true
    }
  }
}
```

Check that violations have confidence < 70%.

---

## State Management Issues

### Cannot Resume Workflow

**Problem:** Workflow cannot be resumed after HITL interrupt

**Solutions:**

**Solution 1: Enable Checkpointing**
```json
{
  "state_management": {
    "checkpoint_enabled": true,
    "save_intermediate_states": true
  }
}
```

**Solution 2: Use Correct Thread ID**
```python
# Resume with same thread_id used for initial execution
config = {"configurable": {"thread_id": "original_thread_id"}}
result = workflow.invoke(state, config)
```

**Solution 3: Check Checkpoint Database**
```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

conn = sqlite3.connect("checkpoints/compliance_workflow.db")
checkpointer = SqliteSaver(conn)

# List checkpoints
checkpoints = checkpointer.list(config)
print(f"Available checkpoints: {len(list(checkpoints))}")
```

---

### State Corruption

**Problem:** State becomes corrupted or invalid

**Solutions:**

**Solution 1: Enable State Validation**
```json
{
  "state_management": {
    "enable_state_validation": true
  }
}
```

**Solution 2: Reset State**
```python
# Start fresh workflow
from data_models_multiagent import initialize_compliance_state

state = initialize_compliance_state(document, document_id, config)
```

**Solution 3: Recover from Checkpoint**
```python
# Load earlier checkpoint
from state_manager import StateManager

state_manager = StateManager()
state = state_manager.load_checkpoint("checkpoint_name")
```

---

## HITL and Review Issues

### Review Queue Not Working

**Problem:** Violations not appearing in review queue

**Solutions:**

**Solution 1: Check Reviewer Agent**
```json
{
  "agents": {
    "reviewer": {
      "enabled": true,
      "auto_queue_enabled": true
    }
  }
}
```

**Solution 2: Verify Confidence Threshold**
```json
{
  "hitl": {
    "review_threshold": 70
  }
}
```

**Solution 3: Check Review Manager**
```python
from review_manager import ReviewManager

review_manager = ReviewManager()
pending = review_manager.get_pending_reviews()
print(f"Pending reviews: {len(pending)}")
```

---

### Cannot Submit Review Feedback

**Problem:** Feedback submission fails

**Solutions:**

**Solution 1: Check Review Item Status**
```python
# Item must be in PENDING or IN_REVIEW status
review_item = review_manager.get_review_by_id(review_id)
print(f"Status: {review_item.status}")
```

**Solution 2: Verify Feedback Format**
```python
feedback = {
    "check_type": "STRUCTURE",
    "document_id": "example.json",
    "predicted_violation": True,
    "actual_violation": False,  # Human correction
    "predicted_confidence": 65,
    "reviewer_notes": "This is a fund description, not advice"
}
```

---

## FAQ

### Q: How do I disable a specific agent?

**A:** Set `enabled: false` in the agent configuration:
```json
{
  "agents": {
    "esg": {
      "enabled": false
    }
  }
}
```

---

### Q: How do I run only specific agents?

**A:** Disable all others and enable only the ones you want:
```json
{
  "agents": {
    "structure": {"enabled": true},
    "performance": {"enabled": true},
    "securities": {"enabled": false},
    "general": {"enabled": false}
  }
}
```

---

### Q: How do I increase the number of parallel agents?

**A:** Adjust `max_parallel_agents`:
```json
{
  "multi_agent": {
    "max_parallel_agents": 8
  }
}
```

Note: More parallel agents require more system resources.

---

### Q: How do I reduce false positives?

**A:** Enable context analysis and lower thresholds:
```json
{
  "routing": {
    "context_threshold": 85,
    "skip_context_if_high_confidence": false
  },
  "agents": {
    "context": {
      "enabled": true,
      "intent_classification_enabled": true,
      "semantic_validation_enabled": true
    }
  }
}
```

---

### Q: How do I speed up execution?

**A:** Enable parallel execution and increase parallel agents:
```json
{
  "multi_agent": {
    "parallel_execution": true,
    "max_parallel_agents": 8
  },
  "agents": {
    "structure": {"parallel_tool_execution": true},
    "performance": {"parallel_tool_execution": true}
  }
}
```

---

### Q: How do I debug a specific agent?

**A:** Run with sequential execution and debug logging:
```json
{
  "multi_agent": {
    "parallel_execution": false
  },
  "agents": {
    "structure": {
      "log_level": "DEBUG"
    }
  }
}
```

Then test the agent in isolation:
```python
from agents.structure_agent import StructureAgent

agent = StructureAgent()
result = agent(state)
```

---

### Q: How do I handle API rate limits?

**A:** Increase retry delays and enable fallback:
```json
{
  "error_handling": {
    "max_agent_retries": 5,
    "retry_delay_seconds": 10.0,
    "fallback_to_rules_on_ai_failure": true
  }
}
```

---

### Q: How do I export metrics?

**A:** Enable metrics export:
```json
{
  "monitoring": {
    "metrics_export_enabled": true,
    "metrics_export_path": "./monitoring/metrics/"
  }
}
```

Then access metrics:
```python
from monitoring.metrics_tracker import MetricsTracker

tracker = MetricsTracker()
metrics = tracker.export_metrics("metrics.json")
```

---

### Q: How do I clear old checkpoints?

**A:** Enable auto-cleanup:
```json
{
  "state_management": {
    "auto_cleanup_old_checkpoints": true,
    "checkpoint_retention_days": 7
  }
}
```

Or manually:
```bash
find checkpoints/ -name "*.db" -mtime +7 -delete
```

---

### Q: How do I test the multi-agent system?

**A:** Run the test suite:
```bash
# Unit tests
pytest tests/test_agents/

# Integration tests
pytest tests/test_workflow.py

# Validation tests
pytest tests/test_validation.py
```

---

### Q: How do I migrate from single-agent to multi-agent?

**A:** See the [Migration Guide](MIGRATION_TO_MULTIAGENT.md) for detailed steps.

Quick migration:
1. Enable multi-agent: `"multi_agent": {"enabled": true}`
2. Run in parallel with old system for validation
3. Compare results
4. Switch fully to multi-agent

---

### Q: How do I get support?

**A:** 
1. Check this troubleshooting guide
2. Review the [Architecture Documentation](MULTI_AGENT_ARCHITECTURE.md)
3. Check the [Configuration Guide](MULTIAGENT_CONFIGURATION.md)
4. Search GitHub issues
5. Create a new issue with:
   - Error message and full traceback
   - Configuration file
   - Log files
   - Steps to reproduce

---

## Additional Resources

- [Multi-Agent Architecture](MULTI_AGENT_ARCHITECTURE.md)
- [Agent API Documentation](AGENT_API.md)
- [Configuration Guide](MULTIAGENT_CONFIGURATION.md)
- [Migration Guide](MIGRATION_TO_MULTIAGENT.md)
- [Quick Start Guide](../QUICK_START.md)

---

## Reporting Issues

When reporting issues, please include:

1. **Error Message**: Full error message and traceback
2. **Configuration**: Your `hybrid_config.json` (remove sensitive data)
3. **Logs**: Relevant log files from `monitoring/logs/`
4. **Environment**: Python version, OS, installed packages
5. **Steps to Reproduce**: Minimal example to reproduce the issue
6. **Expected vs Actual**: What you expected vs what happened

**Example Issue Report:**

```
Title: Context Agent Not Running Despite Low Confidence

Environment:
- Python 3.9.7
- LangGraph 0.0.20
- OS: Ubuntu 20.04

Configuration:
{
  "agents": {
    "context": {"enabled": true, "confidence_threshold": 80}
  },
  "routing": {"context_threshold": 80}
}

Steps to Reproduce:
1. Run check_multiagent.py with exemple.json
2. Observe violations with confidence < 80
3. Context agent is skipped

Expected: Context agent should run for low-confidence violations
Actual: Context agent is skipped, violations go directly to output

Logs:
[2024-01-15 10:30:45] INFO - [aggregator] 3 violations with confidence < 80
[2024-01-15 10:30:45] INFO - [aggregator] Next action: complete
[2024-01-15 10:30:45] INFO - Workflow complete

Error: Context agent not triggered despite low confidence violations
```

---

**Last Updated:** 2024-01-15
**Version:** 1.0.0
