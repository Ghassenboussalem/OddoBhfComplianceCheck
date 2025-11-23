# Multi-Agent System Configuration Guide

## Table of Contents

1. [Overview](#overview)
2. [Configuration File Structure](#configuration-file-structure)
3. [Multi-Agent System Settings](#multi-agent-system-settings)
4. [Agent-Specific Configuration](#agent-specific-configuration)
5. [Routing Configuration](#routing-configuration)
6. [State Management Configuration](#state-management-configuration)
7. [Error Handling Configuration](#error-handling-configuration)
8. [Monitoring Configuration](#monitoring-configuration)
9. [AI Service Configuration](#ai-service-configuration)
10. [Context Analysis Configuration](#context-analysis-configuration)
11. [Whitelist Configuration](#whitelist-configuration)
12. [HITL Configuration](#hitl-configuration)
13. [Configuration Examples](#configuration-examples)
14. [Best Practices](#best-practices)
15. [Troubleshooting](#troubleshooting)

---

## Overview

The Multi-Agent Compliance System uses a comprehensive configuration file (`hybrid_config.json`) to control all aspects of the system behavior. This guide provides detailed documentation of all configuration options, their purposes, and recommended values.

### Configuration Philosophy

- **Modular**: Each agent can be configured independently
- **Flexible**: Enable/disable features without code changes
- **Backward Compatible**: Works with existing single-agent configurations
- **Production-Ready**: Sensible defaults for most use cases

### Quick Start

To enable the multi-agent system, set the following in `hybrid_config.json`:

```json
{
  "multi_agent": {
    "enabled": true
  }
}
```

All other settings have sensible defaults and can be customized as needed.

---

## Configuration File Structure

The configuration file is organized into the following top-level sections:


```json
{
  "multi_agent": {},           // Multi-agent system settings
  "agents": {},                // Individual agent configurations
  "routing": {},               // Conditional routing logic
  "state_management": {},      // State persistence settings
  "error_handling": {},        // Error recovery strategies
  "monitoring": {},            // Observability and metrics
  "ai_service": {},            // AI provider configuration
  "context_analysis": {},      // Context-aware validation
  "whitelist": {},             // Whitelist management
  "hitl": {},                  // Human-in-the-Loop settings
  "confidence": {},            // Confidence scoring
  "features": {}               // Feature flags
}
```

---

## Multi-Agent System Settings

The `multi_agent` section controls the overall multi-agent system behavior.

### Configuration Options

```json
{
  "multi_agent": {
    "enabled": false,
    "parallel_execution": true,
    "max_parallel_agents": 4,
    "agent_timeout_seconds": 30,
    "checkpoint_interval": 5,
    "state_persistence": true,
    "checkpoint_db_path": "./checkpoints/compliance_workflow.db",
    "enable_workflow_visualization": true,
    "enable_agent_logging": true
  }
}
```

### Option Details

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable/disable multi-agent system. When `false`, uses legacy single-agent mode |
| `parallel_execution` | boolean | `true` | Allow independent agents to run in parallel for better performance |
| `max_parallel_agents` | integer | `4` | Maximum number of agents that can execute simultaneously |
| `agent_timeout_seconds` | integer | `30` | Default timeout for agent execution (can be overridden per agent) |
| `checkpoint_interval` | integer | `5` | Number of agent transitions between state checkpoints |
| `state_persistence` | boolean | `true` | Enable state persistence to disk for resumability |
| `checkpoint_db_path` | string | `"./checkpoints/compliance_workflow.db"` | Path to SQLite database for checkpoints |
| `enable_workflow_visualization` | boolean | `true` | Enable workflow graph visualization |
| `enable_agent_logging` | boolean | `true` | Enable detailed agent execution logging |

### Recommendations

- **Development**: Set `enabled: true`, `parallel_execution: true`, `state_persistence: true`
- **Production**: Same as development, but consider increasing `max_parallel_agents` based on available resources
- **Testing**: Set `state_persistence: false` for faster test execution
- **Debugging**: Set `enable_agent_logging: true` and reduce `max_parallel_agents: 1` for sequential execution

---

## Agent-Specific Configuration

The `agents` section allows fine-grained control over each agent's behavior.

### Available Agents


1. **supervisor** - Orchestrates workflow and coordinates agents
2. **preprocessor** - Extracts metadata, builds whitelist, normalizes document
3. **structure** - Checks structural compliance (promotional mention, target audience, etc.)
4. **performance** - Validates performance data and disclaimers
5. **securities** - Checks securities mentions and investment advice
6. **general** - General compliance checks (glossary, sources, dates)
7. **prospectus** - Validates against prospectus data (conditional)
8. **registration** - Checks country authorization (conditional)
9. **esg** - ESG classification and content validation (conditional)
10. **aggregator** - Combines results from all specialist agents
11. **context** - Analyzes context and intent for false-positive elimination
12. **evidence** - Extracts evidence and supporting quotes
13. **reviewer** - Manages Human-in-the-Loop review queue
14. **feedback** - Processes human feedback and calibrates confidence

### Common Agent Configuration

All agents support these common configuration options:

```json
{
  "agents": {
    "agent_name": {
      "enabled": true,
      "timeout_seconds": 30,
      "max_retries": 2,
      "retry_delay_seconds": 1.0,
      "parallel_tool_execution": false,
      "conditional": false,
      "condition": null,
      "confidence_threshold": null
    }
  }
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable/disable this agent |
| `timeout_seconds` | integer | `30` | Maximum execution time for this agent |
| `max_retries` | integer | `2` | Number of retry attempts on failure |
| `retry_delay_seconds` | float | `1.0` | Delay between retry attempts |
| `parallel_tool_execution` | boolean | `false` | Allow agent's tools to run in parallel |
| `conditional` | boolean | `false` | Whether agent runs conditionally |
| `condition` | string | `null` | Condition for execution (see Conditional Agents) |
| `confidence_threshold` | integer | `null` | Minimum confidence for agent's results |

### Supervisor Agent

```json
{
  "agents": {
    "supervisor": {
      "enabled": true,
      "timeout_seconds": 10
    }
  }
}
```

The supervisor agent is always required and should have a short timeout as it only coordinates other agents.

### Preprocessor Agent

```json
{
  "agents": {
    "preprocessor": {
      "enabled": true,
      "timeout_seconds": 15,
      "auto_extract_metadata": true,
      "auto_build_whitelist": true
    }
  }
}
```

| Custom Option | Type | Default | Description |
|---------------|------|---------|-------------|
| `auto_extract_metadata` | boolean | `true` | Automatically extract document metadata |
| `auto_build_whitelist` | boolean | `true` | Automatically build whitelist from document |

### Core Compliance Agents

These agents run in parallel after preprocessing:

```json
{
  "agents": {
    "structure": {
      "enabled": true,
      "timeout_seconds": 30,
      "parallel_tool_execution": true
    },
    "performance": {
      "enabled": true,
      "timeout_seconds": 30,
      "parallel_tool_execution": true
    },
    "securities": {
      "enabled": true,
      "timeout_seconds": 30,
      "parallel_tool_execution": true,
      "use_whitelist_filtering": true
    },
    "general": {
      "enabled": true,
      "timeout_seconds": 30,
      "parallel_tool_execution": true
    }
  }
}
```

**Recommendations**:
- Enable `parallel_tool_execution: true` for faster execution
- Increase `timeout_seconds` if processing large documents
- Securities agent should have `use_whitelist_filtering: true` to reduce false positives


### Specialized Compliance Agents

These agents run conditionally based on document metadata:

```json
{
  "agents": {
    "prospectus": {
      "enabled": true,
      "timeout_seconds": 30,
      "conditional": true,
      "condition": "prospectus_data_available"
    },
    "registration": {
      "enabled": true,
      "timeout_seconds": 30,
      "conditional": true,
      "condition": "fund_isin_available"
    },
    "esg": {
      "enabled": true,
      "timeout_seconds": 30,
      "conditional": true,
      "condition": "esg_classification_not_other"
    }
  }
}
```

#### Conditional Execution

| Condition | When Agent Runs |
|-----------|-----------------|
| `prospectus_data_available` | When `config.prospectus_data` is provided |
| `fund_isin_available` | When `metadata.fund_isin` is present |
| `esg_classification_not_other` | When `metadata.esg_classification` is not "other" |

### Analysis Agents

These agents perform context analysis and evidence extraction:

```json
{
  "agents": {
    "aggregator": {
      "enabled": true,
      "timeout_seconds": 10,
      "deduplication_enabled": true,
      "confidence_calculation_method": "weighted_average"
    },
    "context": {
      "enabled": true,
      "timeout_seconds": 45,
      "confidence_threshold": 80,
      "skip_if_all_high_confidence": true,
      "intent_classification_enabled": true,
      "semantic_validation_enabled": true
    },
    "evidence": {
      "enabled": true,
      "timeout_seconds": 45,
      "extract_quotes": true,
      "find_performance_data": true,
      "track_locations": true
    }
  }
}
```

| Custom Option | Agent | Description |
|---------------|-------|-------------|
| `deduplication_enabled` | aggregator | Remove duplicate violations |
| `confidence_calculation_method` | aggregator | Method for calculating confidence: "weighted_average", "min", "max" |
| `skip_if_all_high_confidence` | context | Skip context analysis if all violations have high confidence |
| `intent_classification_enabled` | context | Enable intent classification (ADVICE, DESCRIPTION, etc.) |
| `semantic_validation_enabled` | context | Enable semantic validation for false-positive elimination |
| `extract_quotes` | evidence | Extract supporting quotes for violations |
| `find_performance_data` | evidence | Find actual performance data (numbers with %) |
| `track_locations` | evidence | Track precise locations of violations |

### Review and Feedback Agents

```json
{
  "agents": {
    "reviewer": {
      "enabled": true,
      "timeout_seconds": 10,
      "confidence_threshold": 70,
      "auto_queue_enabled": true,
      "priority_scoring_enabled": true,
      "batch_operations_enabled": true
    },
    "feedback": {
      "enabled": true,
      "timeout_seconds": 10,
      "real_time_calibration": true,
      "pattern_detection_enabled": true,
      "rule_suggestion_enabled": false
    }
  }
}
```

| Custom Option | Agent | Description |
|---------------|-------|-------------|
| `auto_queue_enabled` | reviewer | Automatically queue low-confidence violations |
| `priority_scoring_enabled` | reviewer | Calculate priority scores for review items |
| `batch_operations_enabled` | reviewer | Enable batch review operations |
| `real_time_calibration` | feedback | Update confidence calibration in real-time |
| `pattern_detection_enabled` | feedback | Detect patterns in false positives |
| `rule_suggestion_enabled` | feedback | Suggest rule modifications (experimental) |

---

## Routing Configuration

The `routing` section controls conditional routing logic between agents.

```json
{
  "routing": {
    "context_threshold": 80,
    "review_threshold": 70,
    "skip_context_if_high_confidence": true,
    "skip_review_if_high_confidence": true,
    "parallel_specialist_agents": [
      "structure",
      "performance",
      "securities",
      "general"
    ],
    "conditional_agents": [
      "prospectus",
      "registration",
      "esg"
    ],
    "sequential_flow": [
      "supervisor",
      "preprocessor",
      "aggregator",
      "context",
      "evidence",
      "reviewer"
    ]
  }
}
```


### Routing Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `context_threshold` | integer | `80` | Violations below this confidence trigger context analysis |
| `review_threshold` | integer | `70` | Violations below this confidence trigger human review |
| `skip_context_if_high_confidence` | boolean | `true` | Skip context agent if all violations have high confidence |
| `skip_review_if_high_confidence` | boolean | `true` | Skip reviewer agent if all violations have high confidence |
| `parallel_specialist_agents` | array | See above | Agents that run in parallel after preprocessing |
| `conditional_agents` | array | See above | Agents that run conditionally |
| `sequential_flow` | array | See above | Agents that must run sequentially |

### Workflow Flow

```
supervisor → preprocessor → [parallel agents] → aggregator
                                                     ↓
                                    (confidence < 80?) → context → evidence
                                                                      ↓
                                                     (confidence < 70?) → reviewer → END
                                                                      ↓
                                                                     END
```

### Tuning Thresholds

- **High Precision** (fewer false positives, more human review):
  - `context_threshold: 90`
  - `review_threshold: 80`

- **Balanced** (recommended):
  - `context_threshold: 80`
  - `review_threshold: 70`

- **High Recall** (catch all violations, accept more false positives):
  - `context_threshold: 70`
  - `review_threshold: 60`

---

## State Management Configuration

The `state_management` section controls state persistence and checkpointing.

```json
{
  "state_management": {
    "checkpoint_enabled": true,
    "checkpoint_interval_agents": 5,
    "save_intermediate_states": true,
    "state_history_max_size": 100,
    "enable_state_validation": true,
    "auto_cleanup_old_checkpoints": true,
    "checkpoint_retention_days": 7
  }
}
```

### State Management Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `checkpoint_enabled` | boolean | `true` | Enable state checkpointing |
| `checkpoint_interval_agents` | integer | `5` | Number of agent transitions between checkpoints |
| `save_intermediate_states` | boolean | `true` | Save state after each agent execution |
| `state_history_max_size` | integer | `100` | Maximum number of historical states to keep |
| `enable_state_validation` | boolean | `true` | Validate state structure after each update |
| `auto_cleanup_old_checkpoints` | boolean | `true` | Automatically delete old checkpoints |
| `checkpoint_retention_days` | integer | `7` | Days to retain old checkpoints |

### Recommendations

- **Production**: Enable all state management features for resumability
- **Development**: Set `checkpoint_interval_agents: 1` for fine-grained debugging
- **Testing**: Set `checkpoint_enabled: false` to speed up tests
- **Long-running workflows**: Increase `state_history_max_size` and `checkpoint_retention_days`

---

## Error Handling Configuration

The `error_handling` section controls how the system responds to agent failures.

```json
{
  "error_handling": {
    "agent_failure_strategy": "continue",
    "max_agent_retries": 2,
    "retry_delay_seconds": 1.0,
    "fallback_to_rules_on_ai_failure": true,
    "partial_results_on_failure": true,
    "circuit_breaker_enabled": true,
    "circuit_breaker_threshold": 5,
    "circuit_breaker_timeout_seconds": 60
  }
}
```

### Error Handling Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `agent_failure_strategy` | string | `"continue"` | Strategy when agent fails: "continue", "stop", "retry" |
| `max_agent_retries` | integer | `2` | Maximum retry attempts for failed agents |
| `retry_delay_seconds` | float | `1.0` | Delay between retry attempts (exponential backoff) |
| `fallback_to_rules_on_ai_failure` | boolean | `true` | Use rule-based checking if AI service fails |
| `partial_results_on_failure` | boolean | `true` | Generate partial report if some agents fail |
| `circuit_breaker_enabled` | boolean | `true` | Enable circuit breaker pattern |
| `circuit_breaker_threshold` | integer | `5` | Number of failures before opening circuit |
| `circuit_breaker_timeout_seconds` | integer | `60` | Time before attempting to close circuit |


### Failure Strategies

#### Continue (Recommended)
```json
{"agent_failure_strategy": "continue"}
```
- Workflow continues even if an agent fails
- Failed agent's checks are skipped
- Partial results are generated
- Best for production environments

#### Stop
```json
{"agent_failure_strategy": "stop"}
```
- Workflow stops immediately on agent failure
- No partial results
- Best for critical compliance checks where all agents must succeed

#### Retry
```json
{"agent_failure_strategy": "retry"}
```
- Failed agent is retried up to `max_agent_retries` times
- Exponential backoff between retries
- Falls back to "continue" if all retries fail
- Best for transient failures (network issues, rate limits)

### Circuit Breaker Pattern

The circuit breaker prevents cascading failures:

1. **Closed** (normal operation): All requests pass through
2. **Open** (failure threshold reached): Requests fail immediately
3. **Half-Open** (after timeout): Test request to check if service recovered

```json
{
  "circuit_breaker_enabled": true,
  "circuit_breaker_threshold": 5,      // Open after 5 failures
  "circuit_breaker_timeout_seconds": 60 // Try again after 60 seconds
}
```

---

## Monitoring Configuration

The `monitoring` section controls observability and metrics collection.

```json
{
  "monitoring": {
    "enabled": true,
    "log_agent_invocations": true,
    "log_level": "INFO",
    "track_execution_times": true,
    "track_success_rates": true,
    "track_cache_hits": true,
    "track_api_calls": true,
    "metrics_export_enabled": true,
    "metrics_export_path": "./monitoring/metrics/",
    "dashboard_enabled": false,
    "alert_on_failures": true,
    "alert_threshold_failure_rate": 0.2
  }
}
```

### Monitoring Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable monitoring system |
| `log_agent_invocations` | boolean | `true` | Log every agent invocation |
| `log_level` | string | `"INFO"` | Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `track_execution_times` | boolean | `true` | Track agent execution times |
| `track_success_rates` | boolean | `true` | Track agent success/failure rates |
| `track_cache_hits` | boolean | `true` | Track cache hit rates |
| `track_api_calls` | boolean | `true` | Track AI API call counts |
| `metrics_export_enabled` | boolean | `true` | Export metrics to files |
| `metrics_export_path` | string | `"./monitoring/metrics/"` | Path for metrics export |
| `dashboard_enabled` | boolean | `false` | Enable real-time dashboard (requires additional setup) |
| `alert_on_failures` | boolean | `true` | Send alerts on agent failures |
| `alert_threshold_failure_rate` | float | `0.2` | Failure rate threshold for alerts (0.0-1.0) |

### Log Levels

- **DEBUG**: Detailed information for debugging (verbose)
- **INFO**: General informational messages (recommended for production)
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failures
- **CRITICAL**: Critical errors requiring immediate attention

### Metrics Collected

When monitoring is enabled, the system tracks:

1. **Agent Metrics**:
   - Execution time per agent
   - Success/failure counts
   - Retry counts
   - Timeout occurrences

2. **Performance Metrics**:
   - Total workflow execution time
   - Parallel execution efficiency
   - Cache hit rates
   - API call counts and costs

3. **Quality Metrics**:
   - Violations detected per agent
   - Confidence score distributions
   - False positive rates (with HITL feedback)
   - Review queue statistics

---

## AI Service Configuration

The `ai_service` section configures the AI provider for enhanced compliance checking.

```json
{
  "ai_service": {
    "provider": "auto",
    "model_name": "auto",
    "api_key_env_var": "",
    "timeout": 30,
    "max_tokens": 2000,
    "temperature": 0.1,
    "retry_attempts": 3,
    "retry_delay": 1.0
  }
}
```


### AI Service Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `provider` | string | `"auto"` | AI provider: "auto", "openai", "anthropic", "azure" |
| `model_name` | string | `"auto"` | Model name (provider-specific) |
| `api_key_env_var` | string | `""` | Environment variable name for API key |
| `timeout` | integer | `30` | Request timeout in seconds |
| `max_tokens` | integer | `2000` | Maximum tokens per request |
| `temperature` | float | `0.1` | Sampling temperature (0.0-1.0, lower = more deterministic) |
| `retry_attempts` | integer | `3` | Number of retry attempts on failure |
| `retry_delay` | float | `1.0` | Delay between retries in seconds |

### Provider Configuration

#### Auto (Recommended)
```json
{"provider": "auto", "model_name": "auto"}
```
Automatically detects available provider from environment variables:
- Checks `OPENAI_API_KEY` → uses OpenAI
- Checks `ANTHROPIC_API_KEY` → uses Anthropic
- Checks `AZURE_OPENAI_API_KEY` → uses Azure OpenAI

#### OpenAI
```json
{
  "provider": "openai",
  "model_name": "gpt-4",
  "api_key_env_var": "OPENAI_API_KEY"
}
```

#### Anthropic
```json
{
  "provider": "anthropic",
  "model_name": "claude-3-opus-20240229",
  "api_key_env_var": "ANTHROPIC_API_KEY"
}
```

#### Azure OpenAI
```json
{
  "provider": "azure",
  "model_name": "gpt-4",
  "api_key_env_var": "AZURE_OPENAI_API_KEY"
}
```

### Temperature Guidelines

- **0.0-0.2**: Highly deterministic, consistent results (recommended for compliance)
- **0.3-0.5**: Balanced creativity and consistency
- **0.6-1.0**: More creative, less consistent (not recommended for compliance)

---

## Context Analysis Configuration

The `context_analysis` section configures context-aware validation for false-positive elimination.

```json
{
  "context_analysis": {
    "enabled": true,
    "min_confidence": 60,
    "use_fallback_rules": true,
    "intent_classification_enabled": true,
    "semantic_validation_enabled": true,
    "evidence_extraction_enabled": true,
    "max_context_length": 2000,
    "context_window_chars": 500
  }
}
```

### Context Analysis Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable context-aware validation |
| `min_confidence` | integer | `60` | Minimum confidence for context analysis |
| `use_fallback_rules` | boolean | `true` | Fall back to rule-based if AI fails |
| `intent_classification_enabled` | boolean | `true` | Classify intent (ADVICE, DESCRIPTION, etc.) |
| `semantic_validation_enabled` | boolean | `true` | Use semantic understanding for validation |
| `evidence_extraction_enabled` | boolean | `true` | Extract supporting evidence |
| `max_context_length` | integer | `2000` | Maximum context length in characters |
| `context_window_chars` | integer | `500` | Context window around violation |

### Intent Classification

When enabled, the system classifies text intent as:

- **ADVICE**: Investment advice or recommendations
- **DESCRIPTION**: Factual description of fund strategy
- **FACT**: Objective facts or data
- **EXAMPLE**: Illustrative examples
- **DISCLAIMER**: Legal disclaimers

This helps eliminate false positives where descriptive text is mistaken for advice.

---

## Whitelist Configuration

The `whitelist` section configures automatic whitelist generation.

```json
{
  "whitelist": {
    "auto_extract_fund_name": true,
    "include_strategy_terms": true,
    "include_regulatory_terms": true,
    "include_benchmark_terms": true,
    "include_generic_financial_terms": true,
    "custom_terms": [],
    "strategy_terms": ["momentum", "quantitative", "systematic", ...],
    "regulatory_terms": ["sri", "srri", "sfdr", "ucits", ...],
    "benchmark_terms": ["s&p", "500", "msci", "stoxx", ...],
    "generic_financial_terms": ["actions", "equities", "bonds", ...]
  }
}
```


### Whitelist Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `auto_extract_fund_name` | boolean | `true` | Automatically extract and whitelist fund name |
| `include_strategy_terms` | boolean | `true` | Include strategy-related terms |
| `include_regulatory_terms` | boolean | `true` | Include regulatory terms |
| `include_benchmark_terms` | boolean | `true` | Include benchmark names |
| `include_generic_financial_terms` | boolean | `true` | Include generic financial terms |
| `custom_terms` | array | `[]` | Additional custom terms to whitelist |

### Custom Terms

Add your own whitelisted terms:

```json
{
  "whitelist": {
    "custom_terms": [
      "my_fund_name",
      "custom_strategy",
      "proprietary_term"
    ]
  }
}
```

---

## HITL Configuration

The `hitl` section configures Human-in-the-Loop review integration.

```json
{
  "hitl": {
    "enabled": true,
    "review_threshold": 70,
    "auto_queue_low_confidence": true,
    "queue_max_size": 10000,
    "batch_similarity_threshold": 0.85,
    "interactive_mode_default": false,
    "audit_log_path": "./audit_logs/",
    "export_formats": ["json", "csv"]
  }
}
```

### HITL Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable HITL review system |
| `review_threshold` | integer | `70` | Violations below this confidence are queued for review |
| `auto_queue_low_confidence` | boolean | `true` | Automatically queue low-confidence violations |
| `queue_max_size` | integer | `10000` | Maximum review queue size |
| `batch_similarity_threshold` | float | `0.85` | Similarity threshold for batch operations |
| `interactive_mode_default` | boolean | `false` | Enable interactive mode by default |
| `audit_log_path` | string | `"./audit_logs/"` | Path for audit logs |
| `export_formats` | array | `["json", "csv"]` | Export formats for review data |

### Review Priorities

```json
{
  "review_priorities": {
    "critical_severity_boost": 20,
    "low_confidence_boost": 10,
    "age_penalty_per_hour": 0.5
  }
}
```

Priority score calculation:
```
priority = 100 - confidence 
         + (critical_severity ? critical_severity_boost : 0)
         + (confidence < 50 ? low_confidence_boost : 0)
         - (age_in_hours * age_penalty_per_hour)
```

---

## Configuration Examples

### Example 1: Development Configuration

Optimized for development with full logging and debugging:

```json
{
  "multi_agent": {
    "enabled": true,
    "parallel_execution": false,
    "max_parallel_agents": 1,
    "state_persistence": true,
    "enable_agent_logging": true
  },
  "monitoring": {
    "enabled": true,
    "log_level": "DEBUG",
    "track_execution_times": true
  },
  "error_handling": {
    "agent_failure_strategy": "stop",
    "max_agent_retries": 0
  }
}
```

### Example 2: Production Configuration

Optimized for production with performance and reliability:

```json
{
  "multi_agent": {
    "enabled": true,
    "parallel_execution": true,
    "max_parallel_agents": 8,
    "state_persistence": true,
    "checkpoint_interval": 5
  },
  "routing": {
    "context_threshold": 80,
    "review_threshold": 70,
    "skip_context_if_high_confidence": true
  },
  "error_handling": {
    "agent_failure_strategy": "continue",
    "max_agent_retries": 3,
    "fallback_to_rules_on_ai_failure": true,
    "circuit_breaker_enabled": true
  },
  "monitoring": {
    "enabled": true,
    "log_level": "INFO",
    "alert_on_failures": true,
    "metrics_export_enabled": true
  }
}
```

### Example 3: High-Precision Configuration

Optimized for maximum accuracy with more human review:

```json
{
  "routing": {
    "context_threshold": 90,
    "review_threshold": 80,
    "skip_context_if_high_confidence": false
  },
  "agents": {
    "context": {
      "enabled": true,
      "confidence_threshold": 90,
      "skip_if_all_high_confidence": false
    },
    "evidence": {
      "enabled": true,
      "extract_quotes": true,
      "find_performance_data": true
    }
  },
  "hitl": {
    "enabled": true,
    "review_threshold": 80,
    "auto_queue_low_confidence": true
  }
}
```


### Example 4: Fast Processing Configuration

Optimized for speed with minimal analysis:

```json
{
  "multi_agent": {
    "enabled": true,
    "parallel_execution": true,
    "max_parallel_agents": 12
  },
  "routing": {
    "context_threshold": 70,
    "review_threshold": 60,
    "skip_context_if_high_confidence": true,
    "skip_review_if_high_confidence": true
  },
  "agents": {
    "context": {
      "enabled": false
    },
    "evidence": {
      "enabled": false
    }
  },
  "state_management": {
    "checkpoint_enabled": false,
    "save_intermediate_states": false
  }
}
```

### Example 5: Testing Configuration

Optimized for automated testing:

```json
{
  "multi_agent": {
    "enabled": true,
    "parallel_execution": false,
    "state_persistence": false,
    "enable_agent_logging": false
  },
  "ai_service": {
    "timeout": 10,
    "retry_attempts": 1
  },
  "monitoring": {
    "enabled": false
  },
  "error_handling": {
    "agent_failure_strategy": "stop"
  }
}
```

---

## Best Practices

### 1. Start with Defaults

Begin with the default configuration and adjust based on your needs:

```json
{
  "multi_agent": {
    "enabled": true
  }
}
```

All other settings have sensible defaults.

### 2. Enable Monitoring in Production

Always enable monitoring in production environments:

```json
{
  "monitoring": {
    "enabled": true,
    "log_level": "INFO",
    "alert_on_failures": true,
    "metrics_export_enabled": true
  }
}
```

### 3. Use State Persistence for Long Workflows

Enable state persistence for workflows that may be interrupted:

```json
{
  "multi_agent": {
    "state_persistence": true,
    "checkpoint_interval": 5
  }
}
```

### 4. Tune Thresholds Based on Feedback

Adjust confidence thresholds based on HITL feedback:

- If too many false positives reach review: **increase** `context_threshold`
- If missing violations: **decrease** `context_threshold` and `review_threshold`
- Monitor false positive rates and adjust accordingly

### 5. Configure Error Handling for Your Environment

Choose error handling strategy based on requirements:

- **Critical compliance**: Use `"stop"` strategy
- **Production systems**: Use `"continue"` strategy with fallbacks
- **Transient failures**: Use `"retry"` strategy

### 6. Optimize Parallel Execution

Adjust `max_parallel_agents` based on available resources:

```python
# Rule of thumb:
max_parallel_agents = min(cpu_cores, 8)
```

### 7. Use Conditional Agents Wisely

Only enable specialized agents when needed:

```json
{
  "agents": {
    "prospectus": {
      "enabled": true,
      "conditional": true
    }
  }
}
```

### 8. Configure Timeouts Appropriately

Set timeouts based on document size and complexity:

- **Small documents** (< 10 pages): 15-30 seconds per agent
- **Medium documents** (10-50 pages): 30-60 seconds per agent
- **Large documents** (> 50 pages): 60-120 seconds per agent

### 9. Enable Caching for Performance

Enable caching to reduce redundant AI calls:

```json
{
  "cache": {
    "enabled": true,
    "max_size": 1000,
    "ttl_seconds": 3600
  }
}
```

### 10. Regular Configuration Reviews

Review and update configuration regularly:

- Monitor metrics to identify bottlenecks
- Adjust thresholds based on accuracy metrics
- Update agent configurations as rules evolve
- Test configuration changes in development first

---

## Troubleshooting

### Issue: Multi-Agent System Not Running

**Symptom**: System uses legacy single-agent mode

**Solution**:
```json
{
  "multi_agent": {
    "enabled": true  // Ensure this is set to true
  }
}
```

### Issue: Agents Timing Out

**Symptom**: Agents frequently exceed timeout

**Solutions**:
1. Increase agent timeout:
```json
{
  "agents": {
    "agent_name": {
      "timeout_seconds": 60  // Increase from default 30
    }
  }
}
```

2. Reduce document size or complexity
3. Enable parallel tool execution:
```json
{
  "agents": {
    "agent_name": {
      "parallel_tool_execution": true
    }
  }
}
```


### Issue: Too Many False Positives

**Symptom**: Many violations are false positives

**Solutions**:
1. Enable context analysis:
```json
{
  "agents": {
    "context": {
      "enabled": true,
      "confidence_threshold": 80
    }
  }
}
```

2. Lower context threshold to analyze more violations:
```json
{
  "routing": {
    "context_threshold": 70  // Lower from 80
  }
}
```

3. Enable semantic validation:
```json
{
  "context_analysis": {
    "semantic_validation_enabled": true,
    "intent_classification_enabled": true
  }
}
```

4. Improve whitelist:
```json
{
  "whitelist": {
    "auto_extract_fund_name": true,
    "include_strategy_terms": true,
    "custom_terms": ["your", "custom", "terms"]
  }
}
```

### Issue: Missing Violations

**Symptom**: Known violations not detected

**Solutions**:
1. Ensure all relevant agents are enabled:
```json
{
  "agents": {
    "structure": {"enabled": true},
    "performance": {"enabled": true},
    "securities": {"enabled": true},
    "general": {"enabled": true}
  }
}
```

2. Check conditional agents are running:
```json
{
  "agents": {
    "prospectus": {
      "enabled": true,
      "conditional": true
    }
  }
}
```

3. Verify document metadata is correct
4. Check agent logs for errors

### Issue: Slow Performance

**Symptom**: Workflow takes too long to complete

**Solutions**:
1. Enable parallel execution:
```json
{
  "multi_agent": {
    "parallel_execution": true,
    "max_parallel_agents": 8
  }
}
```

2. Enable parallel tool execution:
```json
{
  "agents": {
    "structure": {"parallel_tool_execution": true},
    "performance": {"parallel_tool_execution": true}
  }
}
```

3. Skip unnecessary analysis:
```json
{
  "routing": {
    "skip_context_if_high_confidence": true,
    "skip_review_if_high_confidence": true
  }
}
```

4. Disable optional agents:
```json
{
  "agents": {
    "evidence": {"enabled": false}
  }
}
```

5. Enable caching:
```json
{
  "cache": {
    "enabled": true,
    "max_size": 1000
  }
}
```

### Issue: State Persistence Errors

**Symptom**: Errors saving or loading checkpoints

**Solutions**:
1. Check database path is writable:
```json
{
  "multi_agent": {
    "checkpoint_db_path": "./checkpoints/compliance_workflow.db"
  }
}
```

2. Ensure directory exists:
```bash
mkdir -p checkpoints
```

3. Check disk space
4. Verify SQLite is installed

### Issue: AI Service Failures

**Symptom**: AI-enhanced checks failing

**Solutions**:
1. Verify API key is set:
```bash
export OPENAI_API_KEY="your-key"
```

2. Enable fallback to rules:
```json
{
  "error_handling": {
    "fallback_to_rules_on_ai_failure": true
  }
}
```

3. Increase timeout:
```json
{
  "ai_service": {
    "timeout": 60,
    "retry_attempts": 5
  }
}
```

4. Check rate limits and quotas

### Issue: Memory Issues

**Symptom**: Out of memory errors

**Solutions**:
1. Reduce parallel agents:
```json
{
  "multi_agent": {
    "max_parallel_agents": 2
  }
}
```

2. Reduce state history:
```json
{
  "state_management": {
    "state_history_max_size": 10
  }
}
```

3. Disable intermediate state saving:
```json
{
  "state_management": {
    "save_intermediate_states": false
  }
}
```

4. Reduce cache size:
```json
{
  "cache": {
    "max_size": 100
  }
}
```

### Issue: Configuration Not Loading

**Symptom**: Changes to config file not taking effect

**Solutions**:
1. Verify JSON syntax is valid
2. Check file path is correct
3. Restart application after config changes
4. Check for typos in configuration keys
5. Review logs for configuration errors

### Getting Help

If you encounter issues not covered here:

1. **Check Logs**: Review agent logs in `monitoring/logs/`
2. **Check Metrics**: Review metrics in `monitoring/metrics/`
3. **Enable Debug Logging**:
```json
{
  "monitoring": {
    "log_level": "DEBUG"
  }
}
```
4. **Test Individual Agents**: Use agent test scripts
5. **Consult Documentation**: See `docs/MULTI_AGENT_ARCHITECTURE.md`

---

## Configuration Validation

### Validating Your Configuration

Use the configuration manager to validate your configuration:

```python
from config.agent_config_manager import AgentConfigManager

# Load and validate configuration
manager = AgentConfigManager("hybrid_config.json")

# Print summary
manager.print_summary()

# Check specific settings
print(f"Multi-agent enabled: {manager.is_multi_agent_enabled()}")
print(f"Parallel execution: {manager.get_multi_agent_config().parallel_execution}")
```

### Common Validation Errors

1. **Invalid threshold values**: Must be 0-100
2. **Invalid timeout values**: Must be > 0
3. **Invalid log level**: Must be DEBUG, INFO, WARNING, ERROR, or CRITICAL
4. **Invalid failure strategy**: Must be "continue", "stop", or "retry"
5. **Missing required agents**: Supervisor and preprocessor must be enabled

---

## Configuration Migration

### Migrating from Single-Agent to Multi-Agent

1. **Backup existing configuration**:
```bash
cp hybrid_config.json hybrid_config.json.backup
```

2. **Add multi-agent section**:
```json
{
  "multi_agent": {
    "enabled": true
  }
}
```

3. **Test with single document**:
```bash
python check_multiagent.py --document exemple.json
```

4. **Compare results with legacy system**:
```bash
python tests/ab_testing.py
```

5. **Gradually enable features**:
   - Start with parallel execution disabled
   - Enable parallel execution
   - Enable conditional routing
   - Enable state persistence
   - Enable monitoring

### Backward Compatibility

The multi-agent system maintains backward compatibility:

- All existing configuration options are supported
- Legacy output format is preserved
- Command-line interface is unchanged
- Can run in compatibility mode with `--use-legacy` flag

---

## Summary

This configuration guide covers all aspects of the Multi-Agent Compliance System configuration. Key takeaways:

1. **Start Simple**: Begin with defaults and adjust based on needs
2. **Monitor Performance**: Enable monitoring to identify bottlenecks
3. **Tune Thresholds**: Adjust confidence thresholds based on feedback
4. **Handle Errors Gracefully**: Configure appropriate error handling strategies
5. **Optimize for Your Environment**: Adjust parallel execution and timeouts
6. **Test Changes**: Always test configuration changes in development first

For more information, see:
- [Multi-Agent Architecture](MULTI_AGENT_ARCHITECTURE.md)
- [Agent API Documentation](AGENT_API.md)
- [Migration Guide](MIGRATION_TO_MULTIAGENT.md)
- [Troubleshooting Guide](MULTIAGENT_TROUBLESHOOTING.md)

---

**Last Updated**: November 23, 2025  
**Version**: 1.0.0
