# Human-in-the-Loop (HITL) Configuration Guide

## Overview

This guide explains how to configure the Human-in-the-Loop (HITL) system for the AI-Enhanced Compliance Checker. The HITL system enables human reviewers to validate and correct AI predictions, improving accuracy over time through continuous learning.

## Configuration File

The HITL system is configured through `hybrid_config.json`. A template with all available options is provided in `hybrid_config.template.json`.

## HITL Configuration Section

### Basic Settings

```json
{
  "hitl": {
    "enabled": false,
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

### Configuration Options

#### `enabled` (boolean, default: `false`)
- **Description**: Master switch for the HITL system
- **When to enable**: Enable when you want human reviewers to validate low-confidence AI predictions
- **Impact**: When enabled, violations below the review threshold are automatically queued for human review

#### `review_threshold` (integer, default: `70`)
- **Description**: Confidence score threshold below which violations are queued for review
- **Range**: 0-100
- **Recommendations**:
  - `60-70`: Balanced approach, reviews moderate-confidence predictions
  - `70-80`: Conservative, only reviews lower-confidence predictions
  - `50-60`: Aggressive, reviews more predictions (higher workload)
- **Example**: With threshold of 70, violations with confidence 69 or below are queued

#### `auto_queue_low_confidence` (boolean, default: `true`)
- **Description**: Automatically queue violations below review threshold
- **When to disable**: If you want manual control over what gets queued
- **Impact**: When false, violations are not automatically queued even if below threshold

#### `queue_max_size` (integer, default: `10000`)
- **Description**: Maximum number of items that can be in the review queue
- **Behavior**: When queue is full, oldest items are evicted (FIFO)
- **Recommendations**:
  - Small teams (1-5 reviewers): 1,000-5,000
  - Medium teams (5-20 reviewers): 5,000-10,000
  - Large teams (20+ reviewers): 10,000-50,000
- **Performance**: Larger queues use more memory but prevent data loss

#### `batch_similarity_threshold` (float, default: `0.85`)
- **Description**: Similarity threshold for grouping violations in batch operations
- **Range**: 0.0-1.0 (0 = no similarity, 1 = identical)
- **Recommendations**:
  - `0.90-1.0`: Very strict, only groups nearly identical violations
  - `0.80-0.90`: Balanced, groups similar violations
  - `0.70-0.80`: Lenient, groups loosely related violations
- **Use case**: Higher threshold = more precise batching, lower threshold = larger batches

#### `interactive_mode_default` (boolean, default: `false`)
- **Description**: Start in interactive review mode by default when running check.py
- **When to enable**: If your primary workflow involves immediate review after checking
- **Impact**: When true, check.py automatically enters review mode after completion

#### `audit_log_path` (string, default: `"./audit_logs/"`)
- **Description**: Directory path for storing audit logs
- **Requirements**: Directory must be writable
- **Recommendations**:
  - Use absolute paths for production environments
  - Ensure sufficient disk space for log storage
  - Consider log rotation policies
- **Example**: `"/var/log/compliance/audit_logs/"` or `"C:\\Logs\\audit_logs\\"`

#### `export_formats` (array, default: `["json", "csv"]`)
- **Description**: Supported formats for exporting audit logs and reports
- **Available formats**: `"json"`, `"csv"`
- **Use cases**:
  - JSON: Machine-readable, preserves structure, good for APIs
  - CSV: Human-readable, Excel-compatible, good for analysis

## Review Priorities Configuration

```json
{
  "review_priorities": {
    "critical_severity_boost": 20,
    "low_confidence_boost": 10,
    "age_penalty_per_hour": 0.5
  }
}
```

### Priority Scoring Algorithm

The review queue is prioritized using the following formula:

```
priority_score = base_score 
                 + (critical_severity_boost if severity == "critical")
                 + (low_confidence_boost if confidence < 50)
                 - (age_penalty_per_hour * hours_in_queue)
```

Items with **lower** priority scores are reviewed first.

### Configuration Options

#### `critical_severity_boost` (integer, default: `20`)
- **Description**: Priority boost for critical severity violations
- **Impact**: Critical violations are reviewed before non-critical ones
- **Recommendations**: 15-25 for most use cases

#### `low_confidence_boost` (integer, default: `10`)
- **Description**: Priority boost for very low confidence violations (< 50)
- **Impact**: Extremely uncertain predictions are prioritized
- **Recommendations**: 5-15 for most use cases

#### `age_penalty_per_hour` (float, default: `0.5`)
- **Description**: Priority penalty per hour that item has been in queue
- **Impact**: Older items gradually become lower priority
- **Recommendations**:
  - `0.1-0.5`: Slow aging, items stay relevant longer
  - `0.5-1.0`: Moderate aging, balanced approach
  - `1.0-2.0`: Fast aging, prioritizes recent items

## Environment Variable Overrides

You can override configuration values using environment variables:

```bash
# Enable HITL system
export HITL_ENABLED=true

# Set review threshold
export HITL_REVIEW_THRESHOLD=65
```

Available environment variables:
- `HITL_ENABLED`: Enable/disable HITL system (true/false)
- `HITL_REVIEW_THRESHOLD`: Set review threshold (0-100)

## Configuration Examples

### Example 1: Conservative Review (Low Workload)

```json
{
  "hitl": {
    "enabled": true,
    "review_threshold": 80,
    "auto_queue_low_confidence": true,
    "queue_max_size": 5000,
    "batch_similarity_threshold": 0.90
  }
}
```

**Use case**: Small team, only review very uncertain predictions

### Example 2: Aggressive Review (High Accuracy)

```json
{
  "hitl": {
    "enabled": true,
    "review_threshold": 60,
    "auto_queue_low_confidence": true,
    "queue_max_size": 20000,
    "batch_similarity_threshold": 0.80
  }
}
```

**Use case**: Large team, maximize accuracy through extensive review

### Example 3: Batch-Focused Workflow

```json
{
  "hitl": {
    "enabled": true,
    "review_threshold": 70,
    "auto_queue_low_confidence": true,
    "queue_max_size": 10000,
    "batch_similarity_threshold": 0.75,
    "interactive_mode_default": false
  }
}
```

**Use case**: Process similar violations in batches for efficiency

### Example 4: Interactive Review Workflow

```json
{
  "hitl": {
    "enabled": true,
    "review_threshold": 70,
    "auto_queue_low_confidence": true,
    "queue_max_size": 10000,
    "batch_similarity_threshold": 0.85,
    "interactive_mode_default": true
  }
}
```

**Use case**: Immediate review after each compliance check

## Integration with Existing Configuration

The HITL configuration works alongside existing confidence settings:

```json
{
  "confidence": {
    "threshold": 70,
    "review_threshold": 60
  },
  "hitl": {
    "enabled": true,
    "review_threshold": 70
  }
}
```

**Relationship**:
- `confidence.threshold`: Minimum confidence to report violation
- `confidence.review_threshold`: Legacy threshold (deprecated)
- `hitl.review_threshold`: Threshold for queueing to HITL system

**Recommendation**: Set `hitl.review_threshold` equal to or slightly higher than `confidence.threshold`

## Validation Rules

The configuration manager validates all HITL settings:

1. **review_threshold**: Must be 0-100
2. **queue_max_size**: Must be >= 1
3. **batch_similarity_threshold**: Must be 0.0-1.0
4. **critical_severity_boost**: Must be >= 0
5. **low_confidence_boost**: Must be >= 0
6. **age_penalty_per_hour**: Must be >= 0

Invalid configurations will raise a `ValueError` with detailed error messages.

## Programmatic Access

### Using ConfigManager

```python
from config_manager import get_config_manager

# Get configuration manager
config_mgr = get_config_manager()

# Check if HITL is enabled
if config_mgr.is_hitl_enabled():
    print("HITL system is enabled")

# Check if violation should be queued
confidence = 65
if config_mgr.should_queue_for_review(confidence):
    print(f"Violation with confidence {confidence} should be queued")

# Get HITL configuration
hitl_config = config_mgr.get_hitl_config()
print(f"Review threshold: {hitl_config.review_threshold}")
print(f"Queue max size: {hitl_config.queue_max_size}")

# Get review priorities
priorities = config_mgr.get_review_priorities()
print(f"Critical boost: {priorities.critical_severity_boost}")
```

### Updating Configuration at Runtime

```python
# Update HITL settings
config_mgr.update_config(**{
    'hitl.enabled': True,
    'hitl.review_threshold': 75
})

# Save updated configuration
config_mgr.save_config()
```

## Best Practices

1. **Start Conservative**: Begin with a higher review threshold (75-80) and lower it as your team's capacity increases

2. **Monitor Queue Size**: Regularly check queue statistics to ensure it's not growing unbounded

3. **Adjust Priorities**: Tune priority settings based on your team's workflow and violation patterns

4. **Regular Exports**: Export audit logs regularly for compliance and analysis

5. **Batch Similar Items**: Use batch operations for efficiency when reviewing similar violations

6. **Environment-Specific Settings**: Use different configurations for development, staging, and production

7. **Backup Configuration**: Keep backups of your configuration file, especially before major changes

## Troubleshooting

### Queue Growing Too Large

**Problem**: Review queue exceeds capacity
**Solutions**:
- Increase `queue_max_size`
- Increase `review_threshold` to queue fewer items
- Add more reviewers
- Use batch operations for efficiency

### Too Few Items in Queue

**Problem**: Not enough items to review
**Solutions**:
- Lower `review_threshold` to queue more items
- Verify `auto_queue_low_confidence` is enabled
- Check that `hitl.enabled` is true

### Batch Operations Not Grouping Items

**Problem**: Batch operations find no similar items
**Solutions**:
- Lower `batch_similarity_threshold`
- Verify violations have similar characteristics
- Check that enough items are in the queue

### Audit Logs Not Being Created

**Problem**: No audit logs in specified directory
**Solutions**:
- Verify `audit_log_path` exists and is writable
- Check file permissions
- Ensure HITL system is enabled
- Verify reviews are being completed

## See Also

- [HITL Integration Guide](HITL_INTEGRATION.md) - Complete HITL system documentation
- [Review Mode Usage Guide](REVIEW_MODE_USAGE.md) - How to use the review CLI
- [Configuration Guide](CONFIGURATION_GUIDE.md) - General configuration documentation
