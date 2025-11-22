# Compliance Metrics Usage Guide

## Overview

The `compliance_metrics.py` module provides comprehensive performance monitoring and accuracy tracking for the compliance checker system. It tracks false positive rates, false negative rates, precision, recall, AI API usage, cache performance, and processing times.

## Features

- **Accuracy Metrics**: Precision, recall, F1 score, false positive rate, false negative rate
- **AI Performance**: API calls, cache hits, fallback rate
- **Processing Metrics**: Document processing time, average check time
- **Per-Check-Type Metrics**: Detailed metrics for each compliance check type
- **Document History**: Track metrics for individual documents

## Usage

### Command Line

Display metrics after running a compliance check:

```bash
python check.py exemple.json --show-metrics
```

This will:
1. Run the compliance check
2. Display a comprehensive metrics dashboard
3. Export metrics to `exemple_metrics.json`

### Programmatic Usage

```python
from compliance_metrics import get_compliance_metrics

# Get singleton instance
metrics = get_compliance_metrics()

# Start document check
start_time = metrics.start_document_check()

# Record individual check results
metrics.record_check_result(
    check_type='prohibited_phrases',
    predicted_violation=True,
    actual_violation=True,  # If ground truth is known
    duration_ms=45.2,
    ai_call=True,
    cached=False,
    fallback=False
)

# Record complete document result
metrics.record_document_result(
    document_id='exemple.json',
    start_time=start_time,
    violations=violations_list,
    false_positives=0,
    false_negatives=0,
    ai_calls=5,
    cache_hits=3,
    fallback_count=0
)

# Display dashboard
metrics.print_dashboard()

# Export to JSON
metrics.export_metrics('metrics_output.json')
```

## Metrics Output

### Console Dashboard

```
======================================================================
📊 COMPLIANCE METRICS DASHBOARD
======================================================================

⏱️  System Uptime: 0.50 hours
   Documents Processed: 10
   Total Violations: 25
   Avg Violations/Doc: 2.50
   Avg Processing Time: 1250.00ms

🎯 Accuracy Metrics:
   Precision: 95.00%
   Recall: 98.00%
   F1 Score: 0.965
   Overall Accuracy: 96.50%
   False Positive Rate: 5.00%
   False Negative Rate: 2.00%

🤖 AI Performance:
   Total AI Calls: 45
   Cache Hits: 30
   Cache Hit Rate: 40.00%
   Fallback Count: 2
   Fallback Rate: 20.00%

📋 Metrics by Check Type:
   prohibited_phrases:
     Total Checks: 10
     Precision: 100.00%
     Recall: 100.00%
     FP Rate: 0.00%
     FN Rate: 0.00%
     Avg Time: 50.00ms
     AI Calls: 8
     Cache Hit Rate: 60.00%
```

### JSON Export Structure

```json
{
  "timestamp": "2025-11-22T16:22:11.348843",
  "overall": {
    "uptime_seconds": 1800.0,
    "uptime_hours": 0.5,
    "total_documents": 10,
    "total_violations": 25,
    "avg_violations_per_doc": 2.5,
    "avg_processing_time_ms": 1250.0,
    "total_false_positives": 2,
    "total_false_negatives": 1,
    "accuracy_metrics": {
      "precision": 0.95,
      "recall": 0.98,
      "accuracy": 0.965,
      "false_positive_rate": 0.05,
      "false_negative_rate": 0.02,
      "f1_score": 0.965
    },
    "ai_performance": {
      "total_ai_calls": 45,
      "total_cache_hits": 30,
      "cache_hit_rate": 40.0,
      "total_fallbacks": 2,
      "fallback_rate": 20.0
    }
  },
  "by_check_type": {
    "prohibited_phrases": {
      "check_type": "prohibited_phrases",
      "total_checks": 10,
      "precision": 1.0,
      "recall": 1.0,
      "f1_score": 1.0,
      "accuracy": 1.0,
      "false_positive_rate": 0.0,
      "false_negative_rate": 0.0,
      "cache_hit_rate": 60.0,
      "fallback_rate": 0.0
    }
  },
  "recent_documents": [
    {
      "document_id": "exemple.json",
      "timestamp": "2025-11-22T16:22:11.344724",
      "total_violations": 6,
      "false_positives": 0,
      "false_negatives": 0,
      "processing_time_ms": 1250.0,
      "ai_calls": 5,
      "cache_hits": 3,
      "fallback_count": 0,
      "violations_by_type": {
        "STRUCTURE": 3,
        "GENERAL": 2,
        "PERFORMANCE": 1
      }
    }
  ]
}
```

## Key Metrics Explained

### Accuracy Metrics

- **Precision**: TP / (TP + FP) - How many flagged violations are actually violations
- **Recall**: TP / (TP + FN) - How many actual violations are caught
- **F1 Score**: Harmonic mean of precision and recall
- **False Positive Rate**: FP / (FP + TN) - Rate of incorrect violation flags
- **False Negative Rate**: FN / (FN + TP) - Rate of missed violations

### AI Performance Metrics

- **AI Calls**: Number of actual API calls made to AI service
- **Cache Hits**: Number of responses served from cache
- **Cache Hit Rate**: Percentage of requests served from cache
- **Fallback Count**: Number of times system fell back to rule-based checking
- **Fallback Rate**: Percentage of checks that used fallback

## Integration with Check Types

The metrics system automatically tracks performance for all check types:

- `prohibited_phrases` - Investment advice detection
- `repeated_securities` - Company name repetition
- `performance_disclaimers` - Performance data validation
- `structure_validation` - Document structure checks
- `risk_profile_consistency` - Cross-slide validation
- `anglicisms_retail` - English terms in retail docs

## Resetting Metrics

To reset all metrics (useful for testing):

```python
metrics = get_compliance_metrics()
metrics.reset()
```

## Best Practices

1. **Always use --show-metrics during development** to track performance improvements
2. **Export metrics regularly** to track trends over time
3. **Monitor false positive rate** - target < 5% for production
4. **Monitor cache hit rate** - target > 50% for efficiency
5. **Track fallback rate** - high rates indicate AI service issues

## Requirements

Metrics tracking is automatically enabled when:
- `compliance_metrics.py` is available
- No additional dependencies required
- Works with or without AI/hybrid mode

## See Also

- `performance_monitor.py` - Lower-level performance tracking
- `check.py` - Main compliance checker with metrics integration
- `.kiro/specs/false-positive-elimination/design.md` - Design documentation
