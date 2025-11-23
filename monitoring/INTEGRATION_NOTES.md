# PerformanceMonitor Integration with MetricsTracker

## Overview
The MetricsTracker has been integrated with the existing PerformanceMonitor system to provide unified monitoring across the multi-agent compliance checking system.

## Integration Features

### 1. PerformanceMonitor Integration
- **Automatic initialization**: MetricsTracker can automatically initialize a PerformanceMonitor instance
- **Metrics syncing**: Agent execution metrics are synced to PerformanceMonitor in real-time
- **API call tracking**: API calls and cache hits are recorded in both systems
- **Unified metrics**: Combined metrics from both systems available via `get_unified_metrics()`

### 2. Multi-Agent Specific Metrics
- **Agent execution times**: Track individual agent performance
- **Workflow metrics**: Complete workflow execution tracking
- **Success/failure rates**: Monitor agent reliability
- **Cache hit rates**: Track caching efficiency per agent
- **Violation detection**: Track violations found by each agent

### 3. Alerting Integration
- **PerformanceAlerting support**: Integrated with existing alerting system
- **Multi-agent alerts**: Custom alerts for workflow success rates, cache efficiency
- **Threshold monitoring**: Automatic alert generation when thresholds are exceeded
- **Alert dashboard**: Unified view of all active alerts

### 4. Unified Dashboard
- **Combined metrics display**: Shows both multi-agent and PerformanceMonitor metrics
- **Performance summary**: Comprehensive view of system performance
- **Alert notifications**: Real-time alert display in dashboard

## Usage

### Basic Usage
```python
from monitoring.metrics_tracker import MetricsTracker

# Initialize with PerformanceMonitor integration
tracker = MetricsTracker(
    enable_performance_monitor_integration=True
)

# Track workflow
tracker.start_workflow("workflow_id", "document_id")
tracker.start_agent_execution("workflow_id", "agent_name", "exec_id")
tracker.complete_agent_execution("exec_id", status="completed", api_calls=5, cache_hits=2)
tracker.complete_workflow("workflow_id", status="completed")

# Get unified metrics
metrics = tracker.get_unified_metrics()

# Display unified dashboard
tracker.print_unified_dashboard()

# Check alerts
alerts = tracker.check_alerts()
```

### Integration with Existing Code
The integration is backward compatible. Existing code using PerformanceMonitor continues to work without changes.

## Configuration

### Enable/Disable Integration
```python
# Enable integration (default)
tracker = MetricsTracker(enable_performance_monitor_integration=True)

# Disable integration
tracker = MetricsTracker(enable_performance_monitor_integration=False)
```

### Custom PerformanceMonitor Instance
```python
from performance_monitor import PerformanceMonitor

# Use custom instance
pm = PerformanceMonitor(cost_config={'provider': 0.001})
tracker = MetricsTracker(performance_monitor=pm)
```

## Metrics Synced to PerformanceMonitor

1. **API Calls**: Number of API calls per agent execution
2. **Cache Hits**: Number of cache hits per agent execution
3. **Execution Duration**: Agent execution time in milliseconds
4. **Success/Failure**: Execution status for error tracking
5. **Violations Found**: Number of violations detected (for accuracy tracking)

## Alert Thresholds

### Multi-Agent Specific Alerts
- **Workflow Success Rate**: Alert if < 90% (Critical if < 80%)
- **Average Workflow Duration**: Alert if > 30 seconds
- **Cache Hit Rate**: Alert if < 30%

### PerformanceMonitor Alerts
All existing PerformanceMonitor alerts continue to work:
- High latency
- High cost
- Low cache hit rate
- High error rate
- Low accuracy

## Files Modified

1. **monitoring/metrics_tracker.py**
   - Added PerformanceMonitor integration
   - Added alerting support
   - Added unified metrics methods
   - Added unified dashboard

## Testing

The integration has been implemented with the following features verified:
- ✓ MetricsTracker initialization with PerformanceMonitor
- ✓ Metrics syncing during agent execution
- ✓ Unified metrics collection
- ✓ Alert checking
- ✓ Dashboard display

## Notes

- Integration is optional and can be disabled
- Backward compatible with existing code
- Thread-safe operations
- Non-blocking metrics syncing
- Graceful degradation if PerformanceMonitor unavailable
