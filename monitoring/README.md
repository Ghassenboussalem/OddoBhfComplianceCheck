# Monitoring Dashboard

Comprehensive real-time monitoring dashboard for the multi-agent compliance system.

## Features

- **Real-time Agent Status**: Monitor health and performance of all agents
- **Performance Metrics**: Track execution times, success rates, cache hit rates
- **Workflow Execution Paths**: Visualize workflow execution and agent routing
- **Alert Management**: Automatic alerting for failures and performance issues
- **Historical Analysis**: Track trends and analyze system behavior over time

## Components

### 1. Agent Logger (`agent_logger.py`)
Structured logging for agent executions with:
- Input/output state tracking
- Performance metrics (API calls, cache hits)
- Error tracking with stack traces
- Automatic log rotation

### 2. Metrics Tracker (`metrics_tracker.py`)
Performance metrics collection:
- Agent execution statistics
- Workflow-level aggregation
- System-wide metrics
- Real-time metric updates

### 3. Workflow Visualizer (`workflow_visualizer.py`)
Workflow visualization:
- Mermaid diagram generation
- Execution path tracking
- Node status visualization
- Export to multiple formats

### 4. Monitoring Dashboard (`dashboard.py`)
Unified monitoring interface:
- Real-time status display
- Performance summaries
- Alert notifications
- Report generation

## Usage

### Basic Usage

```python
from monitoring.dashboard import get_monitoring_dashboard

# Get dashboard instance
dashboard = get_monitoring_dashboard()

# Display current status
dashboard.display_dashboard()

# Get system health
health = dashboard.get_system_health()
print(f"Overall health: {health['overall_health']}")

# Get agent status
agent_status = dashboard.get_agent_status()
for agent, status in agent_status.items():
    print(f"{agent}: {status['health']} - {status['success_rate']:.1f}% success")
```

### Interactive Dashboard

Run the interactive dashboard with auto-refresh:

```bash
python monitoring/dashboard.py --interactive
```

Or programmatically:

```python
dashboard = get_monitoring_dashboard()
dashboard.run_interactive_dashboard(refresh_interval=5)
```

### Export Reports

```python
# Export dashboard data as JSON
dashboard.export_dashboard_data("dashboard_data.json")

# Generate text report
dashboard.generate_report("monitoring_report.txt", format="text")
```

### Alert Management

```python
# Get all alerts
alerts = dashboard.get_alerts()

# Get critical alerts only
critical = dashboard.get_alerts(level=AlertLevel.CRITICAL)

# Clear alerts
dashboard.clear_alerts()
```

## Alert Thresholds

Default thresholds for automatic alerting:

- **Error Rate**: 20% (triggers ERROR alert)
- **Execution Duration**: 30 seconds (triggers WARNING alert)
- **Cache Hit Rate**: 30% (triggers WARNING alert)

Configure custom thresholds:

```python
from monitoring.dashboard import initialize_monitoring_dashboard

dashboard = initialize_monitoring_dashboard(
    alert_threshold_error_rate=0.15,  # 15%
    alert_threshold_duration=20.0,     # 20 seconds
    alert_threshold_cache_hit_rate=0.4 # 40%
)
```

## Integration with Workflow

The monitoring components integrate automatically with the multi-agent workflow:

```python
from monitoring.agent_logger import get_agent_logger
from monitoring.metrics_tracker import get_metrics_tracker
from monitoring.workflow_visualizer import get_workflow_visualizer

# In your workflow code
logger = get_agent_logger()
tracker = get_metrics_tracker()
visualizer = get_workflow_visualizer()

# Start workflow tracking
workflow_id = "workflow_001"
tracker.start_workflow(workflow_id, document_id)
visualizer.start_execution_tracking(workflow_id, document_id)

# Track agent execution
exec_id = "exec_001"
tracker.start_agent_execution(workflow_id, "structure", exec_id)

# ... agent execution ...

# Complete tracking
tracker.complete_agent_execution(exec_id, status="completed", violations_found=5)
tracker.complete_workflow(workflow_id, status="completed", total_violations=15)
```

## Dashboard Display

The dashboard shows:

```
====================================================================================================
                        MULTI-AGENT SYSTEM MONITORING DASHBOARD
====================================================================================================
Last Updated: 2025-11-23 09:12:19
====================================================================================================

+-- SYSTEM HEALTH --------------------------------------------------------------------------------+
| Overall Status: [OK] HEALTHY             Workflows: 10    Success Rate: 95.0%                   |
| Agents: 8 healthy, 1 degraded, 0 unhealthy                                                      |
| Throughput: 120.5 workflows/hour  Avg Duration: 2.3s  Cache Hit: 65.2%                         |
+--------------------------------------------------------------------------------------------------+

+-- AGENT STATUS ---------------------------------------------------------------------------------+
| Agent                Status       Executions   Success    Avg Time   Cache Hit |
+--------------------------------------------------------------------------------------------------+
| structure            [OK] healthy    150          98.0%      1.2s      70.0%   |
| performance          [OK] healthy    150          97.3%      1.5s      65.0%   |
| securities           [OK] healthy    150          99.0%      1.1s      72.0%   |
| general              [!] degraded    150          88.0%      2.1s      55.0%   |
+--------------------------------------------------------------------------------------------------+

+-- RECENT WORKFLOWS -----------------------------------------------------------------------------+
| Workflow ID               Document ID          Status       Duration   Violations|
+--------------------------------------------------------------------------------------------------+
| workflow_001              doc_001              [OK] completed  2.3s       12        |
| workflow_002              doc_002              [OK] completed  1.9s       8         |
+--------------------------------------------------------------------------------------------------+

+-- ACTIVE ALERTS --------------------------------------------------------------------------------+
| [!] [09:12:15] Low Cache Hit Rate: general    Cache hit rate is 55.0% (threshold: 60.0%)       |
+--------------------------------------------------------------------------------------------------+
```

## Files Generated

- `monitoring/logs/agent_executions.json` - Agent execution logs
- `monitoring/logs/workflow_executions.json` - Workflow execution logs
- `monitoring/metrics/metrics.json` - Performance metrics
- `monitoring/visualizations/*.mmd` - Workflow diagrams

## Demo Mode

Run the demo to see the dashboard in action:

```bash
python monitoring/dashboard.py
```

This will:
1. Initialize all monitoring components
2. Simulate workflow executions
3. Display the dashboard
4. Export sample reports

## Requirements

- Python 3.8+
- No external dependencies (uses only standard library)
- Optional: Mermaid CLI for diagram rendering

## Notes

- All monitoring operations are thread-safe
- Automatic log rotation prevents disk space issues
- Metrics are persisted to disk for historical analysis
- Dashboard can run in background with auto-refresh
