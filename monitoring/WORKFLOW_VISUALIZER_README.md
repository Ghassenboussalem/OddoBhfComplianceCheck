# Workflow Visualizer

The Workflow Visualizer provides comprehensive visualization capabilities for the LangGraph-based multi-agent compliance workflow. It generates visual representations of the workflow graph, highlights execution paths, and tracks real-time execution progress.

## Features

- **Workflow Graph Visualization**: Generate Mermaid diagrams of the workflow structure
- **Execution Path Highlighting**: Highlight the actual path taken through the workflow
- **Real-time Execution Tracking**: Track node execution status in real-time
- **Multiple Export Formats**: Export to Mermaid (.mmd), PNG, SVG
- **Execution History**: Track and export execution history
- **Status Indicators**: Visual indicators for node status (running, completed, failed, skipped)
- **Integration Ready**: Seamlessly integrates with AgentLogger and MetricsTracker

## Installation

The visualizer is included in the monitoring package. No additional installation required.

For converting Mermaid diagrams to images (PNG, SVG), install Mermaid CLI:

```bash
npm install -g @mermaid-js/mermaid-cli
```

## Quick Start

### Basic Usage

```python
from workflow_builder import create_compliance_workflow
from monitoring.workflow_visualizer import WorkflowVisualizer

# Create workflow
workflow = create_compliance_workflow()

# Initialize visualizer
visualizer = WorkflowVisualizer(
    output_dir="./visualizations/",
    track_executions=True
)

# Generate workflow diagram
visualizer.export_diagram(
    workflow=workflow,
    output_path="./visualizations/workflow.mmd",
    format="mermaid"
)
```

### Track Workflow Execution

```python
from monitoring.workflow_visualizer import NodeStatus

# Start tracking
workflow_id = "workflow_001"
document_id = "doc_001"

visualizer.start_execution_tracking(workflow_id, document_id)

# Record node execution
visualizer.record_node_execution(
    workflow_id=workflow_id,
    node_name="structure",
    status=NodeStatus.COMPLETED,
    started_at="2024-01-01T10:00:00",
    completed_at="2024-01-01T10:00:05",
    duration_seconds=5.0
)

# Complete tracking
visualizer.complete_execution_tracking(workflow_id, status="completed")

# Print summary
visualizer.print_execution_summary(workflow_id)
```

### Visualize Execution with Highlighting

```python
# Generate visualization with execution path highlighted
mermaid_code = visualizer.visualize_execution(
    workflow=workflow,
    workflow_id=workflow_id,
    output_path="./visualizations/execution.mmd",
    format="mermaid"
)
```

## API Reference

### WorkflowVisualizer

Main class for workflow visualization.

#### Constructor

```python
WorkflowVisualizer(
    output_dir: str = "./monitoring/visualizations/",
    track_executions: bool = True
)
```

**Parameters:**
- `output_dir`: Directory for visualization outputs
- `track_executions`: Whether to track execution paths

#### Methods

##### generate_mermaid_diagram()

Generate Mermaid diagram for workflow.

```python
generate_mermaid_diagram(
    workflow,
    highlight_path: Optional[List[str]] = None,
    node_statuses: Optional[Dict[str, NodeStatus]] = None
) -> str
```

**Parameters:**
- `workflow`: Compiled LangGraph workflow
- `highlight_path`: Optional list of node names to highlight
- `node_statuses`: Optional dict of node names to their current status

**Returns:** Mermaid diagram code as string

##### export_diagram()

Export workflow diagram to file.

```python
export_diagram(
    workflow,
    output_path: str,
    format: str = "mermaid",
    highlight_path: Optional[List[str]] = None,
    node_statuses: Optional[Dict[str, NodeStatus]] = None
) -> bool
```

**Parameters:**
- `workflow`: Compiled LangGraph workflow
- `output_path`: Output file path
- `format`: Export format (mermaid, png, svg)
- `highlight_path`: Optional nodes to highlight
- `node_statuses`: Optional node status information

**Returns:** True if export successful, False otherwise

##### start_execution_tracking()

Start tracking a workflow execution.

```python
start_execution_tracking(
    workflow_id: str,
    document_id: str
) -> None
```

##### record_node_execution()

Record execution of a node.

```python
record_node_execution(
    workflow_id: str,
    node_name: str,
    status: NodeStatus,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    error: Optional[str] = None
) -> None
```

##### complete_execution_tracking()

Complete tracking of a workflow execution.

```python
complete_execution_tracking(
    workflow_id: str,
    status: str = "completed"
) -> None
```

##### visualize_execution()

Generate visualization of a specific workflow execution.

```python
visualize_execution(
    workflow,
    workflow_id: str,
    output_path: Optional[str] = None,
    format: str = "mermaid"
) -> Optional[str]
```

##### generate_execution_summary()

Generate summary of workflow execution.

```python
generate_execution_summary(
    workflow_id: str
) -> Dict[str, Any]
```

##### export_execution_history()

Export execution history to JSON file.

```python
export_execution_history(
    output_path: str
) -> None
```

##### print_execution_summary()

Print execution summary to console.

```python
print_execution_summary(
    workflow_id: str
) -> None
```

### NodeStatus Enum

Represents the status of a node in the workflow.

- `PENDING`: Node is pending execution
- `RUNNING`: Node is currently executing
- `COMPLETED`: Node completed successfully
- `FAILED`: Node execution failed
- `SKIPPED`: Node was skipped

## Integration with Monitoring System

The visualizer integrates seamlessly with other monitoring components:

```python
from monitoring.workflow_visualizer import get_workflow_visualizer
from monitoring.agent_logger import get_agent_logger
from monitoring.metrics_tracker import get_metrics_tracker

# Get singleton instances
visualizer = get_workflow_visualizer()
logger = get_agent_logger()
tracker = get_metrics_tracker()

# Track execution across all systems
workflow_id = "workflow_001"
document_id = "doc_001"

# Start tracking
visualizer.start_execution_tracking(workflow_id, document_id)
tracker.start_workflow(workflow_id, document_id)

# ... execute workflow ...

# Complete tracking
visualizer.complete_execution_tracking(workflow_id)
tracker.complete_workflow(workflow_id)

# Generate reports
visualizer.print_execution_summary(workflow_id)
tracker.print_summary()
logger.print_statistics()
```

## Converting Mermaid to Images

After generating Mermaid diagrams (.mmd files), convert them to images:

### PNG

```bash
mmdc -i workflow.mmd -o workflow.png
```

### SVG

```bash
mmdc -i workflow.mmd -o workflow.svg
```

### PDF

```bash
mmdc -i workflow.mmd -o workflow.pdf
```

### With Custom Theme

```bash
mmdc -i workflow.mmd -o workflow.png -t dark
```

## Examples

See `monitoring/visualizer_integration_example.py` for comprehensive examples:

1. **Visualize Workflow Structure**: Generate static workflow diagram
2. **Track Workflow Execution**: Real-time execution tracking
3. **Execution with Highlighting**: Highlight execution paths
4. **Full Monitoring Integration**: Integrate with all monitoring components

Run examples:

```bash
python monitoring/visualizer_integration_example.py
```

## Output Files

The visualizer generates the following files:

- `*.mmd`: Mermaid diagram source files
- `execution_history.json`: JSON file with execution history
- `*.png`, `*.svg`, `*.pdf`: Image files (if converted with Mermaid CLI)

## Workflow Diagram Structure

The generated diagrams use color coding:

- **Blue**: Supervisor agent (orchestration)
- **Purple**: Preprocessor agent (data preparation)
- **Green**: Core compliance agents (structure, performance, securities, general)
- **Orange**: Aggregator agent (result consolidation)
- **Pink**: Analysis agents (context, evidence)
- **Red**: Review agent (HITL integration)

### Status Indicators

When tracking execution, nodes are styled based on status:

- **Yellow with thick border**: Currently running
- **Green**: Completed successfully
- **Red**: Failed
- **Gray**: Skipped

## Best Practices

1. **Enable Execution Tracking**: Set `track_executions=True` for real-time monitoring
2. **Export Regularly**: Export execution history periodically for analysis
3. **Use Highlighting**: Highlight execution paths to understand workflow behavior
4. **Integrate with Monitoring**: Use with AgentLogger and MetricsTracker for comprehensive monitoring
5. **Convert to Images**: Convert Mermaid diagrams to images for documentation and presentations

## Troubleshooting

### Mermaid CLI Not Found

If you get "mmdc: command not found":

```bash
npm install -g @mermaid-js/mermaid-cli
```

### Graph Extraction Fails

If the visualizer can't extract the graph structure, it will use a fallback diagram. This is normal for some LangGraph versions.

### Large Execution History

If execution history grows too large, clear it periodically:

```python
visualizer.clear_execution_history()
```

## Requirements

- Python 3.8+
- LangGraph
- Node.js and npm (for Mermaid CLI, optional)

## License

Part of the Multi-Agent Compliance Checker system.
