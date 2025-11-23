#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow Visualizer

This module provides functionality for the multi-agent compliance system.
"""

"""
Workflow Visualizer - LangGraph Workflow Visualization

This module provides comprehensive visualization capabilities for the
LangGraph-based multi-agent compliance workflow. It generates visual
representations of the workflow graph, highlights execution paths,
and tracks real-time execution progress.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    """Status of a node in the workflow"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeExecution:
    """Execution information for a single node"""
    node_name: str
    status: NodeStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ExecutionPath:
    """Represents an execution path through the workflow"""
    workflow_id: str
    document_id: str
    started_at: str
    completed_at: Optional[str] = None
    nodes_executed: List[NodeExecution] = None
    current_node: Optional[str] = None
    status: str = "running"

    def __post_init__(self):
        if self.nodes_executed is None:
            self.nodes_executed = []

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['nodes_executed'] = [n.to_dict() for n in self.nodes_executed]
        return data


class WorkflowVisualizer:
    """
    Visualizer for LangGraph workflows

    Features:
    - Generate Mermaid diagrams of workflow structure
    - Highlight execution paths
    - Track real-time execution progress
    - Export visualizations (PNG, SVG, Mermaid)
    - Execution history tracking
    """

    def __init__(self,
                 output_dir: str = "./monitoring/visualizations/",
                 track_executions: bool = True):
        """
        Initialize workflow visualizer

        Args:
            output_dir: Directory for visualization outputs
            track_executions: Whether to track execution paths
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.track_executions = track_executions
        self.execution_paths: Dict[str, ExecutionPath] = {}

        logger.info(f"WorkflowVisualizer initialized: {output_dir}")


    def generate_mermaid_diagram(self,
                                 workflow,
                                 highlight_path: Optional[List[str]] = None,
                                 node_statuses: Optional[Dict[str, NodeStatus]] = None) -> str:
        """
        Generate Mermaid diagram for workflow

        Args:
            workflow: Compiled LangGraph workflow
            highlight_path: Optional list of node names to highlight
            node_statuses: Optional dict of node names to their current status

        Returns:
            Mermaid diagram code as string
        """
        try:
            # Get graph structure from workflow
            graph = workflow.get_graph()

            # Try to use LangGraph's built-in Mermaid generation
            if hasattr(graph, 'draw_mermaid'):
                base_diagram = graph.draw_mermaid()
                logger.info("Generated base Mermaid diagram using LangGraph")
            else:
                # Fallback: Generate manually
                base_diagram = self._generate_mermaid_manually(graph)
                logger.info("Generated Mermaid diagram manually")

            # Enhance diagram with highlighting and status
            if highlight_path or node_statuses:
                enhanced_diagram = self._enhance_mermaid_diagram(
                    base_diagram,
                    highlight_path,
                    node_statuses
                )
                return enhanced_diagram

            return base_diagram

        except Exception as e:
            logger.error(f"Failed to generate Mermaid diagram: {e}")
            return self._generate_fallback_diagram()


    def _generate_mermaid_manually(self, graph) -> str:
        """
        Manually generate Mermaid diagram from graph structure

        Args:
            graph: Graph object from LangGraph

        Returns:
            Mermaid diagram code
        """
        mermaid = ["graph TD"]

        # Extract nodes
        nodes = []
        if hasattr(graph, 'nodes'):
            if isinstance(graph.nodes, dict):
                nodes = list(graph.nodes.keys())
            else:
                nodes = list(graph.nodes)

        # Add nodes with styling
        for node in nodes:
            if node == "__start__" or node == "__end__":
                continue

            # Style based on node type
            if "supervisor" in node.lower():
                mermaid.append(f'    {node}["{node}"]:::supervisor')
            elif "preprocessor" in node.lower():
                mermaid.append(f'    {node}["{node}"]:::preprocessor')
            elif "aggregator" in node.lower():
                mermaid.append(f'    {node}["{node}"]:::aggregator')
            elif "context" in node.lower() or "evidence" in node.lower():
                mermaid.append(f'    {node}["{node}"]:::analysis')
            elif "reviewer" in node.lower():
                mermaid.append(f'    {node}["{node}"]:::review')
            else:
                mermaid.append(f'    {node}["{node}"]:::agent')

        # Extract edges
        edges = []
        if hasattr(graph, 'edges'):
            if isinstance(graph.edges, list):
                edges = graph.edges
            elif isinstance(graph.edges, dict):
                edges = [(k, v) for k, vals in graph.edges.items() for v in (vals if isinstance(vals, list) else [vals])]

        # Add edges
        for edge in edges:
            if isinstance(edge, tuple) and len(edge) >= 2:
                source, target = edge[0], edge[1]
                if source != "__start__" and target != "__end__":
                    mermaid.append(f'    {source} --> {target}')

        # Add styling
        mermaid.extend([
            "",
            "    classDef supervisor fill:#e1f5ff,stroke:#01579b,stroke-width:3px",
            "    classDef preprocessor fill:#f3e5f5,stroke:#4a148c,stroke-width:2px",
            "    classDef agent fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px",
            "    classDef aggregator fill:#fff3e0,stroke:#e65100,stroke-width:2px",
            "    classDef analysis fill:#fce4ec,stroke:#880e4f,stroke-width:2px",
            "    classDef review fill:#ffebee,stroke:#b71c1c,stroke-width:2px"
        ])

        return "\n".join(mermaid)


    def _enhance_mermaid_diagram(self,
                                base_diagram: str,
                                highlight_path: Optional[List[str]] = None,
                                node_statuses: Optional[Dict[str, NodeStatus]] = None) -> str:
        """
        Enhance Mermaid diagram with highlighting and status indicators

        Args:
            base_diagram: Base Mermaid diagram code
            highlight_path: Nodes to highlight
            node_statuses: Node status information

        Returns:
            Enhanced Mermaid diagram code
        """
        lines = base_diagram.split('\n')
        enhanced = []

        # Add base diagram
        enhanced.extend(lines)

        # Add status-based styling
        if node_statuses:
            enhanced.append("")
            enhanced.append("    %% Status-based styling")

            for node, status in node_statuses.items():
                if status == NodeStatus.RUNNING:
                    enhanced.append(f'    style {node} fill:#fff59d,stroke:#f57f17,stroke-width:4px')
                elif status == NodeStatus.COMPLETED:
                    enhanced.append(f'    style {node} fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px')
                elif status == NodeStatus.FAILED:
                    enhanced.append(f'    style {node} fill:#ffcdd2,stroke:#c62828,stroke-width:3px')
                elif status == NodeStatus.SKIPPED:
                    enhanced.append(f'    style {node} fill:#e0e0e0,stroke:#616161,stroke-width:2px')

        # Add path highlighting
        if highlight_path:
            enhanced.append("")
            enhanced.append("    %% Execution path highlighting")

            for i, node in enumerate(highlight_path):
                # Highlight with gradient based on execution order
                opacity = 0.3 + (0.7 * (i + 1) / len(highlight_path))
                enhanced.append(f'    style {node} stroke-width:5px')

        return "\n".join(enhanced)


    def _generate_fallback_diagram(self) -> str:
        """
        Generate a simple fallback diagram when graph extraction fails

        Returns:
            Basic Mermaid diagram
        """
        return """graph TD
    supervisor["Supervisor"]
    preprocessor["Preprocessor"]
    structure["Structure"]
    performance["Performance"]
    securities["Securities"]
    general["General"]
    aggregator["Aggregator"]
    context["Context"]
    evidence["Evidence"]
    reviewer["Reviewer"]

    supervisor --> preprocessor
    preprocessor --> structure
    preprocessor --> performance
    preprocessor --> securities
    preprocessor --> general
    structure --> aggregator
    performance --> aggregator
    securities --> aggregator
    general --> aggregator
    aggregator --> context
    context --> evidence
    evidence --> reviewer

    classDef supervisor fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    classDef agent fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef aggregator fill:#fff3e0,stroke:#e65100,stroke-width:2px

    class supervisor supervisor
    class structure,performance,securities,general agent
    class aggregator aggregator
"""


    def export_diagram(self,
                      workflow,
                      output_path: str,
                      format: str = "mermaid",
                      highlight_path: Optional[List[str]] = None,
                      node_statuses: Optional[Dict[str, NodeStatus]] = None) -> bool:
        """
        Export workflow diagram to file

        Args:
            workflow: Compiled LangGraph workflow
            output_path: Output file path
            format: Export format (mermaid, png, svg)
            highlight_path: Optional nodes to highlight
            node_statuses: Optional node status information

        Returns:
            True if export successful, False otherwise
        """
        try:
            # Generate Mermaid diagram
            mermaid_code = self.generate_mermaid_diagram(
                workflow,
                highlight_path=highlight_path,
                node_statuses=node_statuses
            )

            if format == "mermaid":
                # Save Mermaid code directly
                output_file = Path(output_path)
                if not output_file.suffix:
                    output_file = output_file.with_suffix('.mmd')

                with open(output_file, 'w') as f:
                    f.write(mermaid_code)

                logger.info(f"Exported Mermaid diagram to {output_file}")
                return True

            elif format in ["png", "svg"]:
                # Save Mermaid code first
                mermaid_file = Path(output_path).with_suffix('.mmd')
                with open(mermaid_file, 'w') as f:
                    f.write(mermaid_code)

                logger.info(f"Saved Mermaid code to {mermaid_file}")
                logger.info(f"Note: To convert to {format.upper()}, use Mermaid CLI:")
                logger.info(f"  mmdc -i {mermaid_file} -o {output_path}")

                return True

            else:
                logger.error(f"Unsupported format: {format}")
                return False

        except Exception as e:
            logger.error(f"Failed to export diagram: {e}")
            return False


    def start_execution_tracking(self,
                                workflow_id: str,
                                document_id: str) -> None:
        """
        Start tracking a workflow execution

        Args:
            workflow_id: Unique workflow identifier
            document_id: Document being processed
        """
        if not self.track_executions:
            return

        execution_path = ExecutionPath(
            workflow_id=workflow_id,
            document_id=document_id,
            started_at=datetime.now().isoformat(),
            status="running"
        )

        self.execution_paths[workflow_id] = execution_path
        logger.debug(f"Started tracking execution: {workflow_id}")

    def record_node_execution(self,
                             workflow_id: str,
                             node_name: str,
                             status: NodeStatus,
                             started_at: Optional[str] = None,
                             completed_at: Optional[str] = None,
                             duration_seconds: Optional[float] = None,
                             error: Optional[str] = None) -> None:
        """
        Record execution of a node

        Args:
            workflow_id: Workflow identifier
            node_name: Name of the node
            status: Node execution status
            started_at: Start timestamp
            completed_at: Completion timestamp
            duration_seconds: Execution duration
            error: Error message if failed
        """
        if not self.track_executions or workflow_id not in self.execution_paths:
            return

        execution_path = self.execution_paths[workflow_id]

        # Update or add node execution
        node_exec = NodeExecution(
            node_name=node_name,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            error=error
        )

        # Check if node already exists in path
        existing_idx = None
        for i, node in enumerate(execution_path.nodes_executed):
            if node.node_name == node_name:
                existing_idx = i
                break

        if existing_idx is not None:
            execution_path.nodes_executed[existing_idx] = node_exec
        else:
            execution_path.nodes_executed.append(node_exec)

        # Update current node if running
        if status == NodeStatus.RUNNING:
            execution_path.current_node = node_name

        logger.debug(f"Recorded node execution: {node_name} ({status})")


    def complete_execution_tracking(self,
                                   workflow_id: str,
                                   status: str = "completed") -> None:
        """
        Complete tracking of a workflow execution

        Args:
            workflow_id: Workflow identifier
            status: Final status (completed, failed, interrupted)
        """
        if not self.track_executions or workflow_id not in self.execution_paths:
            return

        execution_path = self.execution_paths[workflow_id]
        execution_path.completed_at = datetime.now().isoformat()
        execution_path.status = status
        execution_path.current_node = None

        logger.info(f"Completed execution tracking: {workflow_id} ({status})")

    def get_execution_path(self, workflow_id: str) -> Optional[ExecutionPath]:
        """
        Get execution path for a workflow

        Args:
            workflow_id: Workflow identifier

        Returns:
            ExecutionPath object or None
        """
        return self.execution_paths.get(workflow_id)

    def get_current_node_statuses(self, workflow_id: str) -> Dict[str, NodeStatus]:
        """
        Get current status of all nodes in a workflow execution

        Args:
            workflow_id: Workflow identifier

        Returns:
            Dictionary mapping node names to their current status
        """
        if workflow_id not in self.execution_paths:
            return {}

        execution_path = self.execution_paths[workflow_id]
        statuses = {}

        for node_exec in execution_path.nodes_executed:
            statuses[node_exec.node_name] = node_exec.status

        return statuses

    def get_execution_path_nodes(self, workflow_id: str) -> List[str]:
        """
        Get list of nodes executed in order

        Args:
            workflow_id: Workflow identifier

        Returns:
            List of node names in execution order
        """
        if workflow_id not in self.execution_paths:
            return []

        execution_path = self.execution_paths[workflow_id]
        return [node.node_name for node in execution_path.nodes_executed]


    def visualize_execution(self,
                          workflow,
                          workflow_id: str,
                          output_path: Optional[str] = None,
                          format: str = "mermaid") -> Optional[str]:
        """
        Generate visualization of a specific workflow execution

        Args:
            workflow: Compiled LangGraph workflow
            workflow_id: Workflow identifier
            output_path: Optional output file path
            format: Export format (mermaid, png, svg)

        Returns:
            Mermaid code if output_path is None, otherwise None
        """
        if workflow_id not in self.execution_paths:
            logger.warning(f"No execution path found for workflow: {workflow_id}")
            return None

        # Get execution information
        highlight_path = self.get_execution_path_nodes(workflow_id)
        node_statuses = self.get_current_node_statuses(workflow_id)

        # Generate diagram
        mermaid_code = self.generate_mermaid_diagram(
            workflow,
            highlight_path=highlight_path,
            node_statuses=node_statuses
        )

        # Export if path provided
        if output_path:
            self.export_diagram(
                workflow,
                output_path,
                format=format,
                highlight_path=highlight_path,
                node_statuses=node_statuses
            )
            return None

        return mermaid_code

    def generate_execution_summary(self, workflow_id: str) -> Dict[str, Any]:
        """
        Generate summary of workflow execution

        Args:
            workflow_id: Workflow identifier

        Returns:
            Dictionary with execution summary
        """
        if workflow_id not in self.execution_paths:
            return {}

        execution_path = self.execution_paths[workflow_id]

        # Calculate statistics
        total_nodes = len(execution_path.nodes_executed)
        completed_nodes = sum(1 for n in execution_path.nodes_executed if n.status == NodeStatus.COMPLETED)
        failed_nodes = sum(1 for n in execution_path.nodes_executed if n.status == NodeStatus.FAILED)

        total_duration = sum(
            n.duration_seconds for n in execution_path.nodes_executed
            if n.duration_seconds is not None
        )

        return {
            "workflow_id": workflow_id,
            "document_id": execution_path.document_id,
            "status": execution_path.status,
            "started_at": execution_path.started_at,
            "completed_at": execution_path.completed_at,
            "current_node": execution_path.current_node,
            "total_nodes": total_nodes,
            "completed_nodes": completed_nodes,
            "failed_nodes": failed_nodes,
            "total_duration_seconds": total_duration,
            "execution_path": [n.node_name for n in execution_path.nodes_executed]
        }


    def export_execution_history(self, output_path: str) -> None:
        """
        Export execution history to JSON file

        Args:
            output_path: Output file path
        """
        history = {
            "exported_at": datetime.now().isoformat(),
            "total_executions": len(self.execution_paths),
            "executions": {
                workflow_id: path.to_dict()
                for workflow_id, path in self.execution_paths.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"Exported execution history to {output_path}")

    def clear_execution_history(self) -> None:
        """Clear all execution history"""
        self.execution_paths.clear()
        logger.info("Cleared execution history")

    def print_execution_summary(self, workflow_id: str) -> None:
        """
        Print execution summary to console

        Args:
            workflow_id: Workflow identifier
        """
        summary = self.generate_execution_summary(workflow_id)

        if not summary:
            print(f"\nNo execution found for workflow: {workflow_id}")
            return

        print("\n" + "="*70)
        print("WORKFLOW EXECUTION SUMMARY")
        print("="*70)
        print(f"\nWorkflow ID:      {summary['workflow_id']}")
        print(f"Document ID:      {summary['document_id']}")
        print(f"Status:           {summary['status']}")
        print(f"Started:          {summary['started_at']}")
        print(f"Completed:        {summary['completed_at'] or 'In Progress'}")
        print(f"Current Node:     {summary['current_node'] or 'N/A'}")
        print(f"\nProgress:")
        print(f"  Total Nodes:    {summary['total_nodes']}")
        print(f"  Completed:      {summary['completed_nodes']}")
        print(f"  Failed:         {summary['failed_nodes']}")
        print(f"  Duration:       {summary['total_duration_seconds']:.2f}s")
        print(f"\nExecution Path:")
        for i, node in enumerate(summary['execution_path'], 1):
            print(f"  {i}. {node}")
        print("="*70 + "\n")


# Global visualizer instance
_global_visualizer: Optional[WorkflowVisualizer] = None


def get_workflow_visualizer() -> WorkflowVisualizer:
    """
    Get global workflow visualizer instance (singleton)

    Returns:
        WorkflowVisualizer instance
    """
    global _global_visualizer
    if _global_visualizer is None:
        _global_visualizer = WorkflowVisualizer()
    return _global_visualizer


def initialize_workflow_visualizer(output_dir: str = "./monitoring/visualizations/",
                                  track_executions: bool = True) -> WorkflowVisualizer:
    """
    Initialize global workflow visualizer with custom settings

    Args:
        output_dir: Directory for visualization outputs
        track_executions: Whether to track execution paths

    Returns:
        WorkflowVisualizer instance
    """
    global _global_visualizer
    _global_visualizer = WorkflowVisualizer(
        output_dir=output_dir,
        track_executions=track_executions
    )
    return _global_visualizer


# Export all public symbols
__all__ = [
    "WorkflowVisualizer",
    "NodeStatus",
    "NodeExecution",
    "ExecutionPath",
    "get_workflow_visualizer",
    "initialize_workflow_visualizer"
]



if __name__ == "__main__":
    # Example usage and testing
    logger.info("="*70)
    logger.info("WorkflowVisualizer - LangGraph Workflow Visualization")
    logger.info("="*70)

    # Initialize visualizer
    visualizer = WorkflowVisualizer(
        output_dir="./test_visualizations/",
        track_executions=True
    )

    logger.info(f"\n✓ WorkflowVisualizer initialized")
    logger.info(f"  Output directory: ./test_visualizations/")
    logger.info(f"  Execution tracking: enabled")

    # Simulate workflow execution tracking
    logger.info("\n📊 Simulating workflow execution tracking...")

    workflow_id = "workflow_test_001"
    document_id = "doc_001"

    visualizer.start_execution_tracking(workflow_id, document_id)
    logger.info(f"  ✓ Started tracking: {workflow_id}")

    # Simulate node executions
    nodes = [
        "supervisor",
        "preprocessor",
        "structure",
        "performance",
        "securities",
        "general",
        "aggregator",
        "context",
        "evidence"
    ]

    import time
    for node in nodes:
        started = datetime.now().isoformat()
        time.sleep(0.05)  # Simulate work
        completed = datetime.now().isoformat()

        visualizer.record_node_execution(
            workflow_id=workflow_id,
            node_name=node,
            status=NodeStatus.COMPLETED,
            started_at=started,
            completed_at=completed,
            duration_seconds=0.05
        )

    visualizer.complete_execution_tracking(workflow_id, status="completed")
    logger.info(f"  ✓ Tracked {len(nodes)} node executions")

    # Print execution summary
    visualizer.print_execution_summary(workflow_id)

    # Generate fallback diagram (since we don't have actual workflow)
    logger.info("📈 Generating workflow diagram...")
    mermaid_code = visualizer._generate_fallback_diagram()

    # Save diagram
    output_file = "./test_visualizations/workflow_diagram.mmd"
    with open(output_file, 'w') as f:
        f.write(mermaid_code)
    logger.info(f"  ✓ Saved diagram to {output_file}")

    # Export execution history
    logger.info("\n💾 Exporting execution history...")
    visualizer.export_execution_history("./test_visualizations/execution_history.json")
    logger.info(f"  ✓ Exported execution history")

    logger.info("\n" + "="*70)
    logger.info("✓ WorkflowVisualizer test complete")
    logger.info("="*70)
    logger.info("\nTo convert Mermaid diagrams to images, use Mermaid CLI:")
    logger.info("  npm install -g @mermaid-js/mermaid-cli")
    logger.info("  mmdc -i workflow_diagram.mmd -o workflow_diagram.png")
    logger.info("="*70)
