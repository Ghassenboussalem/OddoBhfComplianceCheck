#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizer Integration Example

This module provides functionality for the multi-agent compliance system.
"""

"""
Example: Integrating WorkflowVisualizer with LangGraph Workflow

This example demonstrates how to integrate the WorkflowVisualizer
with the multi-agent compliance workflow for real-time visualization
and execution tracking.
"""

import logging
from datetime import datetime

# Import workflow builder
from workflow_builder import create_compliance_workflow

# Import visualizer
from monitoring.workflow_visualizer import (
    WorkflowVisualizer,
    NodeStatus,
    get_workflow_visualizer,
    initialize_workflow_visualizer
)

# Import monitoring components
from monitoring.agent_logger import get_agent_logger
from monitoring.metrics_tracker import get_metrics_tracker


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_workflow_structure():
    """
    Example 1: Visualize the workflow structure

    This generates a static visualization of the workflow graph
    showing all agents and their connections.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Visualize Workflow Structure")
    print("="*70)

    # Create workflow
    config = {
        "multi_agent": {
            "enabled": True,
            "parallel_execution": True
        }
    }

    workflow = create_compliance_workflow(
        config=config,
        enable_checkpointing=False
    )

    # Initialize visualizer
    visualizer = initialize_workflow_visualizer(
        output_dir="./monitoring/visualizations/",
        track_executions=False
    )

    # Generate and export diagram
    print("\n📊 Generating workflow diagram...")

    success = visualizer.export_diagram(
        workflow=workflow,
        output_path="./monitoring/visualizations/workflow_structure.mmd",
        format="mermaid"
    )

    if success:
        print("✓ Workflow diagram saved to: ./monitoring/visualizations/workflow_structure.mmd")
        print("\nTo convert to PNG:")
        print("  mmdc -i ./monitoring/visualizations/workflow_structure.mmd -o workflow_structure.png")

    print("="*70)


def track_workflow_execution():
    """
    Example 2: Track workflow execution in real-time

    This demonstrates how to track execution progress and generate
    visualizations with execution path highlighting.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Track Workflow Execution")
    print("="*70)

    # Initialize visualizer with execution tracking
    visualizer = initialize_workflow_visualizer(
        output_dir="./monitoring/visualizations/",
        track_executions=True
    )

    # Simulate workflow execution
    workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    document_id = "example_document_001"

    print(f"\n📝 Starting workflow execution: {workflow_id}")

    # Start tracking
    visualizer.start_execution_tracking(workflow_id, document_id)

    # Simulate node executions
    nodes = [
        ("supervisor", 0.1),
        ("preprocessor", 0.2),
        ("structure", 0.3),
        ("performance", 0.3),
        ("securities", 0.3),
        ("general", 0.3),
        ("aggregator", 0.2),
        ("context", 0.4),
        ("evidence", 0.3)
    ]

    import time
    for node_name, duration in nodes:
        started = datetime.now().isoformat()

        # Record node start
        visualizer.record_node_execution(
            workflow_id=workflow_id,
            node_name=node_name,
            status=NodeStatus.RUNNING,
            started_at=started
        )

        print(f"  ⚙️  Executing: {node_name}...")
        time.sleep(duration)

        # Record node completion
        completed = datetime.now().isoformat()
        visualizer.record_node_execution(
            workflow_id=workflow_id,
            node_name=node_name,
            status=NodeStatus.COMPLETED,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration
        )
        print(f"  ✓ Completed: {node_name} ({duration}s)")

    # Complete tracking
    visualizer.complete_execution_tracking(workflow_id, status="completed")

    # Print summary
    print("\n📊 Execution Summary:")
    visualizer.print_execution_summary(workflow_id)

    # Export execution history
    print("💾 Exporting execution history...")
    visualizer.export_execution_history(
        "./monitoring/visualizations/execution_history.json"
    )
    print("✓ Execution history saved")

    print("="*70)


def visualize_execution_with_highlighting():
    """
    Example 3: Generate visualization with execution path highlighting

    This shows how to create a diagram that highlights the actual
    execution path taken through the workflow.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Visualize Execution with Highlighting")
    print("="*70)

    # Create workflow
    config = {
        "multi_agent": {
            "enabled": True,
            "parallel_execution": True
        }
    }

    workflow = create_compliance_workflow(
        config=config,
        enable_checkpointing=False
    )

    # Initialize visualizer
    visualizer = get_workflow_visualizer()

    # Simulate execution tracking
    workflow_id = "workflow_example_003"
    document_id = "doc_003"

    visualizer.start_execution_tracking(workflow_id, document_id)

    # Simulate some node executions
    executed_nodes = [
        "supervisor",
        "preprocessor",
        "structure",
        "performance",
        "aggregator"
    ]

    for node in executed_nodes:
        visualizer.record_node_execution(
            workflow_id=workflow_id,
            node_name=node,
            status=NodeStatus.COMPLETED,
            started_at=datetime.now().isoformat(),
            completed_at=datetime.now().isoformat(),
            duration_seconds=0.1
        )

    visualizer.complete_execution_tracking(workflow_id)

    # Generate visualization with highlighting
    print("\n📊 Generating execution visualization...")

    mermaid_code = visualizer.visualize_execution(
        workflow=workflow,
        workflow_id=workflow_id,
        output_path="./monitoring/visualizations/execution_highlighted.mmd",
        format="mermaid"
    )

    print("✓ Execution visualization saved to: ./monitoring/visualizations/execution_highlighted.mmd")
    print(f"✓ Highlighted path: {' -> '.join(executed_nodes)}")

    print("="*70)


def integrate_with_monitoring():
    """
    Example 4: Full integration with monitoring system

    This demonstrates how to integrate the visualizer with the
    agent logger and metrics tracker for comprehensive monitoring.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Full Monitoring Integration")
    print("="*70)

    # Initialize all monitoring components
    visualizer = get_workflow_visualizer()
    agent_logger = get_agent_logger()
    metrics_tracker = get_metrics_tracker()

    workflow_id = "workflow_integrated_001"
    document_id = "doc_integrated_001"

    print(f"\n📝 Starting integrated monitoring for: {workflow_id}")

    # Start tracking in all systems
    visualizer.start_execution_tracking(workflow_id, document_id)
    metrics_tracker.start_workflow(workflow_id, document_id)

    # Simulate agent execution
    agent_name = "structure"
    execution_id = f"exec_{agent_name}_001"

    print(f"\n⚙️  Executing agent: {agent_name}")

    # Start agent execution tracking
    started_at = datetime.now().isoformat()
    metrics_tracker.start_agent_execution(workflow_id, agent_name, execution_id)
    visualizer.record_node_execution(
        workflow_id=workflow_id,
        node_name=agent_name,
        status=NodeStatus.RUNNING,
        started_at=started_at
    )

    # Simulate work
    import time
    time.sleep(0.2)

    # Complete agent execution
    completed_at = datetime.now().isoformat()
    duration = 0.2

    metrics_tracker.complete_agent_execution(
        execution_id=execution_id,
        status="completed",
        violations_found=2,
        api_calls=3,
        cache_hits=1
    )

    visualizer.record_node_execution(
        workflow_id=workflow_id,
        node_name=agent_name,
        status=NodeStatus.COMPLETED,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration
    )

    agent_logger.log_agent_execution(
        agent_name=agent_name,
        execution_id=execution_id,
        workflow_id=workflow_id,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration,
        status="completed",
        input_state={"document_id": document_id},
        output_state={"violations": [{"rule": "test"}]},
        violations_added=2,
        api_calls=3,
        cache_hits=1
    )

    print(f"✓ Agent execution completed: {agent_name}")

    # Complete workflow
    visualizer.complete_execution_tracking(workflow_id, status="completed")
    metrics_tracker.complete_workflow(
        workflow_id=workflow_id,
        status="completed",
        total_violations=2
    )

    # Print summaries
    print("\n📊 Monitoring Summaries:")
    print("\n1. Execution Path:")
    visualizer.print_execution_summary(workflow_id)

    print("\n2. Performance Metrics:")
    metrics_tracker.print_summary(agent_name=agent_name)

    print("\n3. Agent Logs:")
    agent_logger.print_statistics(agent_name=agent_name)

    print("="*70)


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("WORKFLOW VISUALIZER INTEGRATION EXAMPLES")
    logger.info("="*70)

    # Run examples
    try:
        # Example 1: Static workflow structure
        visualize_workflow_structure()

        # Example 2: Track execution
        track_workflow_execution()

        # Example 3: Execution with highlighting
        visualize_execution_with_highlighting()

        # Example 4: Full integration
        integrate_with_monitoring()

        logger.info("\n" + "="*70)
        logger.info("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info("\nGenerated files:")
        logger.info("  - ./monitoring/visualizations/workflow_structure.mmd")
        logger.info("  - ./monitoring/visualizations/execution_history.json")
        logger.info("  - ./monitoring/visualizations/execution_highlighted.mmd")
        logger.info("\nTo convert Mermaid diagrams to images:")
        logger.info("  npm install -g @mermaid-js/mermaid-cli")
        logger.info("  mmdc -i <input.mmd> -o <output.png>")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        logger.info(f"\n❌ Example failed: {e}")
