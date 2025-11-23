#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph Workflow Builder for Multi-Agent Compliance System

This module provides the infrastructure for building and managing the
LangGraph-based multi-agent compliance checking workflow.

Key Features:
- StateGraph initialization with ComplianceState
- SqliteSaver for state persistence and checkpointing
- Workflow visualization utilities
- Agent node management
- Conditional routing logic
"""

import logging
import os
from typing import Optional, Dict, Any, List, Callable, Literal
from pathlib import Path

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Type hint for compiled graph (use Any if CompiledGraph not available)
try:
    from langgraph.graph.graph import CompiledGraph
except ImportError:
    from typing import Any as CompiledGraph

# Import state models
from data_models_multiagent import (
    ComplianceState,
    WorkflowStatus,
    initialize_compliance_state,
    validate_compliance_state,
    get_state_summary
)

# Import base agent framework
from agents.base_agent import (
    BaseAgent,
    AgentRegistry,
    AgentConfigManager
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkflowBuilder:
    """
    Builder class for constructing LangGraph compliance workflows
    
    Provides a fluent interface for:
    - Adding agent nodes
    - Defining edges and conditional routing
    - Setting up checkpointing
    - Configuring workflow behavior
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize workflow builder
        
        Args:
            config: Configuration dictionary for workflow and agents
        """
        self.config = config or {}
        self.graph = StateGraph(ComplianceState)
        self.agents: Dict[str, BaseAgent] = {}
        self.entry_point: Optional[str] = None
        self.checkpointer: Optional[SqliteSaver] = None
        self.state_manager = None
        self.checkpoint_manager = None
        
        # Agent configuration manager
        self.agent_config_manager = AgentConfigManager(self.config)
        
        logger.info("WorkflowBuilder initialized")
    
    def add_agent_node(
        self,
        name: str,
        agent: BaseAgent,
        set_as_entry: bool = False
    ) -> 'WorkflowBuilder':
        """
        Add an agent as a node in the workflow
        
        Args:
            name: Unique name for the node
            agent: Agent instance to execute at this node
            set_as_entry: If True, set this as the workflow entry point
            
        Returns:
            Self for method chaining
        """
        self.agents[name] = agent
        self.graph.add_node(name, agent)
        
        if set_as_entry:
            self.entry_point = name
        
        logger.info(f"Added agent node: {name} (entry={set_as_entry})")
        return self
    
    def add_edge(self, source: str, target: str) -> 'WorkflowBuilder':
        """
        Add a direct edge between two nodes
        
        Args:
            source: Source node name
            target: Target node name
            
        Returns:
            Self for method chaining
        """
        self.graph.add_edge(source, target)
        logger.debug(f"Added edge: {source} -> {target}")
        return self
    
    def add_conditional_edge(
        self,
        source: str,
        condition: Callable[[ComplianceState], str],
        path_map: Dict[str, str]
    ) -> 'WorkflowBuilder':
        """
        Add a conditional edge with routing logic
        
        Args:
            source: Source node name
            condition: Function that takes state and returns path key
            path_map: Mapping of condition results to target nodes
            
        Returns:
            Self for method chaining
        """
        self.graph.add_conditional_edges(source, condition, path_map)
        logger.debug(f"Added conditional edge from {source} with {len(path_map)} paths")
        return self
    
    def set_entry_point(self, node_name: str) -> 'WorkflowBuilder':
        """
        Set the workflow entry point
        
        Args:
            node_name: Name of the entry node
            
        Returns:
            Self for method chaining
        """
        self.entry_point = node_name
        logger.info(f"Set entry point: {node_name}")
        return self
    
    def setup_checkpointing(
        self,
        db_path: Optional[str] = None,
        checkpoint_interval: int = 1,
        enable_state_manager: bool = True,
        max_history_size: int = 100
    ) -> 'WorkflowBuilder':
        """
        Set up state persistence with SqliteSaver and StateManager
        
        Args:
            db_path: Path to SQLite database file (None for in-memory)
            checkpoint_interval: Number of steps between checkpoints
            enable_state_manager: Whether to enable StateManager for history tracking
            max_history_size: Maximum number of historical states to keep
            
        Returns:
            Self for method chaining
        """
        import sqlite3
        
        if db_path:
            # Ensure directory exists
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            
            logger.info(f"Setting up checkpointing with database: {db_path}")
            # Create connection and saver
            conn = sqlite3.connect(db_path, check_same_thread=False)
        else:
            logger.info("Setting up checkpointing with in-memory database")
            # Create in-memory connection
            conn = sqlite3.connect(":memory:", check_same_thread=False)
        
        # Create SqliteSaver with the connection
        self.checkpointer = SqliteSaver(conn)
        
        # Set up StateManager for additional state management
        if enable_state_manager:
            from state_manager import StateManager, CheckpointManager
            
            checkpoint_dir = os.path.dirname(db_path) if db_path else "checkpoints"
            self.state_manager = StateManager(
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                max_history_size=max_history_size
            )
            
            self.checkpoint_manager = CheckpointManager(
                state_manager=self.state_manager,
                auto_checkpoint=True
            )
            
            logger.info(f"StateManager enabled: dir={checkpoint_dir}, interval={checkpoint_interval}")
        else:
            self.state_manager = None
            self.checkpoint_manager = None
        
        logger.info(f"Checkpointing configured (interval={checkpoint_interval})")
        return self
    
    def compile(self) -> CompiledGraph:
        """
        Compile the workflow graph
        
        Returns:
            Compiled LangGraph workflow ready for execution
        """
        if not self.entry_point:
            raise ValueError("Entry point must be set before compiling")
        
        # Set the entry point in the graph before compiling
        self.graph.set_entry_point(self.entry_point)
        
        # Compile with or without checkpointing
        if self.checkpointer:
            compiled = self.graph.compile(checkpointer=self.checkpointer)
            logger.info("Workflow compiled with checkpointing enabled")
        else:
            compiled = self.graph.compile()
            logger.info("Workflow compiled without checkpointing")
        
        return compiled
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """
        Get an agent by name
        
        Args:
            name: Agent name
            
        Returns:
            Agent instance or None
        """
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """
        Get list of all agent names in the workflow
        
        Returns:
            List of agent names
        """
        return list(self.agents.keys())


def create_compliance_workflow(
    config: Optional[Dict[str, Any]] = None,
    enable_checkpointing: bool = True,
    checkpoint_db_path: Optional[str] = None
) -> CompiledGraph:
    """
    Create the complete multi-agent compliance checking workflow
    
    This is the main entry point for creating a LangGraph workflow.
    It sets up the state graph, adds agent nodes (Supervisor and Preprocessor),
    and configures checkpointing.
    
    Args:
        config: Configuration dictionary for workflow and agents
        enable_checkpointing: Whether to enable state persistence
        checkpoint_db_path: Path to checkpoint database (None for in-memory)
        
    Returns:
        Compiled LangGraph workflow ready for execution
        
    Example:
        >>> config = {"agents": {"supervisor": {"enabled": True}}}
        >>> workflow = create_compliance_workflow(config)
        >>> result = workflow.invoke({"document": doc, "document_id": "123"})
    """
    logger.info("="*70)
    logger.info("CREATING MULTI-AGENT COMPLIANCE WORKFLOW")
    logger.info("="*70)
    
    # Initialize builder
    builder = WorkflowBuilder(config)
    
    # Set up checkpointing if enabled
    if enable_checkpointing:
        if checkpoint_db_path is None:
            # Default to checkpoints directory
            checkpoint_db_path = "checkpoints/compliance_workflow.db"
        
        # Get checkpoint configuration
        checkpoint_config = config.get("multi_agent", {}) if config else {}
        checkpoint_interval = checkpoint_config.get("checkpoint_interval", 1)
        max_history_size = checkpoint_config.get("max_history_size", 100)
        
        builder.setup_checkpointing(
            db_path=checkpoint_db_path,
            checkpoint_interval=checkpoint_interval,
            enable_state_manager=True,
            max_history_size=max_history_size
        )
        logger.info(f"✓ Checkpointing enabled: {checkpoint_db_path}")
        logger.info(f"  - Checkpoint interval: {checkpoint_interval}")
        logger.info(f"  - Max history size: {max_history_size}")
        logger.info(f"  - State manager: enabled")
    else:
        logger.info("✗ Checkpointing disabled")
    
    # Import actual agent implementations
    from agents.supervisor_agent import SupervisorAgent
    from agents.preprocessor_agent import PreprocessorAgent
    from agents.structure_agent import StructureAgent
    from agents.performance_agent import PerformanceAgent
    from agents.securities_agent import SecuritiesAgent
    from agents.general_agent import GeneralAgent
    from agents.prospectus_agent import ProspectusAgent
    from agents.registration_agent import RegistrationAgent
    from agents.esg_agent import ESGAgent
    from agents.aggregator_agent import AggregatorAgent
    from agents.context_agent import ContextAgent
    from agents.evidence_agent import EvidenceAgent
    from agents.reviewer_agent import ReviewerAgent
    
    logger.info("\nInitializing agents...")
    
    # Create Supervisor Agent
    supervisor_config = builder.agent_config_manager.get_config("supervisor")
    supervisor_agent = SupervisorAgent(
        config=supervisor_config,
        enable_parallel_execution=config.get("multi_agent", {}).get("parallel_execution", True) if config else True,
        max_parallel_agents=config.get("multi_agent", {}).get("max_parallel_agents", 4) if config else 4,
        enable_conditional_routing=config.get("routing", {}).get("skip_context_if_high_confidence", True) if config else True
    )
    logger.info(f"✓ Created Supervisor Agent")
    
    # Create Preprocessor Agent
    preprocessor_config = builder.agent_config_manager.get_config("preprocessor")
    preprocessor_agent = PreprocessorAgent(
        config=preprocessor_config,
        validate_before_processing=True,
        fail_on_validation_errors=False
    )
    logger.info(f"✓ Created Preprocessor Agent")
    
    # Create Core Compliance Agents
    # These agents will run in parallel after preprocessing
    
    # Structure Agent
    structure_config = builder.agent_config_manager.get_config("structure")
    structure_agent = StructureAgent(
        config=structure_config,
        parallel_execution=True,
        max_workers=5
    )
    logger.info(f"✓ Created Structure Agent")
    
    # Performance Agent
    performance_config = builder.agent_config_manager.get_config("performance")
    performance_agent = PerformanceAgent(
        config=performance_config,
        parallel_execution=True,
        max_workers=4
    )
    logger.info(f"✓ Created Performance Agent")
    
    # Securities Agent
    securities_config = builder.agent_config_manager.get_config("securities")
    # Initialize AI engine for securities agent if available
    ai_engine = None
    try:
        from ai_engine import create_ai_engine_from_env
        ai_engine = create_ai_engine_from_env()
        if ai_engine:
            logger.info("  AI Engine initialized for Securities Agent")
    except Exception as e:
        logger.warning(f"  AI Engine not available for Securities Agent: {e}")
    
    securities_agent = SecuritiesAgent(
        config=securities_config,
        ai_engine=ai_engine,
        parallel_execution=True,
        max_workers=3
    )
    logger.info(f"✓ Created Securities Agent")
    
    # General Agent
    general_config = builder.agent_config_manager.get_config("general")
    general_agent = GeneralAgent(
        config=general_config,
        parallel_execution=True,
        max_workers=4,
        apply_client_filtering=True
    )
    logger.info(f"✓ Created General Agent")
    
    # Create Specialized Compliance Agents
    # These agents run conditionally based on document metadata
    
    # Prospectus Agent (runs if prospectus data available)
    prospectus_config = builder.agent_config_manager.get_config("prospectus")
    prospectus_agent = ProspectusAgent(
        config=prospectus_config,
        ai_engine=ai_engine,
        parallel_execution=True,
        max_workers=4,
        semantic_matching_enabled=True
    )
    logger.info(f"✓ Created Prospectus Agent")
    
    # Registration Agent (runs if fund ISIN and authorized countries available)
    registration_config = builder.agent_config_manager.get_config("registration")
    registration_agent = RegistrationAgent(
        config=registration_config,
        ai_engine=ai_engine,
        ai_extraction_enabled=True
    )
    logger.info(f"✓ Created Registration Agent")
    
    # ESG Agent (runs if ESG classification is not 'other')
    esg_config = builder.agent_config_manager.get_config("esg")
    esg_agent = ESGAgent(
        config=esg_config,
        ai_engine=ai_engine,
        parallel_execution=False  # Sequential for ESG checks
    )
    logger.info(f"✓ Created ESG Agent")
    
    # Create Aggregator Agent
    # Aggregates results from all specialist agents
    aggregator_config = builder.agent_config_manager.get_config("aggregator")
    aggregator_agent = AggregatorAgent(
        config=aggregator_config,
        context_threshold=config.get("routing", {}).get("context_threshold", 80) if config else 80,
        review_threshold=config.get("routing", {}).get("review_threshold", 70) if config else 70,
        deduplication_enabled=True,
        similarity_threshold=0.85
    )
    logger.info(f"✓ Created Aggregator Agent")
    
    # Create Context Agent
    # Analyzes context and intent for low-confidence violations
    context_config = builder.agent_config_manager.get_config("context")
    context_agent = ContextAgent(
        config=context_config,
        ai_engine=ai_engine,
        confidence_boost_threshold=70,
        false_positive_threshold=85,
        analyze_all_violations=False
    )
    logger.info(f"✓ Created Context Agent")
    
    # Create Evidence Agent
    # Extracts evidence for violations
    evidence_config = builder.agent_config_manager.get_config("evidence")
    evidence_agent = EvidenceAgent(
        config=evidence_config,
        ai_engine=ai_engine,
        min_confidence_for_evidence=0,
        max_violations_to_process=50,
        enhance_all_violations=False
    )
    logger.info(f"✓ Created Evidence Agent")
    
    # Create Reviewer Agent
    # Manages Human-in-the-Loop review process
    from review_manager import ReviewManager
    import hitl_registry
    
    # Use review_manager from registry if available, otherwise create new one
    if config and config.get('hitl_registry_enabled') and hitl_registry.is_registered('review_manager'):
        review_manager = hitl_registry.get_component('review_manager')
        logger.info("  Using ReviewManager from HITL registry (with feedback integration)")
    else:
        review_manager = ReviewManager()
        logger.info("  Created new ReviewManager instance")
    
    reviewer_config = builder.agent_config_manager.get_config("reviewer")
    reviewer_agent = ReviewerAgent(
        config=reviewer_config,
        review_manager=review_manager,
        review_threshold=config.get("routing", {}).get("review_threshold", 70) if config else 70,
        auto_queue_enabled=True,
        batch_operations_enabled=True,
        hitl_interrupt_enabled=True
    )
    logger.info(f"✓ Created Reviewer Agent")
    
    logger.info("\nBuilding workflow graph...")
    
    # Add agent nodes to workflow
    builder.add_agent_node("supervisor", supervisor_agent, set_as_entry=True)
    logger.info("  Added node: supervisor (entry point)")
    
    builder.add_agent_node("preprocessor", preprocessor_agent)
    logger.info("  Added node: preprocessor")
    
    # Add core compliance agent nodes
    builder.add_agent_node("structure", structure_agent)
    logger.info("  Added node: structure")
    
    builder.add_agent_node("performance", performance_agent)
    logger.info("  Added node: performance")
    
    builder.add_agent_node("securities", securities_agent)
    logger.info("  Added node: securities")
    
    builder.add_agent_node("general", general_agent)
    logger.info("  Added node: general")
    
    # Add specialized compliance agent nodes
    builder.add_agent_node("prospectus", prospectus_agent)
    logger.info("  Added node: prospectus")
    
    builder.add_agent_node("registration", registration_agent)
    logger.info("  Added node: registration")
    
    builder.add_agent_node("esg", esg_agent)
    logger.info("  Added node: esg")
    
    # Add aggregator, context, evidence, and reviewer agent nodes
    builder.add_agent_node("aggregator", aggregator_agent)
    logger.info("  Added node: aggregator")
    
    builder.add_agent_node("context", context_agent)
    logger.info("  Added node: context")
    
    builder.add_agent_node("evidence", evidence_agent)
    logger.info("  Added node: evidence")
    
    builder.add_agent_node("reviewer", reviewer_agent)
    logger.info("  Added node: reviewer")
    
    # Define edges between agents
    builder.add_edge("supervisor", "preprocessor")
    logger.info("  Added edge: supervisor -> preprocessor")
    
    # Parallel execution: Preprocessor fans out to all core agents
    # In LangGraph, parallel execution is achieved by having multiple edges
    # from the same source node. All target nodes will execute in parallel.
    logger.info("\n  Setting up parallel execution from preprocessor...")
    builder.add_edge("preprocessor", "structure")
    logger.info("    Added parallel edge: preprocessor -> structure")
    
    builder.add_edge("preprocessor", "performance")
    logger.info("    Added parallel edge: preprocessor -> performance")
    
    builder.add_edge("preprocessor", "securities")
    logger.info("    Added parallel edge: preprocessor -> securities")
    
    builder.add_edge("preprocessor", "general")
    logger.info("    Added parallel edge: preprocessor -> general")
    
    # Add specialized agents to parallel execution group
    # These agents run conditionally based on document metadata
    builder.add_edge("preprocessor", "prospectus")
    logger.info("    Added parallel edge: preprocessor -> prospectus (conditional)")
    
    builder.add_edge("preprocessor", "registration")
    logger.info("    Added parallel edge: preprocessor -> registration (conditional)")
    
    builder.add_edge("preprocessor", "esg")
    logger.info("    Added parallel edge: preprocessor -> esg (conditional)")
    
    # Synchronization point: All agents (core + specialized) converge to AGGREGATOR
    # This creates an implicit barrier where workflow waits for all parallel
    # agents to complete before proceeding to aggregator
    logger.info("\n  Setting up synchronization point at aggregator...")
    builder.add_edge("structure", "aggregator")
    logger.info("    Added edge: structure -> aggregator")
    
    builder.add_edge("performance", "aggregator")
    logger.info("    Added edge: performance -> aggregator")
    
    builder.add_edge("securities", "aggregator")
    logger.info("    Added edge: securities -> aggregator")
    
    builder.add_edge("general", "aggregator")
    logger.info("    Added edge: general -> aggregator")
    
    builder.add_edge("prospectus", "aggregator")
    logger.info("    Added edge: prospectus -> aggregator")
    
    builder.add_edge("registration", "aggregator")
    logger.info("    Added edge: registration -> aggregator")
    
    builder.add_edge("esg", "aggregator")
    logger.info("    Added edge: esg -> aggregator")
    
    logger.info("\n  ✓ Parallel execution configured: 7 agents will run concurrently")
    logger.info("    Core agents: structure, performance, securities, general")
    logger.info("    Specialized agents: prospectus, registration, esg (conditional)")
    logger.info("    All agents converge at aggregator for result consolidation")
    
    # Conditional routing from aggregator based on confidence scores
    logger.info("\n  Setting up conditional routing from aggregator...")
    
    def route_from_aggregator(state: ComplianceState) -> str:
        """
        Route from aggregator based on next_action determined by aggregator
        
        Returns:
            - "context" if violations need context analysis (confidence < 80)
            - "complete" if all violations have acceptable confidence
        """
        next_action = state.get("next_action", "complete")
        
        if next_action == "context_analysis":
            return "context"
        else:
            return "complete"
    
    builder.add_conditional_edge(
        "aggregator",
        route_from_aggregator,
        {
            "context": "context",
            "complete": END
        }
    )
    logger.info("    Added conditional edge: aggregator -> context (if confidence < 80) or END")
    
    # Edge from context to evidence
    builder.add_edge("context", "evidence")
    logger.info("    Added edge: context -> evidence")
    
    # Conditional routing from evidence based on confidence scores
    logger.info("\n  Setting up conditional routing from evidence...")
    
    def route_from_evidence(state: ComplianceState) -> str:
        """
        Route from evidence based on next_action determined by evidence agent
        
        Returns:
            - "review" if violations need human review (confidence < 70)
            - "complete" if all violations have acceptable confidence
        """
        next_action = state.get("next_action", "complete")
        
        if next_action == "review":
            return "review"
        else:
            return "complete"
    
    builder.add_conditional_edge(
        "evidence",
        route_from_evidence,
        {
            "review": "reviewer",  # Route to reviewer agent for HITL
            "complete": END
        }
    )
    logger.info("    Added conditional edge: evidence -> reviewer (if confidence < 70) or END")
    
    # Edge from reviewer to END (workflow completes after HITL review)
    builder.add_edge("reviewer", END)
    logger.info("    Added edge: reviewer -> END")
    
    logger.info("\nCompiling workflow...")
    
    # Compile and return
    workflow = builder.compile()
    
    logger.info("="*70)
    logger.info("WORKFLOW COMPILATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Entry point: {builder.entry_point}")
    logger.info(f"Agents: {', '.join(builder.list_agents())}")
    logger.info(f"Total nodes: {len(builder.list_agents())}")
    logger.info(f"Core parallel agents: structure, performance, securities, general")
    logger.info(f"Specialized agents: prospectus, registration, esg (conditional)")
    logger.info(f"Analysis agents: aggregator, context, evidence (conditional routing)")
    logger.info(f"Review agent: reviewer (HITL integration)")
    logger.info(f"Checkpointing: {'enabled' if enable_checkpointing else 'disabled'}")
    logger.info("")
    logger.info("Workflow flow:")
    logger.info("  1. supervisor -> preprocessor")
    logger.info("  2. preprocessor -> [structure, performance, securities, general, prospectus, registration, esg] (parallel)")
    logger.info("  3. All parallel agents -> aggregator (synchronization)")
    logger.info("  4. aggregator -> context (if confidence < 80) OR END")
    logger.info("  5. context -> evidence")
    logger.info("  6. evidence -> reviewer (if confidence < 70) OR END")
    logger.info("  7. reviewer -> END (HITL interrupt point)")
    logger.info("")
    logger.info("HITL Integration:")
    logger.info("  - Reviewer agent queues low-confidence violations for human review")
    logger.info("  - Workflow can be interrupted at reviewer node for human input")
    logger.info("  - State persistence allows resuming after review completion")
    logger.info("="*70)
    
    return workflow


def visualize_workflow(
    workflow: CompiledGraph,
    output_path: Optional[str] = None,
    format: Literal["png", "svg", "pdf"] = "png"
) -> Optional[bytes]:
    """
    Visualize the workflow graph
    
    Generates a visual representation of the workflow structure showing
    nodes (agents) and edges (transitions).
    
    Args:
        workflow: Compiled workflow to visualize
        output_path: Path to save visualization (None to return bytes)
        format: Output format (png, svg, or pdf)
        
    Returns:
        Image bytes if output_path is None, otherwise None
        
    Note:
        Requires graphviz to be installed for visualization
    """
    try:
        # Get the graph representation
        graph_data = workflow.get_graph()
        
        # Try to use LangGraph's built-in visualization
        if hasattr(graph_data, 'draw_mermaid'):
            mermaid_code = graph_data.draw_mermaid()
            logger.info("Generated Mermaid diagram")
            
            if output_path:
                # Save Mermaid code to file
                with open(output_path.replace(f'.{format}', '.mmd'), 'w') as f:
                    f.write(mermaid_code)
                logger.info(f"Saved Mermaid diagram to {output_path.replace(f'.{format}', '.mmd')}")
            
            return mermaid_code.encode('utf-8')
        
        # Fallback: Generate simple text representation
        logger.warning("Mermaid visualization not available, generating text representation")
        
        nodes = graph_data.nodes if hasattr(graph_data, 'nodes') else []
        edges = graph_data.edges if hasattr(graph_data, 'edges') else []
        
        text_repr = "Workflow Graph:\n"
        text_repr += f"Nodes: {', '.join(str(n) for n in nodes)}\n"
        text_repr += f"Edges: {', '.join(f'{e[0]}->{e[1]}' for e in edges)}\n"
        
        if output_path:
            with open(output_path.replace(f'.{format}', '.txt'), 'w') as f:
                f.write(text_repr)
            logger.info(f"Saved text representation to {output_path.replace(f'.{format}', '.txt')}")
        
        return text_repr.encode('utf-8')
        
    except Exception as e:
        logger.error(f"Failed to visualize workflow: {e}")
        return None


def get_workflow_info(workflow: CompiledGraph) -> Dict[str, Any]:
    """
    Get information about a compiled workflow
    
    Args:
        workflow: Compiled workflow
        
    Returns:
        Dictionary with workflow metadata
    """
    try:
        graph_data = workflow.get_graph()
        
        # Try different ways to get entry point
        entry_point = 'unknown'
        if hasattr(graph_data, '_entry_point'):
            entry_point = graph_data._entry_point
        elif hasattr(graph_data, 'entry_point'):
            entry_point = graph_data.entry_point
        
        # Get nodes - handle different graph structures
        nodes = []
        if hasattr(graph_data, 'nodes'):
            if isinstance(graph_data.nodes, dict):
                nodes = list(graph_data.nodes.keys())
            else:
                nodes = list(graph_data.nodes)
        
        # Get edges - handle different graph structures
        edges = []
        if hasattr(graph_data, 'edges'):
            if isinstance(graph_data.edges, list):
                edges = graph_data.edges
            elif isinstance(graph_data.edges, dict):
                edges = list(graph_data.edges.items())
        
        info = {
            "has_checkpointing": hasattr(workflow, 'checkpointer') and workflow.checkpointer is not None,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "entry_point": entry_point,
            "nodes": nodes,
        }
        
        logger.debug(f"Workflow info: {info}")
        return info
        
    except Exception as e:
        logger.error(f"Failed to get workflow info: {e}")
        return {"error": str(e)}


def resume_workflow(
    workflow: CompiledGraph,
    checkpoint_id: str,
    config: Optional[Dict[str, Any]] = None,
    validate_state: bool = True,
    checkpoint_dir: str = "checkpoints"
) -> ComplianceState:
    """
    Resume a workflow from a checkpoint
    
    This function restores workflow state from a checkpoint and continues
    execution from the interruption point. It supports resuming after:
    - HITL (Human-in-the-Loop) interrupts for review
    - Agent failures with retry
    - Manual workflow pauses
    
    Args:
        workflow: Compiled workflow with checkpointing enabled
        checkpoint_id: ID of the checkpoint to resume from
        config: Optional configuration for resumption
        validate_state: Whether to validate restored state before resuming
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Final state after resuming workflow
        
    Raises:
        ValueError: If workflow doesn't have checkpointing enabled
        ValueError: If checkpoint not found or invalid
        RuntimeError: If state validation fails (when validate_state=True)
        
    Example:
        >>> workflow = create_compliance_workflow(config, enable_checkpointing=True)
        >>> # Workflow interrupted for HITL review
        >>> result = resume_workflow(workflow, "doc123_20240101_120000")
    """
    logger.info("="*70)
    logger.info("RESUMING WORKFLOW FROM CHECKPOINT")
    logger.info("="*70)
    
    # Verify workflow has checkpointing enabled
    if not hasattr(workflow, 'checkpointer') or workflow.checkpointer is None:
        raise ValueError("Workflow must have checkpointing enabled to resume")
    
    logger.info(f"Checkpoint ID: {checkpoint_id}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"State validation: {'enabled' if validate_state else 'disabled'}")
    
    try:
        # Step 1: Load and validate checkpoint state
        logger.info("\nStep 1: Loading checkpoint state...")
        
        from state_manager import StateManager
        state_manager = StateManager(checkpoint_dir=checkpoint_dir)
        
        # Get checkpoint info
        checkpoint_info = state_manager.get_checkpoint_info(checkpoint_id)
        if not checkpoint_info:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        logger.info(f"  Checkpoint found:")
        logger.info(f"    Document ID: {checkpoint_info.get('document_id')}")
        logger.info(f"    Timestamp: {checkpoint_info.get('timestamp')}")
        logger.info(f"    Current agent: {checkpoint_info.get('current_agent')}")
        logger.info(f"    Workflow status: {checkpoint_info.get('workflow_status')}")
        logger.info(f"    Violation count: {checkpoint_info.get('violation_count')}")
        logger.info(f"    File size: {checkpoint_info.get('file_size')} bytes")
        
        # Load the actual state
        restored_state = state_manager.load_state(checkpoint_id)
        if not restored_state:
            raise ValueError(f"Failed to load state from checkpoint: {checkpoint_id}")
        
        logger.info(f"  ✓ State loaded successfully")
        
        # Step 2: Validate restored state
        if validate_state:
            logger.info("\nStep 2: Validating restored state...")
            
            is_valid, validation_errors = validate_compliance_state(restored_state)
            
            if not is_valid:
                logger.error(f"  ✗ State validation failed:")
                for error in validation_errors:
                    logger.error(f"    - {error}")
                raise RuntimeError(f"State validation failed: {validation_errors}")
            
            logger.info(f"  ✓ State validation passed")
            
            # Additional integrity checks
            logger.info("\nPerforming integrity checks...")
            
            # Check for required fields
            required_fields = ["document_id", "document", "workflow_status"]
            missing_fields = [f for f in required_fields if f not in restored_state]
            if missing_fields:
                logger.warning(f"  ⚠ Missing fields: {missing_fields}")
            else:
                logger.info(f"  ✓ All required fields present")
            
            # Check workflow status
            workflow_status = restored_state.get("workflow_status", "unknown")
            logger.info(f"  Workflow status: {workflow_status}")
            
            # Check if this was a HITL interrupt
            hitl_interrupt = restored_state.get("hitl_interrupt_required", False)
            if hitl_interrupt:
                logger.info(f"  ℹ This checkpoint was created during HITL interrupt")
                logger.info(f"    Reason: {restored_state.get('hitl_interrupt_reason', 'N/A')}")
                logger.info(f"    Review queue items: {len(restored_state.get('review_queue', []))}")
            
            # Check for agent failures
            error_log = restored_state.get("error_log", [])
            if error_log:
                logger.warning(f"  ⚠ State contains {len(error_log)} error(s)")
                for i, error in enumerate(error_log[-3:], 1):  # Show last 3 errors
                    logger.warning(f"    {i}. Agent: {error.get('agent')}, Error: {error.get('error')}")
        else:
            logger.info("\nStep 2: State validation skipped")
        
        # Step 3: Prepare resumption configuration
        logger.info("\nStep 3: Preparing resumption configuration...")
        
        # Merge provided config with checkpoint metadata
        resume_config = config or {}
        
        # Set thread_id for LangGraph to resume from correct checkpoint
        # LangGraph uses thread_id to identify the execution thread
        thread_id = checkpoint_info.get("document_id", checkpoint_id)
        
        resume_config_with_thread = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id
            }
        }
        
        # Merge any additional config
        if resume_config:
            resume_config_with_thread.update(resume_config)
        
        logger.info(f"  Thread ID: {thread_id}")
        logger.info(f"  Resume configuration prepared")
        
        # Step 4: Resume workflow execution
        logger.info("\nStep 4: Resuming workflow execution...")
        logger.info(f"  Continuing from agent: {restored_state.get('current_agent', 'unknown')}")
        logger.info(f"  Next action: {restored_state.get('next_action', 'unknown')}")
        
        # Resume execution from checkpoint
        # LangGraph will automatically restore state and continue from interruption point
        result = workflow.invoke(
            None,  # No new input needed - state is restored from checkpoint
            config=resume_config_with_thread
        )
        
        logger.info("\n" + "="*70)
        logger.info("WORKFLOW RESUMPTION COMPLETE")
        logger.info("="*70)
        logger.info(f"Final workflow status: {result.get('workflow_status', 'unknown')}")
        logger.info(f"Final violation count: {len(result.get('violations', []))}")
        logger.info(f"Completed at agent: {result.get('current_agent', 'unknown')}")
        
        # Log summary
        summary = get_state_summary(result)
        logger.info("\nExecution Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        return result
        
    except ValueError as e:
        logger.error(f"✗ Validation error: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"✗ Runtime error: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Failed to resume workflow: {e}", exc_info=True)
        raise RuntimeError(f"Workflow resumption failed: {e}") from e


def resume_after_hitl_review(
    workflow: CompiledGraph,
    checkpoint_id: str,
    review_decisions: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    checkpoint_dir: str = "checkpoints"
) -> ComplianceState:
    """
    Resume workflow after Human-in-the-Loop (HITL) review
    
    This specialized function handles resumption after human review of
    low-confidence violations. It optionally applies review decisions
    before continuing workflow execution.
    
    Args:
        workflow: Compiled workflow with checkpointing enabled
        checkpoint_id: ID of the checkpoint to resume from
        review_decisions: Optional list of review decisions to apply
        config: Optional configuration for resumption
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Final state after resuming workflow
        
    Example:
        >>> decisions = [
        ...     {"review_id": "r1", "actual_violation": True, "reviewer_notes": "Confirmed"},
        ...     {"review_id": "r2", "actual_violation": False, "reviewer_notes": "False positive"}
        ... ]
        >>> result = resume_after_hitl_review(workflow, checkpoint_id, decisions)
    """
    logger.info("="*70)
    logger.info("RESUMING WORKFLOW AFTER HITL REVIEW")
    logger.info("="*70)
    
    try:
        # Load checkpoint state
        from state_manager import StateManager
        state_manager = StateManager(checkpoint_dir=checkpoint_dir)
        
        checkpoint_info = state_manager.get_checkpoint_info(checkpoint_id)
        if not checkpoint_info:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        # Verify this was a HITL interrupt
        restored_state = state_manager.load_state(checkpoint_id)
        if not restored_state:
            raise ValueError(f"Failed to load state from checkpoint: {checkpoint_id}")
        
        hitl_interrupt = restored_state.get("hitl_interrupt_required", False)
        if not hitl_interrupt:
            logger.warning("⚠ Checkpoint was not created during HITL interrupt")
        else:
            logger.info(f"✓ HITL interrupt checkpoint confirmed")
            logger.info(f"  Interrupt reason: {restored_state.get('hitl_interrupt_reason', 'N/A')}")
        
        # Apply review decisions if provided
        if review_decisions:
            logger.info(f"\nApplying {len(review_decisions)} review decisions...")
            
            from review_manager import ReviewManager
            review_manager = ReviewManager()
            
            for decision in review_decisions:
                review_id = decision.get("review_id")
                actual_violation = decision.get("actual_violation")
                reviewer_notes = decision.get("reviewer_notes", "")
                
                logger.info(f"  Review {review_id}: {'VIOLATION' if actual_violation else 'NO VIOLATION'}")
                
                # Update review item
                review_manager.update_review(
                    review_id=review_id,
                    actual_violation=actual_violation,
                    reviewer_notes=reviewer_notes
                )
            
            logger.info(f"✓ Review decisions applied")
            
            # Update state to clear HITL interrupt flag
            restored_state["hitl_interrupt_required"] = False
            restored_state["hitl_interrupt_reason"] = ""
            
            # Save updated state
            state_manager.save_state(
                restored_state,
                checkpoint_id=f"{checkpoint_id}_reviewed",
                metadata={"review_applied": True, "review_count": len(review_decisions)}
            )
        
        # Resume workflow
        logger.info("\nResuming workflow execution...")
        result = resume_workflow(
            workflow=workflow,
            checkpoint_id=checkpoint_id,
            config=config,
            validate_state=True,
            checkpoint_dir=checkpoint_dir
        )
        
        logger.info("\n" + "="*70)
        logger.info("HITL REVIEW RESUMPTION COMPLETE")
        logger.info("="*70)
        
        return result
        
    except Exception as e:
        logger.error(f"✗ Failed to resume after HITL review: {e}", exc_info=True)
        raise


def resume_after_agent_failure(
    workflow: CompiledGraph,
    checkpoint_id: str,
    failed_agent: str,
    retry_agent: bool = True,
    skip_agent: bool = False,
    config: Optional[Dict[str, Any]] = None,
    checkpoint_dir: str = "checkpoints"
) -> ComplianceState:
    """
    Resume workflow after agent failure
    
    This function handles resumption after an agent failure, with options to:
    - Retry the failed agent
    - Skip the failed agent and continue
    - Apply fallback strategies
    
    Args:
        workflow: Compiled workflow with checkpointing enabled
        checkpoint_id: ID of the checkpoint to resume from
        failed_agent: Name of the agent that failed
        retry_agent: Whether to retry the failed agent
        skip_agent: Whether to skip the failed agent
        config: Optional configuration for resumption
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Final state after resuming workflow
        
    Raises:
        ValueError: If both retry_agent and skip_agent are True
    """
    logger.info("="*70)
    logger.info("RESUMING WORKFLOW AFTER AGENT FAILURE")
    logger.info("="*70)
    
    if retry_agent and skip_agent:
        raise ValueError("Cannot both retry and skip agent")
    
    try:
        # Load checkpoint state
        from state_manager import StateManager
        state_manager = StateManager(checkpoint_dir=checkpoint_dir)
        
        checkpoint_info = state_manager.get_checkpoint_info(checkpoint_id)
        if not checkpoint_info:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        logger.info(f"Failed agent: {failed_agent}")
        logger.info(f"Strategy: {'RETRY' if retry_agent else 'SKIP' if skip_agent else 'CONTINUE'}")
        
        # Load state
        restored_state = state_manager.load_state(checkpoint_id)
        if not restored_state:
            raise ValueError(f"Failed to load state from checkpoint: {checkpoint_id}")
        
        # Check error log
        error_log = restored_state.get("error_log", [])
        agent_errors = [e for e in error_log if e.get("agent") == failed_agent]
        
        if agent_errors:
            logger.info(f"\nAgent error history ({len(agent_errors)} errors):")
            for i, error in enumerate(agent_errors[-3:], 1):
                logger.info(f"  {i}. {error.get('error')}")
        
        # Apply recovery strategy
        if skip_agent:
            logger.info(f"\nSkipping failed agent: {failed_agent}")
            
            # Mark agent as skipped in state
            if "skipped_agents" not in restored_state:
                restored_state["skipped_agents"] = []
            restored_state["skipped_agents"].append(failed_agent)
            
            # Update workflow status
            restored_state["workflow_status"] = "recovered_from_failure"
            
            # Save updated state
            state_manager.save_state(
                restored_state,
                checkpoint_id=f"{checkpoint_id}_recovered",
                metadata={"recovery_strategy": "skip", "skipped_agent": failed_agent}
            )
            
            logger.info(f"✓ Agent marked as skipped")
        
        elif retry_agent:
            logger.info(f"\nRetrying failed agent: {failed_agent}")
            
            # Clear previous errors for this agent
            restored_state["error_log"] = [
                e for e in error_log if e.get("agent") != failed_agent
            ]
            
            # Update workflow status
            restored_state["workflow_status"] = "retrying_after_failure"
            
            # Save updated state
            state_manager.save_state(
                restored_state,
                checkpoint_id=f"{checkpoint_id}_retry",
                metadata={"recovery_strategy": "retry", "retry_agent": failed_agent}
            )
            
            logger.info(f"✓ Agent prepared for retry")
        
        # Resume workflow
        logger.info("\nResuming workflow execution...")
        result = resume_workflow(
            workflow=workflow,
            checkpoint_id=checkpoint_id,
            config=config,
            validate_state=True,
            checkpoint_dir=checkpoint_dir
        )
        
        logger.info("\n" + "="*70)
        logger.info("AGENT FAILURE RECOVERY COMPLETE")
        logger.info("="*70)
        
        return result
        
    except Exception as e:
        logger.error(f"✗ Failed to resume after agent failure: {e}", exc_info=True)
        raise


def get_state_history(
    checkpoint_dir: str = "checkpoints",
    document_id: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get state history from checkpoint directory
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        document_id: Optional filter by document ID
        limit: Optional limit on number of records
        
    Returns:
        List of checkpoint metadata (most recent first)
    """
    from state_manager import StateManager
    
    state_manager = StateManager(checkpoint_dir=checkpoint_dir)
    return state_manager.get_state_history(document_id=document_id, limit=limit)


def list_checkpoints(
    checkpoint_dir: str = "checkpoints",
    document_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List all available checkpoints
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        document_id: Optional filter by document ID
        
    Returns:
        List of checkpoint metadata
    """
    from state_manager import StateManager
    
    state_manager = StateManager(checkpoint_dir=checkpoint_dir)
    return state_manager.list_checkpoints(document_id=document_id)


def cleanup_checkpoints(
    checkpoint_dir: str = "checkpoints",
    keep_count: Optional[int] = None,
    older_than_days: Optional[int] = None
) -> int:
    """
    Clean up old checkpoints
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_count: Keep only this many most recent checkpoints
        older_than_days: Delete checkpoints older than this many days
        
    Returns:
        Number of checkpoints deleted
    """
    from state_manager import StateManager
    
    state_manager = StateManager(checkpoint_dir=checkpoint_dir)
    return state_manager.cleanup_old_checkpoints(
        keep_count=keep_count,
        older_than_days=older_than_days
    )


def get_checkpoint_info(
    checkpoint_id: str,
    checkpoint_dir: str = "checkpoints"
) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a checkpoint
    
    Args:
        checkpoint_id: ID of checkpoint
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Checkpoint metadata or None if not found
    """
    from state_manager import StateManager
    
    state_manager = StateManager(checkpoint_dir=checkpoint_dir)
    return state_manager.get_checkpoint_info(checkpoint_id)


def save_workflow_state(
    state: ComplianceState,
    output_path: str
) -> bool:
    """
    Save workflow state to a file
    
    Args:
        state: Compliance state to save
        output_path: Path to save state
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from data_models_multiagent import serialize_state
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Serialize and save
        state_json = serialize_state(state)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(state_json)
        
        logger.info(f"Saved workflow state to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save workflow state: {e}")
        return False


def load_workflow_state(input_path: str) -> Optional[ComplianceState]:
    """
    Load workflow state from a file
    
    Args:
        input_path: Path to load state from
        
    Returns:
        Loaded ComplianceState or None if failed
    """
    try:
        from data_models_multiagent import deserialize_state
        
        with open(input_path, 'r', encoding='utf-8') as f:
            state_json = f.read()
        
        state = deserialize_state(state_json)
        
        # Validate loaded state
        is_valid, errors = validate_compliance_state(state)
        if not is_valid:
            logger.warning(f"Loaded state has validation errors: {errors}")
        
        logger.info(f"Loaded workflow state from {input_path}")
        return state
        
    except Exception as e:
        logger.error(f"Failed to load workflow state: {e}")
        return None


# Export all public symbols
__all__ = [
    "WorkflowBuilder",
    "create_compliance_workflow",
    "visualize_workflow",
    "get_workflow_info",
    "resume_workflow",
    "resume_after_hitl_review",
    "resume_after_agent_failure",
    "save_workflow_state",
    "load_workflow_state",
    "get_state_history",
    "list_checkpoints",
    "cleanup_checkpoints",
    "get_checkpoint_info"
]
