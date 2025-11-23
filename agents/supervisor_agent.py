#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervisor Agent

This module provides functionality for the multi-agent compliance system.
"""

"""
Supervisor Agent for Multi-Agent Compliance System

The Supervisor Agent orchestrates the entire compliance checking workflow,
coordinating specialist agents, handling failures, and generating final reports.

Responsibilities:
- Initialize workflow with document preprocessing
- Determine which specialist agents to invoke based on document type
- Monitor agent execution and handle failures with fallback strategies
- Aggregate results from all specialist agents
- Generate final compliance report with violations, confidence scores, and recommendations
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import base agent framework
from agents.base_agent import BaseAgent, AgentConfig, agent_timing, agent_error_handler

# Import state models
from data_models_multiagent import (
    ComplianceState,
    WorkflowStatus,
    update_state_timestamp,
    mark_state_completed,
    get_state_summary
)


# Configure logging
logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent - Orchestrates the entire compliance workflow

    The Supervisor is the entry point for all compliance checks. It:
    1. Initializes the workflow state
    2. Creates an execution plan based on document type and metadata
    3. Coordinates specialist agent execution
    4. Handles agent failures and implements fallback strategies
    5. Generates the final compliance report

    Requirements addressed:
    - 8.1: Initialize workflow with document preprocessing
    - 8.2: Determine which specialist agents to invoke based on document type
    - 8.3: Monitor agent execution and handle failures
    - 8.4: Aggregate results from all specialist agents
    - 8.5: Generate final compliance report
    """

    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """
        Initialize Supervisor Agent

        Args:
            config: Agent configuration
            **kwargs: Additional configuration options
        """
        if config is None:
            config = AgentConfig(
                name="supervisor",
                enabled=True,
                timeout_seconds=60.0,  # Supervisor gets more time
                retry_attempts=1,  # Supervisor doesn't retry, delegates to agents
                log_level="INFO"
            )

        super().__init__(config, **kwargs)

        # Supervisor-specific configuration
        self.enable_parallel_execution = kwargs.get('enable_parallel_execution', True)
        self.max_parallel_agents = kwargs.get('max_parallel_agents', 4)
        self.enable_conditional_routing = kwargs.get('enable_conditional_routing', True)

        self.logger.info(f"Supervisor Agent initialized")
        self.logger.info(f"  Parallel execution: {self.enable_parallel_execution}")
        self.logger.info(f"  Max parallel agents: {self.max_parallel_agents}")
        self.logger.info(f"  Conditional routing: {self.enable_conditional_routing}")

    def process(self, state: ComplianceState) -> ComplianceState:
        """
        Process the compliance state - main supervisor logic

        This method orchestrates the entire workflow:
        1. Initialize workflow
        2. Create execution plan
        3. Set up for agent coordination
        4. Prepare state for next agents

        Args:
            state: Current compliance state

        Returns:
            Updated compliance state with execution plan
        """
        self.logger.info("="*70)
        self.logger.info("SUPERVISOR: Starting compliance workflow orchestration")
        self.logger.info("="*70)

        # Update workflow status
        state["workflow_status"] = WorkflowStatus.INITIALIZED.value
        state = update_state_timestamp(state)

        # Log document information
        self._log_document_info(state)

        # Create execution plan based on document type and metadata
        execution_plan = self._create_execution_plan(state)
        state["execution_plan"] = execution_plan

        self.logger.info(f"Execution plan created with {len(execution_plan)} agents:")
        for i, agent_name in enumerate(execution_plan, 1):
            self.logger.info(f"  {i}. {agent_name}")

        # Set next action
        state["next_action"] = "preprocess"

        # Initialize performance tracking
        if "agent_timings" not in state:
            state["agent_timings"] = {}
        if "api_calls" not in state:
            state["api_calls"] = 0
        if "cache_hits" not in state:
            state["cache_hits"] = 0

        # Initialize error log
        if "error_log" not in state:
            state["error_log"] = []

        self.logger.info("Supervisor initialization complete")
        self.logger.info(f"Next action: {state['next_action']}")

        return state

    def _log_document_info(self, state: ComplianceState):
        """
        Log document information for monitoring

        Args:
            state: Current compliance state
        """
        self.logger.info("Document Information:")
        self.logger.info(f"  Document ID: {state.get('document_id', 'unknown')}")
        self.logger.info(f"  Document Type: {state.get('document_type', 'unknown')}")
        self.logger.info(f"  Client Type: {state.get('client_type', 'unknown')}")

        # Log metadata if available
        metadata = state.get("metadata", {})
        if metadata:
            self.logger.info("  Metadata:")
            for key, value in metadata.items():
                self.logger.info(f"    {key}: {value}")

    def _create_execution_plan(self, state: ComplianceState) -> List[str]:
        """
        Create execution plan based on document type and configuration

        This method determines which specialist agents should be invoked
        based on:
        - Document type (fund_presentation, factsheet, etc.)
        - Document metadata (fund_isin, esg_classification, etc.)
        - Configuration settings (enabled features)

        Requirements addressed:
        - 8.2: Determine which specialist agents to invoke based on document type

        Args:
            state: Current compliance state

        Returns:
            List of agent names in execution order
        """
        plan = []

        # Get document metadata
        metadata = state.get("metadata", {})
        if not metadata:
            # Try to extract from document
            metadata = state.get("document", {}).get("document_metadata", {})

        document_type = state.get("document_type", "fund_presentation")
        client_type = state.get("client_type", "retail")
        config = state.get("config", {})

        self.logger.info("Creating execution plan...")
        self.logger.info(f"  Document type: {document_type}")
        self.logger.info(f"  Client type: {client_type}")

        # Step 1: Always start with preprocessor
        plan.append("preprocessor")
        self.logger.debug("  Added: preprocessor (required)")

        # Step 2: Core compliance checks (always run)
        # These can run in parallel
        core_agents = []

        if self._is_agent_enabled(config, "structure"):
            core_agents.append("structure")
            self.logger.debug("  Added: structure (core check)")

        if self._is_agent_enabled(config, "performance"):
            core_agents.append("performance")
            self.logger.debug("  Added: performance (core check)")

        if self._is_agent_enabled(config, "securities"):
            core_agents.append("securities")
            self.logger.debug("  Added: securities (core check)")

        if self._is_agent_enabled(config, "general"):
            core_agents.append("general")
            self.logger.debug("  Added: general (core check)")

        plan.extend(core_agents)

        # Step 3: Conditional specialist checks
        specialist_agents = []

        # Prospectus check - only if prospectus data is provided
        if self._should_run_prospectus_check(state, config):
            specialist_agents.append("prospectus")
            self.logger.debug("  Added: prospectus (prospectus data available)")

        # Registration check - only if fund has ISIN
        if self._should_run_registration_check(state, config):
            specialist_agents.append("registration")
            self.logger.debug("  Added: registration (fund ISIN available)")

        # ESG check - only if fund has ESG classification
        if self._should_run_esg_check(state, config):
            specialist_agents.append("esg")
            self.logger.debug("  Added: esg (ESG classification present)")

        plan.extend(specialist_agents)

        # Step 4: Aggregation (always run after specialist checks)
        plan.append("aggregator")
        self.logger.debug("  Added: aggregator (required)")

        # Step 5: Context and evidence analysis (conditional based on confidence)
        # These will be added by conditional routing in the workflow
        # We don't add them to the plan here, as they depend on runtime state

        # Step 6: Review (conditional based on confidence)
        # Also handled by conditional routing

        self.logger.info(f"Execution plan complete: {len(plan)} agents")

        return plan

    def _is_agent_enabled(self, config: Dict[str, Any], agent_name: str) -> bool:
        """
        Check if an agent is enabled in configuration

        Args:
            config: Configuration dictionary
            agent_name: Name of the agent

        Returns:
            True if agent is enabled, False otherwise
        """
        # Check in agents section
        agents_config = config.get("agents", {})
        if agent_name in agents_config:
            return agents_config[agent_name].get("enabled", True)

        # Check in features section (legacy)
        features = config.get("features", {})
        feature_key = f"enable_{agent_name}_ai"
        if feature_key in features:
            return features[feature_key]

        # Default to enabled
        return True

    def _should_run_prospectus_check(self, state: ComplianceState, config: Dict[str, Any]) -> bool:
        """
        Determine if prospectus check should run

        Args:
            state: Current compliance state
            config: Configuration dictionary

        Returns:
            True if prospectus check should run
        """
        # Check if agent is enabled
        if not self._is_agent_enabled(config, "prospectus"):
            return False

        # Check if prospectus data is available
        prospectus_data = config.get("prospectus_data")
        if prospectus_data:
            return True

        # Check in document metadata
        metadata = state.get("metadata", {})
        if not metadata:
            metadata = state.get("document", {}).get("document_metadata", {})

        return bool(metadata.get("prospectus_fund_name") or metadata.get("prospectus_data"))

    def _should_run_registration_check(self, state: ComplianceState, config: Dict[str, Any]) -> bool:
        """
        Determine if registration check should run

        Args:
            state: Current compliance state
            config: Configuration dictionary

        Returns:
            True if registration check should run
        """
        # Check if agent is enabled
        if not self._is_agent_enabled(config, "registration"):
            return False

        # Check if fund ISIN is available
        metadata = state.get("metadata", {})
        if not metadata:
            metadata = state.get("document", {}).get("document_metadata", {})

        fund_isin = metadata.get("fund_isin")

        return bool(fund_isin)

    def _should_run_esg_check(self, state: ComplianceState, config: Dict[str, Any]) -> bool:
        """
        Determine if ESG check should run

        Args:
            state: Current compliance state
            config: Configuration dictionary

        Returns:
            True if ESG check should run
        """
        # Check if agent is enabled
        if not self._is_agent_enabled(config, "esg"):
            return False

        # Check if fund has ESG classification
        metadata = state.get("metadata", {})
        if not metadata:
            metadata = state.get("document", {}).get("document_metadata", {})

        esg_classification = metadata.get("fund_esg_classification", "other")

        # Run ESG check if classification is not "other"
        return esg_classification != "other"

    def generate_final_report(self, state: ComplianceState) -> Dict[str, Any]:
        """
        Generate final compliance report from workflow state

        This method is called at the end of the workflow to create
        a comprehensive report of all findings.

        Requirements addressed:
        - 8.5: Generate final compliance report

        Args:
            state: Final compliance state

        Returns:
            Dictionary with complete compliance report
        """
        self.logger.info("="*70)
        self.logger.info("SUPERVISOR: Generating final compliance report")
        self.logger.info("="*70)

        violations = state.get("violations", [])

        # Categorize violations by status
        detected_violations = [v for v in violations if v.get("status") == "detected"]
        validated_violations = [v for v in violations if v.get("status") == "validated"]
        false_positives = [v for v in violations if v.get("status") == "false_positive_filtered"]
        pending_review = [v for v in violations if v.get("status") == "pending_review"]
        confirmed_violations = [v for v in violations if v.get("status") == "confirmed"]

        # Categorize by severity
        critical_violations = [v for v in violations if v.get("severity") == "critical"]
        high_violations = [v for v in violations if v.get("severity") == "high"]
        medium_violations = [v for v in violations if v.get("severity") == "medium"]
        low_violations = [v for v in violations if v.get("severity") == "low"]

        # Calculate statistics
        total_violations = len(violations)
        total_confirmed = len(confirmed_violations) + len(validated_violations)
        total_false_positives = len(false_positives)
        total_pending = len(pending_review)

        # Get performance metrics
        agent_timings = state.get("agent_timings", {})
        total_time = sum(agent_timings.values())
        api_calls = state.get("api_calls", 0)
        cache_hits = state.get("cache_hits", 0)

        # Get error information
        errors = state.get("error_log", [])

        # Build report
        report = {
            "document_id": state.get("document_id", "unknown"),
            "document_type": state.get("document_type", "unknown"),
            "client_type": state.get("client_type", "unknown"),
            "workflow_status": state.get("workflow_status", "unknown"),
            "timestamp": datetime.now().isoformat(),

            # Summary statistics
            "summary": {
                "total_violations": total_violations,
                "confirmed_violations": total_confirmed,
                "false_positives": total_false_positives,
                "pending_review": total_pending,
                "critical_violations": len(critical_violations),
                "high_violations": len(high_violations),
                "medium_violations": len(medium_violations),
                "low_violations": len(low_violations)
            },

            # All violations
            "violations": violations,

            # Categorized violations
            "violations_by_status": {
                "detected": detected_violations,
                "validated": validated_violations,
                "false_positives": false_positives,
                "pending_review": pending_review,
                "confirmed": confirmed_violations
            },

            "violations_by_severity": {
                "critical": critical_violations,
                "high": high_violations,
                "medium": medium_violations,
                "low": low_violations
            },

            # Performance metrics
            "performance": {
                "total_time_seconds": round(total_time, 2),
                "agent_timings": {k: round(v, 2) for k, v in agent_timings.items()},
                "api_calls": api_calls,
                "cache_hits": cache_hits,
                "cache_hit_rate": round((cache_hits / api_calls * 100) if api_calls > 0 else 0, 2)
            },

            # Execution information
            "execution": {
                "execution_plan": state.get("execution_plan", []),
                "agents_executed": list(agent_timings.keys()),
                "errors": errors,
                "error_count": len(errors)
            },

            # Timestamps
            "started_at": state.get("started_at"),
            "completed_at": state.get("completed_at"),
            "updated_at": state.get("updated_at")
        }

        # Log report summary
        self.logger.info("Report Summary:")
        self.logger.info(f"  Total violations: {total_violations}")
        self.logger.info(f"  Confirmed: {total_confirmed}")
        self.logger.info(f"  False positives: {total_false_positives}")
        self.logger.info(f"  Pending review: {total_pending}")
        self.logger.info(f"  Critical: {len(critical_violations)}")
        self.logger.info(f"  Total time: {total_time:.2f}s")
        self.logger.info(f"  API calls: {api_calls}")
        self.logger.info(f"  Cache hits: {cache_hits}")
        self.logger.info(f"  Errors: {len(errors)}")

        return report

    def handle_agent_failure(
        self,
        state: ComplianceState,
        failed_agent: str,
        error: Exception
    ) -> ComplianceState:
        """
        Handle failure of a specialist agent

        Implements fallback strategies:
        1. Log the error with detailed context
        2. Continue with remaining agents (graceful degradation)
        3. Mark workflow as partially completed
        4. Provide actionable resolution hints

        Requirements addressed:
        - 8.3: Monitor agent execution and handle failures

        Args:
            state: Current compliance state
            failed_agent: Name of the failed agent
            error: Exception that caused the failure

        Returns:
            Updated state with error logged
        """
        error_type = type(error).__name__
        error_msg = str(error)[:300]
        doc_id = state.get('document_id', 'unknown')

        self.logger.error("="*70)
        self.logger.error(f"SUPERVISOR: Agent Failure Detected")
        self.logger.error("="*70)
        self.logger.error(f"Failed Agent: {failed_agent}")
        self.logger.error(f"Error Type: {error_type}")
        self.logger.error(f"Error Message: {error_msg}")
        self.logger.error(f"Document ID: {doc_id}")
        self.logger.error(f"Workflow Status: {state.get('workflow_status', 'unknown')}")
        self.logger.error("="*70)

        # Log error in state with detailed context
        if "error_log" not in state:
            state["error_log"] = []

        state["error_log"].append({
            "agent": failed_agent,
            "error": str(error),
            "error_type": error_type,
            "error_message": f"{error_type}: {error_msg}",
            "timestamp": datetime.now().isoformat(),
            "handled_by": "supervisor",
            "document_id": doc_id,
            "workflow_status": state.get('workflow_status', 'unknown'),
            "resolution_hint": self._get_agent_failure_hint(failed_agent, error)
        })

        # Update state
        state = update_state_timestamp(state)

        # Implement fallback strategy
        self.logger.warning("="*70)
        self.logger.warning(f"SUPERVISOR: Implementing Fallback Strategy")
        self.logger.warning("="*70)
        self.logger.warning(f"Strategy: Graceful Degradation")
        self.logger.warning(f"Action: Continue workflow with remaining agents")
        self.logger.warning(f"Impact: Compliance checks from {failed_agent} will be incomplete")
        self.logger.warning(f"Recommendation: Review error logs and re-run after fixing the issue")
        self.logger.warning("="*70)

        # The workflow will continue with other agents
        # This is handled by LangGraph's error handling

        return state

    def _get_agent_failure_hint(self, agent_name: str, error: Exception) -> str:
        """
        Get resolution hint for agent failure

        Args:
            agent_name: Name of failed agent
            error: Exception that occurred

        Returns:
            Resolution hint string
        """
        error_msg = str(error).lower()

        # Agent-specific hints
        agent_hints = {
            "preprocessor": (
                "Preprocessor failure affects all downstream agents. "
                "Check: 1) Document format is valid, 2) Required metadata fields exist, "
                "3) Whitelist manager is working correctly."
            ),
            "structure": (
                "Structure checks failed. "
                "Check: 1) Document has required sections (cover, slides, back page), "
                "2) Structure rules are properly configured, 3) Document format is valid."
            ),
            "performance": (
                "Performance checks failed. "
                "Check: 1) Performance data format is correct, 2) AI engine is available, "
                "3) Performance rules are properly configured."
            ),
            "securities": (
                "Securities/values checks failed. "
                "Check: 1) Whitelist is properly built, 2) values_rules.json exists and is valid, "
                "3) AI engine is available for context analysis."
            ),
            "context": (
                "Context analysis failed. "
                "Check: 1) AI engine connectivity, 2) API credentials are valid, "
                "3) Context analysis tools are properly configured."
            ),
            "evidence": (
                "Evidence extraction failed. "
                "Check: 1) Document text is accessible, 2) Evidence extraction tools are working, "
                "3) AI engine is available for semantic analysis."
            )
        }

        base_hint = agent_hints.get(agent_name, f"Agent {agent_name} failed. Check agent configuration and logs.")

        # Add error-specific hints
        if "api" in error_msg or "connection" in error_msg:
            base_hint += " API/Network issue detected - check connectivity and API keys."
        elif "auth" in error_msg or "key" in error_msg:
            base_hint += " Authentication issue detected - verify API credentials in .env file."
        elif "json" in error_msg or "parse" in error_msg:
            base_hint += " Data parsing issue detected - verify document format and structure."

        return base_hint

    def get_execution_summary(self, state: ComplianceState) -> Dict[str, Any]:
        """
        Get a summary of workflow execution

        Args:
            state: Current compliance state

        Returns:
            Dictionary with execution summary
        """
        return get_state_summary(state)


# Export
__all__ = ["SupervisorAgent"]
