#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structure Agent

This module provides functionality for the multi-agent compliance system.
"""

"""
Structure Agent for Multi-Agent Compliance System

This agent handles structure compliance checks including:
- Promotional document mention
- Target audience specification
- Management company legal mention
- Fund name validation
- Date validation

The agent integrates all structure checking tools and executes them
in parallel for optimal performance.

Requirements: 1.2, 2.1, 3.2
"""

import logging
import sys
import os
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent, AgentConfig
from data_models_multiagent import ComplianceState, update_state_timestamp
from tools.structure_tools import (
    check_promotional_mention,
    check_target_audience,
    check_management_company,
    check_fund_name,
    check_date_validation,
    STRUCTURE_TOOLS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StructureAgent(BaseAgent):
    """
    Structure Agent - Handles all structure compliance checks

    This agent is responsible for validating document structure requirements:
    - Cover page must have promotional mention
    - Cover page must specify target audience
    - Back page must have management company legal mention
    - Fund name must be present and consistent
    - Document must have valid date

    The agent executes all checks in parallel for optimal performance and
    aggregates results with confidence scoring.

    Requirements: 1.2, 2.1, 3.2
    """

    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """
        Initialize Structure Agent

        Args:
            config: Agent configuration
            **kwargs: Additional configuration options
        """
        # Set default config if not provided
        if config is None:
            config = AgentConfig(
                name="structure",
                enabled=True,
                timeout_seconds=30.0,
                retry_attempts=3,
                log_level="INFO"
            )

        super().__init__(config=config, **kwargs)

        # Initialize tools
        self.tools = {
            "check_promotional_mention": check_promotional_mention,
            "check_target_audience": check_target_audience,
            "check_management_company": check_management_company,
            "check_fund_name": check_fund_name,
            "check_date_validation": check_date_validation
        }

        # Configuration
        self.parallel_execution = kwargs.get('parallel_execution', True)
        self.max_workers = kwargs.get('max_workers', 5)

        self.logger.info(f"Structure Agent initialized with {len(self.tools)} tools")
        self.logger.info(f"Parallel execution: {self.parallel_execution}")

    def process(self, state: ComplianceState) -> ComplianceState:
        """
        Process structure compliance checks

        Executes all structure checking tools and aggregates results.
        Tools can run in parallel for better performance.

        Args:
            state: Current compliance state

        Returns:
            Partial state update with violations and confidence scores
        """
        self.logger.info("Starting structure compliance checks")

        # Get document and metadata
        document = state.get("normalized_document") or state.get("document", {})
        metadata = state.get("metadata", {})
        config = state.get("config", {})

        # Validate inputs
        if not document:
            doc_id = state.get('document_id', 'unknown')
            self.logger.error(
                f"[{self.name}] ❌ CRITICAL: No document found in state for document_id={doc_id}"
            )
            self.logger.error(
                f"[{self.name}] Cannot perform structure checks without document data."
            )
            self.logger.error(
                f"[{self.name}] Resolution: Ensure preprocessor agent ran successfully before structure agent. "
                f"Check that 'normalized_document' or 'document' field exists in state."
            )
            # Return partial state with error
            return {
                "error_log": [{
                    "agent": self.name,
                    "error": "No document found in state - cannot perform structure checks",
                    "error_type": "MissingDocumentError",
                    "severity": "CRITICAL",
                    "timestamp": datetime.now().isoformat(),
                    "document_id": doc_id,
                    "workflow_status": state.get('workflow_status', 'unknown'),
                    "resolution_hint": (
                        "Verify preprocessor agent completed successfully. "
                        "Check workflow execution order. "
                        "Ensure document is properly loaded in state."
                    )
                }]
            }

        # Execute checks
        if self.parallel_execution:
            violations = self._execute_checks_parallel(document, metadata, config)
        else:
            violations = self._execute_checks_sequential(document, metadata, config)

        # Log results
        if violations:
            self.logger.info(f"Found {len(violations)} structure violations")
        else:
            self.logger.info("No structure violations found")

        # Calculate aggregate confidence
        if violations:
            avg_confidence = sum(v.get("confidence", 0) for v in violations) / len(violations)
            confidence = int(avg_confidence)
        else:
            confidence = 100

        self.logger.info(f"Structure checks completed. Violations: {len(violations)}")

        # Return only the fields this agent updates
        # This allows parallel execution without conflicts
        return {
            "violations": violations,
            "confidence_scores": {self.name: confidence}
        }

    def _execute_checks_parallel(
        self,
        document: dict,
        metadata: dict,
        config: dict
    ) -> List[Dict[str, Any]]:
        """
        Execute all structure checks in parallel

        Uses ThreadPoolExecutor to run checks concurrently for better performance.

        Args:
            document: Document to check
            metadata: Document metadata
            config: Configuration dictionary

        Returns:
            List of violations found
        """
        self.logger.info(f"Executing {len(self.tools)} checks in parallel")

        violations = []

        # Create tasks for each check
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all checks
            future_to_check = {}

            # Promotional mention check
            future = executor.submit(
                self._safe_tool_invoke,
                "check_promotional_mention",
                document=document,
                config=config
            )
            future_to_check[future] = "check_promotional_mention"

            # Target audience check
            future = executor.submit(
                self._safe_tool_invoke,
                "check_target_audience",
                document=document,
                client_type=metadata.get("client_type", "retail")
            )
            future_to_check[future] = "check_target_audience"

            # Management company check
            future = executor.submit(
                self._safe_tool_invoke,
                "check_management_company",
                document=document
            )
            future_to_check[future] = "check_management_company"

            # Fund name check
            future = executor.submit(
                self._safe_tool_invoke,
                "check_fund_name",
                document=document,
                metadata=metadata
            )
            future_to_check[future] = "check_fund_name"

            # Date validation check
            future = executor.submit(
                self._safe_tool_invoke,
                "check_date_validation",
                document=document
            )
            future_to_check[future] = "check_date_validation"

            # Collect results as they complete
            for future in as_completed(future_to_check):
                check_name = future_to_check[future]
                try:
                    result = future.result()
                    if result:
                        self.logger.info(f"✗ {check_name}: Violation found")
                        violations.append(result)
                    else:
                        self.logger.info(f"✓ {check_name}: Pass")
                except Exception as e:
                    self.logger.error(f"Error in {check_name}: {e}")

        return violations

    def _execute_checks_sequential(
        self,
        document: dict,
        metadata: dict,
        config: dict
    ) -> List[Dict[str, Any]]:
        """
        Execute all structure checks sequentially

        Runs checks one after another. Useful for debugging or when
        parallel execution is not desired.

        Args:
            document: Document to check
            metadata: Document metadata
            config: Configuration dictionary

        Returns:
            List of violations found
        """
        self.logger.info(f"Executing {len(self.tools)} checks sequentially")

        violations = []

        # Check promotional mention
        result = self._safe_tool_invoke(
            "check_promotional_mention",
            document=document,
            config=config
        )
        if result:
            violations.append(result)
            self.logger.info("✗ check_promotional_mention: Violation found")
        else:
            self.logger.info("✓ check_promotional_mention: Pass")

        # Check target audience
        result = self._safe_tool_invoke(
            "check_target_audience",
            document=document,
            client_type=metadata.get("client_type", "retail")
        )
        if result:
            violations.append(result)
            self.logger.info("✗ check_target_audience: Violation found")
        else:
            self.logger.info("✓ check_target_audience: Pass")

        # Check management company
        result = self._safe_tool_invoke(
            "check_management_company",
            document=document
        )
        if result:
            violations.append(result)
            self.logger.info("✗ check_management_company: Violation found")
        else:
            self.logger.info("✓ check_management_company: Pass")

        # Check fund name
        result = self._safe_tool_invoke(
            "check_fund_name",
            document=document,
            metadata=metadata
        )
        if result:
            violations.append(result)
            self.logger.info("✗ check_fund_name: Violation found")
        else:
            self.logger.info("✓ check_fund_name: Pass")

        # Check date validation
        result = self._safe_tool_invoke(
            "check_date_validation",
            document=document
        )
        if result:
            violations.append(result)
            self.logger.info("✗ check_date_validation: Violation found")
        else:
            self.logger.info("✓ check_date_validation: Pass")

        return violations

    def _safe_tool_invoke(self, tool_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Safely invoke a tool with error handling

        Args:
            tool_name: Name of the tool to invoke
            **kwargs: Arguments to pass to the tool

        Returns:
            Tool result or None if error occurred
        """
        try:
            tool = self.tools.get(tool_name)
            if not tool:
                self.logger.error(f"Tool not found: {tool_name}")
                return None

            # Invoke the tool
            result = tool.func(**kwargs)

            # Add agent name to violation if present
            if result and isinstance(result, dict):
                result["agent"] = self.name
                result["timestamp"] = datetime.now().isoformat()

            return result

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)[:200]
            self.logger.error(
                f"[{self.name}] ❌ Tool execution failed: {tool_name} - {error_type}: {error_msg}"
            )
            self.logger.error(
                f"[{self.name}] Context: document_id={kwargs.get('document', {}).get('document_metadata', {}).get('document_id', 'unknown')}"
            )
            self.logger.debug(f"[{self.name}] Full error details:", exc_info=True)

            # Return error violation with detailed context
            return {
                "type": "STRUCTURE",
                "severity": "ERROR",
                "slide": "Unknown",
                "location": "Unknown",
                "rule": f"STRUCT_ERROR_{tool_name.upper()}",
                "message": (
                    f"Structure check '{tool_name}' failed with {error_type}. "
                    f"Error: {error_msg}. "
                    f"This check could not be completed."
                ),
                "evidence": f"Tool execution failed: {error_type}",
                "confidence": 0,
                "method": "ERROR",
                "agent": self.name,
                "timestamp": datetime.now().isoformat(),
                "error_details": {
                    "tool": tool_name,
                    "error_type": error_type,
                    "error_message": str(e),
                    "resolution_hint": (
                        f"Check {tool_name} implementation and input data. "
                        f"Verify document structure is valid. "
                        f"Review error log for full traceback."
                    )
                }
            }

    def get_tool_names(self) -> List[str]:
        """
        Get list of available tool names

        Returns:
            List of tool names
        """
        return list(self.tools.keys())

    def get_tool_count(self) -> int:
        """
        Get number of available tools

        Returns:
            Number of tools
        """
        return len(self.tools)


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    """Test the Structure Agent with example data"""

    from data_models_multiagent import initialize_compliance_state

    logger.info("=" * 70)
    logger.info("STRUCTURE AGENT TEST")
    logger.info("=" * 70)

    # Create test document
    test_document = {
        'document_metadata': {
            'fund_isin': 'FR0010135103',
            'fund_name': 'ODDO BHF Algo Trend US Fund',
            'client_type': 'retail',
            'document_type': 'fund_presentation',
            'document_date': '2024-01-15'
        },
        'page_de_garde': {
            'title': 'ODDO BHF Algo Trend US Fund',
            'subtitle': 'Document promotionnel destiné aux investisseurs non professionnels'
        },
        'slide_2': {
            'content': 'Investment strategy and objectives...'
        },
        'pages_suivantes': [
            {'content': 'Performance data...'}
        ],
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management SAS - Société de gestion agréée'
        }
    }

    # Initialize state
    config = {
        'ai_enabled': True,
        'agents': {
            'structure': {
                'enabled': True,
                'timeout_seconds': 30.0
            }
        }
    }

    state = initialize_compliance_state(
        document=test_document,
        document_id="test_doc_001",
        config=config
    )

    # Add metadata to state (normally done by preprocessor)
    state["metadata"] = {
        'fund_name': 'ODDO BHF Algo Trend US Fund',
        'client_type': 'retail',
        'fund_isin': 'FR0010135103'
    }
    state["normalized_document"] = test_document

    # Create and run agent
    logger.info("\n1. Creating Structure Agent...")
    agent_config = AgentConfig(
        name="structure",
        enabled=True,
        timeout_seconds=30.0,
        log_level="INFO"
    )

    agent = StructureAgent(config=agent_config, parallel_execution=True)
    logger.info(f"   Agent created: {agent}")
    logger.info(f"   Tools available: {agent.get_tool_count()}")
    logger.info(f"   Tool names: {', '.join(agent.get_tool_names())}")

    # Execute agent
    logger.info("\n2. Executing Structure Agent...")
    logger.info("-" * 70)

    result_state = agent(state)

    logger.info("-" * 70)

    # Display results
    logger.info("\n3. Results:")
    logger.info(f"   Violations found: {len(result_state.get('violations', []))}")
    logger.info(f"   Agent confidence: {result_state.get('confidence_scores', {}).get('structure', 'N/A')}")
    logger.info(f"   Execution time: {result_state.get('agent_timings', {}).get('structure', 'N/A'):.2f}s")

    if result_state.get('violations'):
        logger.info("\n4. Violations:")
        for i, violation in enumerate(result_state['violations'], 1):
            logger.info(f"\n   Violation {i}:")
            logger.info(f"      Rule: {violation.get('rule', 'N/A')}")
            logger.info(f"      Severity: {violation.get('severity', 'N/A')}")
            logger.info(f"      Message: {violation.get('message', 'N/A')}")
            logger.info(f"      Confidence: {violation.get('confidence', 'N/A')}%")
            logger.info(f"      Location: {violation.get('slide', 'N/A')} / {violation.get('location', 'N/A')}")
    else:
        logger.info("\n4. ✓ No violations found - document structure is compliant!")

    # Display errors if any
    if result_state.get('error_log'):
        logger.info("\n5. Errors:")
        for error in result_state['error_log']:
            logger.info(f"   - {error.get('error', 'Unknown error')}")

    logger.info("\n" + "=" * 70)
    logger.info("Structure Agent test completed!")
    logger.info("=" * 70)

    # Display agent statistics
    logger.info("\n6. Agent Statistics:")
    agent.log_execution_stats()
