#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General Agent

This module provides functionality for the multi-agent compliance system.
"""

"""
General Agent for Multi-Agent Compliance System

This agent handles general compliance checks including:
- Glossary requirements for technical terms
- Morningstar rating date validation
- Source citations for external data
- Technical term identification

The agent integrates all general checking tools and executes them
with client type filtering (retail vs professional).

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
from tools.general_tools import (
    check_glossary_requirement,
    check_morningstar_date,
    check_source_citations,
    check_technical_terms,
    GENERAL_TOOLS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeneralAgent(BaseAgent):
    """
    General Agent - Handles all general compliance checks

    This agent is responsible for validating general document requirements:
    - Glossary requirement for retail documents with technical terms
    - Morningstar rating must include calculation date
    - External data must have proper source and date citations
    - Technical term identification for glossary validation

    The agent applies client type filtering (retail vs professional) to
    determine which rules apply. Professional documents have fewer requirements.

    Requirements: 1.2, 2.1, 3.2
    """

    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """
        Initialize General Agent

        Args:
            config: Agent configuration
            **kwargs: Additional configuration options
        """
        # Set default config if not provided
        if config is None:
            config = AgentConfig(
                name="general",
                enabled=True,
                timeout_seconds=30.0,
                retry_attempts=3,
                log_level="INFO"
            )

        super().__init__(config=config, **kwargs)

        # Initialize tools
        self.tools = {
            "check_glossary_requirement": check_glossary_requirement,
            "check_morningstar_date": check_morningstar_date,
            "check_source_citations": check_source_citations,
            "check_technical_terms": check_technical_terms
        }

        # Configuration
        self.parallel_execution = kwargs.get('parallel_execution', True)
        self.max_workers = kwargs.get('max_workers', 4)

        # Client type filtering
        self.apply_client_filtering = kwargs.get('apply_client_filtering', True)

        self.logger.info(f"General Agent initialized with {len(self.tools)} tools")
        self.logger.info(f"Parallel execution: {self.parallel_execution}")
        self.logger.info(f"Client type filtering: {self.apply_client_filtering}")

    def process(self, state: ComplianceState) -> ComplianceState:
        """
        Process general compliance checks

        Executes all general checking tools with client type filtering.
        Some checks only apply to retail documents, not professional.

        Args:
            state: Current compliance state

        Returns:
            Partial state update with violations and confidence scores
        """
        self.logger.info("Starting general compliance checks")

        # Get document and metadata
        document = state.get("normalized_document") or state.get("document", {})
        metadata = state.get("metadata", {})
        config = state.get("config", {})

        # Get client type for filtering
        client_type = metadata.get("client_type", "retail").lower()

        # Validate inputs
        if not document:
            self.logger.error("No document found in state")
            # Return partial state with error
            return {
                "error_log": [{
                    "agent": self.name,
                    "error": "No document found in state",
                    "timestamp": datetime.now().isoformat()
                }]
            }

        self.logger.info(f"Client type: {client_type}")

        # Determine which checks to run based on client type
        checks_to_run = self._determine_applicable_checks(client_type)
        self.logger.info(f"Running {len(checks_to_run)} applicable checks for {client_type} client")

        # Execute checks
        if self.parallel_execution:
            violations = self._execute_checks_parallel(document, metadata, config, client_type, checks_to_run)
        else:
            violations = self._execute_checks_sequential(document, metadata, config, client_type, checks_to_run)

        # Log results
        if violations:
            self.logger.info(f"Found {len(violations)} general violations")
        else:
            self.logger.info("No general violations found")

        # Calculate aggregate confidence
        if violations:
            avg_confidence = sum(v.get("confidence", 0) for v in violations) / len(violations)
            confidence = int(avg_confidence)
        else:
            confidence = 100

        self.logger.info(f"General checks completed. Violations: {len(violations)}")

        # Return only the fields this agent updates
        # This allows parallel execution without conflicts
        return {
            "violations": violations,
            "confidence_scores": {self.name: confidence}
        }

    def _determine_applicable_checks(self, client_type: str) -> List[str]:
        """
        Determine which checks apply based on client type

        Retail documents have stricter requirements than professional documents.

        Rules:
        - Glossary requirement: RETAIL ONLY (professional clients don't need glossary)
        - Morningstar date: ALL CLIENTS (if Morningstar rating is present)
        - Source citations: ALL CLIENTS (external data must be cited)
        - Technical terms: ALL CLIENTS (for information purposes)

        Args:
            client_type: Client type (retail/professional)

        Returns:
            List of check names to execute
        """
        if not self.apply_client_filtering:
            # If filtering disabled, run all checks
            return list(self.tools.keys())

        # Base checks that apply to all clients
        applicable_checks = [
            "check_morningstar_date",
            "check_source_citations",
            "check_technical_terms"
        ]

        # Additional checks for retail clients
        if client_type == "retail":
            applicable_checks.insert(0, "check_glossary_requirement")
            self.logger.info("Retail client: Including glossary requirement check")
        else:
            self.logger.info("Professional client: Skipping glossary requirement check")

        return applicable_checks

    def _execute_checks_parallel(
        self,
        document: dict,
        metadata: dict,
        config: dict,
        client_type: str,
        checks_to_run: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Execute applicable general checks in parallel

        Uses ThreadPoolExecutor to run checks concurrently for better performance.

        Args:
            document: Document to check
            metadata: Document metadata
            config: Configuration dictionary
            client_type: Client type (retail/professional)
            checks_to_run: List of check names to execute

        Returns:
            List of violations found
        """
        self.logger.info(f"Executing {len(checks_to_run)} checks in parallel")

        violations = []

        # Create tasks for each check
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_check = {}

            # Submit applicable checks
            for check_name in checks_to_run:
                if check_name == "check_glossary_requirement":
                    future = executor.submit(
                        self._safe_tool_invoke,
                        check_name,
                        document=document,
                        client_type=client_type
                    )
                    future_to_check[future] = check_name

                elif check_name == "check_morningstar_date":
                    future = executor.submit(
                        self._safe_tool_invoke,
                        check_name,
                        document=document
                    )
                    future_to_check[future] = check_name

                elif check_name == "check_source_citations":
                    future = executor.submit(
                        self._safe_tool_invoke,
                        check_name,
                        document=document
                    )
                    future_to_check[future] = check_name

                elif check_name == "check_technical_terms":
                    future = executor.submit(
                        self._safe_tool_invoke,
                        check_name,
                        document=document,
                        client_type=client_type
                    )
                    future_to_check[future] = check_name

            # Collect results as they complete
            for future in as_completed(future_to_check):
                check_name = future_to_check[future]
                try:
                    result = future.result()

                    # Handle different result types
                    if check_name == "check_technical_terms":
                        # This returns a list of terms, not a violation
                        if result:
                            self.logger.info(f"ℹ {check_name}: Found {len(result)} technical terms")
                        else:
                            self.logger.info(f"ℹ {check_name}: No technical terms found")
                    else:
                        # Regular violation check
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
        config: dict,
        client_type: str,
        checks_to_run: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Execute applicable general checks sequentially

        Runs checks one after another. Useful for debugging or when
        parallel execution is not desired.

        Args:
            document: Document to check
            metadata: Document metadata
            config: Configuration dictionary
            client_type: Client type (retail/professional)
            checks_to_run: List of check names to execute

        Returns:
            List of violations found
        """
        self.logger.info(f"Executing {len(checks_to_run)} checks sequentially")

        violations = []

        # Execute each applicable check
        for check_name in checks_to_run:
            if check_name == "check_glossary_requirement":
                result = self._safe_tool_invoke(
                    check_name,
                    document=document,
                    client_type=client_type
                )
                if result:
                    violations.append(result)
                    self.logger.info("✗ check_glossary_requirement: Violation found")
                else:
                    self.logger.info("✓ check_glossary_requirement: Pass")

            elif check_name == "check_morningstar_date":
                result = self._safe_tool_invoke(
                    check_name,
                    document=document
                )
                if result:
                    violations.append(result)
                    self.logger.info("✗ check_morningstar_date: Violation found")
                else:
                    self.logger.info("✓ check_morningstar_date: Pass")

            elif check_name == "check_source_citations":
                result = self._safe_tool_invoke(
                    check_name,
                    document=document
                )
                if result:
                    violations.append(result)
                    self.logger.info("✗ check_source_citations: Violation found")
                else:
                    self.logger.info("✓ check_source_citations: Pass")

            elif check_name == "check_technical_terms":
                result = self._safe_tool_invoke(
                    check_name,
                    document=document,
                    client_type=client_type
                )
                # This returns a list of terms, not a violation
                if result:
                    self.logger.info(f"ℹ check_technical_terms: Found {len(result)} technical terms")
                else:
                    self.logger.info("ℹ check_technical_terms: No technical terms found")

        return violations

    def _safe_tool_invoke(self, tool_name: str, **kwargs) -> Optional[Any]:
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

            # Add agent name to violation if present and it's a dict
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
                f"[{self.name}] Context: Checking general compliance requirements"
            )
            self.logger.debug(f"[{self.name}] Full error details:", exc_info=True)

            # Return error violation for violation-type checks
            if tool_name != "check_technical_terms":
                return {
                    "type": "GENERAL",
                    "severity": "ERROR",
                    "slide": "Unknown",
                    "location": "Unknown",
                    "rule": f"GEN_ERROR_{tool_name.upper()}",
                    "message": (
                        f"General check '{tool_name}' failed with {error_type}. "
                        f"Error: {error_msg}. "
                        f"This check could not be completed - results may be incomplete."
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
                            f"Check {tool_name} implementation. "
                            f"Verify document structure and required fields. "
                            f"Review error log for full traceback."
                        )
                    }
                }
            else:
                # For technical terms check, return empty list
                self.logger.warning(
                    f"[{self.name}] Technical terms check failed. Returning empty list."
                )
                return []

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

    def get_applicable_checks_for_client(self, client_type: str) -> List[str]:
        """
        Get list of checks that apply to a specific client type

        Args:
            client_type: Client type (retail/professional)

        Returns:
            List of applicable check names
        """
        return self._determine_applicable_checks(client_type)


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    """Test the General Agent with example data"""

    from data_models_multiagent import initialize_compliance_state

    logger.info("=" * 70)
    logger.info("GENERAL AGENT TEST")
    logger.info("=" * 70)

    # Create test document with technical terms and external data
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
            'subtitle': 'Systematic momentum strategy'
        },
        'slide_2': {
            'content': 'The fund uses quantitative analysis and smart momentum indicators. '
                      'Morningstar rating: ★★★★. Performance vs S&P 500 benchmark.'
        },
        'pages_suivantes': [
            {
                'slide_number': 3,
                'content': 'Volatility: 12%. Sharpe ratio: 1.5. Alpha: 2.3%. '
                          'Market data shows strong performance.'
            }
        ],
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management SAS'
        }
    }

    # Initialize state
    config = {
        'ai_enabled': True,
        'agents': {
            'general': {
                'enabled': True,
                'timeout_seconds': 30.0
            }
        }
    }

    # Test with RETAIL client
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: RETAIL CLIENT")
    logger.info("=" * 70)

    state = initialize_compliance_state(
        document=test_document,
        document_id="test_doc_retail",
        config=config
    )

    state["metadata"] = {
        'fund_name': 'ODDO BHF Algo Trend US Fund',
        'client_type': 'retail',
        'fund_isin': 'FR0010135103'
    }
    state["normalized_document"] = test_document

    # Create and run agent
    logger.info("\n1. Creating General Agent for RETAIL client...")
    agent_config = AgentConfig(
        name="general",
        enabled=True,
        timeout_seconds=30.0,
        log_level="INFO"
    )

    agent = GeneralAgent(config=agent_config, parallel_execution=True)
    logger.info(f"   Agent created: {agent}")
    logger.info(f"   Tools available: {agent.get_tool_count()}")
    logger.info(f"   Applicable checks: {', '.join(agent.get_applicable_checks_for_client('retail'))}")

    # Execute agent
    logger.info("\n2. Executing General Agent...")
    logger.info("-" * 70)

    result_state = agent(state)

    logger.info("-" * 70)

    # Display results
    logger.info("\n3. Results:")
    logger.info(f"   Violations found: {len(result_state.get('violations', []))}")
    logger.info(f"   Agent confidence: {result_state.get('confidence_scores', {}).get('general', 'N/A')}")
    logger.info(f"   Execution time: {result_state.get('agent_timings', {}).get('general', 'N/A'):.2f}s")

    if result_state.get('violations'):
        logger.info("\n4. Violations:")
        for i, violation in enumerate(result_state['violations'], 1):
            logger.info(f"\n   Violation {i}:")
            logger.info(f"      Rule: {violation.get('rule', 'N/A')}")
            logger.info(f"      Severity: {violation.get('severity', 'N/A')}")
            logger.info(f"      Message: {violation.get('message', 'N/A')}")
            logger.info(f"      Confidence: {violation.get('confidence', 'N/A')}%")
            logger.info(f"      Evidence: {violation.get('evidence', 'N/A')[:100]}...")
    else:
        logger.info("\n4. ✓ No violations found - document is compliant!")

    # Test with PROFESSIONAL client
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: PROFESSIONAL CLIENT")
    logger.info("=" * 70)

    test_document['document_metadata']['client_type'] = 'professional'

    state2 = initialize_compliance_state(
        document=test_document,
        document_id="test_doc_professional",
        config=config
    )

    state2["metadata"] = {
        'fund_name': 'ODDO BHF Algo Trend US Fund',
        'client_type': 'professional',
        'fund_isin': 'FR0010135103'
    }
    state2["normalized_document"] = test_document

    logger.info("\n1. Creating General Agent for PROFESSIONAL client...")
    logger.info(f"   Applicable checks: {', '.join(agent.get_applicable_checks_for_client('professional'))}")

    # Execute agent
    logger.info("\n2. Executing General Agent...")
    logger.info("-" * 70)

    result_state2 = agent(state2)

    logger.info("-" * 70)

    # Display results
    logger.info("\n3. Results:")
    logger.info(f"   Violations found: {len(result_state2.get('violations', []))}")
    logger.info(f"   Agent confidence: {result_state2.get('confidence_scores', {}).get('general', 'N/A')}")

    if result_state2.get('violations'):
        logger.info("\n4. Violations:")
        for i, violation in enumerate(result_state2['violations'], 1):
            logger.info(f"\n   Violation {i}:")
            logger.info(f"      Rule: {violation.get('rule', 'N/A')}")
            logger.info(f"      Message: {violation.get('message', 'N/A')}")
    else:
        logger.info("\n4. ✓ No violations found - document is compliant!")

    logger.info("\n" + "=" * 70)
    logger.info("General Agent test completed!")
    logger.info("=" * 70)

    # Display agent statistics
    logger.info("\n5. Agent Statistics:")
    agent.log_execution_stats()
