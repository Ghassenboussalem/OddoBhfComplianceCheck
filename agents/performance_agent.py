#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Agent

This module provides functionality for the multi-agent compliance system.
"""

"""
Performance Agent for Multi-Agent Compliance System

This agent handles performance compliance checks including:
- Performance disclaimers (data-aware version)
- Document starts with performance check
- Benchmark comparison validation
- Fund age restrictions

The agent integrates all performance checking tools and executes them
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
from tools.performance_tools import (
    check_performance_disclaimers,
    check_document_starts_with_performance,
    check_benchmark_comparison,
    check_fund_age_restrictions,
    PERFORMANCE_TOOLS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceAgent(BaseAgent):
    """
    Performance Agent - Handles all performance compliance checks

    This agent is responsible for validating performance-related requirements:
    - Performance data must have disclaimers (data-aware detection)
    - Document cannot start with performance data
    - Performance must be compared to benchmark
    - Fund age restrictions for performance display

    The agent executes all checks in parallel for optimal performance and
    aggregates results with confidence scoring. It integrates evidence
    extraction to eliminate false positives by distinguishing actual
    performance data from descriptive keywords.

    Requirements: 1.2, 2.1, 3.2
    """

    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """
        Initialize Performance Agent

        Args:
            config: Agent configuration
            **kwargs: Additional configuration options
        """
        # Set default config if not provided
        if config is None:
            config = AgentConfig(
                name="performance",
                enabled=True,
                timeout_seconds=30.0,
                retry_attempts=3,
                log_level="INFO"
            )

        super().__init__(config=config, **kwargs)

        # Initialize tools
        self.tools = {
            "check_performance_disclaimers": check_performance_disclaimers,
            "check_document_starts_with_performance": check_document_starts_with_performance,
            "check_benchmark_comparison": check_benchmark_comparison,
            "check_fund_age_restrictions": check_fund_age_restrictions
        }

        # Configuration
        self.parallel_execution = kwargs.get('parallel_execution', True)
        self.max_workers = kwargs.get('max_workers', 4)

        self.logger.info(f"Performance Agent initialized with {len(self.tools)} tools")
        self.logger.info(f"Parallel execution: {self.parallel_execution}")

    def process(self, state: ComplianceState) -> ComplianceState:
        """
        Process performance compliance checks

        Executes all performance checking tools and aggregates results.
        Tools can run in parallel for better performance.

        This agent uses evidence extraction integration to:
        - Detect ACTUAL performance data (numbers with %) vs descriptive keywords
        - Find disclaimers using semantic matching
        - Eliminate false positives through context understanding

        Args:
            state: Current compliance state

        Returns:
            Partial state update with violations and confidence scores
        """
        self.logger.info("Starting performance compliance checks")

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
                f"[{self.name}] Cannot perform performance checks without document data."
            )
            self.logger.error(
                f"[{self.name}] Resolution: Ensure preprocessor agent ran successfully. "
                f"Verify workflow execution order is correct."
            )
            # Return partial state with error
            return {
                "error_log": [{
                    "agent": self.name,
                    "error": "No document found in state - cannot perform performance checks",
                    "error_type": "MissingDocumentError",
                    "severity": "CRITICAL",
                    "timestamp": datetime.now().isoformat(),
                    "document_id": doc_id,
                    "workflow_status": state.get('workflow_status', 'unknown'),
                    "resolution_hint": (
                        "Verify preprocessor agent completed successfully. "
                        "Check that document is loaded in state['document'] or state['normalized_document']. "
                        "Review workflow execution logs for preprocessing errors."
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
            self.logger.info(f"Found {len(violations)} performance violations")
        else:
            self.logger.info("No performance violations found")

        # Calculate aggregate confidence
        if violations:
            avg_confidence = sum(v.get("confidence", 0) for v in violations) / len(violations)
            confidence = int(avg_confidence)
        else:
            confidence = 100

        self.logger.info(f"Performance checks completed. Violations: {len(violations)}")

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
        Execute all performance checks in parallel

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

            # Performance disclaimers check (returns list of violations)
            future = executor.submit(
                self._safe_tool_invoke,
                "check_performance_disclaimers",
                document=document,
                config=config,
                returns_list=True
            )
            future_to_check[future] = "check_performance_disclaimers"

            # Document starts with performance check (returns list)
            future = executor.submit(
                self._safe_tool_invoke,
                "check_document_starts_with_performance",
                document=document,
                config=config,
                returns_list=True
            )
            future_to_check[future] = "check_document_starts_with_performance"

            # Benchmark comparison check (returns single violation or None)
            future = executor.submit(
                self._safe_tool_invoke,
                "check_benchmark_comparison",
                document=document,
                config=config,
                returns_list=False
            )
            future_to_check[future] = "check_benchmark_comparison"

            # Fund age restrictions check (returns single violation or None)
            future = executor.submit(
                self._safe_tool_invoke,
                "check_fund_age_restrictions",
                document=document,
                metadata=metadata,
                returns_list=False
            )
            future_to_check[future] = "check_fund_age_restrictions"

            # Collect results as they complete
            for future in as_completed(future_to_check):
                check_name = future_to_check[future]
                try:
                    result = future.result()
                    if result:
                        if isinstance(result, list):
                            # Multiple violations
                            self.logger.info(f"✗ {check_name}: {len(result)} violation(s) found")
                            violations.extend(result)
                        else:
                            # Single violation
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
        Execute all performance checks sequentially

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

        # Check performance disclaimers (returns list)
        result = self._safe_tool_invoke(
            "check_performance_disclaimers",
            document=document,
            config=config,
            returns_list=True
        )
        if result:
            violations.extend(result)
            self.logger.info(f"✗ check_performance_disclaimers: {len(result)} violation(s) found")
        else:
            self.logger.info("✓ check_performance_disclaimers: Pass")

        # Check document starts with performance (returns list)
        result = self._safe_tool_invoke(
            "check_document_starts_with_performance",
            document=document,
            config=config,
            returns_list=True
        )
        if result:
            violations.extend(result)
            self.logger.info(f"✗ check_document_starts_with_performance: {len(result)} violation(s) found")
        else:
            self.logger.info("✓ check_document_starts_with_performance: Pass")

        # Check benchmark comparison (returns single or None)
        result = self._safe_tool_invoke(
            "check_benchmark_comparison",
            document=document,
            config=config,
            returns_list=False
        )
        if result:
            violations.append(result)
            self.logger.info("✗ check_benchmark_comparison: Violation found")
        else:
            self.logger.info("✓ check_benchmark_comparison: Pass")

        # Check fund age restrictions (returns single or None)
        result = self._safe_tool_invoke(
            "check_fund_age_restrictions",
            document=document,
            metadata=metadata,
            returns_list=False
        )
        if result:
            violations.append(result)
            self.logger.info("✗ check_fund_age_restrictions: Violation found")
        else:
            self.logger.info("✓ check_fund_age_restrictions: Pass")

        return violations

    def _safe_tool_invoke(
        self,
        tool_name: str,
        returns_list: bool = False,
        **kwargs
    ) -> Optional[Any]:
        """
        Safely invoke a tool with error handling

        Args:
            tool_name: Name of the tool to invoke
            returns_list: Whether the tool returns a list of violations
            **kwargs: Arguments to pass to the tool

        Returns:
            Tool result (list or single violation) or None if error occurred
        """
        try:
            tool = self.tools.get(tool_name)
            if not tool:
                self.logger.error(f"Tool not found: {tool_name}")
                return [] if returns_list else None

            # Invoke the tool
            result = tool.func(**kwargs)

            # Add agent name and timestamp to violations
            if result:
                if isinstance(result, list):
                    # Multiple violations
                    for violation in result:
                        if isinstance(violation, dict):
                            violation["agent"] = self.name
                            if "timestamp" not in violation:
                                violation["timestamp"] = datetime.now().isoformat()
                elif isinstance(result, dict):
                    # Single violation
                    result["agent"] = self.name
                    if "timestamp" not in result:
                        result["timestamp"] = datetime.now().isoformat()

            return result

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)[:200]
            self.logger.error(
                f"[{self.name}] ❌ Tool execution failed: {tool_name} - {error_type}: {error_msg}"
            )
            self.logger.error(
                f"[{self.name}] Context: Checking performance compliance"
            )
            self.logger.debug(f"[{self.name}] Full error details:", exc_info=True)

            # Return error violation with detailed context
            error_violation = {
                "type": "PERFORMANCE",
                "severity": "ERROR",
                "slide": "Unknown",
                "location": "Unknown",
                "rule": f"PERF_ERROR_{tool_name.upper()}",
                "message": (
                    f"Performance check '{tool_name}' failed with {error_type}. "
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
                        f"Verify document contains performance data in expected format. "
                        f"If using AI features, check API connectivity and credentials. "
                        f"Review error log for full traceback."
                    )
                }
            }

            return [error_violation] if returns_list else error_violation

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
    """Test the Performance Agent with example data"""

    from data_models_multiagent import initialize_compliance_state

    logger.info("=" * 70)
    logger.info("PERFORMANCE AGENT TEST")
    logger.info("=" * 70)

    # Create test document with performance data
    test_document = {
        'document_metadata': {
            'fund_isin': 'FR0010135103',
            'fund_name': 'ODDO BHF Algo Trend US Fund',
            'client_type': 'retail',
            'document_type': 'fund_presentation',
            'fund_inception_date': '2020-01-15'
        },
        'page_de_garde': {
            'title': 'ODDO BHF Algo Trend US Fund',
            'subtitle': 'Document promotionnel'
        },
        'slide_2': {
            'title': 'Performance Overview',
            'content': 'Annual performance: +15.5% in 2023, +10.2% in 2022. Les performances passées ne préjugent pas des performances futures.'
        },
        'pages_suivantes': [
            {
                'slide_number': 3,
                'title': 'Performance History',
                'content': 'Detailed returns: 2023: +15.5%, 2022: +10.2%, 2021: +8.7%. Benchmark (S&P 500): 2023: +12.1%, 2022: +9.5%, 2021: +7.3%. Chart showing comparison.'
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
            'performance': {
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
        'fund_isin': 'FR0010135103',
        'fund_age_years': 4.5,
        'fund_inception_date': '2020-01-15'
    }
    state["normalized_document"] = test_document

    # Create and run agent
    logger.info("\n1. Creating Performance Agent...")
    agent_config = AgentConfig(
        name="performance",
        enabled=True,
        timeout_seconds=30.0,
        log_level="INFO"
    )

    agent = PerformanceAgent(config=agent_config, parallel_execution=True)
    logger.info(f"   Agent created: {agent}")
    logger.info(f"   Tools available: {agent.get_tool_count()}")
    logger.info(f"   Tool names: {', '.join(agent.get_tool_names())}")

    # Execute agent
    logger.info("\n2. Executing Performance Agent...")
    logger.info("-" * 70)

    result_state = agent(state)

    logger.info("-" * 70)

    # Display results
    logger.info("\n3. Results:")
    logger.info(f"   Violations found: {len(result_state.get('violations', []))}")
    logger.info(f"   Agent confidence: {result_state.get('confidence_scores', {}).get('performance', 'N/A')}")

    agent_timing = result_state.get('agent_timings', {}).get('performance')
    if agent_timing:
        logger.info(f"   Execution time: {agent_timing:.2f}s")

    if result_state.get('violations'):
        logger.info("\n4. Violations:")
        for i, violation in enumerate(result_state['violations'], 1):
            logger.info(f"\n   Violation {i}:")
            logger.info(f"      Rule: {violation.get('rule', 'N/A')}")
            logger.info(f"      Severity: {violation.get('severity', 'N/A')}")
            logger.info(f"      Message: {violation.get('message', 'N/A')}")
            logger.info(f"      Confidence: {violation.get('confidence', 'N/A')}%")
            logger.info(f"      Location: {violation.get('slide', 'N/A')} / {violation.get('location', 'N/A')}")
            logger.info(f"      Method: {violation.get('method', 'N/A')}")
            if violation.get('evidence'):
                evidence = violation['evidence']
                if len(evidence) > 100:
                    evidence = evidence[:100] + "..."
                logger.info(f"      Evidence: {evidence}")
    else:
        logger.info("\n4. ✓ No violations found - performance data is compliant!")

    # Display errors if any
    if result_state.get('error_log'):
        logger.info("\n5. Errors:")
        for error in result_state['error_log']:
            logger.info(f"   - {error.get('error', 'Unknown error')}")

    logger.info("\n" + "=" * 70)
    logger.info("Performance Agent test completed!")
    logger.info("=" * 70)

    # Display agent statistics
    logger.info("\n6. Agent Statistics:")
    agent.log_execution_stats()
