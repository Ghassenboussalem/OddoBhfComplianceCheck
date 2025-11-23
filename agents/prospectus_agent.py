#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prospectus Agent

This module provides functionality for the multi-agent compliance system.
"""

"""
Prospectus Agent for Multi-Agent Compliance System

This agent handles prospectus compliance checks including:
- Fund name semantic matching
- Investment strategy consistency
- Benchmark validation
- Investment objective validation

The agent integrates all prospectus checking tools and executes them
with semantic similarity matching and contradiction detection.

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
from tools.prospectus_tools import (
    check_fund_name_match,
    check_strategy_consistency,
    check_benchmark_validation,
    check_investment_objective,
    PROSPECTUS_TOOLS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProspectusAgent(BaseAgent):
    """
    Prospectus Agent - Handles all prospectus compliance checks

    This agent is responsible for validating that marketing documents
    are consistent with the official prospectus:
    - Fund name matches prospectus (semantic matching)
    - Investment strategy is consistent (contradiction detection)
    - Benchmark matches prospectus benchmark
    - Investment objective aligns with prospectus

    The agent uses semantic similarity matching to detect contradictions
    (not just missing details) and provides confidence scoring for each check.

    Requirements: 1.2, 2.1, 3.2
    """

    def __init__(self, config: Optional[AgentConfig] = None, ai_engine=None, **kwargs):
        """
        Initialize Prospectus Agent

        Args:
            config: Agent configuration
            ai_engine: AI engine for semantic analysis
            **kwargs: Additional configuration options
        """
        # Set default config if not provided
        if config is None:
            config = AgentConfig(
                name="prospectus",
                enabled=True,
                timeout_seconds=45.0,  # Longer timeout for AI calls
                retry_attempts=3,
                log_level="INFO"
            )

        super().__init__(config=config, **kwargs)

        # Store AI engine for semantic analysis
        self.ai_engine = ai_engine

        # Initialize tools
        self.tools = {
            "check_fund_name_match": check_fund_name_match,
            "check_strategy_consistency": check_strategy_consistency,
            "check_benchmark_validation": check_benchmark_validation,
            "check_investment_objective": check_investment_objective
        }

        # Configuration
        self.parallel_execution = kwargs.get('parallel_execution', True)
        self.max_workers = kwargs.get('max_workers', 4)
        self.semantic_matching_enabled = kwargs.get('semantic_matching_enabled', True)

        self.logger.info(f"Prospectus Agent initialized with {len(self.tools)} tools")
        self.logger.info(f"Parallel execution: {self.parallel_execution}")
        self.logger.info(f"Semantic matching: {self.semantic_matching_enabled}")
        self.logger.info(f"AI engine available: {self.ai_engine is not None}")


    def process(self, state: ComplianceState) -> ComplianceState:
        """
        Process prospectus compliance checks

        Executes all prospectus checking tools with semantic similarity matching
        and contradiction detection. Only runs if prospectus data is available.

        Args:
            state: Current compliance state

        Returns:
            Partial state update with violations and confidence scores
        """
        self.logger.info("Starting prospectus compliance checks")

        # Get document and configuration
        document = state.get("normalized_document") or state.get("document", {})
        metadata = state.get("metadata", {})
        config = state.get("config", {})

        # Get prospectus data from config
        prospectus_data = config.get("prospectus_data", {})

        # Validate inputs
        if not document:
            self.logger.error("No document found in state")
            return {
                "error_log": [{
                    "agent": self.name,
                    "error": "No document found in state",
                    "timestamp": datetime.now().isoformat()
                }]
            }

        # Check if prospectus data is available
        if not prospectus_data:
            self.logger.info("No prospectus data available, skipping prospectus checks")
            return {
                "confidence_scores": {self.name: 100}
            }

        self.logger.info(f"Prospectus data available: {list(prospectus_data.keys())}")

        # Execute checks
        if self.parallel_execution:
            violations = self._execute_checks_parallel(document, metadata, prospectus_data)
        else:
            violations = self._execute_checks_sequential(document, metadata, prospectus_data)

        # Log results
        if violations:
            self.logger.info(f"Found {len(violations)} prospectus violations")
            for v in violations:
                self.logger.warning(f"  - {v.get('rule', 'Unknown')}: {v.get('message', 'No message')}")
        else:
            self.logger.info("No prospectus violations found - document is consistent with prospectus")

        # Calculate aggregate confidence
        if violations:
            avg_confidence = sum(v.get("confidence", 0) for v in violations) / len(violations)
            confidence = int(avg_confidence)
        else:
            confidence = 100

        self.logger.info(f"Prospectus checks completed. Violations: {len(violations)}, Confidence: {confidence}%")

        # Return only the fields this agent updates
        return {
            "violations": violations,
            "confidence_scores": {self.name: confidence}
        }

    def _execute_checks_parallel(
        self,
        document: dict,
        metadata: dict,
        prospectus_data: dict
    ) -> List[Dict[str, Any]]:
        """
        Execute all prospectus checks in parallel

        Uses ThreadPoolExecutor to run checks concurrently for better performance.
        Each check uses semantic similarity matching when AI engine is available.

        Args:
            document: Document to check
            metadata: Document metadata
            prospectus_data: Prospectus data for comparison

        Returns:
            List of violations found
        """
        self.logger.info(f"Executing {len(self.tools)} checks in parallel")

        violations = []

        # Create tasks for each check
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all checks
            future_to_check = {}

            # Fund name match check
            if prospectus_data.get('fund_name'):
                future = executor.submit(
                    self._safe_tool_invoke,
                    "check_fund_name_match",
                    document=document,
                    prospectus_data=prospectus_data,
                    ai_engine=self.ai_engine if self.semantic_matching_enabled else None
                )
                future_to_check[future] = "check_fund_name_match"

            # Strategy consistency check
            if prospectus_data.get('strategy'):
                future = executor.submit(
                    self._safe_tool_invoke,
                    "check_strategy_consistency",
                    document=document,
                    prospectus_data=prospectus_data,
                    ai_engine=self.ai_engine if self.semantic_matching_enabled else None
                )
                future_to_check[future] = "check_strategy_consistency"

            # Benchmark validation check
            if prospectus_data.get('benchmark'):
                future = executor.submit(
                    self._safe_tool_invoke,
                    "check_benchmark_validation",
                    document=document,
                    prospectus_data=prospectus_data,
                    ai_engine=self.ai_engine if self.semantic_matching_enabled else None
                )
                future_to_check[future] = "check_benchmark_validation"

            # Investment objective check
            if prospectus_data.get('investment_objective'):
                future = executor.submit(
                    self._safe_tool_invoke,
                    "check_investment_objective",
                    document=document,
                    prospectus_data=prospectus_data,
                    ai_engine=self.ai_engine if self.semantic_matching_enabled else None
                )
                future_to_check[future] = "check_investment_objective"

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
        prospectus_data: dict
    ) -> List[Dict[str, Any]]:
        """
        Execute all prospectus checks sequentially

        Runs checks one after another. Useful for debugging or when
        parallel execution is not desired.

        Args:
            document: Document to check
            metadata: Document metadata
            prospectus_data: Prospectus data for comparison

        Returns:
            List of violations found
        """
        self.logger.info(f"Executing {len(self.tools)} checks sequentially")

        violations = []

        # Check fund name match
        if prospectus_data.get('fund_name'):
            result = self._safe_tool_invoke(
                "check_fund_name_match",
                document=document,
                prospectus_data=prospectus_data,
                ai_engine=self.ai_engine if self.semantic_matching_enabled else None
            )
            if result:
                violations.append(result)
                self.logger.info("✗ check_fund_name_match: Violation found")
            else:
                self.logger.info("✓ check_fund_name_match: Pass")

        # Check strategy consistency
        if prospectus_data.get('strategy'):
            result = self._safe_tool_invoke(
                "check_strategy_consistency",
                document=document,
                prospectus_data=prospectus_data,
                ai_engine=self.ai_engine if self.semantic_matching_enabled else None
            )
            if result:
                violations.append(result)
                self.logger.info("✗ check_strategy_consistency: Violation found")
            else:
                self.logger.info("✓ check_strategy_consistency: Pass")

        # Check benchmark validation
        if prospectus_data.get('benchmark'):
            result = self._safe_tool_invoke(
                "check_benchmark_validation",
                document=document,
                prospectus_data=prospectus_data,
                ai_engine=self.ai_engine if self.semantic_matching_enabled else None
            )
            if result:
                violations.append(result)
                self.logger.info("✗ check_benchmark_validation: Violation found")
            else:
                self.logger.info("✓ check_benchmark_validation: Pass")

        # Check investment objective
        if prospectus_data.get('investment_objective'):
            result = self._safe_tool_invoke(
                "check_investment_objective",
                document=document,
                prospectus_data=prospectus_data,
                ai_engine=self.ai_engine if self.semantic_matching_enabled else None
            )
            if result:
                violations.append(result)
                self.logger.info("✗ check_investment_objective: Violation found")
            else:
                self.logger.info("✓ check_investment_objective: Pass")

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
                if "timestamp" not in result:
                    result["timestamp"] = datetime.now().isoformat()

            return result

        except Exception as e:
            self.logger.error(f"Error invoking tool {tool_name}: {e}", exc_info=True)

            # Return error violation
            return {
                "type": "PROSPECTUS",
                "severity": "ERROR",
                "slide": "Unknown",
                "location": "Unknown",
                "rule": f"PROSP_ERROR: Error in {tool_name}",
                "message": f"Error executing {tool_name}: {str(e)}",
                "evidence": "Tool execution failed",
                "confidence": 0,
                "method": "ERROR",
                "agent": self.name,
                "timestamp": datetime.now().isoformat()
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

    def set_ai_engine(self, ai_engine):
        """
        Set or update the AI engine for semantic analysis

        Args:
            ai_engine: AI engine instance
        """
        self.ai_engine = ai_engine
        self.logger.info(f"AI engine {'set' if ai_engine else 'cleared'}")

    def enable_semantic_matching(self, enabled: bool = True):
        """
        Enable or disable semantic matching

        Args:
            enabled: True to enable, False to disable
        """
        self.semantic_matching_enabled = enabled
        self.logger.info(f"Semantic matching {'enabled' if enabled else 'disabled'}")


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    """Test the Prospectus Agent with example data"""

    from data_models_multiagent import initialize_compliance_state

    logger.info("=" * 70)
    logger.info("PROSPECTUS AGENT TEST")
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
            'subtitle': 'Growth-oriented equity fund'
        },
        'slide_2': {
            'content': 'The fund invests in US equities with a focus on growth stocks. Performance is measured against the S&P 500 index.'
        },
        'pages_suivantes': [
            {
                'content': 'Investment objective: Seeks long-term capital appreciation through investments in US growth stocks.'
            }
        ],
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management SAS'
        }
    }

    # Create prospectus data for comparison
    prospectus_data = {
        'fund_name': 'ODDO BHF Algo Trend US Fund',
        'strategy': 'The fund invests at least 70% of its assets in US equity securities with growth characteristics.',
        'benchmark': 'S&P 500 Net Total Return Index',
        'investment_objective': 'The fund seeks long-term capital appreciation by investing primarily in US growth stocks.'
    }

    # Initialize state
    config = {
        'ai_enabled': True,
        'prospectus_data': prospectus_data,
        'agents': {
            'prospectus': {
                'enabled': True,
                'timeout_seconds': 45.0
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

    # Create and run agent (without AI engine for basic test)
    logger.info("\n1. Creating Prospectus Agent...")
    agent_config = AgentConfig(
        name="prospectus",
        enabled=True,
        timeout_seconds=45.0,
        log_level="INFO"
    )

    agent = ProspectusAgent(
        config=agent_config,
        ai_engine=None,  # No AI engine for basic test
        parallel_execution=True,
        semantic_matching_enabled=False
    )
    logger.info(f"   Agent created: {agent}")
    logger.info(f"   Tools available: {agent.get_tool_count()}")
    logger.info(f"   Tool names: {', '.join(agent.get_tool_names())}")

    # Execute agent
    logger.info("\n2. Executing Prospectus Agent...")
    logger.info("-" * 70)

    result_state = agent(state)

    logger.info("-" * 70)

    # Display results
    logger.info("\n3. Results:")
    logger.info(f"   Violations found: {len(result_state.get('violations', []))}")
    logger.info(f"   Agent confidence: {result_state.get('confidence_scores', {}).get('prospectus', 'N/A')}")
    logger.info(f"   Execution time: {result_state.get('agent_timings', {}).get('prospectus', 'N/A'):.2f}s")

    if result_state.get('violations'):
        logger.info("\n4. Violations:")
        for i, violation in enumerate(result_state['violations'], 1):
            logger.info(f"\n   Violation {i}:")
            logger.info(f"      Rule: {violation.get('rule', 'N/A')}")
            logger.info(f"      Severity: {violation.get('severity', 'N/A')}")
            logger.info(f"      Message: {violation.get('message', 'N/A')}")
            logger.info(f"      Confidence: {violation.get('confidence', 'N/A')}%")
            logger.info(f"      Method: {violation.get('method', 'N/A')}")
            if violation.get('ai_reasoning'):
                logger.info(f"      AI Reasoning: {violation.get('ai_reasoning', 'N/A')}")
    else:
        logger.info("\n4. ✓ No violations found - document is consistent with prospectus!")

    # Display errors if any
    if result_state.get('error_log'):
        logger.info("\n5. Errors:")
        for error in result_state['error_log']:
            logger.info(f"   - {error.get('error', 'Unknown error')}")

    logger.info("\n" + "=" * 70)
    logger.info("Prospectus Agent test completed!")
    logger.info("=" * 70)

    # Display agent statistics
    logger.info("\n6. Agent Statistics:")
    agent.log_execution_stats()

    # Test with contradictory data
    logger.info("\n" + "=" * 70)
    logger.info("TESTING WITH CONTRADICTORY DATA")
    logger.info("=" * 70)

    # Create contradictory document
    contradictory_document = {
        'document_metadata': {
            'fund_isin': 'FR0010135103',
            'fund_name': 'Different Fund Name',  # Contradiction
            'client_type': 'retail',
            'document_type': 'fund_presentation'
        },
        'page_de_garde': {
            'title': 'Different Fund Name',
            'subtitle': 'Income-focused bond fund'  # Contradiction
        },
        'slide_2': {
            'content': 'The fund invests exclusively in European bonds.'  # Contradiction
        },
        'pages_suivantes': [
            {
                'content': 'Investment objective: Seeks regular income through bond investments.'  # Contradiction
            }
        ],
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management SAS'
        }
    }

    # Create new state with contradictory document
    state2 = initialize_compliance_state(
        document=contradictory_document,
        document_id="test_doc_002",
        config=config
    )
    state2["metadata"] = {
        'fund_name': 'Different Fund Name',
        'client_type': 'retail',
        'fund_isin': 'FR0010135103'
    }
    state2["normalized_document"] = contradictory_document

    # Reset agent stats
    agent.reset_stats()

    # Execute agent
    logger.info("\n7. Executing with contradictory data...")
    logger.info("-" * 70)

    result_state2 = agent(state2)

    logger.info("-" * 70)

    # Display results
    logger.info("\n8. Results:")
    logger.info(f"   Violations found: {len(result_state2.get('violations', []))}")
    logger.info(f"   Agent confidence: {result_state2.get('confidence_scores', {}).get('prospectus', 'N/A')}")

    if result_state2.get('violations'):
        logger.info("\n9. Violations (should detect contradictions):")
        for i, violation in enumerate(result_state2['violations'], 1):
            logger.info(f"\n   Violation {i}:")
            logger.info(f"      Rule: {violation.get('rule', 'N/A')}")
            logger.info(f"      Message: {violation.get('message', 'N/A')}")
            logger.info(f"      Evidence: {violation.get('evidence', 'N/A')[:100]}...")

    logger.info("\n" + "=" * 70)
