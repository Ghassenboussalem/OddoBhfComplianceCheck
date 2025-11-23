#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esg Agent

This module provides functionality for the multi-agent compliance system.
"""

"""
ESG Agent for Multi-Agent Compliance System

This agent handles ESG compliance checks including:
- ESG classification validation
- Content distribution analysis
- SFDR compliance checking
- ESG terminology validation

The agent integrates all ESG checking tools and executes them
for optimal performance with confidence scoring.

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
from tools.esg_tools import (
    check_esg_classification,
    check_content_distribution,
    check_sfdr_compliance,
    validate_esg_terminology,
    ESG_TOOLS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ESGAgent(BaseAgent):
    """
    ESG Agent - Handles all ESG compliance checks

    This agent is responsible for validating ESG-related requirements:
    - ESG classification compliance (Engaging, Reduced, Prospectus-limited, Other)
    - Content distribution validation (volume limits based on classification)
    - SFDR compliance checking (Article 6/8/9 consistency)
    - ESG terminology validation (appropriate use of ESG labels)

    The agent executes all checks with proper context and aggregates results
    with confidence scoring.

    Requirements: 1.2, 2.1, 3.2
    """

    def __init__(self, config: Optional[AgentConfig] = None, ai_engine=None, **kwargs):
        """
        Initialize ESG Agent

        Args:
            config: Agent configuration
            ai_engine: AI engine instance for content analysis
            **kwargs: Additional configuration options
        """
        # Set default config if not provided
        if config is None:
            config = AgentConfig(
                name="esg",
                enabled=True,
                timeout_seconds=45.0,  # ESG checks may take longer due to AI analysis
                retry_attempts=3,
                log_level="INFO"
            )

        super().__init__(config=config, **kwargs)

        # Store AI engine for content analysis
        self.ai_engine = ai_engine

        # Initialize tools
        self.tools = {
            "check_esg_classification": check_esg_classification,
            "check_content_distribution": check_content_distribution,
            "check_sfdr_compliance": check_sfdr_compliance,
            "validate_esg_terminology": validate_esg_terminology
        }

        # Configuration
        self.parallel_execution = kwargs.get('parallel_execution', False)  # Sequential by default for ESG
        self.max_workers = kwargs.get('max_workers', 4)

        self.logger.info(f"ESG Agent initialized with {len(self.tools)} tools")
        self.logger.info(f"AI Engine: {'Available' if self.ai_engine else 'Not available'}")
        self.logger.info(f"Parallel execution: {self.parallel_execution}")

    def process(self, state: ComplianceState) -> ComplianceState:
        """
        Process ESG compliance checks

        Executes all ESG checking tools and aggregates results.
        ESG checks are typically run sequentially as they may depend on
        each other's results and require AI analysis.

        Args:
            state: Current compliance state

        Returns:
            Partial state update with violations and confidence scores
        """
        self.logger.info("Starting ESG compliance checks")

        # Get document and metadata
        document = state.get("normalized_document") or state.get("document", {})
        metadata = state.get("metadata", {})
        config = state.get("config", {})

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

        # Extract ESG-specific metadata
        esg_classification = metadata.get("esg_classification", "other")
        client_type = metadata.get("client_type", "retail")
        sfdr_classification = metadata.get("sfdr_classification", "")

        self.logger.info(f"ESG Classification: {esg_classification}")
        self.logger.info(f"Client Type: {client_type}")
        self.logger.info(f"SFDR Classification: {sfdr_classification or 'Not specified'}")

        # Check if ESG checks should be skipped
        if self._should_skip_esg_checks(esg_classification, client_type):
            self.logger.info("ESG checks skipped (professional client or engaging approach)")
            return {
                "violations": [],
                "confidence_scores": {self.name: 100}
            }

        # Execute checks
        if self.parallel_execution:
            violations = self._execute_checks_parallel(
                document, metadata, config, esg_classification, client_type
            )
        else:
            violations = self._execute_checks_sequential(
                document, metadata, config, esg_classification, client_type
            )

        # Log results
        if violations:
            self.logger.info(f"Found {len(violations)} ESG violations")
            for v in violations:
                self.logger.info(f"  - {v.get('rule', 'Unknown')}: {v.get('message', '')}")
        else:
            self.logger.info("No ESG violations found")

        # Calculate aggregate confidence
        if violations:
            avg_confidence = sum(v.get("confidence", 0) for v in violations) / len(violations)
            confidence = int(avg_confidence)
        else:
            confidence = 100

        self.logger.info(f"ESG checks completed. Violations: {len(violations)}, Confidence: {confidence}%")

        # Return only the fields this agent updates
        return {
            "violations": violations,
            "confidence_scores": {self.name: confidence}
        }

    def _should_skip_esg_checks(self, esg_classification: str, client_type: str) -> bool:
        """
        Determine if ESG checks should be skipped

        ESG checks are skipped for:
        - Professional clients (exempt from ESG rules)
        - Engaging approach funds (no restrictions)

        Args:
            esg_classification: Fund's ESG classification
            client_type: Client type (retail/professional)

        Returns:
            True if checks should be skipped, False otherwise
        """
        classification_lower = esg_classification.lower()

        # Professional clients are exempt
        if client_type.lower() == 'professional':
            return True

        # Engaging approach has no restrictions (but we still validate terminology)
        # So we don't skip entirely, but some checks will pass quickly
        return False

    def _execute_checks_parallel(
        self,
        document: dict,
        metadata: dict,
        config: dict,
        esg_classification: str,
        client_type: str
    ) -> List[Dict[str, Any]]:
        """
        Execute all ESG checks in parallel

        Uses ThreadPoolExecutor to run checks concurrently.
        Note: ESG checks may have dependencies, so sequential execution
        is often preferred.

        Args:
            document: Document to check
            metadata: Document metadata
            config: Configuration dictionary
            esg_classification: Fund's ESG classification
            client_type: Client type

        Returns:
            List of violations found
        """
        self.logger.info(f"Executing {len(self.tools)} checks in parallel")

        violations = []

        # Create tasks for each check
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all checks
            future_to_check = {}

            # ESG classification check
            future = executor.submit(
                self._safe_tool_invoke,
                "check_esg_classification",
                document=document,
                esg_classification=esg_classification,
                client_type=client_type,
                ai_engine=self.ai_engine
            )
            future_to_check[future] = "check_esg_classification"

            # Content distribution check
            future = executor.submit(
                self._safe_tool_invoke,
                "check_content_distribution",
                document=document,
                esg_classification=esg_classification,
                client_type=client_type,
                ai_engine=self.ai_engine
            )
            future_to_check[future] = "check_content_distribution"

            # SFDR compliance check
            future = executor.submit(
                self._safe_tool_invoke,
                "check_sfdr_compliance",
                document=document,
                esg_classification=esg_classification,
                metadata=metadata,
                ai_engine=self.ai_engine
            )
            future_to_check[future] = "check_sfdr_compliance"

            # ESG terminology validation
            future = executor.submit(
                self._safe_tool_invoke,
                "validate_esg_terminology",
                document=document,
                esg_classification=esg_classification,
                client_type=client_type
            )
            future_to_check[future] = "validate_esg_terminology"

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
        config: dict,
        esg_classification: str,
        client_type: str
    ) -> List[Dict[str, Any]]:
        """
        Execute all ESG checks sequentially

        Runs checks one after another. This is the preferred method for ESG
        checks as they may have dependencies and require AI analysis.

        Args:
            document: Document to check
            metadata: Document metadata
            config: Configuration dictionary
            esg_classification: Fund's ESG classification
            client_type: Client type

        Returns:
            List of violations found
        """
        self.logger.info(f"Executing {len(self.tools)} checks sequentially")

        violations = []

        # 1. Check ESG classification
        self.logger.info("Running check_esg_classification...")
        result = self._safe_tool_invoke(
            "check_esg_classification",
            document=document,
            esg_classification=esg_classification,
            client_type=client_type,
            ai_engine=self.ai_engine
        )
        if result:
            violations.append(result)
            self.logger.info("✗ check_esg_classification: Violation found")
        else:
            self.logger.info("✓ check_esg_classification: Pass")

        # 2. Check content distribution (most important check)
        self.logger.info("Running check_content_distribution...")
        result = self._safe_tool_invoke(
            "check_content_distribution",
            document=document,
            esg_classification=esg_classification,
            client_type=client_type,
            ai_engine=self.ai_engine
        )
        if result:
            violations.append(result)
            self.logger.info("✗ check_content_distribution: Violation found")
        else:
            self.logger.info("✓ check_content_distribution: Pass")

        # 3. Check SFDR compliance
        self.logger.info("Running check_sfdr_compliance...")
        result = self._safe_tool_invoke(
            "check_sfdr_compliance",
            document=document,
            esg_classification=esg_classification,
            metadata=metadata,
            ai_engine=self.ai_engine
        )
        if result:
            violations.append(result)
            self.logger.info("✗ check_sfdr_compliance: Violation found")
        else:
            self.logger.info("✓ check_sfdr_compliance: Pass")

        # 4. Validate ESG terminology
        self.logger.info("Running validate_esg_terminology...")
        result = self._safe_tool_invoke(
            "validate_esg_terminology",
            document=document,
            esg_classification=esg_classification,
            client_type=client_type
        )
        if result:
            violations.append(result)
            self.logger.info("✗ validate_esg_terminology: Violation found")
        else:
            self.logger.info("✓ validate_esg_terminology: Pass")

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
                "type": "ESG",
                "severity": "ERROR",
                "slide": "Unknown",
                "location": "Unknown",
                "rule": f"ESG_ERROR: Error in {tool_name}",
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


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    """Test the ESG Agent with example data"""

    from data_models_multiagent import initialize_compliance_state

    logger.info("=" * 70)
    logger.info("ESG AGENT TEST")
    logger.info("=" * 70)

    # Test Case 1: Reduced approach with excessive ESG content
    logger.info("\n" + "=" * 70)
    logger.info("TEST CASE 1: Reduced Approach with Excessive ESG Content")
    logger.info("=" * 70)

    test_document_1 = {
        'document_metadata': {
            'fund_isin': 'FR0010135103',
            'fund_name': 'ODDO BHF Sustainable Fund',
            'client_type': 'retail',
            'document_type': 'fund_presentation',
            'fund_esg_classification': 'reduced',
            'sfdr_classification': 'article_8'
        },
        'page_de_garde': {
            'title': 'ODDO BHF Sustainable Fund',
            'subtitle': 'ESG Integration Strategy'
        },
        'slide_2': {
            'content': '''ESG Integration Methodology

            Our fund integrates ESG factors through:
            - Environmental analysis: carbon footprint, climate risks
            - Social criteria: labor practices, human rights
            - Governance assessment: board diversity, executive compensation

            We use proprietary ESG scoring and engage with companies on sustainability.'''
        },
        'pages_suivantes': [
            {
                'slide_number': 3,
                'content': '''ESG Performance Metrics

                ESG Score: 8.5/10
                Carbon Intensity: 50% below benchmark
                Sustainable Development Goals alignment: 7 SDGs
                Impact measurement and reporting quarterly'''
            },
            {
                'slide_number': 4,
                'content': '''Investment Strategy

                Traditional investment approach with some ESG considerations.
                Focus on financial performance.'''
            }
        ],
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management SAS'
        }
    }

    # Initialize state
    config_1 = {
        'ai_enabled': True,
        'agents': {
            'esg': {
                'enabled': True,
                'timeout_seconds': 45.0
            }
        }
    }

    state_1 = initialize_compliance_state(
        document=test_document_1,
        document_id="test_esg_001",
        config=config_1
    )

    state_1["metadata"] = {
        'fund_name': 'ODDO BHF Sustainable Fund',
        'client_type': 'retail',
        'fund_isin': 'FR0010135103',
        'esg_classification': 'reduced',
        'sfdr_classification': 'article_8'
    }
    state_1["normalized_document"] = test_document_1

    # Create and run agent (without AI engine for this test)
    logger.info("\n1. Creating ESG Agent...")
    agent_config_1 = AgentConfig(
        name="esg",
        enabled=True,
        timeout_seconds=45.0,
        log_level="INFO"
    )

    agent_1 = ESGAgent(config=agent_config_1, ai_engine=None, parallel_execution=False)
    logger.info(f"   Agent created: {agent_1}")
    logger.info(f"   Tools available: {agent_1.get_tool_count()}")
    logger.info(f"   Tool names: {', '.join(agent_1.get_tool_names())}")

    logger.info("\n2. Executing ESG Agent...")
    logger.info("-" * 70)

    result_state_1 = agent_1(state_1)

    logger.info("-" * 70)

    logger.info("\n3. Results:")
    logger.info(f"   Violations found: {len(result_state_1.get('violations', []))}")
    logger.info(f"   Agent confidence: {result_state_1.get('confidence_scores', {}).get('esg', 'N/A')}")

    if result_state_1.get('violations'):
        logger.info("\n4. Violations:")
        for i, violation in enumerate(result_state_1['violations'], 1):
            logger.info(f"\n   Violation {i}:")
            logger.info(f"      Rule: {violation.get('rule', 'N/A')}")
            logger.info(f"      Severity: {violation.get('severity', 'N/A')}")
            logger.info(f"      Message: {violation.get('message', 'N/A')}")
            logger.info(f"      Confidence: {violation.get('confidence', 'N/A')}%")

    # Test Case 2: Engaging approach (should pass)
    logger.info("\n\n" + "=" * 70)
    logger.info("TEST CASE 2: Engaging Approach (Should Pass)")
    logger.info("=" * 70)

    test_document_2 = {
        'document_metadata': {
            'fund_isin': 'FR0010135104',
            'fund_name': 'ODDO BHF Impact Fund',
            'client_type': 'retail',
            'document_type': 'fund_presentation',
            'fund_esg_classification': 'engaging',
            'sfdr_classification': 'article_9'
        },
        'page_de_garde': {
            'title': 'ODDO BHF Impact Fund',
            'subtitle': 'Sustainable Investment with Measurable Impact'
        },
        'slide_2': {
            'content': 'Comprehensive ESG integration with impact measurement...'
        },
        'pages_suivantes': [],
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management SAS'
        }
    }

    state_2 = initialize_compliance_state(
        document=test_document_2,
        document_id="test_esg_002",
        config=config_1
    )

    state_2["metadata"] = {
        'fund_name': 'ODDO BHF Impact Fund',
        'client_type': 'retail',
        'fund_isin': 'FR0010135104',
        'esg_classification': 'engaging',
        'sfdr_classification': 'article_9'
    }
    state_2["normalized_document"] = test_document_2

    logger.info("\n1. Executing ESG Agent for Engaging fund...")
    logger.info("-" * 70)

    result_state_2 = agent_1(state_2)

    logger.info("-" * 70)

    logger.info("\n2. Results:")
    logger.info(f"   Violations found: {len(result_state_2.get('violations', []))}")
    logger.info(f"   Agent confidence: {result_state_2.get('confidence_scores', {}).get('esg', 'N/A')}")

    if result_state_2.get('violations'):
        logger.info("\n3. Violations:")
        for i, violation in enumerate(result_state_2['violations'], 1):
            logger.info(f"\n   Violation {i}:")
            logger.info(f"      Rule: {violation.get('rule', 'N/A')}")
            logger.info(f"      Message: {violation.get('message', 'N/A')}")
    else:
        logger.info("\n3. ✓ No violations found - Engaging approach allows unlimited ESG content!")

    logger.info("\n" + "=" * 70)
    logger.info("ESG Agent test completed!")
    logger.info("=" * 70)

    # Display agent statistics
    logger.info("\n4. Agent Statistics:")
    agent_1.log_execution_stats()
