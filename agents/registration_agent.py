#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registration Agent

This module provides functionality for the multi-agent compliance system.
"""

"""
Registration Agent for Multi-Agent Compliance System

This agent handles registration compliance checks including:
- Country authorization validation
- Country extraction from documents
- Fund registration validation
- Country name variation matching

The agent integrates all registration checking tools and validates
that countries mentioned for distribution are actually authorized.

Requirements: 1.2, 2.1, 3.2
"""

import logging
import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent, AgentConfig
from data_models_multiagent import ComplianceState, update_state_timestamp
from tools.registration_tools import (
    check_country_authorization,
    extract_countries_from_document,
    validate_fund_registration,
    REGISTRATION_TOOLS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RegistrationAgent(BaseAgent):
    """
    Registration Agent - Handles all registration compliance checks

    This agent is responsible for validating that marketing documents
    only claim distribution authorization in countries where the fund
    is actually registered and authorized:
    - Extract countries mentioned in document
    - Identify distribution authorization claims
    - Validate against registration database
    - Detect unauthorized country mentions

    The agent uses AI-enhanced country extraction when available and
    handles country name variations (e.g., "USA" = "United States").

    Requirements: 1.2, 2.1, 3.2
    """

    def __init__(self, config: Optional[AgentConfig] = None, ai_engine=None, **kwargs):
        """
        Initialize Registration Agent

        Args:
            config: Agent configuration
            ai_engine: AI engine for enhanced country extraction
            **kwargs: Additional configuration options
        """
        # Set default config if not provided
        if config is None:
            config = AgentConfig(
                name="registration",
                enabled=True,
                timeout_seconds=30.0,
                retry_attempts=3,
                log_level="INFO"
            )

        super().__init__(config=config, **kwargs)

        # Store AI engine for enhanced extraction
        self.ai_engine = ai_engine

        # Initialize tools
        self.tools = {
            "check_country_authorization": check_country_authorization,
            "extract_countries_from_document": extract_countries_from_document,
            "validate_fund_registration": validate_fund_registration
        }

        # Configuration
        self.ai_extraction_enabled = kwargs.get('ai_extraction_enabled', True)

        self.logger.info(f"Registration Agent initialized with {len(self.tools)} tools")
        self.logger.info(f"AI extraction: {self.ai_extraction_enabled}")
        self.logger.info(f"AI engine available: {self.ai_engine is not None}")

    def process(self, state: ComplianceState) -> ComplianceState:
        """
        Process registration compliance checks

        Executes registration validation by:
        1. Extracting countries mentioned in document
        2. Identifying distribution authorization claims
        3. Validating against authorized countries list
        4. Returning violations for unauthorized countries

        Only runs if fund ISIN and authorized countries are available.

        Args:
            state: Current compliance state

        Returns:
            Partial state update with violations and confidence scores
        """
        self.logger.info("Starting registration compliance checks")

        # Get document and configuration
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

        # Get fund ISIN and authorized countries
        fund_isin = metadata.get("fund_isin") or document.get("document_metadata", {}).get("fund_isin")
        authorized_countries = config.get("authorized_countries", [])

        # Check if registration data is available
        if not fund_isin:
            self.logger.info("No fund ISIN available, skipping registration checks")
            return {
                "confidence_scores": {self.name: 100}
            }

        if not authorized_countries:
            self.logger.info("No authorized countries list available, skipping registration checks")
            return {
                "confidence_scores": {self.name: 100}
            }

        self.logger.info(f"Checking registration for fund {fund_isin}")
        self.logger.info(f"Authorized countries: {len(authorized_countries)} countries")

        # Execute registration check
        violations = self._execute_registration_check(document, fund_isin, authorized_countries)

        # Log results
        if violations:
            self.logger.warning(f"Found {len(violations)} registration violations")
            for v in violations:
                self.logger.warning(f"  - {v.get('rule', 'Unknown')}: {v.get('message', 'No message')}")
        else:
            self.logger.info("No registration violations found - all mentioned countries are authorized")

        # Calculate confidence
        if violations:
            # Registration violations are typically high confidence
            avg_confidence = sum(v.get("confidence", 0) for v in violations) / len(violations)
            confidence = int(avg_confidence)
        else:
            confidence = 100

        self.logger.info(f"Registration checks completed. Violations: {len(violations)}, Confidence: {confidence}%")

        # Return only the fields this agent updates
        return {
            "violations": violations,
            "confidence_scores": {self.name: confidence}
        }

    def _execute_registration_check(
        self,
        document: dict,
        fund_isin: str,
        authorized_countries: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Execute registration compliance check

        Main check that validates country authorization claims.

        Args:
            document: Document to check
            fund_isin: Fund ISIN code
            authorized_countries: List of authorized countries

        Returns:
            List of violations found
        """
        self.logger.info("Executing country authorization check")

        violations = []

        # Execute main check
        result = self._safe_tool_invoke(
            "check_country_authorization",
            document=document,
            fund_isin=fund_isin,
            authorized_countries=authorized_countries,
            ai_engine=self.ai_engine if self.ai_extraction_enabled else None
        )

        if result:
            violations.append(result)
            self.logger.warning("✗ check_country_authorization: Violation found")
        else:
            self.logger.info("✓ check_country_authorization: Pass")

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
                "type": "REGISTRATION",
                "severity": "ERROR",
                "slide": "Unknown",
                "location": "Unknown",
                "rule": f"REG_ERROR: Error in {tool_name}",
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
        Set or update the AI engine for country extraction

        Args:
            ai_engine: AI engine instance
        """
        self.ai_engine = ai_engine
        self.logger.info(f"AI engine {'set' if ai_engine else 'cleared'}")

    def enable_ai_extraction(self, enabled: bool = True):
        """
        Enable or disable AI-enhanced country extraction

        Args:
            enabled: True to enable, False to disable
        """
        self.ai_extraction_enabled = enabled
        self.logger.info(f"AI extraction {'enabled' if enabled else 'disabled'}")


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    """Test the Registration Agent with example data"""

    from data_models_multiagent import initialize_compliance_state

    logger.info("=" * 70)
    logger.info("REGISTRATION AGENT TEST")
    logger.info("=" * 70)

    # Create test document with authorized countries
    test_document_valid = {
        'document_metadata': {
            'fund_isin': 'FR0010135103',
            'fund_name': 'ODDO BHF Algo Trend US Fund',
            'client_type': 'retail',
            'document_type': 'fund_presentation'
        },
        'page_de_garde': {
            'title': 'ODDO BHF Algo Trend US Fund'
        },
        'slide_2': {
            'content': 'Investment strategy...'
        },
        'pages_suivantes': [],
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management SAS',
            'authorization': 'Autorisé à la distribution en: France, Luxembourg, Belgium'
        }
    }

    # Authorized countries for this fund
    authorized_countries = ['France', 'Luxembourg', 'Belgium', 'Germany']

    # Initialize state
    config = {
        'ai_enabled': True,
        'authorized_countries': authorized_countries,
        'agents': {
            'registration': {
                'enabled': True,
                'timeout_seconds': 30.0
            }
        }
    }

    state = initialize_compliance_state(
        document=test_document_valid,
        document_id="test_doc_001",
        config=config
    )

    # Add metadata to state (normally done by preprocessor)
    state["metadata"] = {
        'fund_name': 'ODDO BHF Algo Trend US Fund',
        'client_type': 'retail',
        'fund_isin': 'FR0010135103'
    }
    state["normalized_document"] = test_document_valid

    # Create and run agent (without AI engine for basic test)
    logger.info("\n1. Creating Registration Agent...")
    agent_config = AgentConfig(
        name="registration",
        enabled=True,
        timeout_seconds=30.0,
        log_level="INFO"
    )

    agent = RegistrationAgent(
        config=agent_config,
        ai_engine=None,  # No AI engine for basic test
        ai_extraction_enabled=False
    )
    logger.info(f"   Agent created: {agent}")
    logger.info(f"   Tools available: {agent.get_tool_count()}")
    logger.info(f"   Tool names: {', '.join(agent.get_tool_names())}")

    # Execute agent
    logger.info("\n2. Executing Registration Agent (valid countries)...")
    logger.info("-" * 70)

    result_state = agent(state)

    logger.info("-" * 70)

    # Display results
    logger.info("\n3. Results:")
    logger.info(f"   Violations found: {len(result_state.get('violations', []))}")
    logger.info(f"   Agent confidence: {result_state.get('confidence_scores', {}).get('registration', 'N/A')}")
    logger.info(f"   Execution time: {result_state.get('agent_timings', {}).get('registration', 'N/A'):.2f}s")

    if result_state.get('violations'):
        logger.info("\n4. Violations:")
        for i, violation in enumerate(result_state['violations'], 1):
            logger.info(f"\n   Violation {i}:")
            logger.info(f"      Rule: {violation.get('rule', 'N/A')}")
            logger.info(f"      Severity: {violation.get('severity', 'N/A')}")
            logger.info(f"      Message: {violation.get('message', 'N/A')}")
            logger.info(f"      Confidence: {violation.get('confidence', 'N/A')}%")
    else:
        logger.info("\n4. ✓ No violations found - all mentioned countries are authorized!")

    # Display errors if any
    if result_state.get('error_log'):
        logger.info("\n5. Errors:")
        for error in result_state['error_log']:
            logger.info(f"   - {error.get('error', 'Unknown error')}")

    logger.info("\n" + "=" * 70)
    logger.info("Registration Agent test completed!")
    logger.info("=" * 70)

    # Display agent statistics
    logger.info("\n6. Agent Statistics:")
    agent.log_execution_stats()

    # Test with unauthorized countries
    logger.info("\n" + "=" * 70)
    logger.info("TESTING WITH UNAUTHORIZED COUNTRIES")
    logger.info("=" * 70)

    # Create document with unauthorized countries
    test_document_invalid = {
        'document_metadata': {
            'fund_isin': 'FR0010135103',
            'fund_name': 'ODDO BHF Algo Trend US Fund',
            'client_type': 'retail',
            'document_type': 'fund_presentation'
        },
        'page_de_garde': {
            'title': 'ODDO BHF Algo Trend US Fund'
        },
        'slide_2': {
            'content': 'Investment strategy...'
        },
        'pages_suivantes': [],
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management SAS',
            'authorization': 'Autorisé à la distribution en: France, Luxembourg, United States, Japan, Australia'
        }
    }

    # Create new state with unauthorized countries
    state2 = initialize_compliance_state(
        document=test_document_invalid,
        document_id="test_doc_002",
        config=config
    )
    state2["metadata"] = {
        'fund_name': 'ODDO BHF Algo Trend US Fund',
        'client_type': 'retail',
        'fund_isin': 'FR0010135103'
    }
    state2["normalized_document"] = test_document_invalid

    # Reset agent stats
    agent.reset_stats()

    # Execute agent
    logger.info("\n7. Executing with unauthorized countries...")
    logger.info("-" * 70)

    result_state2 = agent(state2)

    logger.info("-" * 70)

    # Display results
    logger.info("\n8. Results:")
    logger.info(f"   Violations found: {len(result_state2.get('violations', []))}")
    logger.info(f"   Agent confidence: {result_state2.get('confidence_scores', {}).get('registration', 'N/A')}")

    if result_state2.get('violations'):
        logger.info("\n9. Violations (should detect unauthorized countries):")
        for i, violation in enumerate(result_state2['violations'], 1):
            logger.info(f"\n   Violation {i}:")
            logger.info(f"      Rule: {violation.get('rule', 'N/A')}")
            logger.info(f"      Severity: {violation.get('severity', 'N/A')}")
            logger.info(f"      Message: {violation.get('message', 'N/A')[:200]}...")
            logger.info(f"      Evidence: {violation.get('evidence', 'N/A')[:150]}...")
            logger.info(f"      Confidence: {violation.get('confidence', 'N/A')}%")
    else:
        logger.info("\n9. ✗ Expected violations but none found!")

    logger.info("\n" + "=" * 70)
    logger.info("Test completed!")
    logger.info("=" * 70)
