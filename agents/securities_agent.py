#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Securities Agent

This module provides functionality for the multi-agent compliance system.
"""

"""
Securities Agent for Multi-Agent Compliance System

This agent handles securities/values compliance checks including:
- Prohibited investment advice phrases (context-aware)
- Repeated external company mentions (whitelist-aware)
- Investment advice detection
- Intent classification

The agent integrates all securities checking tools and executes them
with whitelist filtering and context-aware validation to eliminate
false positives.

Requirements: 1.2, 2.1, 2.3, 3.2
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
from tools.securities_tools import (
    check_prohibited_phrases,
    check_repeated_securities,
    check_investment_advice,
    classify_text_intent,
    SECURITIES_TOOLS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecuritiesAgent(BaseAgent):
    """
    Securities Agent - Handles all securities/values compliance checks

    This agent is responsible for validating securities-related requirements:
    - No prohibited investment advice phrases (context-aware detection)
    - No repeated external company mentions (whitelist-aware filtering)
    - No investment advice to clients (intent classification)

    The agent uses advanced AI techniques to eliminate false positives:
    - Context analysis to distinguish fund descriptions from client advice
    - Intent classification to identify ADVICE vs DESCRIPTION
    - Whitelist filtering to exclude fund names and strategy terms
    - Semantic validation to verify actual company names

    Key features:
    - Whitelist filtering: Excludes fund names, strategy terms, regulatory terms
    - Context-aware validation: Distinguishes "Le fonds investit" from "Vous devriez investir"
    - Confidence scoring: Provides confidence levels for each violation

    Requirements: 1.2, 2.1, 2.3, 3.2
    """

    def __init__(self, config: Optional[AgentConfig] = None, ai_engine=None, **kwargs):
        """
        Initialize Securities Agent

        Args:
            config: Agent configuration
            ai_engine: AI engine for context analysis and intent classification
            **kwargs: Additional configuration options
        """
        # Set default config if not provided
        if config is None:
            config = AgentConfig(
                name="securities",
                enabled=True,
                timeout_seconds=30.0,
                retry_attempts=3,
                log_level="INFO"
            )

        super().__init__(config=config, **kwargs)

        # Store AI engine
        self.ai_engine = ai_engine

        # Initialize tools
        self.tools = {
            "check_prohibited_phrases": check_prohibited_phrases,
            "check_repeated_securities": check_repeated_securities,
            "check_investment_advice": check_investment_advice,
            "classify_text_intent": classify_text_intent
        }

        # Configuration
        self.parallel_execution = kwargs.get('parallel_execution', True)
        self.max_workers = kwargs.get('max_workers', 3)

        self.logger.info(f"Securities Agent initialized with {len(self.tools)} tools")
        self.logger.info(f"Parallel execution: {self.parallel_execution}")
        self.logger.info(f"AI engine available: {self.ai_engine is not None}")

    def process(self, state: ComplianceState) -> ComplianceState:
        """
        Process securities compliance checks

        Executes all securities checking tools with whitelist filtering
        and context-aware validation. Tools can run in parallel for
        better performance.

        This agent uses advanced AI techniques:
        - ContextAnalyzer: Determines WHO performs actions (fund vs client)
        - IntentClassifier: Identifies WHAT the intent is (advice vs description)
        - SemanticValidator: Verifies if terms are actual company names
        - WhitelistManager: Filters out fund names and strategy terms

        Args:
            state: Current compliance state

        Returns:
            Partial state update with violations and confidence scores
        """
        self.logger.info("Starting securities compliance checks")

        # Get document, metadata, and whitelist
        document = state.get("normalized_document") or state.get("document", {})
        metadata = state.get("metadata", {})
        config = state.get("config", {})
        whitelist = state.get("whitelist", set())

        # Validate inputs
        if not document:
            doc_id = state.get('document_id', 'unknown')
            self.logger.error(
                f"[{self.name}] ❌ CRITICAL: No document found in state for document_id={doc_id}"
            )
            self.logger.error(
                f"[{self.name}] Cannot perform securities/values checks without document data."
            )
            self.logger.error(
                f"[{self.name}] Resolution: Ensure preprocessor agent ran successfully. "
                f"Verify workflow execution order is correct."
            )
            # Return partial state with error
            return {
                "error_log": [{
                    "agent": self.name,
                    "error": "No document found in state - cannot perform securities checks",
                    "error_type": "MissingDocumentError",
                    "severity": "CRITICAL",
                    "timestamp": datetime.now().isoformat(),
                    "document_id": doc_id,
                    "workflow_status": state.get('workflow_status', 'unknown'),
                    "resolution_hint": (
                        "Verify preprocessor agent completed successfully. "
                        "Check that document is loaded in state. "
                        "Review workflow execution logs for preprocessing errors."
                    )
                }]
            }

        # Log whitelist info
        self.logger.info(f"Using whitelist with {len(whitelist)} terms")

        # Initialize AI engine if not provided
        if not self.ai_engine:
            self.ai_engine = self._initialize_ai_engine(config)

        # Execute checks
        if self.parallel_execution:
            violations = self._execute_checks_parallel(document, metadata, config, whitelist)
        else:
            violations = self._execute_checks_sequential(document, metadata, config, whitelist)

        # Log results
        if violations:
            self.logger.info(f"Found {len(violations)} securities violations")
        else:
            self.logger.info("No securities violations found")

        # Calculate aggregate confidence
        if violations:
            avg_confidence = sum(v.get("confidence", 0) for v in violations) / len(violations)
            confidence = int(avg_confidence)
        else:
            confidence = 100

        self.logger.info(f"Securities checks completed. Violations: {len(violations)}")

        # Return only the fields this agent updates
        # This allows parallel execution without conflicts
        return {
            "violations": violations,
            "confidence_scores": {self.name: confidence}
        }

    def _initialize_ai_engine(self, config: dict):
        """
        Initialize AI engine from config if not provided

        Args:
            config: Configuration dictionary

        Returns:
            AI engine instance or None
        """
        try:
            from ai_engine import create_ai_engine_from_env
            ai_engine = create_ai_engine_from_env()
            if ai_engine:
                self.logger.info("AI engine initialized from environment")
            else:
                self.logger.warning("AI engine not available - using rule-based fallback")
            return ai_engine
        except Exception as e:
            self.logger.warning(f"Could not initialize AI engine: {e}")
            return None

    def _execute_checks_parallel(
        self,
        document: dict,
        metadata: dict,
        config: dict,
        whitelist: set
    ) -> List[Dict[str, Any]]:
        """
        Execute all securities checks in parallel

        Uses ThreadPoolExecutor to run checks concurrently for better performance.

        Args:
            document: Document to check
            metadata: Document metadata
            config: Configuration dictionary
            whitelist: Set of whitelisted terms

        Returns:
            List of violations found
        """
        self.logger.info(f"Executing {len(self.tools) - 1} checks in parallel")  # -1 for classify_text_intent

        violations = []

        # Load rules for prohibited phrases check
        rules = self._load_securities_rules(config)

        # Create tasks for each check
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all checks
            future_to_check = {}

            # Prohibited phrases check (returns list of violations)
            if rules:
                for rule in rules:
                    if rule.get('rule_id', '').startswith('VAL_'):
                        future = executor.submit(
                            self._safe_tool_invoke,
                            "check_prohibited_phrases",
                            document=document,
                            rule=rule,
                            ai_engine=self.ai_engine,
                            returns_list=True
                        )
                        future_to_check[future] = f"check_prohibited_phrases ({rule.get('rule_id', 'unknown')})"

            # Repeated securities check (returns list)
            future = executor.submit(
                self._safe_tool_invoke,
                "check_repeated_securities",
                document=document,
                whitelist=whitelist,
                ai_engine=self.ai_engine,
                returns_list=True
            )
            future_to_check[future] = "check_repeated_securities"

            # Investment advice check (returns list)
            future = executor.submit(
                self._safe_tool_invoke,
                "check_investment_advice",
                document=document,
                ai_engine=self.ai_engine,
                returns_list=True
            )
            future_to_check[future] = "check_investment_advice"

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
        config: dict,
        whitelist: set
    ) -> List[Dict[str, Any]]:
        """
        Execute all securities checks sequentially

        Runs checks one after another. Useful for debugging or when
        parallel execution is not desired.

        Args:
            document: Document to check
            metadata: Document metadata
            config: Configuration dictionary
            whitelist: Set of whitelisted terms

        Returns:
            List of violations found
        """
        self.logger.info(f"Executing {len(self.tools) - 1} checks sequentially")

        violations = []

        # Load rules for prohibited phrases check
        rules = self._load_securities_rules(config)

        # Check prohibited phrases for each rule
        if rules:
            for rule in rules:
                if rule.get('rule_id', '').startswith('VAL_'):
                    result = self._safe_tool_invoke(
                        "check_prohibited_phrases",
                        document=document,
                        rule=rule,
                        ai_engine=self.ai_engine,
                        returns_list=True
                    )
                    if result:
                        violations.extend(result)
                        self.logger.info(f"✗ check_prohibited_phrases ({rule.get('rule_id')}): {len(result)} violation(s) found")
                    else:
                        self.logger.info(f"✓ check_prohibited_phrases ({rule.get('rule_id')}): Pass")

        # Check repeated securities (returns list)
        result = self._safe_tool_invoke(
            "check_repeated_securities",
            document=document,
            whitelist=whitelist,
            ai_engine=self.ai_engine,
            returns_list=True
        )
        if result:
            violations.extend(result)
            self.logger.info(f"✗ check_repeated_securities: {len(result)} violation(s) found")
        else:
            self.logger.info("✓ check_repeated_securities: Pass")

        # Check investment advice (returns list)
        result = self._safe_tool_invoke(
            "check_investment_advice",
            document=document,
            ai_engine=self.ai_engine,
            returns_list=True
        )
        if result:
            violations.extend(result)
            self.logger.info(f"✗ check_investment_advice: {len(result)} violation(s) found")
        else:
            self.logger.info("✓ check_investment_advice: Pass")

        return violations

    def _load_securities_rules(self, config: dict) -> List[Dict[str, Any]]:
        """
        Load securities rules from configuration

        Args:
            config: Configuration dictionary

        Returns:
            List of rule dictionaries
        """
        try:
            import json

            # Try to load from config
            rules_file = config.get('values_rules_file', 'values_rules.json')

            if os.path.exists(rules_file):
                with open(rules_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    rules = data.get('rules', [])
                    self.logger.info(f"Loaded {len(rules)} securities rules from {rules_file}")
                    return rules
            else:
                self.logger.warning(f"Securities rules file not found: {rules_file}")
                return []

        except Exception as e:
            self.logger.error(f"Error loading securities rules: {e}")
            return []

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
                f"[{self.name}] Context: Checking securities/values compliance with whitelist filtering"
            )
            self.logger.debug(f"[{self.name}] Full error details:", exc_info=True)

            # Return error violation with detailed context
            error_violation = {
                "type": "SECURITIES/VALUES",
                "severity": "ERROR",
                "slide": "Unknown",
                "location": "Unknown",
                "rule": f"SEC_ERROR_{tool_name.upper()}",
                "message": (
                    f"Securities check '{tool_name}' failed with {error_type}. "
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
                        f"Verify whitelist is properly built. "
                        f"If using AI features, check API connectivity and credentials. "
                        f"Ensure values_rules.json file exists and is valid. "
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
    """Test the Securities Agent with example data"""

    from data_models_multiagent import initialize_compliance_state

    logger.info("=" * 70)
    logger.info("SECURITIES AGENT TEST")
    logger.info("=" * 70)

    # Create test document
    test_document = {
        'document_metadata': {
            'fund_isin': 'FR0010135103',
            'fund_name': 'ODDO BHF Algo Trend US Fund',
            'client_type': 'retail',
            'document_type': 'fund_presentation'
        },
        'page_de_garde': {
            'title': 'ODDO BHF Algo Trend US Fund',
            'subtitle': 'Document promotionnel'
        },
        'slide_2': {
            'title': 'Investment Strategy',
            'content': 'Le fonds investit dans des actions américaines à forte capitalisation. La stratégie quantitative vise à tirer parti du momentum des marchés.'
        },
        'pages_suivantes': [
            {
                'slide_number': 3,
                'title': 'Fund Description',
                'content': 'ODDO BHF Asset Management utilise une approche quantitative pour identifier les tendances. Le fonds se concentre sur les valeurs momentum.'
            }
        ],
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management SAS'
        }
    }

    # Initialize state
    config = {
        'ai_enabled': True,
        'values_rules_file': 'values_rules.json',
        'agents': {
            'securities': {
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

    # Add metadata and whitelist to state (normally done by preprocessor)
    state["metadata"] = {
        'fund_name': 'ODDO BHF Algo Trend US Fund',
        'client_type': 'retail',
        'fund_isin': 'FR0010135103'
    }
    state["normalized_document"] = test_document
    state["whitelist"] = {'oddo', 'bhf', 'momentum', 'quantitative', 'trend', 'algo'}

    # Create and run agent
    logger.info("\n1. Creating Securities Agent...")
    agent_config = AgentConfig(
        name="securities",
        enabled=True,
        timeout_seconds=30.0,
        log_level="INFO"
    )

    # Initialize AI engine
    try:
        from ai_engine import create_ai_engine_from_env
        ai_engine = create_ai_engine_from_env()
        if ai_engine:
            logger.info("   ✓ AI Engine initialized")
        else:
            logger.info("   ⚠️  AI Engine not available - using rule-based fallback")
    except Exception as e:
        logger.info(f"   ⚠️  Could not initialize AI Engine: {e}")
        ai_engine = None

    agent = SecuritiesAgent(config=agent_config, ai_engine=ai_engine, parallel_execution=True)
    logger.info(f"   Agent created: {agent}")
    logger.info(f"   Tools available: {agent.get_tool_count()}")
    logger.info(f"   Tool names: {', '.join(agent.get_tool_names())}")

    # Execute agent
    logger.info("\n2. Executing Securities Agent...")
    logger.info("-" * 70)

    result_state = agent(state)

    logger.info("-" * 70)

    # Display results
    logger.info("\n3. Results:")
    logger.info(f"   Violations found: {len(result_state.get('violations', []))}")
    logger.info(f"   Agent confidence: {result_state.get('confidence_scores', {}).get('securities', 'N/A')}")

    agent_timing = result_state.get('agent_timings', {}).get('securities')
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
                if len(evidence) > 150:
                    evidence = evidence[:150] + "..."
                logger.info(f"      Evidence: {evidence}")
    else:
        logger.info("\n4. ✓ No violations found - securities compliance OK!")

    # Display errors if any
    if result_state.get('error_log'):
        logger.info("\n5. Errors:")
        for error in result_state['error_log']:
            logger.info(f"   - {error.get('error', 'Unknown error')}")

    logger.info("\n" + "=" * 70)
    logger.info("Securities Agent test completed!")
    logger.info("=" * 70)

    # Display agent statistics
    logger.info("\n6. Agent Statistics:")
    agent.log_execution_stats()
