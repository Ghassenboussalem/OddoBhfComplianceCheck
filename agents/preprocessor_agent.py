#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor Agent

This module provides functionality for the multi-agent compliance system.
"""

"""
Preprocessor Agent for Multi-Agent Compliance System

This agent handles document preprocessing before compliance checks:
- Metadata extraction
- Whitelist building
- Document normalization
- Document validation

The preprocessor runs first in the workflow and prepares the document
for downstream compliance checking agents.

Requirements: 1.2, 2.3, 6.1
"""

import logging
import sys
import os
from typing import Dict, Set, Optional, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import base agent framework
from agents.base_agent import BaseAgent, AgentConfig

# Import preprocessing tools
from tools.preprocessing_tools import (
    extract_metadata,
    build_whitelist,
    normalize_document,
    validate_document,
    PREPROCESSING_TOOLS
)

# Import state models
from data_models_multiagent import (
    ComplianceState,
    WorkflowStatus,
    update_state_timestamp
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessorAgent(BaseAgent):
    """
    Preprocessor Agent - First agent in the compliance workflow

    Responsibilities:
    1. Extract metadata from document (fund ISIN, client type, etc.)
    2. Build whitelist of allowed terms to prevent false positives
    3. Normalize document structure for consistent processing
    4. Validate document has required sections and content

    The preprocessor prepares the document and state for all downstream
    compliance checking agents.

    Requirements: 1.2, 2.3, 6.1
    """

    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """
        Initialize Preprocessor Agent

        Args:
            config: Agent configuration
            **kwargs: Additional configuration options
        """
        # Initialize with default config if not provided
        if config is None:
            config = AgentConfig(
                name="preprocessor",
                enabled=True,
                timeout_seconds=30.0,
                log_level="INFO"
            )

        super().__init__(config=config, **kwargs)

        # Store tools as instance attributes for easy access
        self.tools = {tool.name: tool for tool in PREPROCESSING_TOOLS}

        # Configuration options
        self.validate_before_processing = kwargs.get('validate_before_processing', True)
        self.fail_on_validation_errors = kwargs.get('fail_on_validation_errors', False)

        self.logger.info(f"PreprocessorAgent initialized with {len(self.tools)} tools")
        self.logger.info(f"Tools available: {list(self.tools.keys())}")

    def process(self, state: ComplianceState) -> ComplianceState:
        """
        Process the compliance state through preprocessing steps.

        Workflow:
        1. Validate document structure (optional)
        2. Extract metadata
        3. Normalize document
        4. Build whitelist
        5. Update state with preprocessing results

        Args:
            state: Current compliance state

        Returns:
            Updated state with preprocessing results

        Requirements: 1.2, 2.3, 6.1
        """
        self.logger.info("=" * 70)
        self.logger.info("PREPROCESSOR AGENT - Starting document preprocessing")
        self.logger.info("=" * 70)

        # Update workflow status
        state["workflow_status"] = WorkflowStatus.PREPROCESSING.value
        state = update_state_timestamp(state)

        # Step 1: Validate document (optional)
        if self.validate_before_processing:
            state = self._validate_document(state)

        # Step 2: Extract metadata
        state = self._extract_metadata(state)

        # Step 3: Normalize document structure
        state = self._normalize_document(state)

        # Step 4: Build whitelist
        state = self._build_whitelist(state)

        # Step 5: Log preprocessing summary
        self._log_preprocessing_summary(state)

        # Update next action
        state["next_action"] = "parallel_checks"

        self.logger.info("=" * 70)
        self.logger.info("PREPROCESSOR AGENT - Preprocessing completed successfully")
        self.logger.info("=" * 70)

        return state

    def _validate_document(self, state: ComplianceState) -> ComplianceState:
        """
        Validate document structure and content.

        Args:
            state: Current state

        Returns:
            Updated state (may include validation errors in error_log)
        """
        self.logger.info("\n[Step 1/4] Validating document structure...")

        try:
            # Invoke validation tool
            validation_result = self.tools["validate_document"].invoke({
                "document": state["document"]
            })

            # Log validation results
            if validation_result["valid"]:
                self.logger.info("✓ Document validation passed")
            else:
                self.logger.warning(f"✗ Document validation failed with {validation_result['total_errors']} errors")

                # Log errors
                for error in validation_result["errors"]:
                    self.logger.error(f"  - {error}")

                # Add to error log
                if "error_log" not in state:
                    state["error_log"] = []

                state["error_log"].append({
                    "agent": self.name,
                    "step": "validation",
                    "errors": validation_result["errors"],
                    "warnings": validation_result["warnings"]
                })

                # Fail if configured to do so
                if self.fail_on_validation_errors:
                    raise ValueError(f"Document validation failed: {validation_result['errors']}")

            # Log warnings
            if validation_result["warnings"]:
                self.logger.info(f"  Warnings ({validation_result['total_warnings']}):")
                for warning in validation_result["warnings"]:
                    self.logger.warning(f"  - {warning}")

            # Log sections present
            sections = validation_result["sections_present"]
            self.logger.info(f"  Sections present: {sum(sections.values())}/{len(sections)}")

        except Exception as e:
            self.logger.error(f"Error during document validation: {e}")

            # Add to error log but don't fail
            if "error_log" not in state:
                state["error_log"] = []

            state["error_log"].append({
                "agent": self.name,
                "step": "validation",
                "error": str(e)
            })

        return state

    def _extract_metadata(self, state: ComplianceState) -> ComplianceState:
        """
        Extract metadata from document.

        Args:
            state: Current state

        Returns:
            Updated state with metadata field populated
        """
        self.logger.info("\n[Step 2/4] Extracting metadata...")

        try:
            # Invoke metadata extraction tool
            metadata = self.tools["extract_metadata"].invoke({
                "document": state["document"]
            })

            # Store in state
            state["metadata"] = metadata

            # Update top-level fields for convenience
            state["document_type"] = metadata.get("document_type", "fund_presentation")
            state["client_type"] = metadata.get("client_type", "retail")

            # Log extracted metadata
            self.logger.info("✓ Metadata extracted successfully:")
            self.logger.info(f"  - Fund ISIN: {metadata.get('fund_isin', 'N/A')}")
            self.logger.info(f"  - Fund Name: {metadata.get('fund_name', 'N/A')}")
            self.logger.info(f"  - Client Type: {metadata.get('client_type', 'N/A')}")
            self.logger.info(f"  - Document Type: {metadata.get('document_type', 'N/A')}")
            self.logger.info(f"  - ESG Classification: {metadata.get('esg_classification', 'N/A')}")

            if metadata.get('fund_age_years'):
                self.logger.info(f"  - Fund Age: {metadata['fund_age_years']} years")

        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")

            # Use default metadata
            state["metadata"] = {
                "fund_isin": None,
                "client_type": "retail",
                "document_type": "fund_presentation",
                "fund_name": None,
                "esg_classification": "other"
            }

            # Add to error log
            if "error_log" not in state:
                state["error_log"] = []

            state["error_log"].append({
                "agent": self.name,
                "step": "metadata_extraction",
                "error": str(e)
            })

        return state

    def _normalize_document(self, state: ComplianceState) -> ComplianceState:
        """
        Normalize document structure.

        Args:
            state: Current state

        Returns:
            Updated state with normalized_document field populated
        """
        self.logger.info("\n[Step 3/4] Normalizing document structure...")

        try:
            # Invoke normalization tool
            normalized_doc = self.tools["normalize_document"].invoke({
                "document": state["document"]
            })

            # Store in state
            state["normalized_document"] = normalized_doc

            # Log normalization results
            self.logger.info("✓ Document normalized successfully:")
            self.logger.info(f"  - Cover page: {'✓' if normalized_doc.get('page_de_garde') else '✗'}")
            self.logger.info(f"  - Slide 2: {'✓' if normalized_doc.get('slide_2') else '✗'}")
            self.logger.info(f"  - Following pages: {len(normalized_doc.get('pages_suivantes', []))}")
            self.logger.info(f"  - Back page: {'✓' if normalized_doc.get('page_de_fin') else '✗'}")

        except Exception as e:
            self.logger.error(f"Error normalizing document: {e}")

            # Use original document as fallback
            state["normalized_document"] = state["document"]

            # Add to error log
            if "error_log" not in state:
                state["error_log"] = []

            state["error_log"].append({
                "agent": self.name,
                "step": "normalization",
                "error": str(e)
            })

        return state

    def _build_whitelist(self, state: ComplianceState) -> ComplianceState:
        """
        Build whitelist of allowed terms.

        Args:
            state: Current state

        Returns:
            Updated state with whitelist field populated
        """
        self.logger.info("\n[Step 4/4] Building whitelist...")

        try:
            # Invoke whitelist building tool
            whitelist = self.tools["build_whitelist"].invoke({
                "document": state["document"],
                "metadata": state.get("metadata", {})
            })

            # Store in state
            state["whitelist"] = whitelist

            # Log whitelist statistics
            self.logger.info(f"✓ Whitelist built successfully:")
            self.logger.info(f"  - Total terms: {len(whitelist)}")

            # Sample some terms for logging
            if whitelist:
                sample_terms = sorted(list(whitelist))[:10]
                self.logger.info(f"  - Sample terms: {', '.join(sample_terms)}")

        except Exception as e:
            self.logger.error(f"Error building whitelist: {e}")

            # Use empty whitelist as fallback
            state["whitelist"] = set()

            # Add to error log
            if "error_log" not in state:
                state["error_log"] = []

            state["error_log"].append({
                "agent": self.name,
                "step": "whitelist_building",
                "error": str(e)
            })

        return state

    def _log_preprocessing_summary(self, state: ComplianceState):
        """
        Log summary of preprocessing results.

        Args:
            state: Current state with preprocessing results
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("PREPROCESSING SUMMARY")
        self.logger.info("=" * 70)

        # Document info
        self.logger.info(f"Document ID: {state.get('document_id', 'N/A')}")
        self.logger.info(f"Document Type: {state.get('document_type', 'N/A')}")
        self.logger.info(f"Client Type: {state.get('client_type', 'N/A')}")

        # Metadata
        metadata = state.get("metadata", {})
        if metadata.get("fund_name"):
            self.logger.info(f"Fund Name: {metadata['fund_name']}")
        if metadata.get("fund_isin"):
            self.logger.info(f"Fund ISIN: {metadata['fund_isin']}")

        # Whitelist
        whitelist_size = len(state.get("whitelist", set()))
        self.logger.info(f"Whitelist Terms: {whitelist_size}")

        # Document structure
        normalized = state.get("normalized_document", {})
        pages_count = len(normalized.get("pages_suivantes", []))
        self.logger.info(f"Document Pages: {pages_count + 3}")  # +3 for cover, slide2, back

        # Errors
        error_count = len(state.get("error_log", []))
        if error_count > 0:
            self.logger.warning(f"Errors Encountered: {error_count}")
        else:
            self.logger.info("Errors Encountered: 0")

        self.logger.info("=" * 70)


# Convenience function to create preprocessor agent
def create_preprocessor_agent(config: Optional[Dict[str, Any]] = None) -> PreprocessorAgent:
    """
    Create a PreprocessorAgent instance with configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured PreprocessorAgent instance
    """
    if config is None:
        agent_config = AgentConfig(name="preprocessor")
    else:
        agent_config = AgentConfig(
            name="preprocessor",
            enabled=config.get("enabled", True),
            timeout_seconds=config.get("timeout_seconds", 30.0),
            log_level=config.get("log_level", "INFO"),
            custom_settings=config.get("custom_settings", {})
        )

    return PreprocessorAgent(config=agent_config)


# Export public symbols
__all__ = [
    "PreprocessorAgent",
    "create_preprocessor_agent"
]


# Example usage and testing
if __name__ == "__main__":
    from data_models_multiagent import initialize_compliance_state

    logger.info("=" * 70)
    logger.info("PREPROCESSOR AGENT TEST")
    logger.info("=" * 70)

    # Create test document
    test_doc = {
        'document_metadata': {
            'fund_isin': 'FR0010135103',
            'fund_name': 'ODDO BHF Algo Trend US Fund',
            'client_type': 'retail',
            'document_type': 'fund_presentation',
            'fund_esg_classification': 'article_8',
            'fund_age_years': 5
        },
        'page_de_garde': {
            'title': 'Fund Presentation',
            'subtitle': 'ODDO BHF Algo Trend US',
            'date': '2024-01-15'
        },
        'slide_2': {
            'content': 'Investment strategy focuses on momentum and trend following...'
        },
        'pages_suivantes': [
            {
                'slide_number': 3,
                'title': 'Performance',
                'content': 'Historical performance data...'
            },
            {
                'slide_number': 4,
                'title': 'Risk Profile',
                'content': 'Risk and volatility analysis...'
            }
        ],
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management SAS',
            'disclaimers': 'Past performance is not indicative of future results.'
        }
    }

    # Initialize state
    config = {
        "agents": {
            "preprocessor": {
                "enabled": True,
                "timeout_seconds": 30.0
            }
        }
    }

    state = initialize_compliance_state(
        document=test_doc,
        document_id="test_doc_001",
        config=config
    )

    # Create and run preprocessor agent
    logger.info("\nCreating PreprocessorAgent...")
    agent = create_preprocessor_agent(config.get("agents", {}).get("preprocessor"))

    logger.info("\nRunning preprocessing...")
    result_state = agent(state)

    # Display results
    logger.info("\n" + "=" * 70)
    logger.info("PREPROCESSING RESULTS")
    logger.info("=" * 70)

    logger.info(f"\nWorkflow Status: {result_state['workflow_status']}")
    logger.info(f"Current Agent: {result_state['current_agent']}")
    logger.info(f"Next Action: {result_state['next_action']}")

    logger.info(f"\nMetadata Extracted:")
    for key, value in result_state['metadata'].items():
        logger.info(f"  {key}: {value}")

    logger.info(f"\nWhitelist Size: {len(result_state['whitelist'])}")
    logger.info(f"Sample Terms: {list(result_state['whitelist'])[:10]}")

    logger.info(f"\nNormalized Document Sections:")
    for section in ['page_de_garde', 'slide_2', 'pages_suivantes', 'page_de_fin']:
        if section in result_state['normalized_document']:
            if isinstance(result_state['normalized_document'][section], list):
                logger.info(f"  {section}: {len(result_state['normalized_document'][section])} items")
            else:
                logger.info(f"  {section}: ✓")

    logger.info(f"\nErrors: {len(result_state.get('error_log', []))}")

    logger.info(f"\nAgent Timing: {result_state.get('agent_timings', {}).get('preprocessor', 0):.3f}s")

    logger.info("\n" + "=" * 70)
    logger.info("TEST COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
