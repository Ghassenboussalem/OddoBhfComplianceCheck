#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evidence Agent

This module provides functionality for the multi-agent compliance system.
"""

"""
Evidence Agent for Multi-Agent Compliance System

This agent is responsible for extracting and tracking evidence supporting compliance findings:
- Extract evidence for each violation
- Find actual performance data (numbers with percentages vs descriptive keywords)
- Implement semantic disclaimer matching
- Add location and context tracking
- Enhance violations with supporting evidence

The Evidence Agent works after the Context Agent to provide concrete evidence
for violations, helping to validate findings and support human review.

Requirements: 1.2, 2.3, 9.3, 9.4, 9.5
"""

import logging
import sys
import os
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent, AgentConfig
from data_models_multiagent import ComplianceState, update_state_timestamp
from tools.evidence_tools import (
    extract_evidence,
    find_performance_data,
    find_disclaimer,
    track_location,
    extract_quotes,
    EVIDENCE_TOOLS
)

# Configure logging
logger = logging.getLogger(__name__)


class EvidenceAgent(BaseAgent):
    """
    Evidence Agent - Extract and track evidence for compliance violations

    This agent enhances violations with concrete evidence:
    - Extracts specific quotes and text passages
    - Finds actual performance data (numbers with %)
    - Locates disclaimers using semantic matching
    - Tracks locations within documents
    - Adds context to support findings

    The agent processes violations that need evidence enhancement,
    particularly those with lower confidence scores or those flagged
    for human review.

    Requirements: 1.2, 2.3, 9.3, 9.4, 9.5
    """

    def __init__(self, config: Optional[AgentConfig] = None, ai_engine=None, **kwargs):
        """
        Initialize Evidence Agent

        Args:
            config: Agent configuration
            ai_engine: Optional AIEngine for semantic analysis
            **kwargs: Additional configuration options
        """
        if config is None:
            config = AgentConfig(name="evidence")

        super().__init__(config, **kwargs)

        self.ai_engine = ai_engine

        # Store tools as functions (they are already decorated and registered)
        self.tools = {
            "extract_evidence": extract_evidence,
            "find_performance_data": find_performance_data,
            "find_disclaimer": find_disclaimer,
            "track_location": track_location,
            "extract_quotes": extract_quotes
        }

        # Configuration
        self.min_confidence_for_evidence = self.get_config_value("min_confidence_for_evidence", 0)
        self.max_violations_to_process = self.get_config_value("max_violations_to_process", 50)
        self.enhance_all_violations = self.get_config_value("enhance_all_violations", False)

        self.logger.info(f"EvidenceAgent initialized with {len(self.tools)} tools")
        self.logger.info(f"  Min confidence for evidence: {self.min_confidence_for_evidence}")
        self.logger.info(f"  Max violations to process: {self.max_violations_to_process}")

    def process(self, state: ComplianceState) -> ComplianceState:
        """
        Process compliance state to extract evidence for violations

        This method:
        1. Identifies violations needing evidence enhancement
        2. Extracts evidence based on violation type
        3. Finds performance data for performance-related violations
        4. Locates disclaimers for disclaimer violations
        5. Tracks locations and adds context
        6. Updates violations with evidence

        Args:
            state: Current compliance state

        Returns:
            Updated state with evidence-enhanced violations

        Requirements: 1.2, 2.3, 9.3, 9.4, 9.5
        """
        self.logger.info("="*70)
        self.logger.info("Evidence Agent - Starting Evidence Extraction")
        self.logger.info("="*70)

        # Get violations from state
        violations = list(state.get("violations", []))

        if not violations:
            self.logger.info("No violations to process")
            state["next_action"] = "complete"
            return update_state_timestamp(state)

        self.logger.info(f"Processing {len(violations)} violations for evidence extraction")

        # Filter violations that need evidence enhancement
        violations_to_enhance = self._filter_violations_for_evidence(violations)

        self.logger.info(f"Selected {len(violations_to_enhance)} violations for evidence enhancement")

        # Extract evidence for each violation
        enhanced_violations = []
        evidence_extractions = state.get("evidence_extractions", {})

        for i, violation in enumerate(violations):
            if violation in violations_to_enhance:
                self.logger.debug(f"Enhancing violation {i+1}/{len(violations)}: {violation.get('rule', 'unknown')}")

                # Extract evidence based on violation type
                enhanced_violation = self._enhance_violation_with_evidence(
                    violation,
                    state.get("normalized_document", state.get("document", {}))
                )

                # Store evidence extraction results
                violation_key = f"{violation.get('rule', 'unknown')}_{violation.get('slide', 'unknown')}"
                evidence_extractions[violation_key] = {
                    "evidence": enhanced_violation.get("evidence", ""),
                    "evidence_quotes": enhanced_violation.get("evidence_quotes", []),
                    "evidence_locations": enhanced_violation.get("evidence_locations", []),
                    "evidence_confidence": enhanced_violation.get("evidence_confidence", 0)
                }

                enhanced_violations.append(enhanced_violation)
            else:
                # Keep violation as-is
                enhanced_violations.append(violation)

        # Update state with enhanced violations
        state["violations"] = enhanced_violations
        state["evidence_extractions"] = evidence_extractions

        # Log summary
        self.logger.info("="*70)
        self.logger.info("Evidence Extraction Summary:")
        self.logger.info(f"  Total violations: {len(violations)}")
        self.logger.info(f"  Enhanced with evidence: {len(violations_to_enhance)}")
        self.logger.info(f"  Evidence extractions stored: {len(evidence_extractions)}")
        self.logger.info("="*70)

        # Set next action
        state["next_action"] = "review" if self._needs_review(enhanced_violations) else "complete"

        return update_state_timestamp(state)

    def _filter_violations_for_evidence(self, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter violations that need evidence enhancement

        Criteria:
        - Confidence below threshold
        - Flagged for review
        - Missing evidence field
        - Performance or disclaimer related

        Args:
            violations: List of violations

        Returns:
            Filtered list of violations needing evidence
        """
        filtered = []

        for violation in violations[:self.max_violations_to_process]:
            # Always enhance if configured to do so
            if self.enhance_all_violations:
                filtered.append(violation)
                continue

            # Check if violation needs evidence
            confidence = violation.get("confidence", 100)
            has_evidence = bool(violation.get("evidence", "").strip())
            violation_type = violation.get("type", "").lower()
            status = violation.get("status", "")

            # Enhance if:
            # 1. Low confidence
            # 2. Missing evidence
            # 3. Performance or disclaimer related
            # 4. Pending review
            needs_evidence = (
                confidence < 80 or
                not has_evidence or
                "performance" in violation_type or
                "disclaimer" in violation_type or
                "pending_review" in status.lower()
            )

            if needs_evidence:
                filtered.append(violation)

        return filtered

    def _enhance_violation_with_evidence(
        self,
        violation: Dict[str, Any],
        document: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance a violation with extracted evidence

        Args:
            violation: Violation to enhance
            document: Document to extract evidence from

        Returns:
            Enhanced violation with evidence
        """
        violation_type = violation.get("type", "").lower()
        slide = violation.get("slide", "")
        location = violation.get("location", "")
        existing_evidence = violation.get("evidence", "")

        # Get text from relevant slide/location
        text = self._extract_text_from_location(document, slide, location)

        # If no text found, use existing evidence
        if not text:
            text = existing_evidence

        # Extract evidence based on violation type
        if "performance" in violation_type:
            evidence_result = self._extract_performance_evidence(text, violation)
        elif "disclaimer" in violation_type:
            evidence_result = self._extract_disclaimer_evidence(text, violation)
        else:
            evidence_result = self._extract_generic_evidence(text, violation_type, location)

        # Enhance violation with evidence
        enhanced = violation.copy()

        # Add evidence fields
        if evidence_result:
            enhanced["evidence_quotes"] = evidence_result.get("quotes", [])
            enhanced["evidence_locations"] = evidence_result.get("locations", [])
            enhanced["evidence_context"] = evidence_result.get("context", "")
            enhanced["evidence_confidence"] = evidence_result.get("confidence", 0)

            # Update main evidence field if better evidence found
            if evidence_result.get("quotes") and (not existing_evidence or len(existing_evidence) < 50):
                enhanced["evidence"] = " | ".join(evidence_result["quotes"][:2])

        return enhanced

    def _extract_performance_evidence(
        self,
        text: str,
        violation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract evidence for performance-related violations

        Finds actual performance data (numbers with percentages) vs
        descriptive keywords like "attractive performance".

        Args:
            text: Text to analyze
            violation: Violation details

        Returns:
            Evidence dictionary with quotes, locations, context, confidence
        """
        self.logger.debug("Extracting performance evidence...")

        try:
            # Find actual performance data
            perf_data_result = find_performance_data(text, self.ai_engine)
            perf_data = perf_data_result.result if hasattr(perf_data_result, 'result') else perf_data_result

            if perf_data:
                # Extract quotes from performance data
                quotes = []
                locations = []

                for pd in perf_data[:3]:  # Top 3 performance data points
                    quote = f"{pd.value} - {pd.context[:100]}"
                    quotes.append(quote)
                    locations.append(pd.location)

                return {
                    "quotes": quotes,
                    "locations": locations,
                    "context": f"Found {len(perf_data)} actual performance data points with percentages",
                    "confidence": max([pd.confidence for pd in perf_data]) if perf_data else 0
                }
            else:
                # No actual performance data found, extract descriptive text
                quotes_result = extract_quotes(text, max_quotes=2)
                quotes = quotes_result.result if hasattr(quotes_result, 'result') else quotes_result

                return {
                    "quotes": quotes,
                    "locations": [violation.get("location", "Unknown")],
                    "context": "No actual performance data found, only descriptive text",
                    "confidence": 50
                }

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)[:200]
            self.logger.error(
                f"[{self.name}] ❌ Performance evidence extraction failed: {error_type}: {error_msg}"
            )
            self.logger.warning(
                f"[{self.name}] Falling back to generic evidence extraction"
            )
            self.logger.debug(f"[{self.name}] Full error details:", exc_info=True)
            return self._extract_generic_evidence(text, "performance", violation.get("location", ""))

    def _extract_disclaimer_evidence(
        self,
        text: str,
        violation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract evidence for disclaimer-related violations

        Uses semantic matching to find disclaimers or identify
        where disclaimers are missing.

        Args:
            text: Text to analyze
            violation: Violation details

        Returns:
            Evidence dictionary with quotes, locations, context, confidence
        """
        self.logger.debug("Extracting disclaimer evidence...")

        try:
            # Get required disclaimer from violation or use default
            required_disclaimer = violation.get(
                "required_disclaimer",
                "Les performances passées ne préjugent pas des performances futures"
            )

            # Search for disclaimer
            disclaimer_result = find_disclaimer(text, required_disclaimer, self.ai_engine)
            disclaimer = disclaimer_result.result if hasattr(disclaimer_result, 'result') else disclaimer_result

            if disclaimer.found:
                return {
                    "quotes": [disclaimer.text],
                    "locations": [disclaimer.location],
                    "context": f"Disclaimer found with {disclaimer.similarity_score}% similarity",
                    "confidence": disclaimer.confidence
                }
            else:
                # Disclaimer not found, extract context where it should be
                quotes_result = extract_quotes(text, max_quotes=3)
                quotes = quotes_result.result if hasattr(quotes_result, 'result') else quotes_result

                return {
                    "quotes": quotes,
                    "locations": [violation.get("location", "Unknown")],
                    "context": "Required disclaimer not found in text",
                    "confidence": 85
                }

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)[:200]
            self.logger.error(
                f"[{self.name}] ❌ Disclaimer evidence extraction failed: {error_type}: {error_msg}"
            )
            self.logger.warning(
                f"[{self.name}] Falling back to generic evidence extraction"
            )
            self.logger.debug(f"[{self.name}] Full error details:", exc_info=True)
            return self._extract_generic_evidence(text, "disclaimer", violation.get("location", ""))

    def _extract_generic_evidence(
        self,
        text: str,
        violation_type: str,
        location: str
    ) -> Dict[str, Any]:
        """
        Extract generic evidence for any violation type

        Args:
            text: Text to analyze
            violation_type: Type of violation
            location: Location in document

        Returns:
            Evidence dictionary with quotes, locations, context, confidence
        """
        self.logger.debug(f"Extracting generic evidence for {violation_type}...")

        try:
            # Use the extract_evidence tool
            evidence_result = extract_evidence(text, violation_type, location)
            evidence = evidence_result.result if hasattr(evidence_result, 'result') else evidence_result

            return {
                "quotes": evidence.quotes,
                "locations": evidence.locations,
                "context": evidence.context,
                "confidence": evidence.confidence
            }

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)[:200]
            self.logger.error(
                f"[{self.name}] ❌ Generic evidence extraction failed: {error_type}: {error_msg}"
            )
            self.logger.warning(
                f"[{self.name}] Attempting simple quote extraction as last resort"
            )

            # Fallback: extract simple quotes
            try:
                quotes_result = extract_quotes(text, max_quotes=3)
                quotes = quotes_result.result if hasattr(quotes_result, 'result') else quotes_result

                return {
                    "quotes": quotes,
                    "locations": [location],
                    "context": f"Evidence for {violation_type} (fallback extraction)",
                    "confidence": 60
                }
            except Exception as fallback_error:
                self.logger.error(
                    f"[{self.name}] ❌ Even fallback quote extraction failed: {type(fallback_error).__name__}"
                )
                self.logger.error(
                    f"[{self.name}] Resolution: Check text format and evidence extraction tools. "
                    f"Violation will have no evidence quotes."
                )
                return {
                    "quotes": [],
                    "locations": [location],
                    "context": f"Evidence extraction failed: {error_type}",
                    "confidence": 0,
                    "error": error_msg
                }

    def _extract_text_from_location(
        self,
        document: Dict[str, Any],
        slide: str,
        location: str
    ) -> str:
        """
        Extract text from a specific location in the document

        Args:
            document: Document dictionary
            slide: Slide identifier
            location: Location within slide

        Returns:
            Extracted text
        """
        try:
            import json

            # Try to find the slide in the document
            slide_key = None

            # Common slide key patterns
            slide_patterns = [
                slide.lower(),
                f"slide_{slide}",
                f"slide{slide}",
                f"page_{slide}",
                f"diapositive_{slide}"
            ]

            for key in document.keys():
                key_lower = key.lower()
                if any(pattern in key_lower for pattern in slide_patterns):
                    slide_key = key
                    break

            # If slide found, extract text
            if slide_key and slide_key in document:
                slide_content = document[slide_key]

                if isinstance(slide_content, dict):
                    return json.dumps(slide_content, ensure_ascii=False)
                elif isinstance(slide_content, str):
                    return slide_content

            # Fallback: return all document text
            return json.dumps(document, ensure_ascii=False)

        except Exception as e:
            error_type = type(e).__name__
            self.logger.error(
                f"[{self.name}] ❌ Text extraction from location failed: {error_type}: {str(e)[:100]}"
            )
            self.logger.error(
                f"[{self.name}] Location: slide={slide}, location={location}"
            )
            self.logger.warning(
                f"[{self.name}] Returning empty string. Evidence extraction for this location will be incomplete."
            )
            return ""

    def _needs_review(self, violations: List[Dict[str, Any]]) -> bool:
        """
        Determine if violations need human review

        Args:
            violations: List of violations

        Returns:
            True if any violation needs review
        """
        review_threshold = 70

        for violation in violations:
            confidence = violation.get("confidence", 100)
            evidence_confidence = violation.get("evidence_confidence", 100)

            # Need review if confidence is low
            if confidence < review_threshold or evidence_confidence < review_threshold:
                return True

        return False


# ============================================================================
# AGENT REGISTRATION
# ============================================================================

def create_evidence_agent(config: Optional[Dict[str, Any]] = None, ai_engine=None) -> EvidenceAgent:
    """
    Factory function to create an Evidence Agent

    Args:
        config: Configuration dictionary
        ai_engine: Optional AIEngine for semantic analysis

    Returns:
        Configured EvidenceAgent instance
    """
    agent_config = AgentConfig(
        name="evidence",
        enabled=config.get("enabled", True) if config else True,
        timeout_seconds=config.get("timeout_seconds", 30.0) if config else 30.0,
        retry_attempts=config.get("retry_attempts", 3) if config else 3,
        log_level=config.get("log_level", "INFO") if config else "INFO",
        custom_settings=config.get("custom_settings", {}) if config else {}
    )

    return EvidenceAgent(config=agent_config, ai_engine=ai_engine)


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    import json
    from data_models_multiagent import initialize_compliance_state

    logger.info("="*70)
    logger.info("Evidence Agent - Standalone Test")
    logger.info("="*70)

    # Create test document
    test_document = {
        "document_metadata": {
            "fund_isin": "FR0000000000",
            "client_type": "retail",
            "document_type": "fund_presentation"
        },
        "slide_2": {
            "content": "Le fonds a généré une performance de +15.5% en 2023. "
                      "Le rendement annualisé est de 8.2% sur 5 ans."
        },
        "slide_3": {
            "content": "L'objectif de performance est d'obtenir des résultats attractifs."
        }
    }

    # Create test violations
    test_violations = [
        {
            "rule": "PERF_DATA_WITHOUT_DISCLAIMER",
            "type": "PERFORMANCE",
            "severity": "HIGH",
            "slide": "2",
            "location": "Slide 2",
            "evidence": "Performance data found",
            "ai_reasoning": "Performance data without disclaimer",
            "confidence": 75
        },
        {
            "rule": "MISSING_DISCLAIMER",
            "type": "DISCLAIMER",
            "severity": "HIGH",
            "slide": "2",
            "location": "Slide 2",
            "evidence": "Disclaimer missing",
            "ai_reasoning": "Required disclaimer not found",
            "confidence": 70
        },
        {
            "rule": "DESCRIPTIVE_PERFORMANCE",
            "type": "PERFORMANCE",
            "severity": "LOW",
            "slide": "3",
            "location": "Slide 3",
            "evidence": "Descriptive performance text",
            "ai_reasoning": "Only descriptive text, no actual data",
            "confidence": 90
        }
    ]

    # Initialize state
    config = {
        "agents": {
            "evidence": {
                "enabled": True,
                "min_confidence_for_evidence": 0,
                "enhance_all_violations": True
            }
        }
    }

    state = initialize_compliance_state(test_document, "test_doc_001", config)
    state["violations"] = test_violations
    state["normalized_document"] = test_document

    # Create and run agent
    logger.info("\n🤖 Creating Evidence Agent...")
    agent = create_evidence_agent(config.get("agents", {}).get("evidence", {}))

    logger.info("\n🔍 Processing violations for evidence extraction...")
    result_state = agent(state)

    # Display results
    logger.info("\n" + "="*70)
    logger.info("Evidence Extraction Results:")
    logger.info("="*70)

    enhanced_violations = result_state.get("violations", [])
    evidence_extractions = result_state.get("evidence_extractions", {})

    logger.info(f"\nTotal violations: {len(enhanced_violations)}")
    logger.info(f"Evidence extractions: {len(evidence_extractions)}")

    for i, violation in enumerate(enhanced_violations, 1):
        logger.info(f"\n--- Violation {i} ---")
        logger.info(f"Rule: {violation.get('rule', 'unknown')}")
        logger.info(f"Type: {violation.get('type', 'unknown')}")
        logger.info(f"Confidence: {violation.get('confidence', 0)}%")

        if "evidence_quotes" in violation:
            logger.info(f"Evidence Quotes: {len(violation['evidence_quotes'])}")
            for quote in violation["evidence_quotes"][:2]:
                logger.info(f"  - {quote[:100]}...")

        if "evidence_locations" in violation:
            logger.info(f"Evidence Locations: {violation['evidence_locations']}")

        if "evidence_context" in violation:
            logger.info(f"Evidence Context: {violation['evidence_context']}")

        if "evidence_confidence" in violation:
            logger.info(f"Evidence Confidence: {violation['evidence_confidence']}%")

    logger.info("\n" + "="*70)
    logger.info("✅ Evidence Agent test completed successfully!")
    logger.info("="*70)
