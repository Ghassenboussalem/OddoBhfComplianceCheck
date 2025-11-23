#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context Agent

This module provides functionality for the multi-agent compliance system.
"""

"""
Context Agent for Multi-Agent Compliance System

The Context Agent analyzes text context and intent to eliminate false positives
by understanding semantic meaning. It distinguishes between fund strategy
descriptions (ALLOWED) and client investment advice (PROHIBITED).

Responsibilities:
- Analyze context for low-confidence violations
- Classify intent (ADVICE, DESCRIPTION, FACT, EXAMPLE)
- Extract subject (WHO performs the action)
- Update violation confidence based on context
- Filter false positives through semantic understanding

Requirements: 1.2, 2.3, 9.1, 9.2, 9.3
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import base agent framework
from agents.base_agent import BaseAgent, AgentConfig

# Import state models
from data_models_multiagent import (
    ComplianceState,
    WorkflowStatus,
    ViolationStatus,
    update_state_timestamp
)

# Import context tools
from tools.context_tools import (
    analyze_context,
    classify_intent,
    extract_subject,
    is_fund_strategy_description,
    is_investment_advice
)

# Import AI engine
from ai_engine import AIEngine

# Import performance optimizer
from performance_optimizer import get_profiler, get_deduplicator, profile_function


# Configure logging
logger = logging.getLogger(__name__)


class ContextAgent(BaseAgent):
    """
    Context Agent - Analyzes context and intent to eliminate false positives

    This agent runs after the Aggregator Agent when violations have confidence
    scores below the context threshold (default 80%). It performs deep semantic
    analysis to understand the true meaning and intent of flagged text.

    Key capabilities:
    1. Analyze semantic context using AI
    2. Classify intent (ADVICE, DESCRIPTION, FACT, EXAMPLE)
    3. Extract subject (fund, client, general)
    4. Distinguish fund descriptions from client advice
    5. Update violation confidence based on context
    6. Filter false positives

    Requirements addressed:
    - 1.2: Agent-based architecture with specialized responsibilities
    - 2.3: AI-enhanced features (context analysis, intent classification)
    - 9.1: Context analysis to determine WHO performs actions and WHAT the intent is
    - 9.2: Intent classification as ADVICE, DESCRIPTION, FACT, or EXAMPLE
    - 9.3: Semantic understanding to eliminate false positives
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        ai_engine: Optional[AIEngine] = None,
        **kwargs
    ):
        """
        Initialize Context Agent

        Args:
            config: Agent configuration
            ai_engine: AIEngine instance for LLM calls
            **kwargs: Additional configuration options
        """
        if config is None:
            config = AgentConfig(
                name="context",
                enabled=True,
                timeout_seconds=60.0,  # Context analysis can take longer
                retry_attempts=2,
                log_level="INFO"
            )

        super().__init__(config, **kwargs)

        # Store AI engine
        self.ai_engine = ai_engine
        if not self.ai_engine:
            self.logger.warning("No AI engine provided - context analysis will use fallback rules")

        # Context-specific configuration
        self.confidence_boost_threshold = kwargs.get('confidence_boost_threshold', 70)
        self.false_positive_threshold = kwargs.get('false_positive_threshold', 85)
        self.analyze_all_violations = kwargs.get('analyze_all_violations', False)

        self.logger.info(f"Context Agent initialized")
        self.logger.info(f"  AI engine: {'available' if self.ai_engine else 'not available (using fallback)'}")
        self.logger.info(f"  Confidence boost threshold: {self.confidence_boost_threshold}%")
        self.logger.info(f"  False positive threshold: {self.false_positive_threshold}%")

    def process(self, state: ComplianceState) -> ComplianceState:
        """
        Process the compliance state - analyze context for low-confidence violations

        This method:
        1. Filters violations that need context analysis
        2. Analyzes context and intent for each violation
        3. Updates violation confidence based on semantic understanding
        4. Filters false positives
        5. Stores analysis results in state

        Args:
            state: Current compliance state with violations from aggregator

        Returns:
            Updated compliance state with context-analyzed violations
        """
        self.logger.info("="*70)
        self.logger.info("CONTEXT AGENT: Starting context analysis")
        self.logger.info("="*70)

        # Update workflow status
        state["workflow_status"] = WorkflowStatus.ANALYZING.value
        state = update_state_timestamp(state)

        # Get violations
        violations = list(state.get("violations", []))

        if not violations:
            self.logger.info("No violations to analyze")
            return state

        self.logger.info(f"Total violations: {len(violations)}")

        # Filter violations that need context analysis
        violations_to_analyze = self._filter_violations_for_analysis(violations, state)

        if not violations_to_analyze:
            self.logger.info("No violations need context analysis")
            return state

        self.logger.info(f"Analyzing {len(violations_to_analyze)} violations with low confidence")

        # Initialize context analysis storage
        if "context_analysis" not in state:
            state["context_analysis"] = {}
        if "intent_classifications" not in state:
            state["intent_classifications"] = {}

        # Analyze each violation
        analyzed_count = 0
        false_positives_filtered = 0
        confidence_boosted = 0

        for i, violation in enumerate(violations_to_analyze, 1):
            self.logger.info(f"\n[{i}/{len(violations_to_analyze)}] Analyzing violation:")
            self.logger.info(f"  Rule: {violation.get('rule', 'unknown')}")
            self.logger.info(f"  Type: {violation.get('type', 'unknown')}")
            self.logger.info(f"  Current confidence: {violation.get('confidence', 0)}%")

            # Analyze context
            analysis_result = self._analyze_violation_context(violation, state)

            if analysis_result:
                analyzed_count += 1

                # Store analysis in state
                violation_key = self._get_violation_key(violation)
                state["context_analysis"][violation_key] = analysis_result["context"]
                state["intent_classifications"][violation_key] = analysis_result["intent"]

                # Update violation based on analysis
                updated = self._update_violation_from_analysis(
                    violation,
                    analysis_result,
                    state
                )

                if updated["filtered_as_false_positive"]:
                    false_positives_filtered += 1
                    self.logger.info(f"  ✓ Filtered as FALSE POSITIVE")
                elif updated["confidence_increased"]:
                    confidence_boosted += 1
                    self.logger.info(f"  ✓ Confidence boosted: {violation.get('confidence')}% → {updated['new_confidence']}%")
                else:
                    self.logger.info(f"  → No confidence change")

        # Log summary
        self._log_analysis_summary(
            analyzed_count,
            false_positives_filtered,
            confidence_boosted,
            violations
        )

        self.logger.info("Context analysis complete")

        return state

    def _filter_violations_for_analysis(
        self,
        violations: List[Dict[str, Any]],
        state: ComplianceState
    ) -> List[Dict[str, Any]]:
        """
        Filter violations that need context analysis

        By default, only analyzes violations with confidence < 80%.
        Can be configured to analyze all violations.

        Args:
            violations: List of all violations
            state: Current compliance state

        Returns:
            List of violations that need context analysis
        """
        if self.analyze_all_violations:
            self.logger.info("Analyzing ALL violations (analyze_all_violations=True)")
            return violations

        # Get context threshold from config
        config = state.get("config", {})
        routing_config = config.get("routing", {})
        context_threshold = routing_config.get("context_threshold", 80)

        # Filter low-confidence violations
        to_analyze = [
            v for v in violations
            if v.get("confidence", 100) < context_threshold
            and v.get("status", "") != ViolationStatus.FALSE_POSITIVE_FILTERED.value
        ]

        self.logger.info(f"Filtering violations with confidence < {context_threshold}%")
        self.logger.info(f"  Found {len(to_analyze)} violations needing analysis")

        return to_analyze

    @profile_function("context_agent.analyze_violation")
    def _analyze_violation_context(
        self,
        violation: Dict[str, Any],
        state: ComplianceState
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze context and intent for a single violation (optimized with deduplication)

        Args:
            violation: Violation to analyze
            state: Current compliance state

        Returns:
            Dictionary with context and intent analysis results, or None if analysis fails
        """
        try:
            # Get evidence text
            evidence = violation.get("evidence", "")
            if not evidence:
                self.logger.warning(
                    f"  ⚠️  No evidence text to analyze for violation: {violation.get('rule', 'unknown')}"
                )
                self.logger.warning(
                    f"  Skipping context analysis. Violation will retain original confidence: {violation.get('confidence', 0)}%"
                )
                return None

            # Get check type
            check_type = violation.get("type", "general")

            # Create deduplication key
            import hashlib
            dedup_key = hashlib.md5(f"{evidence}:{check_type}".encode()).hexdigest()

            # Use deduplicator to avoid redundant AI calls
            deduplicator = get_deduplicator()

            def analyze():
                # Analyze context using context tools
                self.logger.info(f"  Analyzing context...")
                context_result = analyze_context(evidence, check_type, self.ai_engine)

                # Handle ToolResult wrapper if present
                context_data = context_result.result if hasattr(context_result, 'result') else context_result

                self.logger.info(f"    Subject: {context_data.subject}")
                self.logger.info(f"    Intent: {context_data.intent}")
                self.logger.info(f"    Fund description: {context_data.is_fund_description}")
                self.logger.info(f"    Client advice: {context_data.is_client_advice}")
                self.logger.info(f"    Confidence: {context_data.confidence}%")

                # Classify intent
                self.logger.info(f"  Classifying intent...")
                intent_result = classify_intent(evidence, self.ai_engine)

                # Handle ToolResult wrapper if present
                intent_data = intent_result.result if hasattr(intent_result, 'result') else intent_result

                self.logger.info(f"    Intent type: {intent_data.intent_type}")
                self.logger.info(f"    Confidence: {intent_data.confidence}%")

                # Return analysis results
                return {
                    "context": {
                        "subject": context_data.subject,
                        "intent": context_data.intent,
                        "is_fund_description": context_data.is_fund_description,
                        "is_client_advice": context_data.is_client_advice,
                        "confidence": context_data.confidence,
                        "reasoning": context_data.reasoning,
                        "evidence": context_data.evidence
                    },
                    "intent": {
                        "intent_type": intent_data.intent_type,
                        "confidence": intent_data.confidence,
                        "subject": intent_data.subject,
                        "reasoning": intent_data.reasoning,
                        "evidence": intent_data.evidence
                    }
                }

            # Get or call with deduplication
            return deduplicator.get_or_call(dedup_key, analyze)

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)[:200]
            self.logger.error(
                f"  ❌ Context analysis failed: {error_type}: {error_msg}"
            )
            self.logger.error(
                f"  Violation: {violation.get('rule', 'unknown')} will retain original confidence"
            )
            self.logger.error(
                f"  Resolution: Check AI engine connectivity. Verify API credentials. "
                f"Ensure context analysis tools are properly configured."
            )
            self.logger.debug(f"  Full error details:", exc_info=True)
            return None

    def _update_violation_from_analysis(
        self,
        violation: Dict[str, Any],
        analysis: Dict[str, Any],
        state: ComplianceState
    ) -> Dict[str, Any]:
        """
        Update violation confidence and status based on context analysis

        Logic:
        - If context shows fund description (not client advice), boost confidence or filter
        - If context shows client advice, keep or reduce confidence
        - If intent is DESCRIPTION/FACT/EXAMPLE, likely false positive
        - If intent is ADVICE, likely true positive

        Args:
            violation: Violation to update
            analysis: Context and intent analysis results
            state: Current compliance state

        Returns:
            Dictionary with update results
        """
        context = analysis["context"]
        intent = analysis["intent"]

        original_confidence = violation.get("confidence", 0)
        new_confidence = original_confidence
        filtered_as_false_positive = False
        confidence_increased = False

        # Get violation type
        violation_type = violation.get("type", "").upper()

        # Apply context-based logic

        # Case 1: Investment advice violations
        if "ADVICE" in violation_type or "SECURITIES" in violation_type:
            # If context shows fund description (not client advice)
            if context["is_fund_description"] and not context["is_client_advice"]:
                if context["confidence"] >= self.false_positive_threshold:
                    # High confidence fund description - filter as false positive
                    violation["status"] = ViolationStatus.FALSE_POSITIVE_FILTERED.value
                    violation["confidence"] = 0
                    violation["context_reasoning"] = (
                        f"Filtered by context analysis: Text describes fund strategy, not client advice. "
                        f"Subject: {context['subject']}, Intent: {context['intent']}. "
                        f"Context confidence: {context['confidence']}%"
                    )
                    filtered_as_false_positive = True
                    new_confidence = 0
                elif context["confidence"] >= self.confidence_boost_threshold:
                    # Medium confidence - reduce violation confidence
                    new_confidence = max(original_confidence - 20, 30)
                    violation["confidence"] = new_confidence
                    violation["context_reasoning"] = (
                        f"Confidence reduced by context analysis: Likely fund description. "
                        f"Context confidence: {context['confidence']}%"
                    )

            # If context confirms client advice
            elif context["is_client_advice"]:
                if context["confidence"] >= self.confidence_boost_threshold:
                    # Boost confidence
                    new_confidence = min(original_confidence + 15, 95)
                    violation["confidence"] = new_confidence
                    violation["context_reasoning"] = (
                        f"Confidence boosted by context analysis: Confirmed client advice. "
                        f"Context confidence: {context['confidence']}%"
                    )
                    confidence_increased = True

        # Case 2: Intent-based filtering
        if intent["intent_type"] in ["DESCRIPTION", "FACT", "EXAMPLE"]:
            if intent["confidence"] >= self.false_positive_threshold:
                # High confidence non-advice intent
                if violation.get("status") != ViolationStatus.FALSE_POSITIVE_FILTERED.value:
                    # Reduce confidence significantly
                    new_confidence = max(original_confidence - 25, 20)
                    violation["confidence"] = new_confidence
                    violation["intent_reasoning"] = (
                        f"Confidence reduced: Intent classified as {intent['intent_type']} "
                        f"(not ADVICE). Intent confidence: {intent['confidence']}%"
                    )

        elif intent["intent_type"] == "ADVICE":
            if intent["confidence"] >= self.confidence_boost_threshold:
                # High confidence advice intent - boost confidence
                new_confidence = min(original_confidence + 10, 95)
                violation["confidence"] = new_confidence
                violation["intent_reasoning"] = (
                    f"Confidence boosted: Intent classified as ADVICE. "
                    f"Intent confidence: {intent['confidence']}%"
                )
                confidence_increased = True

        # Case 3: Subject-based filtering
        if context["subject"] in ["fund", "strategy"] and violation_type in ["ADVICE", "SECURITIES"]:
            # Fund/strategy as subject for advice violation - likely false positive
            if context["confidence"] >= self.confidence_boost_threshold:
                new_confidence = max(original_confidence - 15, 25)
                violation["confidence"] = new_confidence
                if not violation.get("context_reasoning"):
                    violation["context_reasoning"] = (
                        f"Confidence reduced: Subject is {context['subject']}, not client. "
                        f"Context confidence: {context['confidence']}%"
                    )

        # Update violation metadata
        violation["context_analyzed"] = True
        violation["context_analysis_timestamp"] = datetime.now().isoformat()

        return {
            "filtered_as_false_positive": filtered_as_false_positive,
            "confidence_increased": confidence_increased,
            "original_confidence": original_confidence,
            "new_confidence": new_confidence,
            "confidence_change": new_confidence - original_confidence
        }

    def _get_violation_key(self, violation: Dict[str, Any]) -> str:
        """
        Generate unique key for violation

        Args:
            violation: Violation dictionary

        Returns:
            Unique key string
        """
        rule = violation.get("rule", "unknown")
        slide = violation.get("slide", "unknown")
        location = violation.get("location", "unknown")

        return f"{rule}_{slide}_{location}"

    def _log_analysis_summary(
        self,
        analyzed_count: int,
        false_positives_filtered: int,
        confidence_boosted: int,
        all_violations: List[Dict[str, Any]]
    ):
        """
        Log summary of context analysis results

        Args:
            analyzed_count: Number of violations analyzed
            false_positives_filtered: Number filtered as false positives
            confidence_boosted: Number with confidence boosted
            all_violations: All violations (for final stats)
        """
        self.logger.info("="*70)
        self.logger.info("CONTEXT ANALYSIS SUMMARY")
        self.logger.info("="*70)
        self.logger.info(f"Violations analyzed: {analyzed_count}")
        self.logger.info(f"False positives filtered: {false_positives_filtered}")
        self.logger.info(f"Confidence boosted: {confidence_boosted}")

        # Count remaining violations by confidence
        remaining_violations = [
            v for v in all_violations
            if v.get("status") != ViolationStatus.FALSE_POSITIVE_FILTERED.value
        ]

        if remaining_violations:
            high_conf = len([v for v in remaining_violations if v.get("confidence", 0) >= 80])
            medium_conf = len([v for v in remaining_violations if 50 <= v.get("confidence", 0) < 80])
            low_conf = len([v for v in remaining_violations if v.get("confidence", 0) < 50])

            self.logger.info(f"\nRemaining violations: {len(remaining_violations)}")
            self.logger.info(f"  High confidence (>=80%): {high_conf}")
            self.logger.info(f"  Medium confidence (50-79%): {medium_conf}")
            self.logger.info(f"  Low confidence (<50%): {low_conf}")

        self.logger.info("="*70)

    def get_context_statistics(self, state: ComplianceState) -> Dict[str, Any]:
        """
        Get detailed context analysis statistics from state

        Args:
            state: Current compliance state

        Returns:
            Dictionary with context analysis statistics
        """
        violations = state.get("violations", [])
        context_analysis = state.get("context_analysis", {})
        intent_classifications = state.get("intent_classifications", {})

        # Count violations by status
        analyzed = len([v for v in violations if v.get("context_analyzed", False)])
        filtered = len([
            v for v in violations
            if v.get("status") == ViolationStatus.FALSE_POSITIVE_FILTERED.value
        ])

        # Count by intent type
        intent_counts = {}
        for intent_data in intent_classifications.values():
            intent_type = intent_data.get("intent_type", "UNKNOWN")
            intent_counts[intent_type] = intent_counts.get(intent_type, 0) + 1

        # Count by subject
        subject_counts = {}
        for context_data in context_analysis.values():
            subject = context_data.get("subject", "unknown")
            subject_counts[subject] = subject_counts.get(subject, 0) + 1

        return {
            "total_violations": len(violations),
            "analyzed": analyzed,
            "false_positives_filtered": filtered,
            "context_analyses": len(context_analysis),
            "intent_classifications": len(intent_classifications),
            "intent_distribution": intent_counts,
            "subject_distribution": subject_counts,
            "ai_engine_available": self.ai_engine is not None
        }


# Export
__all__ = ["ContextAgent"]
