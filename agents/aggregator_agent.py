#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregator Agent

This module provides functionality for the multi-agent compliance system.
"""

"""
Aggregator Agent for Multi-Agent Compliance System

The Aggregator Agent collects and consolidates violations from all specialist
agents, calculates confidence scores, deduplicates results, categorizes violations,
and determines the next workflow action based on confidence thresholds.

Responsibilities:
- Collect violations from all specialist agents
- Calculate aggregate confidence scores
- Deduplicate violations based on rule, location, and evidence
- Categorize violations by type, severity, and status
- Determine next action (context analysis, review, or completion)

Requirements: 1.2, 4.1, 4.2, 6.1
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict

# Import base agent framework
from agents.base_agent import BaseAgent, AgentConfig

# Import state models
from data_models_multiagent import (
    ComplianceState,
    WorkflowStatus,
    ViolationStatus,
    update_state_timestamp
)


# Configure logging
logger = logging.getLogger(__name__)


class AggregatorAgent(BaseAgent):
    """
    Aggregator Agent - Consolidates results from all specialist agents

    This agent runs after all specialist compliance agents (Structure, Performance,
    Securities, General, Prospectus, Registration, ESG) have completed their checks.

    It performs the following operations:
    1. Collects all violations from the state
    2. Deduplicates violations based on rule, location, and evidence similarity
    3. Calculates confidence scores for each violation and overall
    4. Categorizes violations by type, severity, and status
    5. Determines next workflow action based on confidence thresholds

    Requirements addressed:
    - 1.2: Agent-based architecture with specialized responsibilities
    - 4.1: Conditional routing based on confidence scores
    - 4.2: Aggregation of results from multiple agents
    - 6.1: Standard state interface for agent communication
    """

    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """
        Initialize Aggregator Agent

        Args:
            config: Agent configuration
            **kwargs: Additional configuration options
        """
        if config is None:
            config = AgentConfig(
                name="aggregator",
                enabled=True,
                timeout_seconds=30.0,
                retry_attempts=1,
                log_level="INFO"
            )

        super().__init__(config, **kwargs)

        # Aggregator-specific configuration
        self.context_threshold = kwargs.get('context_threshold', 80)
        self.review_threshold = kwargs.get('review_threshold', 70)
        self.deduplication_enabled = kwargs.get('deduplication_enabled', True)
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.85)

        self.logger.info(f"Aggregator Agent initialized")
        self.logger.info(f"  Context threshold: {self.context_threshold}%")
        self.logger.info(f"  Review threshold: {self.review_threshold}%")
        self.logger.info(f"  Deduplication: {self.deduplication_enabled}")

    def process(self, state: ComplianceState) -> ComplianceState:
        """
        Process the compliance state - aggregate and analyze violations

        This method:
        1. Collects all violations from specialist agents
        2. Deduplicates violations
        3. Calculates confidence scores
        4. Categorizes violations
        5. Determines next workflow action

        Args:
            state: Current compliance state with violations from all agents

        Returns:
            Updated compliance state with aggregated results and next action
        """
        self.logger.info("="*70)
        self.logger.info("AGGREGATOR: Starting violation aggregation and analysis")
        self.logger.info("="*70)

        # Update workflow status
        state["workflow_status"] = WorkflowStatus.AGGREGATING.value
        state = update_state_timestamp(state)

        # Get all violations
        violations = list(state.get("violations", []))

        self.logger.info(f"Collected {len(violations)} violations from specialist agents")

        # Log violations by agent
        self._log_violations_by_agent(violations)

        # Deduplicate violations if enabled
        if self.deduplication_enabled and violations:
            original_count = len(violations)
            violations = self._deduplicate_violations(violations)
            deduped_count = len(violations)

            if original_count != deduped_count:
                self.logger.info(f"Deduplication: {original_count} → {deduped_count} violations")
                self.logger.info(f"  Removed {original_count - deduped_count} duplicates")

        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(violations)
        aggregated_confidence = self._calculate_aggregated_confidence(violations)

        self.logger.info(f"Calculated confidence scores:")
        self.logger.info(f"  Aggregated confidence: {aggregated_confidence}%")

        # Categorize violations
        categorization = self._categorize_violations(violations)

        self.logger.info(f"Violation categorization:")
        self.logger.info(f"  By severity: {categorization['by_severity']}")
        self.logger.info(f"  By type: {categorization['by_type']}")
        self.logger.info(f"  By status: {categorization['by_status']}")

        # Determine next action based on confidence
        next_action = self._determine_next_action(violations, aggregated_confidence, state)

        self.logger.info(f"Next action determined: {next_action}")

        # Update state with aggregated results
        state["violations"] = violations
        state["aggregated_confidence"] = aggregated_confidence
        state["next_action"] = next_action

        # Store categorization in state for reporting
        if "violation_categorization" not in state:
            state["violation_categorization"] = {}
        state["violation_categorization"] = categorization

        # Log summary
        self._log_aggregation_summary(violations, aggregated_confidence, next_action)

        self.logger.info("Aggregator processing complete")

        return state

    def _log_violations_by_agent(self, violations: List[Dict[str, Any]]):
        """
        Log violations grouped by agent for monitoring

        Args:
            violations: List of all violations
        """
        by_agent = defaultdict(int)
        by_severity = defaultdict(lambda: defaultdict(int))

        for violation in violations:
            agent = violation.get("agent", "unknown")
            severity = violation.get("severity", "unknown")
            by_agent[agent] += 1
            by_severity[agent][severity] += 1

        if by_agent:
            self.logger.info("="*70)
            self.logger.info("Violations by Agent:")
            self.logger.info("="*70)
            for agent, count in sorted(by_agent.items()):
                self.logger.info(f"  {agent}: {count} violation(s)")
                # Show severity breakdown
                severities = by_severity[agent]
                if severities:
                    severity_str = ", ".join([f"{sev}: {cnt}" for sev, cnt in sorted(severities.items())])
                    self.logger.info(f"    └─ Severity breakdown: {severity_str}")
            self.logger.info("="*70)
        else:
            self.logger.info("✓ No violations found by any agent - document is compliant!")

    def _deduplicate_violations(self, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate violations based on rule, location, and evidence similarity

        Two violations are considered duplicates if they have:
        - Same rule
        - Same slide/location
        - Similar evidence (>85% similarity)

        When duplicates are found, keeps the one with higher confidence.

        Args:
            violations: List of violations to deduplicate

        Returns:
            Deduplicated list of violations
        """
        if not violations:
            return violations

        self.logger.info("Starting deduplication process...")

        # Group violations by (rule, slide, location) key
        grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

        for violation in violations:
            key = (
                violation.get("rule", ""),
                violation.get("slide", ""),
                violation.get("location", "")
            )
            grouped[key].append(violation)

        # Process each group
        deduplicated = []
        duplicates_removed = 0

        for key, group in grouped.items():
            if len(group) == 1:
                # No duplicates in this group
                deduplicated.append(group[0])
            else:
                # Multiple violations with same key - check evidence similarity
                self.logger.info(f"Found {len(group)} violations with key {key}")

                # Keep track of which violations to keep
                to_keep = []
                processed = set()

                for i, v1 in enumerate(group):
                    if i in processed:
                        continue

                    # Find all similar violations
                    similar_group = [v1]
                    processed.add(i)

                    for j, v2 in enumerate(group[i+1:], start=i+1):
                        if j in processed:
                            continue

                        # Check evidence similarity
                        if self._are_violations_similar(v1, v2):
                            similar_group.append(v2)
                            processed.add(j)

                    # Keep the one with highest confidence
                    if len(similar_group) > 1:
                        best = max(similar_group, key=lambda v: v.get("confidence", 0))
                        to_keep.append(best)
                        duplicates_removed += len(similar_group) - 1
                        self.logger.info(f"  Merged {len(similar_group)} similar violations, kept highest confidence ({best.get('confidence')}%)")
                    else:
                        to_keep.append(v1)

                deduplicated.extend(to_keep)

        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate violations")

        return deduplicated

    def _are_violations_similar(self, v1: Dict[str, Any], v2: Dict[str, Any]) -> bool:
        """
        Check if two violations are similar based on evidence

        Args:
            v1: First violation
            v2: Second violation

        Returns:
            True if violations are similar enough to be considered duplicates
        """
        evidence1 = v1.get("evidence", "").lower().strip()
        evidence2 = v2.get("evidence", "").lower().strip()

        # If evidence is identical, they're similar
        if evidence1 == evidence2:
            return True

        # If either is empty, not similar
        if not evidence1 or not evidence2:
            return False

        # Calculate simple similarity (Jaccard similarity on words)
        words1 = set(evidence1.split())
        words2 = set(evidence2.split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        similarity = intersection / union if union > 0 else 0

        self.logger.debug(f"Similarity between violations: {similarity:.2f}")
        self.logger.debug(f"  Evidence 1: {evidence1[:50]}...")
        self.logger.debug(f"  Evidence 2: {evidence2[:50]}...")

        return similarity >= self.similarity_threshold

    def _calculate_confidence_scores(self, violations: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Calculate confidence scores for each violation category

        Args:
            violations: List of violations

        Returns:
            Dictionary mapping category to confidence score
        """
        scores = {}

        # Calculate by type
        by_type = defaultdict(list)
        for violation in violations:
            vtype = violation.get("type", "UNKNOWN")
            confidence = violation.get("confidence", 0)
            by_type[vtype].append(confidence)

        for vtype, confidences in by_type.items():
            if confidences:
                scores[f"type_{vtype}"] = int(sum(confidences) / len(confidences))

        # Calculate by severity
        by_severity = defaultdict(list)
        for violation in violations:
            severity = violation.get("severity", "UNKNOWN")
            confidence = violation.get("confidence", 0)
            by_severity[severity].append(confidence)

        for severity, confidences in by_severity.items():
            if confidences:
                scores[f"severity_{severity}"] = int(sum(confidences) / len(confidences))

        return scores

    def _calculate_aggregated_confidence(self, violations: List[Dict[str, Any]]) -> int:
        """
        Calculate overall aggregated confidence score

        Uses weighted average based on severity:
        - Critical: weight 4
        - High: weight 3
        - Medium: weight 2
        - Low: weight 1

        Args:
            violations: List of violations

        Returns:
            Aggregated confidence score (0-100)
        """
        if not violations:
            return 100  # No violations = 100% confidence

        severity_weights = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }

        total_weighted_confidence = 0
        total_weight = 0

        for violation in violations:
            confidence = violation.get("confidence", 0)
            severity = violation.get("severity", "medium").lower()
            weight = severity_weights.get(severity, 2)

            total_weighted_confidence += confidence * weight
            total_weight += weight

        if total_weight == 0:
            return 0

        aggregated = int(total_weighted_confidence / total_weight)

        return aggregated

    def _categorize_violations(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Categorize violations by type, severity, and status

        Args:
            violations: List of violations

        Returns:
            Dictionary with categorization results
        """
        categorization = {
            "by_type": defaultdict(int),
            "by_severity": defaultdict(int),
            "by_status": defaultdict(int),
            "by_agent": defaultdict(int),
            "total": len(violations)
        }

        for violation in violations:
            # By type
            vtype = violation.get("type", "UNKNOWN")
            categorization["by_type"][vtype] += 1

            # By severity
            severity = violation.get("severity", "UNKNOWN")
            categorization["by_severity"][severity] += 1

            # By status
            status = violation.get("status", ViolationStatus.DETECTED.value)
            categorization["by_status"][status] += 1

            # By agent
            agent = violation.get("agent", "unknown")
            categorization["by_agent"][agent] += 1

        # Convert defaultdicts to regular dicts for JSON serialization
        categorization["by_type"] = dict(categorization["by_type"])
        categorization["by_severity"] = dict(categorization["by_severity"])
        categorization["by_status"] = dict(categorization["by_status"])
        categorization["by_agent"] = dict(categorization["by_agent"])

        return categorization

    def _determine_next_action(
        self,
        violations: List[Dict[str, Any]],
        aggregated_confidence: int,
        state: ComplianceState
    ) -> str:
        """
        Determine next workflow action based on confidence scores

        Decision logic:
        - If any violation has confidence < context_threshold (80): route to context agent
        - Else if any violation has confidence < review_threshold (70): route to reviewer
        - Else: workflow complete

        Requirements addressed:
        - 4.1: Conditional routing based on confidence scores

        Args:
            violations: List of violations
            aggregated_confidence: Overall confidence score
            state: Current compliance state

        Returns:
            Next action: "context_analysis", "review", or "complete"
        """
        if not violations:
            self.logger.info("No violations found - workflow complete")
            return "complete"

        # Get thresholds from config if available
        config = state.get("config", {})
        routing_config = config.get("routing", {})

        context_threshold = routing_config.get("context_threshold", self.context_threshold)
        review_threshold = routing_config.get("review_threshold", self.review_threshold)

        # Check if any violation needs context analysis
        low_confidence_violations = [
            v for v in violations
            if v.get("confidence", 100) < context_threshold
        ]

        if low_confidence_violations:
            self.logger.info(f"Found {len(low_confidence_violations)} violations with confidence < {context_threshold}%")
            self.logger.info("Routing to context agent for deeper analysis")
            return "context_analysis"

        # Check if any violation needs human review
        very_low_confidence_violations = [
            v for v in violations
            if v.get("confidence", 100) < review_threshold
        ]

        if very_low_confidence_violations:
            self.logger.info(f"Found {len(very_low_confidence_violations)} violations with confidence < {review_threshold}%")
            self.logger.info("Routing to reviewer for human validation")
            return "review"

        # All violations have acceptable confidence
        self.logger.info(f"All violations have confidence >= {context_threshold}%")
        self.logger.info("Workflow complete - no further analysis needed")
        return "complete"

    def _log_aggregation_summary(
        self,
        violations: List[Dict[str, Any]],
        aggregated_confidence: int,
        next_action: str
    ):
        """
        Log summary of aggregation results

        Args:
            violations: List of violations
            aggregated_confidence: Overall confidence score
            next_action: Determined next action
        """
        self.logger.info("="*70)
        self.logger.info("AGGREGATION SUMMARY")
        self.logger.info("="*70)
        self.logger.info(f"Total violations: {len(violations)}")
        self.logger.info(f"Aggregated confidence: {aggregated_confidence}%")
        self.logger.info(f"Next action: {next_action}")

        if violations:
            # Confidence distribution
            high_conf = len([v for v in violations if v.get("confidence", 0) >= 80])
            medium_conf = len([v for v in violations if 50 <= v.get("confidence", 0) < 80])
            low_conf = len([v for v in violations if v.get("confidence", 0) < 50])

            self.logger.info(f"Confidence distribution:")
            self.logger.info(f"  High (>=80%): {high_conf}")
            self.logger.info(f"  Medium (50-79%): {medium_conf}")
            self.logger.info(f"  Low (<50%): {low_conf}")

        self.logger.info("="*70)

    def get_aggregation_statistics(self, state: ComplianceState) -> Dict[str, Any]:
        """
        Get detailed aggregation statistics from state

        Args:
            state: Current compliance state

        Returns:
            Dictionary with aggregation statistics
        """
        violations = state.get("violations", [])
        categorization = state.get("violation_categorization", {})

        return {
            "total_violations": len(violations),
            "aggregated_confidence": state.get("aggregated_confidence", 0),
            "next_action": state.get("next_action", "unknown"),
            "categorization": categorization,
            "confidence_scores": state.get("confidence_scores", {}),
            "deduplication_enabled": self.deduplication_enabled,
            "thresholds": {
                "context": self.context_threshold,
                "review": self.review_threshold
            }
        }


# Export
__all__ = ["AggregatorAgent"]
