#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reviewer Agent

This module provides functionality for the multi-agent compliance system.
"""

"""
Reviewer Agent for Multi-Agent Compliance System

The Reviewer Agent manages the Human-in-the-Loop (HITL) review process for
low-confidence violations. It queues violations for human review, manages
priority scoring, supports batch operations, and implements HITL interrupt
mechanisms for LangGraph workflow integration.

Responsibilities:
- Queue low-confidence violations for human review
- Calculate and manage priority scores
- Filter and batch review items by various criteria
- Implement HITL interrupt mechanism for LangGraph
- Track review queue statistics
- Support batch review operations

Requirements: 1.2, 2.4, 10.1, 10.2, 10.3, 10.4, 10.5
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

# Import review tools
from tools.review_tools import (
    queue_for_review,
    calculate_priority_score,
    filter_reviews,
    batch_review_by_check_type,
    batch_review_by_document,
    batch_review_by_severity,
    batch_review_by_confidence_range,
    get_similar_reviews,
    get_queue_statistics,
    get_next_review
)

# Import review manager
from review_manager import ReviewManager, ReviewItem, ReviewStatus


# Configure logging
logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    """
    Reviewer Agent - Manages Human-in-the-Loop review process

    This agent runs when violations have confidence scores below the review
    threshold (typically 70%). It queues these violations for human review,
    manages priority scoring, and supports batch operations.

    The agent integrates with LangGraph's interrupt mechanism to pause the
    workflow for human input and resume after review is complete.

    Key Features:
    - Automatic queuing of low-confidence violations
    - Priority-based queue management
    - Filtering by check type, severity, confidence range
    - Batch operations for similar violations
    - HITL interrupt integration with LangGraph
    - Queue statistics and monitoring

    Requirements addressed:
    - 1.2: Agent-based architecture with specialized responsibilities
    - 2.4: Preserve HITL integration with review queue and feedback loop
    - 10.1: Review queue management with priority scoring
    - 10.2: Present violations to human reviewers with full context
    - 10.3: Process human corrections and update confidence calibration
    - 10.4: Detect patterns in false positives
    - 10.5: Support batch operations for similar violations
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        review_manager: Optional[ReviewManager] = None,
        **kwargs
    ):
        """
        Initialize Reviewer Agent

        Args:
            config: Agent configuration
            review_manager: ReviewManager instance for queue management
            **kwargs: Additional configuration options
        """
        if config is None:
            config = AgentConfig(
                name="reviewer",
                enabled=True,
                timeout_seconds=60.0,
                retry_attempts=1,
                log_level="INFO"
            )

        super().__init__(config, **kwargs)

        # Review manager for queue operations
        self.review_manager = review_manager or ReviewManager()

        # Reviewer-specific configuration
        self.review_threshold = kwargs.get('review_threshold', 70)
        self.auto_queue_enabled = kwargs.get('auto_queue_enabled', True)
        self.batch_operations_enabled = kwargs.get('batch_operations_enabled', True)
        self.hitl_interrupt_enabled = kwargs.get('hitl_interrupt_enabled', True)

        # Initialize review tools
        self.tools = {
            "queue_for_review": queue_for_review,
            "calculate_priority_score": calculate_priority_score,
            "filter_reviews": filter_reviews,
            "batch_review_by_check_type": batch_review_by_check_type,
            "batch_review_by_document": batch_review_by_document,
            "batch_review_by_severity": batch_review_by_severity,
            "batch_review_by_confidence_range": batch_review_by_confidence_range,
            "get_similar_reviews": get_similar_reviews,
            "get_queue_statistics": get_queue_statistics,
            "get_next_review": get_next_review
        }

        self.logger.info(f"Reviewer Agent initialized")
        self.logger.info(f"  Review threshold: {self.review_threshold}%")
        self.logger.info(f"  Auto-queue: {self.auto_queue_enabled}")
        self.logger.info(f"  Batch operations: {self.batch_operations_enabled}")
        self.logger.info(f"  HITL interrupt: {self.hitl_interrupt_enabled}")

    def process(self, state: ComplianceState) -> ComplianceState:
        """
        Process the compliance state - queue violations for review

        This method:
        1. Identifies violations needing human review (confidence < threshold)
        2. Calculates priority scores for each violation
        3. Queues violations in the review manager
        4. Identifies similar violations for batch processing
        5. Updates state with review queue information
        6. Triggers HITL interrupt if enabled

        Args:
            state: Current compliance state with violations

        Returns:
            Updated compliance state with review queue information
        """
        self.logger.info("="*70)
        self.logger.info("REVIEWER: Starting review queue management")
        self.logger.info("="*70)

        # Update workflow status
        state["workflow_status"] = WorkflowStatus.REVIEWING.value
        state = update_state_timestamp(state)

        # Get violations from state
        violations = list(state.get("violations", []))

        if not violations:
            self.logger.info("No violations to review")
            return state

        # Get review threshold from config or use default
        config = state.get("config", {})
        hitl_config = config.get("hitl", {})
        review_threshold = hitl_config.get("review_threshold", self.review_threshold)

        # Filter violations needing review
        violations_needing_review = self._filter_violations_for_review(
            violations,
            review_threshold
        )

        self.logger.info(f"Found {len(violations_needing_review)} violations needing review")
        self.logger.info(f"  (confidence < {review_threshold}%)")

        if not violations_needing_review:
            self.logger.info("No violations need human review")
            return state

        # Queue violations for review
        if self.auto_queue_enabled:
            queued_items = self._queue_violations(
                violations_needing_review,
                state.get("document_id", "")
            )

            self.logger.info(f"Queued {len(queued_items)} violations for review")

            # Update state with review queue information
            if "review_queue" not in state:
                state["review_queue"] = []

            state["review_queue"] = list(state.get("review_queue", [])) + queued_items

        # Identify similar violations for batch processing
        if self.batch_operations_enabled:
            batch_opportunities = self._identify_batch_opportunities(
                violations_needing_review
            )

            if batch_opportunities:
                self.logger.info(f"Identified {len(batch_opportunities)} batch processing opportunities")
                state["batch_opportunities"] = batch_opportunities

        # Get queue statistics
        queue_stats = self._get_queue_statistics()
        state["queue_statistics"] = queue_stats

        self.logger.info(f"Queue statistics:")
        self.logger.info(f"  Total pending: {queue_stats.get('total_pending', 0)}")
        self.logger.info(f"  Total in review: {queue_stats.get('total_in_review', 0)}")
        self.logger.info(f"  Average confidence: {queue_stats.get('avg_confidence', 0):.1f}%")

        # Mark violations as queued for review
        for violation in violations_needing_review:
            violation["status"] = ViolationStatus.PENDING_REVIEW.value
            violation["queued_at"] = datetime.now().isoformat()

        # Update violations in state
        state["violations"] = violations

        # Set HITL interrupt flag if enabled
        if self.hitl_interrupt_enabled:
            state["hitl_interrupt_required"] = True
            state["hitl_interrupt_reason"] = f"{len(violations_needing_review)} violations need human review"
            self.logger.info("HITL interrupt flag set - workflow will pause for human review")

        # Log summary
        self._log_review_summary(violations_needing_review, queue_stats)

        self.logger.info("Reviewer processing complete")

        return state

    def _filter_violations_for_review(
        self,
        violations: List[Dict[str, Any]],
        threshold: int
    ) -> List[Dict[str, Any]]:
        """
        Filter violations that need human review based on confidence threshold

        Args:
            violations: List of all violations
            threshold: Confidence threshold (violations below this need review)

        Returns:
            List of violations needing review
        """
        needs_review = []

        for violation in violations:
            confidence = violation.get("confidence", 100)
            status = violation.get("status", ViolationStatus.DETECTED.value)

            # Skip if already reviewed or filtered
            if status in [
                ViolationStatus.CONFIRMED.value,
                ViolationStatus.REJECTED.value,
                ViolationStatus.FALSE_POSITIVE_FILTERED.value,
                ViolationStatus.PENDING_REVIEW.value
            ]:
                continue

            # Check if confidence is below threshold
            if 0 < confidence < threshold:
                needs_review.append(violation)
                self.logger.debug(f"Violation needs review: {violation.get('rule', 'unknown')} "
                                f"(confidence: {confidence}%)")

        return needs_review

    def _queue_violations(
        self,
        violations: List[Dict[str, Any]],
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Queue violations for human review

        Args:
            violations: List of violations to queue
            document_id: ID of the document being reviewed

        Returns:
            List of queued item information
        """
        queued_items = []

        for violation in violations:
            try:
                # Calculate priority score
                confidence = violation.get("confidence", 50)
                severity = violation.get("severity", "MEDIUM")

                priority_score = self.tools["calculate_priority_score"].invoke({
                    "confidence": confidence,
                    "severity": severity,
                    "age_hours": 0.0
                })

                # Queue for review
                review_id = self.tools["queue_for_review"].invoke({
                    "violation": violation,
                    "review_manager": self.review_manager,
                    "document_id": document_id,
                    "confidence": confidence,
                    "severity": severity
                })

                # Store queued item info
                queued_item = {
                    "review_id": review_id,
                    "violation_rule": violation.get("rule", ""),
                    "check_type": violation.get("type", ""),
                    "confidence": confidence,
                    "severity": severity,
                    "priority_score": priority_score,
                    "queued_at": datetime.now().isoformat()
                }

                queued_items.append(queued_item)

                self.logger.debug(f"Queued violation: {review_id} "
                                f"(priority: {priority_score:.2f})")

            except Exception as e:
                self.logger.error(f"Failed to queue violation: {e}")
                # Continue with other violations
                continue

        return queued_items

    def _identify_batch_opportunities(
        self,
        violations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify opportunities for batch processing of similar violations

        Groups violations by:
        - Same check type
        - Same severity
        - Similar confidence range

        Args:
            violations: List of violations

        Returns:
            List of batch opportunity descriptions
        """
        batch_opportunities = []

        # Group by check type
        by_check_type = {}
        for violation in violations:
            check_type = violation.get("type", "UNKNOWN")
            if check_type not in by_check_type:
                by_check_type[check_type] = []
            by_check_type[check_type].append(violation)

        # Identify batches with 2+ items
        for check_type, group in by_check_type.items():
            if len(group) >= 2:
                avg_confidence = sum(v.get("confidence", 0) for v in group) / len(group)

                batch_opportunities.append({
                    "type": "check_type",
                    "check_type": check_type,
                    "count": len(group),
                    "avg_confidence": avg_confidence,
                    "violations": [v.get("rule", "") for v in group]
                })

                self.logger.info(f"Batch opportunity: {len(group)} {check_type} violations")

        # Group by severity
        by_severity = {}
        for violation in violations:
            severity = violation.get("severity", "MEDIUM")
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(violation)

        # Identify batches with 3+ items
        for severity, group in by_severity.items():
            if len(group) >= 3:
                avg_confidence = sum(v.get("confidence", 0) for v in group) / len(group)

                batch_opportunities.append({
                    "type": "severity",
                    "severity": severity,
                    "count": len(group),
                    "avg_confidence": avg_confidence,
                    "violations": [v.get("rule", "") for v in group]
                })

                self.logger.info(f"Batch opportunity: {len(group)} {severity} severity violations")

        return batch_opportunities

    def _get_queue_statistics(self) -> Dict[str, Any]:
        """
        Get current review queue statistics

        Returns:
            Dictionary with queue statistics
        """
        try:
            stats = self.tools["get_queue_statistics"].invoke({
                "review_manager": self.review_manager
            })
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get queue statistics: {e}")
            return {
                "total_pending": 0,
                "total_in_review": 0,
                "total_reviewed": 0,
                "avg_confidence": 0.0,
                "by_check_type": {},
                "by_severity": {},
                "oldest_pending_age_hours": 0.0
            }

    def _log_review_summary(
        self,
        violations_needing_review: List[Dict[str, Any]],
        queue_stats: Dict[str, Any]
    ):
        """
        Log summary of review operations

        Args:
            violations_needing_review: List of violations queued for review
            queue_stats: Current queue statistics
        """
        self.logger.info("="*70)
        self.logger.info("REVIEW SUMMARY")
        self.logger.info("="*70)
        self.logger.info(f"Violations queued for review: {len(violations_needing_review)}")
        self.logger.info(f"Total pending in queue: {queue_stats.get('total_pending', 0)}")
        self.logger.info(f"Total in review: {queue_stats.get('total_in_review', 0)}")
        self.logger.info(f"Total reviewed: {queue_stats.get('total_reviewed', 0)}")

        if violations_needing_review:
            # Confidence distribution
            very_low = len([v for v in violations_needing_review if v.get("confidence", 0) < 30])
            low = len([v for v in violations_needing_review if 30 <= v.get("confidence", 0) < 50])
            medium_low = len([v for v in violations_needing_review if 50 <= v.get("confidence", 0) < 70])

            self.logger.info(f"Confidence distribution:")
            self.logger.info(f"  Very low (<30%): {very_low}")
            self.logger.info(f"  Low (30-49%): {low}")
            self.logger.info(f"  Medium-low (50-69%): {medium_low}")

            # Severity distribution
            by_severity = {}
            for v in violations_needing_review:
                severity = v.get("severity", "UNKNOWN")
                by_severity[severity] = by_severity.get(severity, 0) + 1

            self.logger.info(f"Severity distribution: {by_severity}")

        self.logger.info("="*70)

    def get_next_review_item(self, reviewer_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the next highest-priority review item for a reviewer

        This method is used by the review CLI or UI to retrieve the next
        item for human review.

        Args:
            reviewer_id: ID of the reviewer requesting an item

        Returns:
            Review item dictionary or None if queue is empty
        """
        try:
            review_item = self.tools["get_next_review"].invoke({
                "review_manager": self.review_manager,
                "reviewer_id": reviewer_id
            })

            if review_item:
                self.logger.info(f"Retrieved review item {review_item.get('review_id')} "
                               f"for reviewer {reviewer_id}")
            else:
                self.logger.info(f"No pending reviews for reviewer {reviewer_id}")

            return review_item

        except Exception as e:
            self.logger.error(f"Failed to get next review item: {e}")
            return None

    def filter_reviews_by_criteria(
        self,
        check_type: Optional[str] = None,
        severity: Optional[str] = None,
        min_confidence: Optional[int] = None,
        max_confidence: Optional[int] = None,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter pending reviews by various criteria

        Args:
            check_type: Filter by check type
            severity: Filter by severity level
            min_confidence: Minimum confidence score
            max_confidence: Maximum confidence score
            document_id: Filter by document ID

        Returns:
            List of filtered review items
        """
        try:
            filtered = self.tools["filter_reviews"].invoke({
                "review_manager": self.review_manager,
                "check_type": check_type,
                "severity": severity,
                "min_confidence": min_confidence,
                "max_confidence": max_confidence,
                "document_id": document_id
            })

            self.logger.info(f"Filtered reviews: {len(filtered)} items")

            return filtered

        except Exception as e:
            self.logger.error(f"Failed to filter reviews: {e}")
            return []

    def get_batch_for_review(
        self,
        batch_type: str,
        batch_value: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get a batch of review items for batch processing

        Args:
            batch_type: Type of batch ("check_type", "document", "severity", "confidence_range")
            batch_value: Value for the batch type
            limit: Optional maximum number of items

        Returns:
            List of review items for batch processing
        """
        try:
            if batch_type == "check_type":
                batch = self.tools["batch_review_by_check_type"].invoke({
                    "review_manager": self.review_manager,
                    "check_type": batch_value,
                    "limit": limit
                })
            elif batch_type == "document":
                batch = self.tools["batch_review_by_document"].invoke({
                    "review_manager": self.review_manager,
                    "document_id": batch_value,
                    "limit": limit
                })
            elif batch_type == "severity":
                batch = self.tools["batch_review_by_severity"].invoke({
                    "review_manager": self.review_manager,
                    "severity": batch_value,
                    "limit": limit
                })
            else:
                self.logger.error(f"Unknown batch type: {batch_type}")
                return []

            self.logger.info(f"Retrieved batch of {len(batch)} items for {batch_type}={batch_value}")

            return batch

        except Exception as e:
            self.logger.error(f"Failed to get batch: {e}")
            return []

    def find_similar_reviews(
        self,
        review_id: str,
        similarity_threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Find reviews similar to a reference review for batch processing

        Args:
            review_id: Reference review ID
            similarity_threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of similar review items
        """
        try:
            similar = self.tools["get_similar_reviews"].invoke({
                "review_manager": self.review_manager,
                "review_id": review_id,
                "similarity_threshold": similarity_threshold
            })

            self.logger.info(f"Found {len(similar)} similar reviews to {review_id}")

            return similar

        except Exception as e:
            self.logger.error(f"Failed to find similar reviews: {e}")
            return []

    def get_review_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive review queue statistics

        Returns:
            Dictionary with detailed queue statistics
        """
        return self._get_queue_statistics()


# Export
__all__ = ["ReviewerAgent"]
