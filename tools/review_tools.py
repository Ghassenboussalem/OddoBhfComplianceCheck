#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Review Tools

This module provides functionality for the multi-agent compliance system.
"""

"""
Review Tools - LangChain Tools for Human-in-the-Loop Review Management
Provides tools for queue management, prioritization, filtering, and batch operations
"""

from langchain.tools import tool
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def queue_for_review(
    violation: Dict[str, Any],
    review_manager: Any,
    document_id: str = "",
    confidence: int = 0,
    severity: str = "MEDIUM"
) -> str:
    """
    Add a violation to the review queue for human review.

    This tool queues low-confidence violations for human-in-the-loop review,
    calculating priority scores and managing the review queue.

    Args:
        violation: Dictionary containing violation details with keys:
            - type: Check type (e.g., "STRUCTURE", "PERFORMANCE")
            - slide: Slide location
            - location: Specific location in slide
            - rule: Rule that was violated
            - evidence: Evidence text
            - ai_reasoning: AI's reasoning for the violation
            - confidence: Confidence score (0-100)
            - severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
        review_manager: ReviewManager instance for queue management
        document_id: ID of the document being reviewed
        confidence: Override confidence score
        severity: Override severity level

    Returns:
        Review ID string for tracking the queued item

    Example:
        >>> review_id = queue_for_review(
        ...     violation={
        ...         "type": "STRUCTURE",
        ...         "slide": "Slide 1",
        ...         "location": "Header",
        ...         "rule": "Promotional mention required",
        ...         "evidence": "Document promotional",
        ...         "ai_reasoning": "Missing required mention",
        ...         "confidence": 65,
        ...         "severity": "HIGH"
        ...     },
        ...     review_manager=manager,
        ...     document_id="doc_123"
        ... )
    """
    from review_manager import ReviewItem, ReviewStatus
    import uuid

    # Extract violation details
    check_type = violation.get("type", "UNKNOWN")
    slide = violation.get("slide", "")
    location = violation.get("location", "")
    rule = violation.get("rule", "")
    evidence = violation.get("evidence", "")
    ai_reasoning = violation.get("ai_reasoning", "")

    # Use provided confidence/severity or extract from violation
    conf = confidence if confidence > 0 else violation.get("confidence", 50)
    sev = severity if severity != "MEDIUM" else violation.get("severity", "MEDIUM")

    # Calculate priority score
    priority_score = review_manager.calculate_priority_score(
        confidence=conf,
        severity=sev,
        age_hours=0.0
    )

    # Create review item
    review_item = ReviewItem(
        review_id=str(uuid.uuid4()),
        document_id=document_id or violation.get("document_id", ""),
        check_type=check_type,
        slide=slide,
        location=location,
        predicted_violation=True,
        confidence=conf,
        ai_reasoning=ai_reasoning,
        evidence=evidence,
        rule=rule,
        severity=sev,
        created_at=datetime.now().isoformat(),
        priority_score=priority_score,
        status=ReviewStatus.PENDING
    )

    # Add to queue
    review_id = review_manager.add_to_queue(review_item)

    logger.info(f"Queued violation for review: {review_id} "
                f"(confidence: {conf}%, priority: {priority_score:.2f})")

    return review_id


@tool
def calculate_priority_score(
    confidence: int,
    severity: str,
    age_hours: float = 0.0
) -> float:
    """
    Calculate priority score for a review item.

    Priority algorithm:
    - Lower confidence = higher priority (inverse relationship)
    - Higher severity = higher priority
    - Older items = higher priority (time-based boost)

    Args:
        confidence: Confidence score (0-100), lower means more uncertain
        severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
        age_hours: Age of the item in hours (default: 0.0)

    Returns:
        Priority score as float (higher = more urgent)

    Example:
        >>> score = calculate_priority_score(
        ...     confidence=65,
        ...     severity="HIGH",
        ...     age_hours=2.5
        ... )
        >>> print(f"Priority: {score:.2f}")
        Priority: 66.25
    """
    # Confidence component (inverse - lower confidence = higher priority)
    # Range: 0-100 (confidence 0% = 100 points, confidence 100% = 0 points)
    confidence_score = 100 - confidence

    # Severity component
    # CRITICAL = 50 points, HIGH = 30, MEDIUM = 15, LOW = 5
    severity_weights = {
        'CRITICAL': 50,
        'HIGH': 30,
        'MEDIUM': 15,
        'LOW': 5
    }
    severity_score = severity_weights.get(severity.upper(), 10)

    # Age component (0.5 points per hour)
    # Ensures older items gradually increase in priority
    age_score = age_hours * 0.5

    # Combined score
    priority = confidence_score + severity_score + age_score

    logger.debug(f"Priority calculation: confidence={confidence_score}, "
                 f"severity={severity_score}, age={age_score}, total={priority}")

    return priority


@tool
def filter_reviews(
    review_manager: Any,
    check_type: Optional[str] = None,
    severity: Optional[str] = None,
    min_confidence: Optional[int] = None,
    max_confidence: Optional[int] = None,
    document_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filter pending reviews based on multiple criteria.

    This tool allows filtering the review queue by check type, severity,
    confidence range, or document ID to focus on specific subsets of reviews.

    Args:
        review_manager: ReviewManager instance
        check_type: Filter by check type (e.g., "STRUCTURE", "PERFORMANCE")
        severity: Filter by severity level (CRITICAL, HIGH, MEDIUM, LOW)
        min_confidence: Minimum confidence score (0-100)
        max_confidence: Maximum confidence score (0-100)
        document_id: Filter by specific document ID

    Returns:
        List of review item dictionaries matching the filter criteria

    Example:
        >>> # Get all high-severity structure checks with low confidence
        >>> reviews = filter_reviews(
        ...     review_manager=manager,
        ...     check_type="STRUCTURE",
        ...     severity="HIGH",
        ...     max_confidence=70
        ... )
        >>> print(f"Found {len(reviews)} matching reviews")
    """
    # Build filter dictionary
    filters = {}

    if check_type:
        filters['check_type'] = check_type

    if severity:
        filters['severity'] = severity

    if min_confidence is not None:
        filters['min_confidence'] = min_confidence

    if max_confidence is not None:
        filters['max_confidence'] = max_confidence

    # Get filtered reviews
    review_items = review_manager.get_pending_reviews(filters if filters else None)

    # Additional document_id filtering if specified
    if document_id:
        review_items = [item for item in review_items if item.document_id == document_id]

    # Convert to dictionaries
    reviews = [item.to_dict() for item in review_items]

    logger.info(f"Filtered reviews: {len(reviews)} items matching criteria")

    return reviews


@tool
def batch_review_by_check_type(
    review_manager: Any,
    check_type: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get a batch of review items by check type for batch processing.

    This tool retrieves all pending reviews of a specific check type,
    sorted by priority, for efficient batch review operations.

    Args:
        review_manager: ReviewManager instance
        check_type: Check type to filter by (e.g., "STRUCTURE", "PERFORMANCE")
        limit: Optional maximum number of items to return

    Returns:
        List of review item dictionaries sorted by priority (highest first)

    Example:
        >>> # Get top 10 structure checks for batch review
        >>> batch = batch_review_by_check_type(
        ...     review_manager=manager,
        ...     check_type="STRUCTURE",
        ...     limit=10
        ... )
        >>> print(f"Batch contains {len(batch)} items")
    """
    review_items = review_manager.get_batch_by_check_type(check_type, limit)

    # Convert to dictionaries
    batch = [item.to_dict() for item in review_items]

    logger.info(f"Retrieved batch of {len(batch)} items for check_type={check_type}")

    return batch


@tool
def batch_review_by_document(
    review_manager: Any,
    document_id: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get a batch of review items by document ID for batch processing.

    This tool retrieves all pending reviews for a specific document,
    sorted by priority, allowing reviewers to process all issues
    in a single document together.

    Args:
        review_manager: ReviewManager instance
        document_id: Document ID to filter by
        limit: Optional maximum number of items to return

    Returns:
        List of review item dictionaries sorted by priority (highest first)

    Example:
        >>> # Get all reviews for a specific document
        >>> batch = batch_review_by_document(
        ...     review_manager=manager,
        ...     document_id="doc_123"
        ... )
        >>> print(f"Document has {len(batch)} pending reviews")
    """
    review_items = review_manager.get_batch_by_document(document_id, limit)

    # Convert to dictionaries
    batch = [item.to_dict() for item in review_items]

    logger.info(f"Retrieved batch of {len(batch)} items for document={document_id}")

    return batch


@tool
def batch_review_by_severity(
    review_manager: Any,
    severity: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get a batch of review items by severity level for batch processing.

    This tool retrieves all pending reviews of a specific severity level,
    sorted by priority, allowing reviewers to focus on critical issues first.

    Args:
        review_manager: ReviewManager instance
        severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
        limit: Optional maximum number of items to return

    Returns:
        List of review item dictionaries sorted by priority (highest first)

    Example:
        >>> # Get all critical severity reviews
        >>> batch = batch_review_by_severity(
        ...     review_manager=manager,
        ...     severity="CRITICAL"
        ... )
        >>> print(f"Found {len(batch)} critical reviews")
    """
    review_items = review_manager.get_batch_by_severity(severity, limit)

    # Convert to dictionaries
    batch = [item.to_dict() for item in review_items]

    logger.info(f"Retrieved batch of {len(batch)} items with severity={severity}")

    return batch


@tool
def batch_review_by_confidence_range(
    review_manager: Any,
    min_confidence: int,
    max_confidence: int,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get a batch of review items by confidence range for batch processing.

    This tool retrieves all pending reviews within a specific confidence range,
    sorted by priority, allowing reviewers to focus on uncertain predictions.

    Args:
        review_manager: ReviewManager instance
        min_confidence: Minimum confidence score (0-100)
        max_confidence: Maximum confidence score (0-100)
        limit: Optional maximum number of items to return

    Returns:
        List of review item dictionaries sorted by priority (highest first)

    Example:
        >>> # Get all reviews with very low confidence (0-50%)
        >>> batch = batch_review_by_confidence_range(
        ...     review_manager=manager,
        ...     min_confidence=0,
        ...     max_confidence=50
        ... )
        >>> print(f"Found {len(batch)} low-confidence reviews")
    """
    review_items = review_manager.get_batch_by_confidence_range(
        min_confidence, max_confidence, limit
    )

    # Convert to dictionaries
    batch = [item.to_dict() for item in review_items]

    logger.info(f"Retrieved batch of {len(batch)} items with "
                f"confidence {min_confidence}-{max_confidence}%")

    return batch


@tool
def get_similar_reviews(
    review_manager: Any,
    review_id: str,
    similarity_threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """
    Find similar review items for batch processing.

    This tool identifies reviews similar to a reference review based on
    check type, severity, confidence, and document. Useful for applying
    the same decision to multiple similar violations.

    Similarity calculation:
    - Same check type: 40% weight
    - Same severity: 20% weight
    - Similar confidence (±10%): 20% weight
    - Same document: 20% weight

    Args:
        review_manager: ReviewManager instance
        review_id: Reference review item ID
        similarity_threshold: Minimum similarity score (0.0-1.0, default: 0.85)

    Returns:
        List of similar review item dictionaries

    Example:
        >>> # Find reviews similar to a specific one
        >>> similar = get_similar_reviews(
        ...     review_manager=manager,
        ...     review_id="review_123",
        ...     similarity_threshold=0.80
        ... )
        >>> print(f"Found {len(similar)} similar reviews")
    """
    review_items = review_manager.get_similar_items(review_id, similarity_threshold)

    # Convert to dictionaries
    similar = [item.to_dict() for item in review_items]

    logger.info(f"Found {len(similar)} similar items to {review_id} "
                f"(threshold: {similarity_threshold})")

    return similar


@tool
def get_queue_statistics(review_manager: Any) -> Dict[str, Any]:
    """
    Get comprehensive statistics about the review queue.

    This tool provides an overview of the review queue including counts,
    average confidence, distribution by check type and severity, and
    age of oldest pending item.

    Args:
        review_manager: ReviewManager instance

    Returns:
        Dictionary containing queue statistics:
        - total_pending: Number of pending reviews
        - total_in_review: Number of reviews currently being reviewed
        - total_reviewed: Number of completed reviews
        - avg_confidence: Average confidence of pending reviews
        - by_check_type: Distribution by check type
        - by_severity: Distribution by severity level
        - oldest_pending_age_hours: Age of oldest pending item in hours

    Example:
        >>> stats = get_queue_statistics(review_manager=manager)
        >>> print(f"Pending: {stats['total_pending']}")
        >>> print(f"Avg confidence: {stats['avg_confidence']:.1f}%")
    """
    stats = review_manager.get_queue_stats()

    # Convert to dictionary
    stats_dict = {
        'total_pending': stats.total_pending,
        'total_in_review': stats.total_in_review,
        'total_reviewed': stats.total_reviewed,
        'avg_confidence': stats.avg_confidence,
        'by_check_type': stats.by_check_type,
        'by_severity': stats.by_severity,
        'oldest_pending_age_hours': stats.oldest_pending_age_hours
    }

    logger.info(f"Queue stats: {stats.total_pending} pending, "
                f"{stats.total_in_review} in review, "
                f"{stats.total_reviewed} reviewed")

    return stats_dict


@tool
def get_next_review(
    review_manager: Any,
    reviewer_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get the next highest-priority review item from the queue.

    This tool retrieves the next review item based on priority score,
    assigns it to the reviewer, and moves it to the in-review state.

    Args:
        review_manager: ReviewManager instance
        reviewer_id: ID of the reviewer requesting the item

    Returns:
        Review item dictionary or None if queue is empty

    Example:
        >>> review = get_next_review(
        ...     review_manager=manager,
        ...     reviewer_id="reviewer_001"
        ... )
        >>> if review:
        ...     print(f"Reviewing: {review['review_id']}")
    """
    review_item = review_manager.get_next_review(reviewer_id)

    if review_item:
        review_dict = review_item.to_dict()
        logger.info(f"Retrieved next review {review_item.review_id} for {reviewer_id}")
        return review_dict
    else:
        logger.info(f"No pending reviews available for {reviewer_id}")
        return None


# Export all tools as a list for easy registration
REVIEW_TOOLS = [
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
]


if __name__ == "__main__":
    # Example usage and testing
    logger.info("="*70)
    logger.info("Review Tools - LangChain Tool Testing")
    logger.info("="*70)

    from review_manager import ReviewManager

    # Initialize review manager
    manager = ReviewManager(queue_file="test_review_tools_queue.json")

    logger.info("\n📝 Testing queue_for_review tool...")
    violation = {
        "type": "STRUCTURE",
        "slide": "Slide 1",
        "location": "Header",
        "rule": "Promotional mention required",
        "evidence": "Document promotional",
        "ai_reasoning": "Missing required promotional mention",
        "confidence": 65,
        "severity": "HIGH"
    }

    review_id = queue_for_review.invoke({
        "violation": violation,
        "review_manager": manager,
        "document_id": "doc_test_001"
    })
    logger.info(f"  ✓ Queued review: {review_id}")

    logger.info("\n📊 Testing calculate_priority_score tool...")
    priority = calculate_priority_score.invoke({
        "confidence": 65,
        "severity": "HIGH",
        "age_hours": 2.5
    })
    logger.info(f"  ✓ Priority score: {priority:.2f}")

    logger.info("\n🔍 Testing filter_reviews tool...")
    filtered = filter_reviews.invoke({
        "review_manager": manager,
        "check_type": "STRUCTURE",
        "max_confidence": 70
    })
    logger.info(f"  ✓ Found {len(filtered)} filtered reviews")

    logger.info("\n📦 Testing batch_review_by_check_type tool...")
    batch = batch_review_by_check_type.invoke({
        "review_manager": manager,
        "check_type": "STRUCTURE",
        "limit": 10
    })
    logger.info(f"  ✓ Retrieved batch of {len(batch)} items")

    logger.info("\n📊 Testing get_queue_statistics tool...")
    stats = get_queue_statistics.invoke({
        "review_manager": manager
    })
    logger.info(f"  ✓ Queue stats:")
    logger.info(f"    Pending: {stats['total_pending']}")
    logger.info(f"    Avg confidence: {stats['avg_confidence']:.1f}%")

    logger.info("\n🔍 Testing get_next_review tool...")
    next_review = get_next_review.invoke({
        "review_manager": manager,
        "reviewer_id": "test_reviewer"
    })
    if next_review:
        logger.info(f"  ✓ Next review: {next_review['review_id']}")
        logger.info(f"    Check type: {next_review['check_type']}")
        logger.info(f"    Confidence: {next_review['confidence']}%")

    logger.info("\n" + "="*70)
    logger.info("✓ Review Tools test complete")
    logger.info("="*70)
