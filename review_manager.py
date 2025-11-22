#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Review Manager - Queue Management for Human-in-the-Loop System
Manages review queue, prioritization, filtering, and batch operations
"""

import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
from persistence_utils import PersistenceManager, SchemaMigrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    """Status of a review item"""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    REVIEWED = "reviewed"
    SKIPPED = "skipped"


@dataclass
class ReviewItem:
    """Single item in the review queue"""
    review_id: str
    document_id: str
    check_type: str
    slide: str
    location: str
    predicted_violation: bool
    confidence: int
    ai_reasoning: str
    evidence: str
    rule: str
    severity: str
    created_at: str  # ISO format datetime
    priority_score: float
    status: ReviewStatus
    assigned_to: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with enum values as strings"""
        d = asdict(self)
        d['status'] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ReviewItem':
        """Create ReviewItem from dictionary"""
        # Convert status string to enum
        if isinstance(data.get('status'), str):
            data['status'] = ReviewStatus(data['status'])
        return cls(**data)


@dataclass
class ReviewDecision:
    """Human reviewer decision on a review item"""
    review_id: str
    reviewer_id: str
    decision: str  # APPROVE, REJECT, MODIFY
    actual_violation: bool
    corrected_confidence: Optional[int]
    reviewer_notes: str
    tags: List[str]
    reviewed_at: str  # ISO format datetime
    review_duration_seconds: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ReviewDecision':
        """Create ReviewDecision from dictionary"""
        return cls(**data)


@dataclass
class QueueStats:
    """Statistics about the review queue"""
    total_pending: int
    total_in_review: int
    total_reviewed: int
    avg_confidence: float
    by_check_type: Dict[str, int]
    by_severity: Dict[str, int]
    oldest_pending_age_hours: float


class ReviewManager:
    """
    Manages the review queue for human-in-the-loop compliance checking
    
    Features:
    - Queue management (add, retrieve, prioritize)
    - Filtering by check type, severity, confidence range
    - Batch operations for similar items
    - Queue statistics tracking
    - Thread-safe operations
    - Integration with feedback learning system
    """
    
    def __init__(self, queue_file: str = "review_queue.json", max_queue_size: int = 10000,
                 feedback_integration=None, audit_logger=None):
        """
        Initialize review manager
        
        Args:
            queue_file: Path to JSON file for queue persistence
            max_queue_size: Maximum number of items in queue
            feedback_integration: Optional FeedbackIntegration instance for learning
            audit_logger: Optional AuditLogger instance for audit trail
        """
        self.queue_file = queue_file
        self.max_queue_size = max_queue_size
        self.feedback_integration = feedback_integration
        self.audit_logger = audit_logger
        
        # Queue storage
        self.pending_items: Dict[str, ReviewItem] = {}
        self.in_review_items: Dict[str, ReviewItem] = {}
        self.reviewed_items: Dict[str, ReviewItem] = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Load existing queue
        self._load_queue()
        
        logger.info(f"ReviewManager initialized with {len(self.pending_items)} pending items")
    
    def add_to_queue(self, review_item: ReviewItem) -> str:
        """
        Add item to review queue
        
        Args:
            review_item: ReviewItem to add
            
        Returns:
            Review ID
        """
        with self.lock:
            # Check queue size limit
            if len(self.pending_items) >= self.max_queue_size:
                logger.warning(f"Queue at max size ({self.max_queue_size}), removing oldest item")
                self._evict_oldest()
            
            # Add to pending queue
            self.pending_items[review_item.review_id] = review_item
            
            # Save queue
            self._save_queue()
            
            # Log to audit trail
            if self.audit_logger:
                self.audit_logger.log_queue_action("ITEM_ADDED", {
                    'review_id': review_item.review_id,
                    'check_type': review_item.check_type,
                    'confidence': review_item.confidence,
                    'priority_score': review_item.priority_score
                })
            
            logger.info(f"Added to review queue: {review_item.review_id} "
                       f"(confidence: {review_item.confidence}%, priority: {review_item.priority_score:.2f})")
            
            return review_item.review_id
    
    def get_next_review(self, reviewer_id: str) -> Optional[ReviewItem]:
        """
        Get next review item based on priority
        
        Args:
            reviewer_id: ID of reviewer requesting item
            
        Returns:
            Next ReviewItem or None if queue is empty
        """
        with self.lock:
            if not self.pending_items:
                return None
            
            # Get highest priority item
            sorted_items = sorted(
                self.pending_items.values(),
                key=lambda x: x.priority_score,
                reverse=True
            )
            
            next_item = sorted_items[0]
            
            # Move to in_review
            next_item.status = ReviewStatus.IN_REVIEW
            next_item.assigned_to = reviewer_id
            
            del self.pending_items[next_item.review_id]
            self.in_review_items[next_item.review_id] = next_item
            
            # Save queue
            self._save_queue()
            
            # Log to audit trail
            if self.audit_logger:
                self.audit_logger.log_queue_action("ITEM_ASSIGNED", {
                    'review_id': next_item.review_id,
                    'reviewer_id': reviewer_id,
                    'check_type': next_item.check_type
                })
            
            logger.info(f"Assigned review {next_item.review_id} to {reviewer_id}")
            
            return next_item
    
    def get_pending_reviews(self, filters: Optional[Dict] = None) -> List[ReviewItem]:
        """
        Get list of pending reviews with optional filtering
        
        Args:
            filters: Optional dict with filter criteria:
                - check_type: str
                - severity: str
                - min_confidence: int
                - max_confidence: int
                
        Returns:
            List of ReviewItem objects
        """
        with self.lock:
            items = list(self.pending_items.values())
            
            if not filters:
                return items
            
            # Apply filters
            if 'check_type' in filters:
                items = [i for i in items if i.check_type == filters['check_type']]
            
            if 'severity' in filters:
                items = [i for i in items if i.severity == filters['severity']]
            
            if 'min_confidence' in filters:
                items = [i for i in items if i.confidence >= filters['min_confidence']]
            
            if 'max_confidence' in filters:
                items = [i for i in items if i.confidence <= filters['max_confidence']]
            
            return items
    
    def mark_reviewed(self, review_id: str, decision: ReviewDecision) -> bool:
        """
        Mark review as completed and submit to feedback system
        
        Args:
            review_id: ID of review item
            decision: ReviewDecision with reviewer's decision
            
        Returns:
            True if successful, False if review not found
        """
        with self.lock:
            # Find item in in_review queue
            if review_id not in self.in_review_items:
                logger.error(f"Review {review_id} not found in in_review queue")
                return False
            
            item = self.in_review_items[review_id]
            
            # Update status
            item.status = ReviewStatus.REVIEWED
            
            # Move to reviewed queue
            del self.in_review_items[review_id]
            self.reviewed_items[review_id] = item
            
            # Save queue
            self._save_queue()
            
            logger.info(f"Marked review {review_id} as reviewed by {decision.reviewer_id}")
        
        # Log to audit trail (outside lock to avoid deadlock)
        if self.audit_logger:
            try:
                self.audit_logger.log_review(item, decision)
            except Exception as e:
                logger.error(f"Failed to log to audit trail: {e}")
        
        # Submit to feedback system (outside lock to avoid deadlock)
        if self.feedback_integration:
            try:
                self._submit_to_feedback_system(item, decision)
            except Exception as e:
                logger.error(f"Failed to submit feedback: {e}")
        
        return True
    
    def _submit_to_feedback_system(self, item: ReviewItem, decision: ReviewDecision):
        """
        Submit review to feedback system for learning
        
        Args:
            item: ReviewItem that was reviewed
            decision: ReviewDecision from reviewer
        """
        # Submit to feedback interface for review
        feedback_id = self.feedback_integration.feedback_interface.submit_for_review(
            check_type=item.check_type,
            document_id=item.document_id,
            slide=item.slide,
            predicted_violation=item.predicted_violation,
            predicted_confidence=item.confidence,
            predicted_reasoning=item.ai_reasoning,
            predicted_evidence=item.evidence,
            processing_time_ms=decision.review_duration_seconds * 1000
        )
        
        # Provide the correction
        self.feedback_integration.feedback_interface.provide_correction(
            feedback_id=feedback_id,
            actual_violation=decision.actual_violation,
            reviewer_notes=decision.reviewer_notes,
            corrected_confidence=decision.corrected_confidence,
            reviewer_id=decision.reviewer_id
        )
        
        logger.info(f"Submitted review {item.review_id} to feedback system as {feedback_id}")
    
    def get_similar_items(self, review_id: str, threshold: float = 0.85) -> List[ReviewItem]:
        """
        Find similar items in the queue for batch processing
        
        Args:
            review_id: Reference review item ID
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar ReviewItem objects
        """
        with self.lock:
            # Get reference item
            reference = None
            if review_id in self.pending_items:
                reference = self.pending_items[review_id]
            elif review_id in self.in_review_items:
                reference = self.in_review_items[review_id]
            
            if not reference:
                logger.error(f"Review {review_id} not found")
                return []
            
            # Find similar items in pending queue
            similar = []
            for item in self.pending_items.values():
                if item.review_id == review_id:
                    continue
                
                similarity = self._calculate_similarity(reference, item)
                if similarity >= threshold:
                    similar.append(item)
            
            logger.info(f"Found {len(similar)} similar items to {review_id}")
            
            return similar
    
    def get_batch_by_check_type(self, check_type: str, limit: Optional[int] = None) -> List[ReviewItem]:
        """
        Get batch of items by check type
        
        Args:
            check_type: Check type to filter by
            limit: Optional maximum number of items to return
            
        Returns:
            List of ReviewItem objects
        """
        with self.lock:
            items = [item for item in self.pending_items.values() 
                    if item.check_type == check_type]
            
            # Sort by priority
            items.sort(key=lambda x: x.priority_score, reverse=True)
            
            if limit:
                items = items[:limit]
            
            logger.info(f"Found {len(items)} items with check_type={check_type}")
            
            return items
    
    def get_batch_by_document(self, document_id: str, limit: Optional[int] = None) -> List[ReviewItem]:
        """
        Get batch of items by document
        
        Args:
            document_id: Document ID to filter by
            limit: Optional maximum number of items to return
            
        Returns:
            List of ReviewItem objects
        """
        with self.lock:
            items = [item for item in self.pending_items.values() 
                    if item.document_id == document_id]
            
            # Sort by priority
            items.sort(key=lambda x: x.priority_score, reverse=True)
            
            if limit:
                items = items[:limit]
            
            logger.info(f"Found {len(items)} items for document={document_id}")
            
            return items
    
    def get_batch_by_severity(self, severity: str, limit: Optional[int] = None) -> List[ReviewItem]:
        """
        Get batch of items by severity
        
        Args:
            severity: Severity level to filter by
            limit: Optional maximum number of items to return
            
        Returns:
            List of ReviewItem objects
        """
        with self.lock:
            items = [item for item in self.pending_items.values() 
                    if item.severity.upper() == severity.upper()]
            
            # Sort by priority
            items.sort(key=lambda x: x.priority_score, reverse=True)
            
            if limit:
                items = items[:limit]
            
            logger.info(f"Found {len(items)} items with severity={severity}")
            
            return items
    
    def get_batch_by_confidence_range(self, min_confidence: int, max_confidence: int, 
                                     limit: Optional[int] = None) -> List[ReviewItem]:
        """
        Get batch of items by confidence range
        
        Args:
            min_confidence: Minimum confidence score
            max_confidence: Maximum confidence score
            limit: Optional maximum number of items to return
            
        Returns:
            List of ReviewItem objects
        """
        with self.lock:
            items = [item for item in self.pending_items.values() 
                    if min_confidence <= item.confidence <= max_confidence]
            
            # Sort by priority
            items.sort(key=lambda x: x.priority_score, reverse=True)
            
            if limit:
                items = items[:limit]
            
            logger.info(f"Found {len(items)} items with confidence {min_confidence}-{max_confidence}%")
            
            return items
    
    def batch_review(self, review_ids: List[str], decision: ReviewDecision) -> int:
        """
        Apply decision to multiple review items
        
        Args:
            review_ids: List of review IDs to process
            decision: ReviewDecision to apply to all items
            
        Returns:
            Number of items successfully processed
        """
        processed = 0
        
        with self.lock:
            for review_id in review_ids:
                # Move item from pending to in_review if needed
                if review_id in self.pending_items:
                    item = self.pending_items[review_id]
                    item.status = ReviewStatus.IN_REVIEW
                    item.assigned_to = decision.reviewer_id
                    self.in_review_items[review_id] = item
                    del self.pending_items[review_id]
                
                # Create individual decision for each item
                item_decision = ReviewDecision(
                    review_id=review_id,
                    reviewer_id=decision.reviewer_id,
                    decision=decision.decision,
                    actual_violation=decision.actual_violation,
                    corrected_confidence=decision.corrected_confidence,
                    reviewer_notes=decision.reviewer_notes,
                    tags=decision.tags,
                    reviewed_at=datetime.now().isoformat(),
                    review_duration_seconds=decision.review_duration_seconds
                )
        
        # Mark as reviewed (outside lock to avoid deadlock)
        for review_id in review_ids:
            item_decision = ReviewDecision(
                review_id=review_id,
                reviewer_id=decision.reviewer_id,
                decision=decision.decision,
                actual_violation=decision.actual_violation,
                corrected_confidence=decision.corrected_confidence,
                reviewer_notes=decision.reviewer_notes,
                tags=decision.tags,
                reviewed_at=datetime.now().isoformat(),
                review_duration_seconds=decision.review_duration_seconds
            )
            
            if self.mark_reviewed(review_id, item_decision):
                processed += 1
        
        # Log batch operation to audit trail
        if self.audit_logger:
            self.audit_logger.log_queue_action("BATCH_PROCESSED", {
                'reviewer_id': decision.reviewer_id,
                'total_items': len(review_ids),
                'processed_items': processed,
                'decision': decision.decision
            })
        
        logger.info(f"Batch processed {processed}/{len(review_ids)} items")
        
        return processed
    
    def get_queue_stats(self) -> QueueStats:
        """
        Get statistics about the review queue
        
        Returns:
            QueueStats object
        """
        with self.lock:
            # Calculate average confidence
            all_pending = list(self.pending_items.values())
            avg_confidence = 0.0
            if all_pending:
                avg_confidence = sum(item.confidence for item in all_pending) / len(all_pending)
            
            # Group by check type
            by_check_type = defaultdict(int)
            for item in all_pending:
                by_check_type[item.check_type] += 1
            
            # Group by severity
            by_severity = defaultdict(int)
            for item in all_pending:
                by_severity[item.severity] += 1
            
            # Calculate oldest pending age
            oldest_age_hours = 0.0
            if all_pending:
                oldest_item = min(all_pending, key=lambda x: x.created_at)
                oldest_time = datetime.fromisoformat(oldest_item.created_at)
                age_delta = datetime.now() - oldest_time
                oldest_age_hours = age_delta.total_seconds() / 3600
            
            return QueueStats(
                total_pending=len(self.pending_items),
                total_in_review=len(self.in_review_items),
                total_reviewed=len(self.reviewed_items),
                avg_confidence=avg_confidence,
                by_check_type=dict(by_check_type),
                by_severity=dict(by_severity),
                oldest_pending_age_hours=oldest_age_hours
            )
    
    def calculate_priority_score(self, confidence: int, severity: str, 
                                 age_hours: float = 0.0) -> float:
        """
        Calculate priority score for a review item
        
        Priority algorithm:
        - Lower confidence = higher priority (inverse)
        - Higher severity = higher priority
        - Older items = higher priority
        
        Args:
            confidence: Confidence score (0-100)
            severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
            age_hours: Age of item in hours
            
        Returns:
            Priority score (higher = more urgent)
        """
        # Confidence component (inverse - lower confidence = higher priority)
        confidence_score = 100 - confidence
        
        # Severity component
        severity_weights = {
            'CRITICAL': 50,
            'HIGH': 30,
            'MEDIUM': 15,
            'LOW': 5
        }
        severity_score = severity_weights.get(severity.upper(), 10)
        
        # Age component (0.5 points per hour)
        age_score = age_hours * 0.5
        
        # Combined score
        priority = confidence_score + severity_score + age_score
        
        return priority
    
    def _calculate_similarity(self, item1: ReviewItem, item2: ReviewItem) -> float:
        """
        Calculate similarity between two review items
        
        Args:
            item1: First ReviewItem
            item2: Second ReviewItem
            
        Returns:
            Similarity score (0-1)
        """
        similarity = 0.0
        
        # Same check type (40% weight)
        if item1.check_type == item2.check_type:
            similarity += 0.4
        
        # Same severity (20% weight)
        if item1.severity == item2.severity:
            similarity += 0.2
        
        # Similar confidence (20% weight)
        confidence_diff = abs(item1.confidence - item2.confidence)
        if confidence_diff <= 10:
            similarity += 0.2
        elif confidence_diff <= 20:
            similarity += 0.1
        
        # Same document (20% weight)
        if item1.document_id == item2.document_id:
            similarity += 0.2
        
        return similarity
    
    def _evict_oldest(self):
        """Remove oldest item from pending queue"""
        if not self.pending_items:
            return
        
        oldest_item = min(
            self.pending_items.values(),
            key=lambda x: x.created_at
        )
        
        del self.pending_items[oldest_item.review_id]
        logger.warning(f"Evicted oldest item from queue: {oldest_item.review_id}")
    
    def _load_queue(self):
        """Load queue from file with thread-safe file locking"""
        try:
            # Setup schema migrator
            migrator = SchemaMigrator(current_version=2)
            migrator.register_migration(1, 2, self._migrate_queue_v1_to_v2)
            
            # Load data with migration support
            data = PersistenceManager.load_json(
                self.queue_file,
                migration_func=migrator.migrate
            )
            
            if not data:
                logger.info(f"No existing queue file found at {self.queue_file}")
                return
            
            # Load pending items
            for item_data in data.get('pending', []):
                item = ReviewItem.from_dict(item_data)
                self.pending_items[item.review_id] = item
            
            # Load in_review items
            for item_data in data.get('in_review', []):
                item = ReviewItem.from_dict(item_data)
                self.in_review_items[item.review_id] = item
            
            # Load reviewed items
            for item_data in data.get('reviewed', []):
                item = ReviewItem.from_dict(item_data)
                self.reviewed_items[item.review_id] = item
            
            schema_version = data.get('schema_version', 1)
            logger.info(f"Loaded queue from {self.queue_file} (schema v{schema_version})")
            
        except Exception as e:
            logger.error(f"Error loading queue: {e}")
    
    def _save_queue(self):
        """Save queue to file with thread-safe file locking"""
        try:
            # Prepare data
            data = {
                'schema_version': 2,  # Current schema version
                'pending': [item.to_dict() for item in self.pending_items.values()],
                'in_review': [item.to_dict() for item in self.in_review_items.values()],
                'reviewed': [item.to_dict() for item in self.reviewed_items.values()],
                'last_updated': datetime.now().isoformat()
            }
            
            # Save using persistence manager
            PersistenceManager.save_json(self.queue_file, data, backup=True)
            
            logger.debug(f"Saved queue to {self.queue_file}")
            
        except Exception as e:
            logger.error(f"Error saving queue: {e}")
    
    def _migrate_queue_v1_to_v2(self, data: Dict) -> Dict:
        """
        Migrate queue data from v1 to v2
        
        Args:
            data: Queue data in v1 schema
            
        Returns:
            Migrated data in v2 schema
        """
        logger.info("Migrating queue schema from v1 to v2")
        
        # v1 didn't have schema_version field
        # v2 adds schema_version and ensures all items have required fields
        
        # Ensure all items have default values for new fields
        for queue_type in ['pending', 'in_review', 'reviewed']:
            items = data.get(queue_type, [])
            for item in items:
                # Add any missing fields with defaults
                if 'assigned_to' not in item:
                    item['assigned_to'] = None
                if 'priority_score' not in item:
                    # Recalculate priority if missing
                    item['priority_score'] = self.calculate_priority_score(
                        item.get('confidence', 50),
                        item.get('severity', 'MEDIUM'),
                        0.0
                    )
        
        data['schema_version'] = 2
        
        return data


if __name__ == "__main__":
    # Example usage and testing
    print("="*70)
    print("Review Manager - Queue Management System")
    print("="*70)
    
    # Initialize manager
    manager = ReviewManager(queue_file="test_review_queue.json")
    
    # Create test review items
    print("\n📝 Adding test items to queue...")
    for i in range(5):
        created_time = datetime.now().isoformat()
        
        item = ReviewItem(
            review_id=f"review_{i}",
            document_id=f"doc_{i // 2}",  # Group some items by document
            check_type="STRUCTURE" if i % 2 == 0 else "PERFORMANCE",
            slide=f"Slide {i+1}",
            location="Header section",
            predicted_violation=True,
            confidence=60 + (i * 5),  # Varying confidence
            ai_reasoning=f"AI detected potential violation in document {i}",
            evidence=f"Found suspicious pattern in slide {i+1}",
            rule="Promotional mention required",
            severity="CRITICAL" if i < 2 else "HIGH",
            created_at=created_time,
            priority_score=manager.calculate_priority_score(
                confidence=60 + (i * 5),
                severity="CRITICAL" if i < 2 else "HIGH",
                age_hours=0.0
            ),
            status=ReviewStatus.PENDING
        )
        
        manager.add_to_queue(item)
    
    print(f"  ✓ Added 5 items to queue")
    
    # Get queue statistics
    print("\n📊 Queue Statistics:")
    stats = manager.get_queue_stats()
    print(f"  Pending: {stats.total_pending}")
    print(f"  In Review: {stats.total_in_review}")
    print(f"  Reviewed: {stats.total_reviewed}")
    print(f"  Avg Confidence: {stats.avg_confidence:.1f}%")
    print(f"  By Check Type: {stats.by_check_type}")
    print(f"  By Severity: {stats.by_severity}")
    
    # Get next review
    print("\n🔍 Getting next review...")
    next_item = manager.get_next_review("reviewer_001")
    if next_item:
        print(f"  ✓ Assigned: {next_item.review_id}")
        print(f"    Check Type: {next_item.check_type}")
        print(f"    Confidence: {next_item.confidence}%")
        print(f"    Priority: {next_item.priority_score:.2f}")
    
    # Test filtering
    print("\n🔎 Testing filters...")
    filtered = manager.get_pending_reviews({'check_type': 'STRUCTURE'})
    print(f"  ✓ Found {len(filtered)} STRUCTURE checks")
    
    # Test similarity detection
    print("\n🔗 Testing similarity detection...")
    if next_item:
        similar = manager.get_similar_items(next_item.review_id, threshold=0.3)
        print(f"  ✓ Found {len(similar)} similar items")
    
    # Mark as reviewed
    print("\n✅ Marking review as complete...")
    if next_item:
        decision = ReviewDecision(
            review_id=next_item.review_id,
            reviewer_id="reviewer_001",
            decision="APPROVE",
            actual_violation=True,
            corrected_confidence=None,
            reviewer_notes="Confirmed violation",
            tags=["confirmed"],
            reviewed_at=datetime.now().isoformat(),
            review_duration_seconds=120
        )
        
        success = manager.mark_reviewed(next_item.review_id, decision)
        print(f"  ✓ Review marked as complete: {success}")
    
    # Final statistics
    print("\n📊 Final Queue Statistics:")
    stats = manager.get_queue_stats()
    print(f"  Pending: {stats.total_pending}")
    print(f"  In Review: {stats.total_in_review}")
    print(f"  Reviewed: {stats.total_reviewed}")
    
    print("\n" + "="*70)
    print("✓ Review Manager test complete")
    print("="*70)
