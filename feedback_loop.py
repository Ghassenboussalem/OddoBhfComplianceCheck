#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feedback Loop System for Human Corrections
Enables learning from human reviewer corrections to improve future predictions
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
from enum import Enum
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorrectionType(Enum):
    """Types of corrections that can be made"""
    FALSE_POSITIVE = "false_positive"  # System flagged violation, but was incorrect
    FALSE_NEGATIVE = "false_negative"  # System missed a violation
    CONFIDENCE_ADJUSTMENT = "confidence_adjustment"  # Confidence score was wrong
    REASONING_IMPROVEMENT = "reasoning_improvement"  # Reasoning needs improvement


class ReviewStatus(Enum):
    """Status of a review"""
    PENDING = "pending"
    APPROVED = "approved"
    CORRECTED = "corrected"
    REJECTED = "rejected"


@dataclass
class FeedbackRecord:
    """Single feedback record from human reviewer"""
    feedback_id: str
    timestamp: str
    check_type: str
    document_id: str
    slide: str
    
    # Original prediction
    predicted_violation: bool
    predicted_confidence: int
    predicted_reasoning: str
    predicted_evidence: str
    
    # Human correction
    actual_violation: bool
    corrected_confidence: Optional[int]
    reviewer_notes: str
    correction_type: CorrectionType
    review_status: ReviewStatus
    
    # Metadata
    reviewer_id: Optional[str] = None
    processing_time_ms: Optional[float] = None
    ai_provider: Optional[str] = None


@dataclass
class PatternInsight:
    """Discovered pattern from feedback analysis"""
    pattern_id: str
    check_type: str
    pattern_description: str
    occurrence_count: int
    confidence: float
    examples: List[str]
    suggested_rule: Optional[str] = None
    impact_score: float = 0.0  # How much this pattern affects accuracy


@dataclass
class LearningMetrics:
    """Metrics for learning system performance"""
    total_feedback: int
    false_positives: int
    false_negatives: int
    confidence_adjustments: int
    patterns_discovered: int
    rules_suggested: int
    accuracy_improvement: float
    avg_correction_time_ms: float


class FeedbackInterface:
    """
    Interface for human reviewers to provide corrections
    Stores feedback and makes it available for analysis
    """
    
    def __init__(self, db_path: str = "feedback_data.json"):
        """
        Initialize feedback interface
        
        Args:
            db_path: Path to JSON file for storing feedback
        """
        self.db_path = db_path
        self.feedback_records: List[FeedbackRecord] = []
        self.pending_reviews: Dict[str, FeedbackRecord] = {}
        
        # Load existing feedback
        self._load_data()
        
        logger.info(f"FeedbackInterface initialized with {len(self.feedback_records)} historical records")
    
    def submit_for_review(self, check_type: str, document_id: str, slide: str,
                         predicted_violation: bool, predicted_confidence: int,
                         predicted_reasoning: str, predicted_evidence: str,
                         processing_time_ms: Optional[float] = None,
                         ai_provider: Optional[str] = None) -> str:
        """
        Submit a compliance check result for human review
        
        Args:
            check_type: Type of compliance check
            document_id: Document identifier
            slide: Slide identifier
            predicted_violation: System's prediction
            predicted_confidence: Confidence score
            predicted_reasoning: AI reasoning
            predicted_evidence: Evidence found
            processing_time_ms: Processing time
            ai_provider: AI provider used
            
        Returns:
            Feedback ID for tracking
        """
        feedback_id = f"{check_type}_{document_id}_{slide}_{datetime.now().timestamp()}"
        
        record = FeedbackRecord(
            feedback_id=feedback_id,
            timestamp=datetime.now().isoformat(),
            check_type=check_type,
            document_id=document_id,
            slide=slide,
            predicted_violation=predicted_violation,
            predicted_confidence=predicted_confidence,
            predicted_reasoning=predicted_reasoning,
            predicted_evidence=predicted_evidence,
            actual_violation=predicted_violation,  # Default to prediction
            corrected_confidence=None,
            reviewer_notes="",
            correction_type=CorrectionType.CONFIDENCE_ADJUSTMENT,
            review_status=ReviewStatus.PENDING,
            processing_time_ms=processing_time_ms,
            ai_provider=ai_provider
        )
        
        self.pending_reviews[feedback_id] = record
        
        logger.info(f"Submitted for review: {feedback_id}")
        return feedback_id
    
    def provide_correction(self, feedback_id: str, actual_violation: bool,
                          reviewer_notes: str, corrected_confidence: Optional[int] = None,
                          reviewer_id: Optional[str] = None) -> bool:
        """
        Provide human correction for a pending review
        
        Args:
            feedback_id: ID of the feedback record
            actual_violation: Actual ground truth
            reviewer_notes: Notes from reviewer
            corrected_confidence: Corrected confidence score
            reviewer_id: ID of reviewer
            
        Returns:
            True if correction was recorded successfully
        """
        if feedback_id not in self.pending_reviews:
            logger.error(f"Feedback ID not found: {feedback_id}")
            return False
        
        record = self.pending_reviews[feedback_id]
        
        # Update record with correction
        record.actual_violation = actual_violation
        record.corrected_confidence = corrected_confidence
        record.reviewer_notes = reviewer_notes
        record.reviewer_id = reviewer_id
        record.review_status = ReviewStatus.CORRECTED
        
        # Determine correction type
        if record.predicted_violation and not actual_violation:
            record.correction_type = CorrectionType.FALSE_POSITIVE
        elif not record.predicted_violation and actual_violation:
            record.correction_type = CorrectionType.FALSE_NEGATIVE
        elif corrected_confidence and abs(corrected_confidence - record.predicted_confidence) > 15:
            record.correction_type = CorrectionType.CONFIDENCE_ADJUSTMENT
        else:
            record.correction_type = CorrectionType.REASONING_IMPROVEMENT
        
        # Move to feedback records
        self.feedback_records.append(record)
        del self.pending_reviews[feedback_id]
        
        # Save data
        self._save_data()
        
        logger.info(f"Correction recorded: {feedback_id} ({record.correction_type.value})")
        return True
    
    def approve_prediction(self, feedback_id: str, reviewer_id: Optional[str] = None) -> bool:
        """
        Approve a prediction as correct
        
        Args:
            feedback_id: ID of the feedback record
            reviewer_id: ID of reviewer
            
        Returns:
            True if approval was recorded successfully
        """
        if feedback_id not in self.pending_reviews:
            logger.error(f"Feedback ID not found: {feedback_id}")
            return False
        
        record = self.pending_reviews[feedback_id]
        record.review_status = ReviewStatus.APPROVED
        record.reviewer_id = reviewer_id
        record.reviewer_notes = "Approved by reviewer"
        
        # Move to feedback records
        self.feedback_records.append(record)
        del self.pending_reviews[feedback_id]
        
        # Save data
        self._save_data()
        
        logger.info(f"Prediction approved: {feedback_id}")
        return True
    
    def get_pending_reviews(self, check_type: Optional[str] = None) -> List[FeedbackRecord]:
        """
        Get list of pending reviews
        
        Args:
            check_type: Filter by check type
            
        Returns:
            List of pending feedback records
        """
        pending = list(self.pending_reviews.values())
        
        if check_type:
            pending = [r for r in pending if r.check_type == check_type]
        
        return pending
    
    def get_feedback_history(self, check_type: Optional[str] = None,
                            days: Optional[int] = None) -> List[FeedbackRecord]:
        """
        Get feedback history
        
        Args:
            check_type: Filter by check type
            days: Only include records from last N days
            
        Returns:
            List of feedback records
        """
        records = self.feedback_records
        
        # Filter by check type
        if check_type:
            records = [r for r in records if r.check_type == check_type]
        
        # Filter by time range
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            records = [
                r for r in records
                if datetime.fromisoformat(r.timestamp) >= cutoff
            ]
        
        return records
    
    def export_feedback(self, filepath: str, check_type: Optional[str] = None):
        """
        Export feedback data to JSON file
        
        Args:
            filepath: Path to output file
            check_type: Filter by check type
        """
        records = self.get_feedback_history(check_type=check_type)
        
        data = {
            'total_records': len(records),
            'export_timestamp': datetime.now().isoformat(),
            'check_type_filter': check_type,
            'records': [asdict(r) for r in records]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(records)} feedback records to {filepath}")
    
    def _load_data(self):
        """Load feedback data from file"""
        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)
            
            self.feedback_records = []
            for record in data.get('feedback_records', []):
                try:
                    # Convert string enum values back to enums
                    correction_type_str = record.get('correction_type', 'confidence_adjustment')
                    review_status_str = record.get('review_status', 'pending')
                    
                    # Handle both enum value strings and full enum strings
                    if '.' in correction_type_str:
                        correction_type_str = correction_type_str.split('.')[-1]
                    if '.' in review_status_str:
                        review_status_str = review_status_str.split('.')[-1]
                    
                    record_obj = FeedbackRecord(
                        **{**record, 
                           'correction_type': CorrectionType(correction_type_str),
                           'review_status': ReviewStatus(review_status_str)}
                    )
                    self.feedback_records.append(record_obj)
                except Exception as e:
                    logger.warning(f"Skipping invalid record: {e}")
                    continue
            
            self.pending_reviews = {}
            for record in data.get('pending_reviews', []):
                try:
                    correction_type_str = record.get('correction_type', 'confidence_adjustment')
                    review_status_str = record.get('review_status', 'pending')
                    
                    if '.' in correction_type_str:
                        correction_type_str = correction_type_str.split('.')[-1]
                    if '.' in review_status_str:
                        review_status_str = review_status_str.split('.')[-1]
                    
                    record_obj = FeedbackRecord(
                        **{**record,
                           'correction_type': CorrectionType(correction_type_str),
                           'review_status': ReviewStatus(review_status_str)}
                    )
                    self.pending_reviews[record['feedback_id']] = record_obj
                except Exception as e:
                    logger.warning(f"Skipping invalid pending record: {e}")
                    continue
            
            logger.info(f"Loaded {len(self.feedback_records)} feedback records from {self.db_path}")
        
        except FileNotFoundError:
            logger.info(f"No existing feedback data found at {self.db_path}")
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
    
    def _save_data(self):
        """Save feedback data to file"""
        try:
            # Convert records to dicts with enum values as strings
            def record_to_dict(record):
                d = asdict(record)
                d['correction_type'] = record.correction_type.value
                d['review_status'] = record.review_status.value
                return d
            
            data = {
                'feedback_records': [record_to_dict(record) for record in self.feedback_records],
                'pending_reviews': [record_to_dict(record) for record in self.pending_reviews.values()]
            }
            
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug(f"Saved {len(self.feedback_records)} feedback records to {self.db_path}")
        
        except Exception as e:
            logger.error(f"Error saving feedback data: {e}")


class PatternAnalyzer:
    """
    Analyzes feedback to discover patterns and suggest improvements
    """
    
    def __init__(self, feedback_interface: FeedbackInterface):
        """
        Initialize pattern analyzer
        
        Args:
            feedback_interface: FeedbackInterface instance
        """
        self.feedback_interface = feedback_interface
        self.discovered_patterns: List[PatternInsight] = []
    
    def analyze_false_positives(self, check_type: Optional[str] = None,
                                min_occurrences: int = 3) -> List[PatternInsight]:
        """
        Analyze false positive patterns
        
        Args:
            check_type: Filter by check type
            min_occurrences: Minimum occurrences to consider a pattern
            
        Returns:
            List of discovered patterns
        """
        # Get false positive corrections
        records = self.feedback_interface.get_feedback_history(check_type=check_type)
        false_positives = [
            r for r in records
            if r.correction_type == CorrectionType.FALSE_POSITIVE
        ]
        
        if len(false_positives) < min_occurrences:
            logger.info(f"Not enough false positives to analyze ({len(false_positives)} < {min_occurrences})")
            return []
        
        # Group by check type and look for common patterns
        patterns_by_type = defaultdict(list)
        for fp in false_positives:
            patterns_by_type[fp.check_type].append(fp)
        
        patterns = []
        for check_type, fps in patterns_by_type.items():
            if len(fps) >= min_occurrences:
                # Analyze common elements in evidence and reasoning
                common_terms = self._extract_common_terms([fp.predicted_evidence for fp in fps])
                
                pattern = PatternInsight(
                    pattern_id=f"fp_{check_type}_{len(patterns)}",
                    check_type=check_type,
                    pattern_description=f"False positive pattern in {check_type}: {', '.join(common_terms[:3])}",
                    occurrence_count=len(fps),
                    confidence=len(fps) / len(records) if records else 0,
                    examples=[fp.predicted_evidence[:100] for fp in fps[:3]],
                    suggested_rule=f"Filter out cases containing: {', '.join(common_terms[:2])}",
                    impact_score=len(fps) / len(records) if records else 0
                )
                patterns.append(pattern)
        
        self.discovered_patterns.extend(patterns)
        logger.info(f"Discovered {len(patterns)} false positive patterns")
        return patterns
    
    def analyze_false_negatives(self, check_type: Optional[str] = None,
                               min_occurrences: int = 3) -> List[PatternInsight]:
        """
        Analyze false negative patterns (missed violations)
        
        Args:
            check_type: Filter by check type
            min_occurrences: Minimum occurrences to consider a pattern
            
        Returns:
            List of discovered patterns
        """
        # Get false negative corrections
        records = self.feedback_interface.get_feedback_history(check_type=check_type)
        false_negatives = [
            r for r in records
            if r.correction_type == CorrectionType.FALSE_NEGATIVE
        ]
        
        if len(false_negatives) < min_occurrences:
            logger.info(f"Not enough false negatives to analyze ({len(false_negatives)} < {min_occurrences})")
            return []
        
        # Group by check type
        patterns_by_type = defaultdict(list)
        for fn in false_negatives:
            patterns_by_type[fn.check_type].append(fn)
        
        patterns = []
        for check_type, fns in patterns_by_type.items():
            if len(fns) >= min_occurrences:
                # Extract common terms from reviewer notes
                common_terms = self._extract_common_terms([fn.reviewer_notes for fn in fns])
                
                pattern = PatternInsight(
                    pattern_id=f"fn_{check_type}_{len(patterns)}",
                    check_type=check_type,
                    pattern_description=f"Missed violation pattern in {check_type}: {', '.join(common_terms[:3])}",
                    occurrence_count=len(fns),
                    confidence=len(fns) / len(records) if records else 0,
                    examples=[fn.reviewer_notes[:100] for fn in fns[:3]],
                    suggested_rule=f"Add detection for: {', '.join(common_terms[:2])}",
                    impact_score=len(fns) / len(records) if records else 0
                )
                patterns.append(pattern)
        
        self.discovered_patterns.extend(patterns)
        logger.info(f"Discovered {len(patterns)} false negative patterns")
        return patterns
    
    def analyze_confidence_patterns(self, check_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze confidence score accuracy patterns
        
        Args:
            check_type: Filter by check type
            
        Returns:
            Dict with confidence analysis
        """
        records = self.feedback_interface.get_feedback_history(check_type=check_type)
        confidence_corrections = [
            r for r in records
            if r.corrected_confidence is not None
        ]
        
        if not confidence_corrections:
            return {'error': 'No confidence corrections available'}
        
        # Calculate confidence errors
        errors = [
            abs(r.predicted_confidence - r.corrected_confidence)
            for r in confidence_corrections
        ]
        
        # Analyze over/under confidence
        over_confident = [
            r for r in confidence_corrections
            if r.predicted_confidence > r.corrected_confidence
        ]
        under_confident = [
            r for r in confidence_corrections
            if r.predicted_confidence < r.corrected_confidence
        ]
        
        analysis = {
            'total_corrections': len(confidence_corrections),
            'mean_error': statistics.mean(errors),
            'median_error': statistics.median(errors),
            'max_error': max(errors),
            'over_confident_count': len(over_confident),
            'under_confident_count': len(under_confident),
            'over_confident_rate': len(over_confident) / len(confidence_corrections),
            'under_confident_rate': len(under_confident) / len(confidence_corrections),
            'suggested_adjustment': -int(statistics.mean([
                r.predicted_confidence - r.corrected_confidence
                for r in confidence_corrections
            ]))
        }
        
        logger.info(f"Confidence analysis: mean error = {analysis['mean_error']:.1f}%")
        return analysis
    
    def get_rule_suggestions(self, min_impact: float = 0.05) -> List[PatternInsight]:
        """
        Get rule suggestions based on discovered patterns
        
        Args:
            min_impact: Minimum impact score to include
            
        Returns:
            List of high-impact patterns with rule suggestions
        """
        suggestions = [
            p for p in self.discovered_patterns
            if p.suggested_rule and p.impact_score >= min_impact
        ]
        
        # Sort by impact score
        suggestions.sort(key=lambda p: p.impact_score, reverse=True)
        
        return suggestions
    
    def _extract_common_terms(self, texts: List[str], min_frequency: int = 2) -> List[str]:
        """
        Extract common terms from a list of texts
        
        Args:
            texts: List of text strings
            min_frequency: Minimum frequency to consider a term common
            
        Returns:
            List of common terms
        """
        # Simple word frequency analysis
        word_counts = defaultdict(int)
        
        for text in texts:
            # Simple tokenization (lowercase, split on whitespace)
            words = text.lower().split()
            for word in words:
                # Filter out very short words and common stop words
                if len(word) > 3 and word not in {'the', 'and', 'for', 'with', 'this', 'that'}:
                    word_counts[word] += 1
        
        # Get words that appear in multiple texts
        common = [
            word for word, count in word_counts.items()
            if count >= min_frequency
        ]
        
        # Sort by frequency
        common.sort(key=lambda w: word_counts[w], reverse=True)
        
        return common


class LearningEngine:
    """
    Learning engine that uses feedback to improve predictions
    """
    
    def __init__(self, feedback_interface: FeedbackInterface,
                 pattern_analyzer: PatternAnalyzer,
                 confidence_calibrator=None):
        """
        Initialize learning engine
        
        Args:
            feedback_interface: FeedbackInterface instance
            pattern_analyzer: PatternAnalyzer instance
            confidence_calibrator: Optional ConfidenceCalibrator instance
        """
        self.feedback_interface = feedback_interface
        self.pattern_analyzer = pattern_analyzer
        self.confidence_calibrator = confidence_calibrator
        
        logger.info("LearningEngine initialized")
    
    def process_feedback_batch(self, check_type: Optional[str] = None):
        """
        Process a batch of feedback to update learning models
        
        Args:
            check_type: Filter by check type
        """
        logger.info("Processing feedback batch...")
        
        # Get recent feedback
        records = self.feedback_interface.get_feedback_history(check_type=check_type, days=30)
        
        if not records:
            logger.info("No feedback records to process")
            return
        
        # Update confidence calibrator if available
        if self.confidence_calibrator:
            for record in records:
                self.confidence_calibrator.record_prediction(
                    check_type=record.check_type,
                    predicted_confidence=record.predicted_confidence,
                    predicted_violation=record.predicted_violation,
                    actual_violation=record.actual_violation
                )
            logger.info(f"Updated confidence calibrator with {len(records)} records")
        
        # Analyze patterns
        fp_patterns = self.pattern_analyzer.analyze_false_positives(check_type=check_type)
        fn_patterns = self.pattern_analyzer.analyze_false_negatives(check_type=check_type)
        
        logger.info(f"Discovered {len(fp_patterns)} false positive patterns")
        logger.info(f"Discovered {len(fn_patterns)} false negative patterns")
        
        # Analyze confidence patterns
        confidence_analysis = self.pattern_analyzer.analyze_confidence_patterns(check_type=check_type)
        if 'suggested_adjustment' in confidence_analysis:
            logger.info(f"Suggested confidence adjustment: {confidence_analysis['suggested_adjustment']:+d}%")
    
    def get_learning_metrics(self) -> LearningMetrics:
        """
        Get metrics about learning system performance
        
        Returns:
            LearningMetrics object
        """
        records = self.feedback_interface.get_feedback_history()
        
        if not records:
            return LearningMetrics(
                total_feedback=0,
                false_positives=0,
                false_negatives=0,
                confidence_adjustments=0,
                patterns_discovered=0,
                rules_suggested=0,
                accuracy_improvement=0.0,
                avg_correction_time_ms=0.0
            )
        
        false_positives = sum(1 for r in records if r.correction_type == CorrectionType.FALSE_POSITIVE)
        false_negatives = sum(1 for r in records if r.correction_type == CorrectionType.FALSE_NEGATIVE)
        confidence_adjustments = sum(1 for r in records if r.correction_type == CorrectionType.CONFIDENCE_ADJUSTMENT)
        
        # Calculate accuracy improvement (compare early vs recent records)
        if len(records) >= 20:
            early_records = records[:len(records)//2]
            recent_records = records[len(records)//2:]
            
            early_accuracy = sum(1 for r in early_records if r.predicted_violation == r.actual_violation) / len(early_records)
            recent_accuracy = sum(1 for r in recent_records if r.predicted_violation == r.actual_violation) / len(recent_records)
            
            accuracy_improvement = recent_accuracy - early_accuracy
        else:
            accuracy_improvement = 0.0
        
        # Average correction time
        times = [r.processing_time_ms for r in records if r.processing_time_ms]
        avg_time = statistics.mean(times) if times else 0.0
        
        # Get rule suggestions
        rule_suggestions = self.pattern_analyzer.get_rule_suggestions()
        
        return LearningMetrics(
            total_feedback=len(records),
            false_positives=false_positives,
            false_negatives=false_negatives,
            confidence_adjustments=confidence_adjustments,
            patterns_discovered=len(self.pattern_analyzer.discovered_patterns),
            rules_suggested=len(rule_suggestions),
            accuracy_improvement=accuracy_improvement,
            avg_correction_time_ms=avg_time
        )
    
    def print_learning_report(self):
        """Print comprehensive learning report"""
        metrics = self.get_learning_metrics()
        
        print("\n" + "="*70)
        print("Learning Engine Report")
        print("="*70)
        print(f"Total Feedback: {metrics.total_feedback}")
        print(f"False Positives: {metrics.false_positives}")
        print(f"False Negatives: {metrics.false_negatives}")
        print(f"Confidence Adjustments: {metrics.confidence_adjustments}")
        print(f"Patterns Discovered: {metrics.patterns_discovered}")
        print(f"Rules Suggested: {metrics.rules_suggested}")
        print(f"Accuracy Improvement: {metrics.accuracy_improvement:+.1%}")
        print(f"Avg Correction Time: {metrics.avg_correction_time_ms:.0f}ms")
        
        # Show top rule suggestions
        suggestions = self.pattern_analyzer.get_rule_suggestions()
        if suggestions:
            print(f"\n📋 Top Rule Suggestions:")
            for i, pattern in enumerate(suggestions[:5], 1):
                print(f"\n  {i}. {pattern.pattern_description}")
                print(f"     Impact: {pattern.impact_score:.1%}")
                print(f"     Occurrences: {pattern.occurrence_count}")
                print(f"     Suggested Rule: {pattern.suggested_rule}")
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Example usage and testing
    print("="*70)
    print("Feedback Loop System")
    print("="*70)
    
    # Initialize components
    feedback_interface = FeedbackInterface(db_path="test_feedback.json")
    pattern_analyzer = PatternAnalyzer(feedback_interface)
    learning_engine = LearningEngine(feedback_interface, pattern_analyzer)
    
    print("\n📝 Simulating feedback workflow...")
    
    # Submit some predictions for review
    feedback_ids = []
    for i in range(10):
        feedback_id = feedback_interface.submit_for_review(
            check_type="promotional_mention",
            document_id=f"doc_{i}",
            slide=f"slide_{i}",
            predicted_violation=(i % 3 == 0),
            predicted_confidence=85,
            predicted_reasoning="AI detected promotional mention",
            predicted_evidence=f"Found phrase: 'document promotionnel' on slide {i}",
            processing_time_ms=1500.0
        )
        feedback_ids.append(feedback_id)
    
    print(f"  ✓ Submitted {len(feedback_ids)} predictions for review")
    
    # Provide corrections for some
    print("\n✏️ Providing human corrections...")
    for i, feedback_id in enumerate(feedback_ids[:7]):
        if i % 3 == 0:
            # False positive
            feedback_interface.provide_correction(
                feedback_id=feedback_id,
                actual_violation=False,
                reviewer_notes="This was an example, not actual promotional mention",
                reviewer_id="reviewer_1"
            )
        else:
            # Approve
            feedback_interface.approve_prediction(
                feedback_id=feedback_id,
                reviewer_id="reviewer_1"
            )
    
    print(f"  ✓ Processed {7} reviews")
    
    # Process feedback batch
    print("\n🔄 Processing feedback batch...")
    learning_engine.process_feedback_batch()
    
    # Print learning report
    learning_engine.print_learning_report()
    
    # Export feedback
    feedback_interface.export_feedback("test_feedback_export.json")
    print("\n✓ Exported feedback to test_feedback_export.json")
    
    print("\n" + "="*70)
    print("✓ Feedback Loop System test complete")
    print("="*70)
