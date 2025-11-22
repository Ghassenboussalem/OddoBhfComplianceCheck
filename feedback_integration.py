#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feedback Integration System
Connects review feedback to learning components for real-time model updates
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeedbackProcessingResult:
    """Result of processing a feedback record"""
    feedback_id: str
    processing_time_ms: float
    calibration_updated: bool
    patterns_analyzed: bool
    confidence_adjusted: bool
    error: Optional[str] = None


class FeedbackIntegration:
    """
    Integrates human feedback into the learning system
    
    Features:
    - Real-time confidence calibration updates (< 1 second)
    - Pattern detection for false positives/negatives
    - Confidence adjustment based on corrections
    - Audit trail maintenance
    - Batch processing support
    """
    
    def __init__(self, feedback_interface, confidence_calibrator, 
                 pattern_detector=None, audit_logger=None):
        """
        Initialize feedback integration
        
        Args:
            feedback_interface: FeedbackInterface instance
            confidence_calibrator: ConfidenceCalibrator instance
            pattern_detector: Optional AIPatternDetector instance
            audit_logger: Optional AuditLogger instance
        """
        self.feedback_interface = feedback_interface
        self.confidence_calibrator = confidence_calibrator
        self.pattern_detector = pattern_detector
        self.audit_logger = audit_logger
        
        # Performance tracking
        self.processing_times: List[float] = []
        self.total_processed = 0
        
        # Register callback with feedback interface
        self.feedback_interface.register_learning_callback(
            self._on_feedback_received
        )
        
        logger.info("FeedbackIntegration initialized with real-time learning")
    
    def _on_feedback_received(self, feedback_record):
        """
        Callback triggered when feedback is received
        Performs real-time learning updates
        
        Args:
            feedback_record: FeedbackRecord from reviewer
        """
        start_time = time.time()
        
        try:
            # Process the feedback immediately
            result = self.process_review_decision(feedback_record)
            
            # Track processing time
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            self.total_processed += 1
            
            logger.info(f"Processed feedback {feedback_record.feedback_id} in {processing_time:.1f}ms")
            
            # Log to audit if available
            if self.audit_logger:
                self.audit_logger.log_feedback_processing(feedback_record, result)
        
        except Exception as e:
            logger.error(f"Error processing feedback callback: {e}")
    
    def process_review_decision(self, feedback_record) -> FeedbackProcessingResult:
        """
        Process a review decision and update learning models
        
        Args:
            feedback_record: FeedbackRecord with reviewer's decision
            
        Returns:
            FeedbackProcessingResult with processing details
        """
        start_time = time.time()
        result = FeedbackProcessingResult(
            feedback_id=feedback_record.feedback_id,
            processing_time_ms=0.0,
            calibration_updated=False,
            patterns_analyzed=False,
            confidence_adjusted=False,
            error=None
        )
        
        try:
            # 1. Update confidence calibration (MUST be < 1 second)
            result.calibration_updated = self._update_confidence_calibration(feedback_record)
            
            # 2. Analyze patterns (async-friendly, can be slower)
            if self.pattern_detector:
                result.patterns_analyzed = self._analyze_patterns(feedback_record)
            
            # 3. Apply real-time confidence adjustments
            result.confidence_adjusted = self._apply_confidence_adjustment(feedback_record)
            
            # Calculate total processing time
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            # Verify we met the 1-second requirement
            if result.processing_time_ms > 1000:
                logger.warning(f"Feedback processing took {result.processing_time_ms:.1f}ms (> 1000ms target)")
            
            logger.info(f"Processed feedback: calibration={result.calibration_updated}, "
                       f"patterns={result.patterns_analyzed}, "
                       f"adjustment={result.confidence_adjusted}, "
                       f"time={result.processing_time_ms:.1f}ms")
        
        except Exception as e:
            result.error = str(e)
            logger.error(f"Error processing review decision: {e}")
        
        return result
    
    def _update_confidence_calibration(self, feedback_record) -> bool:
        """
        Update confidence calibration model with feedback
        
        Args:
            feedback_record: FeedbackRecord
            
        Returns:
            True if successful
        """
        try:
            # Record the prediction outcome
            self.confidence_calibrator.record_prediction(
                check_type=feedback_record.check_type,
                predicted_confidence=feedback_record.predicted_confidence,
                predicted_violation=feedback_record.predicted_violation,
                actual_violation=feedback_record.actual_violation
            )
            
            logger.debug(f"Updated calibration for {feedback_record.check_type}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update calibration: {e}")
            return False
    
    def _analyze_patterns(self, feedback_record) -> bool:
        """
        Analyze feedback for patterns (false positives/negatives)
        
        Args:
            feedback_record: FeedbackRecord
            
        Returns:
            True if analysis was performed
        """
        try:
            from feedback_loop import CorrectionType
            
            # Trigger pattern analysis based on correction type
            if feedback_record.correction_type == CorrectionType.FALSE_POSITIVE:
                # Analyze false positive patterns
                patterns = self.pattern_detector.discover_false_positive_patterns(
                    check_type=feedback_record.check_type,
                    min_occurrences=3
                )
                if patterns:
                    logger.info(f"Discovered {len(patterns)} false positive patterns")
            
            elif feedback_record.correction_type == CorrectionType.FALSE_NEGATIVE:
                # Analyze false negative patterns
                patterns = self.pattern_detector.discover_missed_violation_patterns(
                    check_type=feedback_record.check_type,
                    min_occurrences=3
                )
                if patterns:
                    logger.info(f"Discovered {len(patterns)} missed violation patterns")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to analyze patterns: {e}")
            return False
    
    def _apply_confidence_adjustment(self, feedback_record) -> bool:
        """
        Apply real-time confidence adjustment based on correction
        
        Args:
            feedback_record: FeedbackRecord
            
        Returns:
            True if adjustment was applied
        """
        try:
            # If reviewer provided corrected confidence, use it for adjustment
            if feedback_record.corrected_confidence is not None:
                adjustment = feedback_record.corrected_confidence - feedback_record.predicted_confidence
                
                # Apply adjustment to calibrator's threshold adjustments
                current_adjustments = self.confidence_calibrator.threshold_adjustments.get(
                    feedback_record.check_type, {}
                )
                
                # Update confidence adjustment (weighted average with existing)
                existing_adj = current_adjustments.get('confidence_adjustment', 0)
                new_adj = int((existing_adj * 0.9) + (adjustment * 0.1))  # 10% weight to new feedback
                
                self.confidence_calibrator.threshold_adjustments[feedback_record.check_type] = {
                    **current_adjustments,
                    'confidence_adjustment': new_adj
                }
                
                # Save updated adjustments
                self.confidence_calibrator._save_data()
                
                logger.info(f"Applied confidence adjustment for {feedback_record.check_type}: {new_adj:+d}%")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to apply confidence adjustment: {e}")
            return False
    
    def get_accuracy_metrics(self, check_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate AI accuracy metrics from feedback
        
        Args:
            check_type: Optional filter by check type
            
        Returns:
            Dict with accuracy metrics (precision, recall, F1)
        """
        records = self.feedback_interface.get_feedback_history(check_type=check_type)
        
        if not records:
            return {
                'total_reviews': 0,
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0
            }
        
        # Calculate confusion matrix
        true_positives = sum(
            1 for r in records
            if r.predicted_violation and r.actual_violation
        )
        false_positives = sum(
            1 for r in records
            if r.predicted_violation and not r.actual_violation
        )
        true_negatives = sum(
            1 for r in records
            if not r.predicted_violation and not r.actual_violation
        )
        false_negatives = sum(
            1 for r in records
            if not r.predicted_violation and r.actual_violation
        )
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (true_positives + true_negatives) / len(records) if records else 0.0
        
        # Calculate average confidence for correct vs incorrect predictions
        correct_records = [r for r in records if r.predicted_violation == r.actual_violation]
        incorrect_records = [r for r in records if r.predicted_violation != r.actual_violation]
        
        avg_confidence_correct = (
            sum(r.predicted_confidence for r in correct_records) / len(correct_records)
            if correct_records else 0.0
        )
        avg_confidence_incorrect = (
            sum(r.predicted_confidence for r in incorrect_records) / len(incorrect_records)
            if incorrect_records else 0.0
        )
        
        return {
            'total_reviews': len(records),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'avg_confidence_correct': avg_confidence_correct,
            'avg_confidence_incorrect': avg_confidence_incorrect
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about feedback processing performance
        
        Returns:
            Dict with processing statistics
        """
        if not self.processing_times:
            return {
                'total_processed': 0,
                'avg_processing_time_ms': 0.0,
                'max_processing_time_ms': 0.0,
                'min_processing_time_ms': 0.0,
                'under_1s_rate': 0.0
            }
        
        under_1s = sum(1 for t in self.processing_times if t < 1000)
        
        return {
            'total_processed': self.total_processed,
            'avg_processing_time_ms': sum(self.processing_times) / len(self.processing_times),
            'max_processing_time_ms': max(self.processing_times),
            'min_processing_time_ms': min(self.processing_times),
            'under_1s_rate': under_1s / len(self.processing_times)
        }
    
    def process_batch_feedback(self, check_type: Optional[str] = None, days: int = 30):
        """
        Process a batch of historical feedback for pattern analysis
        
        Args:
            check_type: Optional filter by check type
            days: Number of days of history to process
        """
        logger.info(f"Processing batch feedback for last {days} days...")
        
        records = self.feedback_interface.get_feedback_history(check_type=check_type, days=days)
        
        if not records:
            logger.info("No feedback records to process")
            return
        
        # Update calibration for all records
        for record in records:
            self._update_confidence_calibration(record)
        
        logger.info(f"Updated calibration with {len(records)} historical records")
        
        # Analyze patterns if detector available
        if self.pattern_detector:
            from feedback_loop import CorrectionType
            
            # Analyze false positives
            fp_records = [r for r in records if r.correction_type == CorrectionType.FALSE_POSITIVE]
            if fp_records:
                fp_patterns = self.pattern_detector.discover_false_positive_patterns(
                    check_type=check_type,
                    min_occurrences=3
                )
                logger.info(f"Discovered {len(fp_patterns)} false positive patterns from {len(fp_records)} records")
            
            # Analyze false negatives
            fn_records = [r for r in records if r.correction_type == CorrectionType.FALSE_NEGATIVE]
            if fn_records:
                fn_patterns = self.pattern_detector.discover_missed_violation_patterns(
                    check_type=check_type,
                    min_occurrences=3
                )
                logger.info(f"Discovered {len(fn_patterns)} false negative patterns from {len(fn_records)} records")
    
    def print_integration_report(self):
        """Print comprehensive integration report"""
        print("\n" + "="*70)
        print("Feedback Integration Report")
        print("="*70)
        
        # Processing stats
        stats = self.get_processing_stats()
        print(f"\n📊 Processing Performance:")
        print(f"  Total Processed: {stats['total_processed']}")
        print(f"  Avg Processing Time: {stats['avg_processing_time_ms']:.1f}ms")
        print(f"  Max Processing Time: {stats['max_processing_time_ms']:.1f}ms")
        print(f"  Under 1s Rate: {stats['under_1s_rate']:.1%}")
        
        # Accuracy metrics
        metrics = self.get_accuracy_metrics()
        print(f"\n🎯 Accuracy Metrics:")
        print(f"  Total Reviews: {metrics['total_reviews']}")
        print(f"  Accuracy: {metrics['accuracy']:.1%}")
        print(f"  Precision: {metrics['precision']:.1%}")
        print(f"  Recall: {metrics['recall']:.1%}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
        print(f"  Avg Confidence (Correct): {metrics['avg_confidence_correct']:.1f}%")
        print(f"  Avg Confidence (Incorrect): {metrics['avg_confidence_incorrect']:.1f}%")
        
        # Calibration info
        print(f"\n🔧 Calibration Status:")
        check_types = list(set(r.check_type for r in self.feedback_interface.feedback_records))
        for check_type in check_types[:5]:  # Show top 5
            reliability = self.confidence_calibrator.get_reliability_score(check_type)
            adjustments = self.confidence_calibrator.threshold_adjustments.get(check_type, {})
            print(f"  {check_type}:")
            print(f"    Reliability: {reliability:.3f}")
            if adjustments:
                for key, value in adjustments.items():
                    print(f"    {key}: {value:+d}")
        
        print("\n" + "="*70 + "\n")
    
    def export_integration_metrics(self, filepath: str):
        """
        Export integration metrics to JSON file
        
        Args:
            filepath: Path to output file
        """
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'processing_stats': self.get_processing_stats(),
            'accuracy_metrics': self.get_accuracy_metrics(),
            'calibration_adjustments': dict(self.confidence_calibrator.threshold_adjustments),
            'feedback_history_count': len(self.feedback_interface.feedback_records)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported integration metrics to {filepath}")


if __name__ == "__main__":
    # Example usage and testing
    print("="*70)
    print("Feedback Integration System")
    print("="*70)
    
    # Import required components
    from feedback_loop import FeedbackInterface
    from confidence_calibrator import ConfidenceCalibrator
    
    # Initialize components
    feedback_interface = FeedbackInterface(db_path="test_feedback_integration.json")
    confidence_calibrator = ConfidenceCalibrator(db_path="test_calibration_integration.json")
    
    # Initialize integration
    integration = FeedbackIntegration(
        feedback_interface=feedback_interface,
        confidence_calibrator=confidence_calibrator
    )
    
    print("\n✓ Feedback integration initialized with real-time learning")
    
    # Simulate some feedback
    print("\n📝 Simulating feedback workflow...")
    
    # Submit predictions
    feedback_ids = []
    for i in range(5):
        feedback_id = feedback_interface.submit_for_review(
            check_type="PROMOTIONAL_MENTION",
            document_id=f"doc_{i}",
            slide=f"slide_{i}",
            predicted_violation=True,
            predicted_confidence=75 + (i * 3),
            predicted_reasoning="AI detected promotional mention",
            predicted_evidence=f"Found phrase on slide {i}",
            processing_time_ms=1200.0
        )
        feedback_ids.append(feedback_id)
    
    print(f"  ✓ Submitted {len(feedback_ids)} predictions")
    
    # Provide corrections (triggers real-time learning)
    print("\n✏️ Providing corrections (triggers real-time learning)...")
    for i, feedback_id in enumerate(feedback_ids):
        if i % 2 == 0:
            # False positive
            feedback_interface.provide_correction(
                feedback_id=feedback_id,
                actual_violation=False,
                reviewer_notes="This was an example",
                corrected_confidence=30,
                reviewer_id="reviewer_1"
            )
        else:
            # Correct prediction
            feedback_interface.provide_correction(
                feedback_id=feedback_id,
                actual_violation=True,
                reviewer_notes="Confirmed violation",
                corrected_confidence=90,
                reviewer_id="reviewer_1"
            )
    
    print(f"  ✓ Processed {len(feedback_ids)} corrections")
    
    # Print integration report
    integration.print_integration_report()
    
    # Export metrics
    integration.export_integration_metrics("test_integration_metrics.json")
    print("\n✓ Exported metrics to test_integration_metrics.json")
    
    print("\n" + "="*70)
    print("✓ Feedback Integration System test complete")
    print("="*70)
