#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confidence Calibration System
Tracks accuracy over time and dynamically adjusts confidence thresholds
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CalibrationRecord:
    """Single calibration data point"""
    timestamp: str
    check_type: str
    predicted_confidence: int
    predicted_violation: bool
    actual_violation: bool
    correct_prediction: bool
    confidence_error: float  # Difference between predicted and actual


@dataclass
class CalibrationMetrics:
    """Aggregated calibration metrics"""
    total_predictions: int
    correct_predictions: int
    accuracy: float
    mean_confidence_error: float
    calibration_score: float  # How well confidence matches actual accuracy
    over_confident_rate: float  # Rate of high confidence but wrong predictions
    under_confident_rate: float  # Rate of low confidence but correct predictions


class ConfidenceCalibrator:
    """
    Tracks prediction accuracy and calibrates confidence scores
    
    Features:
    - Accuracy tracking database for all predictions
    - Calibration algorithms based on historical performance
    - Dynamic threshold adjustment
    - Confidence score reliability metrics
    """
    
    def __init__(self, db_path: str = "calibration_data.json"):
        """
        Initialize confidence calibrator
        
        Args:
            db_path: Path to JSON file for storing calibration data
        """
        self.db_path = db_path
        self.records: List[CalibrationRecord] = []
        self.metrics_by_check_type: Dict[str, CalibrationMetrics] = {}
        self.threshold_adjustments: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        # Load existing data
        self._load_data()
        
        logger.info(f"ConfidenceCalibrator initialized with {len(self.records)} historical records")
    
    def record_prediction(self, check_type: str, predicted_confidence: int,
                         predicted_violation: bool, actual_violation: bool):
        """
        Record a prediction and its actual outcome
        
        Args:
            check_type: Type of compliance check
            predicted_confidence: Confidence score (0-100) that was predicted
            predicted_violation: Whether violation was predicted
            actual_violation: Actual outcome (from human review)
        """
        correct = (predicted_violation == actual_violation)
        
        # Calculate confidence error
        # If correct: error = 0 if confidence was 100%, increases as confidence decreases
        # If incorrect: error = confidence (higher confidence on wrong prediction = worse)
        if correct:
            confidence_error = (100 - predicted_confidence) / 100.0
        else:
            confidence_error = predicted_confidence / 100.0
        
        record = CalibrationRecord(
            timestamp=datetime.now().isoformat(),
            check_type=check_type,
            predicted_confidence=predicted_confidence,
            predicted_violation=predicted_violation,
            actual_violation=actual_violation,
            correct_prediction=correct,
            confidence_error=confidence_error
        )
        
        self.records.append(record)
        
        logger.info(f"Recorded prediction: {check_type}, confidence={predicted_confidence}%, "
                   f"correct={correct}")
        
        # Recalibrate if we have enough data
        if len(self.records) % 10 == 0:
            self._recalibrate()
    
    def get_calibration_metrics(self, check_type: Optional[str] = None,
                               days: Optional[int] = None) -> CalibrationMetrics:
        """
        Get calibration metrics for a specific check type or overall
        
        Args:
            check_type: Specific check type (None for overall)
            days: Only include records from last N days (None for all time)
            
        Returns:
            CalibrationMetrics object
        """
        # Filter records
        filtered_records = self._filter_records(check_type, days)
        
        if not filtered_records:
            return CalibrationMetrics(
                total_predictions=0,
                correct_predictions=0,
                accuracy=0.0,
                mean_confidence_error=0.0,
                calibration_score=0.0,
                over_confident_rate=0.0,
                under_confident_rate=0.0
            )
        
        # Calculate metrics
        total = len(filtered_records)
        correct = sum(1 for r in filtered_records if r.correct_prediction)
        accuracy = correct / total
        
        mean_error = statistics.mean(r.confidence_error for r in filtered_records)
        
        # Calibration score: how well confidence matches actual accuracy
        # Group by confidence buckets and compare predicted vs actual accuracy
        calibration_score = self._calculate_calibration_score(filtered_records)
        
        # Over-confident: high confidence (>80%) but wrong
        over_confident = sum(
            1 for r in filtered_records 
            if r.predicted_confidence > 80 and not r.correct_prediction
        )
        over_confident_rate = over_confident / total
        
        # Under-confident: low confidence (<70%) but correct
        under_confident = sum(
            1 for r in filtered_records
            if r.predicted_confidence < 70 and r.correct_prediction
        )
        under_confident_rate = under_confident / total
        
        metrics = CalibrationMetrics(
            total_predictions=total,
            correct_predictions=correct,
            accuracy=accuracy,
            mean_confidence_error=mean_error,
            calibration_score=calibration_score,
            over_confident_rate=over_confident_rate,
            under_confident_rate=under_confident_rate
        )
        
        # Cache metrics
        if check_type:
            self.metrics_by_check_type[check_type] = metrics
        
        return metrics
    
    def get_adjusted_confidence(self, check_type: str, raw_confidence: int) -> int:
        """
        Get calibrated confidence score based on historical performance
        
        Args:
            check_type: Type of compliance check
            raw_confidence: Raw confidence score from AI/rules
            
        Returns:
            Adjusted confidence score
        """
        # Get adjustment for this check type
        adjustment = self.threshold_adjustments.get(check_type, {}).get('confidence_adjustment', 0)
        
        # Apply adjustment
        adjusted = raw_confidence + adjustment
        
        # Clamp to valid range
        adjusted = max(0, min(100, adjusted))
        
        logger.debug(f"Adjusted confidence for {check_type}: {raw_confidence}% -> {adjusted}%")
        
        return adjusted
    
    def get_adjusted_threshold(self, check_type: str, threshold_type: str = 'review') -> int:
        """
        Get adjusted threshold for a specific check type
        
        Args:
            check_type: Type of compliance check
            threshold_type: Type of threshold ('review', 'high', 'medium')
            
        Returns:
            Adjusted threshold value
        """
        key = f'{threshold_type}_threshold'
        adjustment = self.threshold_adjustments.get(check_type, {}).get(key, 0)
        
        # Default thresholds
        defaults = {
            'review_threshold': 60,
            'high_threshold': 85,
            'medium_threshold': 70
        }
        
        base = defaults.get(key, 70)
        adjusted = base + adjustment
        
        # Clamp to valid range
        adjusted = max(40, min(95, adjusted))
        
        return adjusted
    
    def get_reliability_score(self, check_type: str) -> float:
        """
        Get reliability score for a specific check type
        
        Args:
            check_type: Type of compliance check
            
        Returns:
            Reliability score (0.0 to 1.0)
        """
        metrics = self.get_calibration_metrics(check_type, days=30)
        
        if metrics.total_predictions < 10:
            return 0.5  # Not enough data
        
        # Reliability = accuracy * calibration_score * (1 - over_confident_rate)
        reliability = (
            metrics.accuracy * 
            metrics.calibration_score * 
            (1.0 - metrics.over_confident_rate)
        )
        
        return reliability
    
    def export_metrics(self, filepath: str):
        """
        Export calibration metrics to JSON file
        
        Args:
            filepath: Path to output file
        """
        data = {
            'total_records': len(self.records),
            'overall_metrics': asdict(self.get_calibration_metrics()),
            'metrics_by_check_type': {
                check_type: asdict(self.get_calibration_metrics(check_type))
                for check_type in self._get_check_types()
            },
            'threshold_adjustments': dict(self.threshold_adjustments),
            'reliability_scores': {
                check_type: self.get_reliability_score(check_type)
                for check_type in self._get_check_types()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported calibration metrics to {filepath}")
    
    def print_calibration_report(self, check_type: Optional[str] = None):
        """
        Print calibration report to console
        
        Args:
            check_type: Specific check type (None for overall)
        """
        metrics = self.get_calibration_metrics(check_type, days=30)
        
        print("\n" + "="*70)
        print(f"Confidence Calibration Report - {check_type or 'Overall'}")
        print("="*70)
        print(f"Total Predictions: {metrics.total_predictions}")
        print(f"Correct Predictions: {metrics.correct_predictions}")
        print(f"Accuracy: {metrics.accuracy:.1%}")
        print(f"Mean Confidence Error: {metrics.mean_confidence_error:.3f}")
        print(f"Calibration Score: {metrics.calibration_score:.3f}")
        print(f"Over-Confident Rate: {metrics.over_confident_rate:.1%}")
        print(f"Under-Confident Rate: {metrics.under_confident_rate:.1%}")
        
        if check_type:
            reliability = self.get_reliability_score(check_type)
            print(f"Reliability Score: {reliability:.3f}")
            
            adjustments = self.threshold_adjustments.get(check_type, {})
            if adjustments:
                print(f"\nThreshold Adjustments:")
                for key, value in adjustments.items():
                    print(f"  {key}: {value:+d}")
        
        print("="*70 + "\n")
    
    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================
    
    def _recalibrate(self):
        """Recalibrate thresholds based on recent performance"""
        logger.info("Recalibrating confidence thresholds...")
        
        for check_type in self._get_check_types():
            metrics = self.get_calibration_metrics(check_type, days=30)
            
            if metrics.total_predictions < 20:
                continue  # Not enough data
            
            adjustments = {}
            
            # Adjust confidence if consistently over/under confident
            if metrics.over_confident_rate > 0.2:
                # Too many high-confidence wrong predictions - reduce confidence
                adjustments['confidence_adjustment'] = -5
                logger.info(f"{check_type}: Reducing confidence (over-confident rate: "
                           f"{metrics.over_confident_rate:.1%})")
            elif metrics.under_confident_rate > 0.2:
                # Too many low-confidence correct predictions - increase confidence
                adjustments['confidence_adjustment'] = +5
                logger.info(f"{check_type}: Increasing confidence (under-confident rate: "
                           f"{metrics.under_confident_rate:.1%})")
            
            # Adjust review threshold based on accuracy
            if metrics.accuracy > 0.9:
                # Very accurate - can lower review threshold
                adjustments['review_threshold'] = -5
                logger.info(f"{check_type}: Lowering review threshold (accuracy: "
                           f"{metrics.accuracy:.1%})")
            elif metrics.accuracy < 0.7:
                # Not accurate enough - raise review threshold
                adjustments['review_threshold'] = +5
                logger.info(f"{check_type}: Raising review threshold (accuracy: "
                           f"{metrics.accuracy:.1%})")
            
            if adjustments:
                self.threshold_adjustments[check_type] = adjustments
        
        # Save updated adjustments
        self._save_data()
    
    def _calculate_calibration_score(self, records: List[CalibrationRecord]) -> float:
        """
        Calculate calibration score (how well confidence matches actual accuracy)
        
        Returns:
            Score from 0.0 (poorly calibrated) to 1.0 (perfectly calibrated)
        """
        if not records:
            return 0.0
        
        # Group records by confidence buckets
        buckets = defaultdict(list)
        for record in records:
            bucket = (record.predicted_confidence // 10) * 10
            buckets[bucket].append(record)
        
        # Calculate calibration error for each bucket
        calibration_errors = []
        for bucket, bucket_records in buckets.items():
            if len(bucket_records) < 3:
                continue  # Skip buckets with too few samples
            
            predicted_confidence = bucket / 100.0
            actual_accuracy = sum(1 for r in bucket_records if r.correct_prediction) / len(bucket_records)
            
            error = abs(predicted_confidence - actual_accuracy)
            calibration_errors.append(error)
        
        if not calibration_errors:
            return 0.5  # Not enough data
        
        # Average calibration error
        mean_error = statistics.mean(calibration_errors)
        
        # Convert to score (lower error = higher score)
        score = max(0.0, 1.0 - mean_error)
        
        return score
    
    def _filter_records(self, check_type: Optional[str] = None,
                       days: Optional[int] = None) -> List[CalibrationRecord]:
        """Filter records by check type and time range"""
        filtered = self.records
        
        # Filter by check type
        if check_type:
            filtered = [r for r in filtered if r.check_type == check_type]
        
        # Filter by time range
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            filtered = [
                r for r in filtered
                if datetime.fromisoformat(r.timestamp) >= cutoff
            ]
        
        return filtered
    
    def _get_check_types(self) -> List[str]:
        """Get list of unique check types in records"""
        return list(set(r.check_type for r in self.records))
    
    def _load_data(self):
        """Load calibration data from file"""
        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)
            
            self.records = [
                CalibrationRecord(**record)
                for record in data.get('records', [])
            ]
            
            self.threshold_adjustments = defaultdict(
                dict,
                data.get('threshold_adjustments', {})
            )
            
            logger.info(f"Loaded {len(self.records)} calibration records from {self.db_path}")
        
        except FileNotFoundError:
            logger.info(f"No existing calibration data found at {self.db_path}")
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
    
    def _save_data(self):
        """Save calibration data to file"""
        try:
            data = {
                'records': [asdict(record) for record in self.records],
                'threshold_adjustments': dict(self.threshold_adjustments)
            }
            
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.records)} calibration records to {self.db_path}")
        
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")


if __name__ == "__main__":
    # Example usage and testing
    print("="*70)
    print("Confidence Calibration System")
    print("="*70)
    
    calibrator = ConfidenceCalibrator(db_path="test_calibration.json")
    
    # Simulate some predictions
    print("\n📊 Simulating predictions...")
    
    # Good predictions (high confidence, correct)
    for i in range(15):
        calibrator.record_prediction(
            check_type="PROMOTIONAL_MENTION",
            predicted_confidence=90,
            predicted_violation=True,
            actual_violation=True
        )
    
    # Over-confident wrong predictions
    for i in range(5):
        calibrator.record_prediction(
            check_type="PROMOTIONAL_MENTION",
            predicted_confidence=85,
            predicted_violation=True,
            actual_violation=False
        )
    
    # Under-confident correct predictions
    for i in range(8):
        calibrator.record_prediction(
            check_type="FUND_NAME_MATCH",
            predicted_confidence=65,
            predicted_violation=True,
            actual_violation=True
        )
    
    # Print calibration report
    calibrator.print_calibration_report("PROMOTIONAL_MENTION")
    calibrator.print_calibration_report("FUND_NAME_MATCH")
    
    # Test adjusted confidence
    print("\n🔧 Testing confidence adjustments...")
    raw_confidence = 85
    adjusted = calibrator.get_adjusted_confidence("PROMOTIONAL_MENTION", raw_confidence)
    print(f"Raw confidence: {raw_confidence}%")
    print(f"Adjusted confidence: {adjusted}%")
    
    # Export metrics
    calibrator.export_metrics("test_calibration_metrics.json")
    print("\n✓ Exported metrics to test_calibration_metrics.json")
    
    print("\n" + "="*70)
    print("✓ Confidence Calibration System test complete")
    print("="*70)
