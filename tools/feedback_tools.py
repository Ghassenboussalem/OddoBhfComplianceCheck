"""
Feedback Tools

This module provides functionality for the multi-agent compliance system.
"""

﻿#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feedback Tools - LangChain Tools for Feedback Processing and Learning
Provides tools for processing feedback, updating confidence calibration, detecting patterns, and suggesting rule modifications
"""

from langchain.tools import tool
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def process_feedback(
    feedback_record: Dict[str, Any],
    confidence_calibrator: Any,
    pattern_analyzer: Any
) -> bool:
    """Process a single feedback record to update learning models."""
    try:
        check_type = feedback_record.get("check_type", "UNKNOWN")
        predicted_confidence = feedback_record.get("predicted_confidence", 50)
        predicted_violation = feedback_record.get("predicted_violation", False)
        actual_violation = feedback_record.get("actual_violation", False)

        confidence_calibrator.record_prediction(
            check_type=check_type,
            predicted_confidence=predicted_confidence,
            predicted_violation=predicted_violation,
            actual_violation=actual_violation
        )

        logger.info(f"Processed feedback for {check_type}")
        return True
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return False


@tool
def process_feedback_batch(
    feedback_records: List[Dict[str, Any]],
    confidence_calibrator: Any,
    pattern_analyzer: Any
) -> Dict[str, Any]:
    """Process a batch of feedback records to update learning models."""
    successful = 0
    failed = 0

    for record in feedback_records:
        try:
            success = process_feedback.invoke({
                "feedback_record": record,
                "confidence_calibrator": confidence_calibrator,
                "pattern_analyzer": pattern_analyzer
            })
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Error processing record: {e}")
            failed += 1

    fp_patterns = pattern_analyzer.analyze_false_positives(min_occurrences=3)
    fn_patterns = pattern_analyzer.analyze_false_negatives(min_occurrences=3)
    patterns_discovered = len(fp_patterns) + len(fn_patterns)

    results = {
        "total_processed": len(feedback_records),
        "successful": successful,
        "failed": failed,
        "patterns_discovered": patterns_discovered
    }

    logger.info(f"Batch processing complete: {successful} successful, {failed} failed")
    return results


@tool
def update_confidence_calibration(
    check_type: str,
    predicted_confidence: int,
    predicted_violation: bool,
    actual_violation: bool,
    confidence_calibrator: Any
) -> Dict[str, Any]:
    """Update confidence calibration based on a prediction outcome."""
    try:
        confidence_calibrator.record_prediction(
            check_type=check_type,
            predicted_confidence=predicted_confidence,
            predicted_violation=predicted_violation,
            actual_violation=actual_violation
        )

        adjusted_confidence = confidence_calibrator.get_adjusted_confidence(
            check_type=check_type,
            raw_confidence=predicted_confidence
        )

        reliability_score = confidence_calibrator.get_reliability_score(check_type)

        result = {
            "recorded": True,
            "adjusted_confidence": adjusted_confidence,
            "reliability_score": reliability_score
        }

        logger.info(f"Updated calibration for {check_type}")
        return result
    except Exception as e:
        logger.error(f"Error updating confidence calibration: {e}")
        return {
            "recorded": False,
            "adjusted_confidence": predicted_confidence,
            "reliability_score": 0.0
        }


@tool
def get_calibration_metrics(
    confidence_calibrator: Any,
    check_type: Optional[str] = None,
    days: Optional[int] = 30
) -> Dict[str, Any]:
    """Get calibration metrics for a specific check type or overall."""
    try:
        metrics = confidence_calibrator.get_calibration_metrics(
            check_type=check_type,
            days=days
        )

        result = {
            "total_predictions": metrics.total_predictions,
            "correct_predictions": metrics.correct_predictions,
            "accuracy": metrics.accuracy,
            "mean_confidence_error": metrics.mean_confidence_error,
            "calibration_score": metrics.calibration_score,
            "over_confident_rate": metrics.over_confident_rate,
            "under_confident_rate": metrics.under_confident_rate
        }

        logger.info(f"Calibration metrics for {check_type or 'overall'}")
        return result
    except Exception as e:
        logger.error(f"Error getting calibration metrics: {e}")
        return {
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy": 0.0,
            "mean_confidence_error": 0.0,
            "calibration_score": 0.0,
            "over_confident_rate": 0.0,
            "under_confident_rate": 0.0
        }


@tool
def detect_false_positive_patterns(
    pattern_analyzer: Any,
    check_type: Optional[str] = None,
    min_occurrences: int = 3
) -> List[Dict[str, Any]]:
    """Detect patterns in false positive violations for filtering."""
    try:
        patterns = pattern_analyzer.analyze_false_positives(
            check_type=check_type,
            min_occurrences=min_occurrences
        )

        result = []
        for pattern in patterns:
            result.append({
                "pattern_id": pattern.pattern_id,
                "check_type": pattern.check_type,
                "pattern_description": pattern.pattern_description,
                "occurrence_count": pattern.occurrence_count,
                "confidence": pattern.confidence,
                "examples": pattern.examples,
                "suggested_rule": pattern.suggested_rule,
                "impact_score": pattern.impact_score
            })

        logger.info(f"Detected {len(patterns)} false positive patterns")
        return result
    except Exception as e:
        logger.error(f"Error detecting false positive patterns: {e}")
        return []


@tool
def detect_false_negative_patterns(
    pattern_analyzer: Any,
    check_type: Optional[str] = None,
    min_occurrences: int = 3
) -> List[Dict[str, Any]]:
    """Detect patterns in false negative violations (missed detections)."""
    try:
        patterns = pattern_analyzer.analyze_false_negatives(
            check_type=check_type,
            min_occurrences=min_occurrences
        )

        result = []
        for pattern in patterns:
            result.append({
                "pattern_id": pattern.pattern_id,
                "check_type": pattern.check_type,
                "pattern_description": pattern.pattern_description,
                "occurrence_count": pattern.occurrence_count,
                "confidence": pattern.confidence,
                "examples": pattern.examples,
                "suggested_rule": pattern.suggested_rule,
                "impact_score": pattern.impact_score
            })

        logger.info(f"Detected {len(patterns)} false negative patterns")
        return result
    except Exception as e:
        logger.error(f"Error detecting false negative patterns: {e}")
        return []


@tool
def detect_patterns(
    pattern_analyzer: Any,
    check_type: Optional[str] = None,
    min_occurrences: int = 3
) -> Dict[str, Any]:
    """Detect all patterns (both false positives and false negatives)."""
    try:
        fp_patterns = detect_false_positive_patterns.invoke({
            "pattern_analyzer": pattern_analyzer,
            "check_type": check_type,
            "min_occurrences": min_occurrences
        })

        fn_patterns = detect_false_negative_patterns.invoke({
            "pattern_analyzer": pattern_analyzer,
            "check_type": check_type,
            "min_occurrences": min_occurrences
        })

        all_patterns = fp_patterns + fn_patterns
        high_impact = [p for p in all_patterns if p["impact_score"] > 0.1]

        result = {
            "false_positive_patterns": fp_patterns,
            "false_negative_patterns": fn_patterns,
            "total_patterns": len(all_patterns),
            "high_impact_patterns": high_impact
        }

        logger.info(f"Pattern detection complete: {len(all_patterns)} total patterns")
        return result
    except Exception as e:
        logger.error(f"Error detecting patterns: {e}")
        return {
            "false_positive_patterns": [],
            "false_negative_patterns": [],
            "total_patterns": 0,
            "high_impact_patterns": []
        }


@tool
def suggest_rule_modifications(
    pattern_analyzer: Any,
    min_impact: float = 0.05
) -> List[Dict[str, Any]]:
    """Generate rule modification suggestions based on discovered patterns."""
    try:
        patterns = pattern_analyzer.get_rule_suggestions(min_impact=min_impact)

        suggestions = []
        for pattern in patterns:
            if "false_positive" in pattern.pattern_id:
                suggestion_type = "filter"
                implementation = f"Add filtering logic to exclude: {pattern.suggested_rule}"
            else:
                suggestion_type = "detection"
                implementation = f"Add detection logic for: {pattern.suggested_rule}"

            suggestions.append({
                "pattern_id": pattern.pattern_id,
                "check_type": pattern.check_type,
                "suggestion_type": suggestion_type,
                "description": pattern.pattern_description,
                "suggested_rule": pattern.suggested_rule,
                "impact_score": pattern.impact_score,
                "priority": "high" if pattern.impact_score > 0.15 else "medium",
                "implementation_notes": implementation
            })

        suggestions.sort(key=lambda s: s["impact_score"], reverse=True)
        logger.info(f"Generated {len(suggestions)} rule modification suggestions")
        return suggestions
    except Exception as e:
        logger.error(f"Error suggesting rule modifications: {e}")
        return []


@tool
def get_learning_metrics(
    feedback_interface: Any,
    pattern_analyzer: Any
) -> Dict[str, Any]:
    """Get comprehensive learning system metrics."""
    try:
        from feedback_loop import CorrectionType
        records = feedback_interface.get_feedback_history()

        false_positives = sum(
            1 for r in records
            if r.correction_type == CorrectionType.FALSE_POSITIVE
        )
        false_negatives = sum(
            1 for r in records
            if r.correction_type == CorrectionType.FALSE_NEGATIVE
        )

        patterns = pattern_analyzer.get_rule_suggestions(min_impact=0.0)

        accuracy_improvement = 0.0
        if len(records) >= 20:
            early_records = records[:len(records)//2]
            recent_records = records[len(records)//2:]

            early_accuracy = sum(
                1 for r in early_records
                if r.predicted_violation == r.actual_violation
            ) / len(early_records)

            recent_accuracy = sum(
                1 for r in recent_records
                if r.predicted_violation == r.actual_violation
            ) / len(recent_records)

            accuracy_improvement = recent_accuracy - early_accuracy

        result = {
            "total_feedback": len(records),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "patterns_discovered": len(pattern_analyzer.discovered_patterns),
            "rules_suggested": len(patterns),
            "accuracy_improvement": accuracy_improvement
        }

        logger.info(f"Learning metrics: {len(records)} feedback records")
        return result
    except Exception as e:
        logger.error(f"Error getting learning metrics: {e}")
        return {
            "total_feedback": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "patterns_discovered": 0,
            "rules_suggested": 0,
            "accuracy_improvement": 0.0
        }


@tool
def analyze_confidence_patterns(
    pattern_analyzer: Any,
    check_type: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze confidence score accuracy patterns."""
    try:
        analysis = pattern_analyzer.analyze_confidence_patterns(check_type=check_type)

        if "error" in analysis:
            logger.warning(f"No confidence corrections available")
            return {
                "total_corrections": 0,
                "mean_error": 0.0,
                "median_error": 0.0,
                "max_error": 0.0,
                "over_confident_count": 0,
                "under_confident_count": 0,
                "over_confident_rate": 0.0,
                "under_confident_rate": 0.0,
                "suggested_adjustment": 0
            }

        logger.info(f"Confidence analysis complete")
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing confidence patterns: {e}")
        return {
            "total_corrections": 0,
            "mean_error": 0.0,
            "median_error": 0.0,
            "max_error": 0.0,
            "over_confident_count": 0,
            "under_confident_count": 0,
            "over_confident_rate": 0.0,
            "under_confident_rate": 0.0,
            "suggested_adjustment": 0
        }


# Export all tools as a list for easy registration
FEEDBACK_TOOLS = [
    process_feedback,
    process_feedback_batch,
    update_confidence_calibration,
    get_calibration_metrics,
    detect_false_positive_patterns,
    detect_false_negative_patterns,
    detect_patterns,
    suggest_rule_modifications,
    get_learning_metrics,
    analyze_confidence_patterns
]


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("Feedback Tools - LangChain Tool Testing")
    logger.info("="*70)

    from feedback_loop import FeedbackInterface, PatternAnalyzer
    from confidence_calibrator import ConfidenceCalibrator

    feedback_interface = FeedbackInterface(db_path="test_feedback_tools.json")
    confidence_calibrator = ConfidenceCalibrator(db_path="test_calibration_tools.json")
    pattern_analyzer = PatternAnalyzer(feedback_interface)

    logger.info("\n✓ Feedback Tools module loaded successfully")
    logger.info(f"✓ {len(FEEDBACK_TOOLS)} tools available")
    logger.info("="*70)
