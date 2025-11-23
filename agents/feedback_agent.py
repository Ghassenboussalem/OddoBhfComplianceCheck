#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feedback Agent

This module provides functionality for the multi-agent compliance system.
"""

"""
Feedback Agent - Multi-Agent Compliance System

This agent processes human feedback to improve system accuracy through:
- Real-time confidence calibration updates
- Pattern detection in false positives and false negatives
- Rule suggestion generation
- Accuracy metrics calculation
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import base agent framework
from agents.base_agent import BaseAgent, AgentConfig

# Import state models
from data_models_multiagent import ComplianceState, update_state_timestamp

# Import feedback tools
from tools.feedback_tools import (
    process_feedback,
    process_feedback_batch,
    update_confidence_calibration,
    get_calibration_metrics,
    detect_patterns,
    suggest_rule_modifications,
    get_learning_metrics,
    analyze_confidence_patterns
)

# Import feedback system components
from feedback_loop import FeedbackInterface, PatternAnalyzer, LearningEngine
from confidence_calibrator import ConfidenceCalibrator
from pattern_detector import AIPatternDetector, RuleSuggestionEngine

# Configure logging
logger = logging.getLogger(__name__)


class FeedbackAgent(BaseAgent):
    """
    Feedback Agent for processing human corrections and learning

    Responsibilities:
    - Process feedback from human reviewers
    - Update confidence calibration models in real-time
    - Detect patterns in false positives and false negatives
    - Generate rule modification suggestions
    - Calculate and track accuracy metrics
    - Provide learning insights for system improvement

    This agent is typically invoked after review cycles to incorporate
    human feedback and improve future predictions.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        feedback_interface: Optional[FeedbackInterface] = None,
        confidence_calibrator: Optional[ConfidenceCalibrator] = None,
        pattern_analyzer: Optional[PatternAnalyzer] = None,
        ai_engine: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize Feedback Agent

        Args:
            config: Agent configuration
            feedback_interface: FeedbackInterface for accessing feedback data
            confidence_calibrator: ConfidenceCalibrator for calibration updates
            pattern_analyzer: PatternAnalyzer for pattern detection
            ai_engine: Optional AIEngine for advanced pattern analysis
            **kwargs: Additional configuration options
        """
        # Initialize base agent
        if config is None:
            config = AgentConfig(name="feedback")
        super().__init__(config=config, **kwargs)

        # Initialize feedback components
        self.feedback_interface = feedback_interface or FeedbackInterface()
        self.confidence_calibrator = confidence_calibrator or ConfidenceCalibrator()
        self.pattern_analyzer = pattern_analyzer or PatternAnalyzer(self.feedback_interface)
        self.ai_engine = ai_engine

        # Initialize AI-powered pattern detector if AI engine available
        self.ai_pattern_detector = None
        if self.ai_engine:
            self.ai_pattern_detector = AIPatternDetector(
                ai_engine=self.ai_engine,
                feedback_interface=self.feedback_interface
            )
            self.rule_suggestion_engine = RuleSuggestionEngine(
                pattern_detector=self.ai_pattern_detector,
                ai_engine=self.ai_engine
            )

        # Initialize learning engine
        self.learning_engine = LearningEngine(
            feedback_interface=self.feedback_interface,
            pattern_analyzer=self.pattern_analyzer,
            confidence_calibrator=self.confidence_calibrator
        )

        # Feedback processing settings
        self.min_pattern_occurrences = self.get_config_value('min_pattern_occurrences', 3)
        self.min_impact_threshold = self.get_config_value('min_impact_threshold', 0.05)
        self.auto_calibrate = self.get_config_value('auto_calibrate', True)
        self.pattern_detection_enabled = self.get_config_value('pattern_detection_enabled', True)

        self.logger.info(f"FeedbackAgent initialized with auto_calibrate={self.auto_calibrate}")

    def process(self, state: ComplianceState) -> ComplianceState:
        """
        Process feedback and update learning models

        This method:
        1. Processes any pending feedback records
        2. Updates confidence calibration in real-time
        3. Detects patterns in false positives/negatives
        4. Generates rule modification suggestions
        5. Calculates accuracy metrics
        6. Updates state with learning insights

        Args:
            state: Current compliance state

        Returns:
            Updated compliance state with feedback processing results
        """
        self.logger.info("Processing feedback and updating learning models")

        # Initialize feedback history in state if not present
        if "feedback_history" not in state:
            state["feedback_history"] = []

        # Get feedback records to process
        feedback_records = self._get_feedback_to_process(state)

        if not feedback_records:
            self.logger.info("No new feedback to process")
            state["feedback_processing"] = {
                "processed": 0,
                "patterns_discovered": 0,
                "rules_suggested": 0,
                "timestamp": datetime.now().isoformat()
            }
            return state

        self.logger.info(f"Processing {len(feedback_records)} feedback records")

        # Process feedback batch
        batch_results = self._process_feedback_batch(feedback_records)

        # Update confidence calibration
        if self.auto_calibrate:
            calibration_results = self._update_calibration(feedback_records)
        else:
            calibration_results = {"calibration_updated": False}

        # Detect patterns
        pattern_results = {}
        if self.pattern_detection_enabled:
            pattern_results = self._detect_patterns(state)

        # Generate rule suggestions
        rule_suggestions = self._generate_rule_suggestions()

        # Calculate accuracy metrics
        accuracy_metrics = self._calculate_accuracy_metrics()

        # Update state with results
        state = self._update_state_with_results(
            state=state,
            batch_results=batch_results,
            calibration_results=calibration_results,
            pattern_results=pattern_results,
            rule_suggestions=rule_suggestions,
            accuracy_metrics=accuracy_metrics
        )

        # Log summary
        self._log_processing_summary(state)

        return state

    def _get_feedback_to_process(self, state: ComplianceState) -> List[Dict[str, Any]]:
        """
        Get feedback records that need processing

        Args:
            state: Current compliance state

        Returns:
            List of feedback records to process
        """
        # Get recent feedback from interface
        recent_feedback = self.feedback_interface.get_feedback_history(days=30)

        # Filter to only unprocessed feedback
        processed_ids = set(state.get("feedback_history", []))
        unprocessed = [
            {
                "feedback_id": record.feedback_id,
                "check_type": record.check_type,
                "predicted_confidence": record.predicted_confidence,
                "predicted_violation": record.predicted_violation,
                "actual_violation": record.actual_violation,
                "correction_type": record.correction_type.value,
                "timestamp": record.timestamp
            }
            for record in recent_feedback
            if record.feedback_id not in processed_ids
        ]

        return unprocessed

    def _process_feedback_batch(self, feedback_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of feedback records

        Args:
            feedback_records: List of feedback records

        Returns:
            Batch processing results
        """
        self.logger.info(f"Processing batch of {len(feedback_records)} feedback records")

        try:
            # Use feedback tool to process batch
            results = process_feedback_batch.invoke({
                "feedback_records": feedback_records,
                "confidence_calibrator": self.confidence_calibrator,
                "pattern_analyzer": self.pattern_analyzer
            })

            self.logger.info(f"Batch processing complete: {results.get('successful', 0)} successful")
            return results

        except Exception as e:
            self.logger.error(f"Error processing feedback batch: {e}")
            return {
                "total_processed": len(feedback_records),
                "successful": 0,
                "failed": len(feedback_records),
                "patterns_discovered": 0,
                "error": str(e)
            }

    def _update_calibration(self, feedback_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update confidence calibration based on feedback

        Args:
            feedback_records: List of feedback records

        Returns:
            Calibration update results
        """
        self.logger.info("Updating confidence calibration")

        calibration_updates = []

        for record in feedback_records:
            try:
                result = update_confidence_calibration.invoke({
                    "check_type": record["check_type"],
                    "predicted_confidence": record["predicted_confidence"],
                    "predicted_violation": record["predicted_violation"],
                    "actual_violation": record["actual_violation"],
                    "confidence_calibrator": self.confidence_calibrator
                })
                calibration_updates.append(result)

            except Exception as e:
                self.logger.error(f"Error updating calibration for {record['check_type']}: {e}")

        # Get overall calibration metrics
        try:
            overall_metrics = get_calibration_metrics.invoke({
                "confidence_calibrator": self.confidence_calibrator,
                "check_type": None,
                "days": 30
            })
        except Exception as e:
            self.logger.error(f"Error getting calibration metrics: {e}")
            overall_metrics = {}

        return {
            "calibration_updated": True,
            "records_processed": len(calibration_updates),
            "overall_metrics": overall_metrics
        }

    def _detect_patterns(self, state: ComplianceState) -> Dict[str, Any]:
        """
        Detect patterns in false positives and false negatives

        Args:
            state: Current compliance state

        Returns:
            Pattern detection results
        """
        self.logger.info("Detecting patterns in feedback")

        try:
            # Use pattern detection tool
            patterns = detect_patterns.invoke({
                "pattern_analyzer": self.pattern_analyzer,
                "check_type": None,
                "min_occurrences": self.min_pattern_occurrences
            })

            # If AI pattern detector available, use it for deeper analysis
            if self.ai_pattern_detector:
                self.logger.info("Running AI-powered pattern detection")

                # Discover missed violation patterns
                ai_fn_patterns = self.ai_pattern_detector.discover_missed_violation_patterns(
                    check_type=None,
                    min_occurrences=self.min_pattern_occurrences
                )

                # Discover false positive patterns
                ai_fp_patterns = self.ai_pattern_detector.discover_false_positive_patterns(
                    check_type=None,
                    min_occurrences=self.min_pattern_occurrences
                )

                patterns["ai_false_negative_patterns"] = [
                    {
                        "pattern_id": p.pattern_id,
                        "check_type": p.check_type,
                        "description": p.description,
                        "occurrence_count": p.occurrence_count,
                        "confidence": p.confidence,
                        "impact_score": p.impact_score
                    }
                    for p in ai_fn_patterns
                ]

                patterns["ai_false_positive_patterns"] = [
                    {
                        "pattern_id": p.pattern_id,
                        "check_type": p.check_type,
                        "description": p.description,
                        "occurrence_count": p.occurrence_count,
                        "confidence": p.confidence,
                        "impact_score": p.impact_score
                    }
                    for p in ai_fp_patterns
                ]

            self.logger.info(f"Pattern detection complete: {patterns.get('total_patterns', 0)} patterns found")
            return patterns

        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return {
                "false_positive_patterns": [],
                "false_negative_patterns": [],
                "total_patterns": 0,
                "high_impact_patterns": [],
                "error": str(e)
            }

    def _generate_rule_suggestions(self) -> List[Dict[str, Any]]:
        """
        Generate rule modification suggestions

        Returns:
            List of rule suggestions
        """
        self.logger.info("Generating rule modification suggestions")

        try:
            # Use rule suggestion tool
            suggestions = suggest_rule_modifications.invoke({
                "pattern_analyzer": self.pattern_analyzer,
                "min_impact": self.min_impact_threshold
            })

            # If rule suggestion engine available, generate detailed recommendations
            if hasattr(self, 'rule_suggestion_engine') and self.rule_suggestion_engine:
                self.logger.info("Generating AI-powered rule recommendations")

                recommendations = self.rule_suggestion_engine.generate_recommendations(
                    min_impact=self.min_impact_threshold
                )

                # Convert to dict format
                detailed_suggestions = [
                    {
                        "recommendation_id": rec.recommendation_id,
                        "check_type": rec.check_type,
                        "recommendation_type": rec.recommendation_type,
                        "title": rec.title,
                        "description": rec.description,
                        "rationale": rec.rationale,
                        "expected_impact": rec.expected_impact,
                        "priority": rec.priority,
                        "implementation_details": rec.implementation_details,
                        "code_snippet": rec.code_snippet
                    }
                    for rec in recommendations
                ]

                # Merge with basic suggestions
                suggestions.extend(detailed_suggestions)

            self.logger.info(f"Generated {len(suggestions)} rule suggestions")
            return suggestions

        except Exception as e:
            self.logger.error(f"Error generating rule suggestions: {e}")
            return []

    def _calculate_accuracy_metrics(self) -> Dict[str, Any]:
        """
        Calculate accuracy and learning metrics

        Returns:
            Accuracy metrics
        """
        self.logger.info("Calculating accuracy metrics")

        try:
            # Get learning metrics
            metrics = get_learning_metrics.invoke({
                "feedback_interface": self.feedback_interface,
                "pattern_analyzer": self.pattern_analyzer
            })

            # Get confidence pattern analysis
            confidence_analysis = analyze_confidence_patterns.invoke({
                "pattern_analyzer": self.pattern_analyzer,
                "check_type": None
            })

            # Combine metrics
            combined_metrics = {
                **metrics,
                "confidence_analysis": confidence_analysis,
                "timestamp": datetime.now().isoformat()
            }

            self.logger.info(f"Accuracy metrics calculated: {metrics.get('total_feedback', 0)} feedback records")
            return combined_metrics

        except Exception as e:
            self.logger.error(f"Error calculating accuracy metrics: {e}")
            return {
                "total_feedback": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "patterns_discovered": 0,
                "rules_suggested": 0,
                "accuracy_improvement": 0.0,
                "error": str(e)
            }

    def _update_state_with_results(
        self,
        state: ComplianceState,
        batch_results: Dict[str, Any],
        calibration_results: Dict[str, Any],
        pattern_results: Dict[str, Any],
        rule_suggestions: List[Dict[str, Any]],
        accuracy_metrics: Dict[str, Any]
    ) -> ComplianceState:
        """
        Update state with feedback processing results

        Args:
            state: Current state
            batch_results: Batch processing results
            calibration_results: Calibration update results
            pattern_results: Pattern detection results
            rule_suggestions: Rule suggestions
            accuracy_metrics: Accuracy metrics

        Returns:
            Updated state
        """
        # Update feedback processing summary
        state["feedback_processing"] = {
            "processed": batch_results.get("successful", 0),
            "failed": batch_results.get("failed", 0),
            "patterns_discovered": pattern_results.get("total_patterns", 0),
            "rules_suggested": len(rule_suggestions),
            "timestamp": datetime.now().isoformat()
        }

        # Store calibration results
        if "calibration_metrics" not in state:
            state["calibration_metrics"] = {}
        state["calibration_metrics"] = calibration_results.get("overall_metrics", {})

        # Store discovered patterns
        if "discovered_patterns" not in state:
            state["discovered_patterns"] = []

        # Add new patterns
        new_patterns = (
            pattern_results.get("false_positive_patterns", []) +
            pattern_results.get("false_negative_patterns", [])
        )
        state["discovered_patterns"].extend(new_patterns)

        # Store rule suggestions
        if "rule_suggestions" not in state:
            state["rule_suggestions"] = []
        state["rule_suggestions"].extend(rule_suggestions)

        # Store accuracy metrics
        state["learning_metrics"] = accuracy_metrics

        # Update timestamp
        state = update_state_timestamp(state)

        return state

    def _log_processing_summary(self, state: ComplianceState):
        """
        Log summary of feedback processing

        Args:
            state: Current state with processing results
        """
        processing = state.get("feedback_processing", {})
        metrics = state.get("learning_metrics", {})

        self.logger.info("="*70)
        self.logger.info("Feedback Processing Summary")
        self.logger.info("="*70)
        self.logger.info(f"Feedback Processed: {processing.get('processed', 0)}")
        self.logger.info(f"Patterns Discovered: {processing.get('patterns_discovered', 0)}")
        self.logger.info(f"Rules Suggested: {processing.get('rules_suggested', 0)}")
        self.logger.info(f"Total Feedback: {metrics.get('total_feedback', 0)}")
        self.logger.info(f"False Positives: {metrics.get('false_positives', 0)}")
        self.logger.info(f"False Negatives: {metrics.get('false_negatives', 0)}")
        self.logger.info(f"Accuracy Improvement: {metrics.get('accuracy_improvement', 0.0):.1%}")
        self.logger.info("="*70)

    def get_learning_report(self) -> Dict[str, Any]:
        """
        Get comprehensive learning report

        Returns:
            Learning report with all metrics and insights
        """
        metrics = self.learning_engine.get_learning_metrics()

        report = {
            "total_feedback": metrics.total_feedback,
            "false_positives": metrics.false_positives,
            "false_negatives": metrics.false_negatives,
            "confidence_adjustments": metrics.confidence_adjustments,
            "patterns_discovered": metrics.patterns_discovered,
            "rules_suggested": metrics.rules_suggested,
            "accuracy_improvement": metrics.accuracy_improvement,
            "avg_correction_time_ms": metrics.avg_correction_time_ms,
            "timestamp": datetime.now().isoformat()
        }

        return report

    def export_learning_data(self, base_path: str = "."):
        """
        Export all learning data for analysis

        Args:
            base_path: Base directory for exports
        """
        self.logger.info(f"Exporting learning data to {base_path}")

        # Export feedback data
        self.feedback_interface.export_feedback(f"{base_path}/feedback_export.json")

        # Export calibration metrics
        self.confidence_calibrator.export_metrics(f"{base_path}/calibration_metrics.json")

        # Export patterns if AI detector available
        if self.ai_pattern_detector:
            self.ai_pattern_detector.export_patterns(f"{base_path}/discovered_patterns.json")

        # Export rule recommendations if available
        if hasattr(self, 'rule_suggestion_engine') and self.rule_suggestion_engine:
            self.rule_suggestion_engine.export_recommendations(f"{base_path}/rule_recommendations.json")

        self.logger.info("Learning data export complete")


# Export agent class
__all__ = ["FeedbackAgent"]


if __name__ == "__main__":
    # Example usage and testing
    logger.info("="*70)
    logger.info("Feedback Agent - Multi-Agent Compliance System")
    logger.info("="*70)

    # Create test configuration
    config = AgentConfig(
        name="feedback",
        enabled=True,
        timeout_seconds=60.0,
        custom_settings={
            "min_pattern_occurrences": 3,
            "min_impact_threshold": 0.05,
            "auto_calibrate": True,
            "pattern_detection_enabled": True
        }
    )

    # Initialize agent
    agent = FeedbackAgent(config=config)

    logger.info(f"\n✓ FeedbackAgent initialized: {agent.name}")
    logger.info(f"  - Auto-calibrate: {agent.auto_calibrate}")
    logger.info(f"  - Pattern detection: {agent.pattern_detection_enabled}")
    logger.info(f"  - Min pattern occurrences: {agent.min_pattern_occurrences}")
    logger.info(f"  - Min impact threshold: {agent.min_impact_threshold}")

    # Create test state
    from data_models_multiagent import initialize_compliance_state

    test_document = {
        "document_metadata": {
            "fund_isin": "LU1234567890",
            "client_type": "retail"
        }
    }

    test_state = initialize_compliance_state(
        document=test_document,
        document_id="test_doc_001",
        config={}
    )

    logger.info("\n📊 Processing test state...")
    result_state = agent(test_state)

    logger.info(f"\n✓ Feedback processing complete")
    logger.info(f"  - Feedback processed: {result_state.get('feedback_processing', {}).get('processed', 0)}")
    logger.info(f"  - Patterns discovered: {result_state.get('feedback_processing', {}).get('patterns_discovered', 0)}")
    logger.info(f"  - Rules suggested: {result_state.get('feedback_processing', {}).get('rules_suggested', 0)}")

    # Get learning report
    report = agent.get_learning_report()
    logger.info(f"\n📈 Learning Report:")
    logger.info(f"  - Total feedback: {report['total_feedback']}")
    logger.info(f"  - Accuracy improvement: {report['accuracy_improvement']:.1%}")

    logger.info("\n" + "="*70)
    logger.info("✓ FeedbackAgent test complete")
    logger.info("="*70)
