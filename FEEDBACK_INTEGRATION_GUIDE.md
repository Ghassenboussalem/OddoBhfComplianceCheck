# Feedback Integration Guide

## Overview

The Feedback Integration System connects human review decisions to the learning components, enabling real-time model updates and continuous improvement of the AI-Enhanced Compliance Checker.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ReviewManager                             │
│  - Manages review queue                                      │
│  - Assigns reviews to reviewers                              │
│  - Collects reviewer decisions                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              FeedbackIntegration                             │
│  - Real-time learning coordination                           │
│  - Performance tracking (< 1s requirement)                   │
│  - Metrics calculation                                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│Feedback  │ │Confidence│ │Pattern       │
│Interface │ │Calibrator│ │Detector      │
│          │ │          │ │              │
│- History │ │- Adjust  │ │- Analyze FP  │
│- Audit   │ │  scores  │ │- Analyze FN  │
└──────────┘ └──────────┘ └──────────────┘
```

## Key Features

### 1. Real-Time Learning (< 1 second)

When a reviewer provides feedback, the system:
- Updates confidence calibration models immediately
- Applies confidence adjustments based on corrections
- Records feedback for audit trail
- All within 1 second (typically < 5ms)

### 2. Confidence Calibration

The system tracks:
- Prediction accuracy by check type
- Over-confident vs under-confident patterns
- Confidence score adjustments

Example:
```python
# Before correction: 75% confidence
# After reviewer correction: 30% actual confidence
# System learns: -45% adjustment for this pattern
# Next similar case: 75% → 72% (gradual adjustment)
```

### 3. Pattern Detection

Analyzes feedback to discover:
- **False Positive Patterns**: Cases where AI incorrectly flagged violations
- **False Negative Patterns**: Cases where AI missed violations
- **Common Features**: Shared characteristics in errors

### 4. Accuracy Metrics

Calculates standard ML metrics:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: (TP + TN) / Total

## Usage

### Basic Setup

```python
from feedback_integration import FeedbackIntegration
from feedback_loop import FeedbackInterface
from confidence_calibrator import ConfidenceCalibrator
from pattern_detector import AIPatternDetector

# Initialize components
feedback_interface = FeedbackInterface(db_path="feedback_data.json")
confidence_calibrator = ConfidenceCalibrator(db_path="calibration_data.json")
pattern_detector = AIPatternDetector(
    feedback_interface=feedback_interface
)

# Create integration
integration = FeedbackIntegration(
    feedback_interface=feedback_interface,
    confidence_calibrator=confidence_calibrator,
    pattern_detector=pattern_detector
)
```

### With Review Manager

```python
from review_manager import ReviewManager

# Initialize review manager with feedback integration
review_manager = ReviewManager(
    queue_file="review_queue.json",
    feedback_integration=integration
)

# When a review is completed, feedback is automatically submitted
decision = ReviewDecision(
    review_id="review_123",
    reviewer_id="reviewer_001",
    decision="REJECT",
    actual_violation=False,
    corrected_confidence=30,
    reviewer_notes="This was an example",
    tags=["false_positive"],
    reviewed_at=datetime.now().isoformat(),
    review_duration_seconds=45
)

# This triggers real-time learning
review_manager.mark_reviewed("review_123", decision)
```

### Manual Feedback Submission

```python
# Submit prediction for review
feedback_id = feedback_interface.submit_for_review(
    check_type="PROMOTIONAL_MENTION",
    document_id="doc_123",
    slide="slide_5",
    predicted_violation=True,
    predicted_confidence=75,
    predicted_reasoning="AI detected promotional mention",
    predicted_evidence="Found phrase 'document promotionnel'",
    processing_time_ms=1200.0
)

# Provide correction (triggers real-time learning)
feedback_interface.provide_correction(
    feedback_id=feedback_id,
    actual_violation=False,
    reviewer_notes="This was an example, not actual promotional mention",
    corrected_confidence=30,
    reviewer_id="reviewer_001"
)
```

## Performance Metrics

### Processing Time

The system tracks processing time for each feedback:

```python
stats = integration.get_processing_stats()
print(f"Avg Processing Time: {stats['avg_processing_time_ms']:.1f}ms")
print(f"Under 1s Rate: {stats['under_1s_rate']:.1%}")
```

Expected performance:
- Average: < 5ms
- Maximum: < 50ms
- Under 1s rate: 100%

### Accuracy Metrics

```python
metrics = integration.get_accuracy_metrics(check_type="PROMOTIONAL_MENTION")
print(f"Accuracy: {metrics['accuracy']:.1%}")
print(f"Precision: {metrics['precision']:.1%}")
print(f"Recall: {metrics['recall']:.1%}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

## Batch Processing

For historical feedback analysis:

```python
# Process last 30 days of feedback
integration.process_batch_feedback(
    check_type="PROMOTIONAL_MENTION",
    days=30
)
```

This will:
1. Update calibration with all historical records
2. Discover patterns in false positives
3. Discover patterns in false negatives
4. Generate rule suggestions

## Reports

### Integration Report

```python
integration.print_integration_report()
```

Output:
```
======================================================================
Feedback Integration Report
======================================================================

📊 Processing Performance:
  Total Processed: 150
  Avg Processing Time: 2.3ms
  Max Processing Time: 8.5ms
  Under 1s Rate: 100.0%

🎯 Accuracy Metrics:
  Total Reviews: 150
  Accuracy: 87.3%
  Precision: 89.2%
  Recall: 84.5%
  F1 Score: 0.868
  Avg Confidence (Correct): 82.4%
  Avg Confidence (Incorrect): 71.2%

🔧 Calibration Status:
  PROMOTIONAL_MENTION:
    Reliability: 0.873
    confidence_adjustment: -5
    review_threshold: +3
```

### Export Metrics

```python
integration.export_integration_metrics("integration_metrics.json")
```

## Configuration

### Confidence Adjustment Weight

The system uses a weighted average for confidence adjustments:

```python
# In feedback_integration.py
new_adj = int((existing_adj * 0.9) + (adjustment * 0.1))
```

- 90% weight to existing adjustment
- 10% weight to new feedback
- Prevents over-reaction to single corrections

### Pattern Detection Thresholds

```python
# Minimum occurrences to consider a pattern
min_occurrences = 3

# Discover patterns
patterns = pattern_detector.discover_false_positive_patterns(
    check_type="PROMOTIONAL_MENTION",
    min_occurrences=min_occurrences
)
```

## Audit Trail

All feedback is automatically recorded for audit purposes:

```python
# Get feedback history
history = feedback_interface.get_feedback_history(
    check_type="PROMOTIONAL_MENTION",
    days=30
)

# Export for audit
feedback_interface.export_feedback(
    filepath="audit_feedback.json",
    check_type="PROMOTIONAL_MENTION"
)
```

Each record includes:
- Original prediction (violation, confidence, reasoning, evidence)
- Human correction (actual violation, corrected confidence, notes)
- Metadata (timestamp, reviewer ID, correction type)
- Processing time

## Best Practices

### 1. Regular Calibration

Run batch processing weekly to update calibration:

```python
# Weekly calibration update
integration.process_batch_feedback(days=7)
```

### 2. Monitor Processing Time

Check that feedback processing stays under 1 second:

```python
stats = integration.get_processing_stats()
if stats['under_1s_rate'] < 0.95:
    logger.warning("Processing time exceeding 1s threshold")
```

### 3. Review Accuracy Trends

Track accuracy over time to measure improvement:

```python
# Compare early vs recent accuracy
early_metrics = integration.get_accuracy_metrics()  # All time
recent_metrics = integration.get_accuracy_metrics()  # Last 30 days

improvement = recent_metrics['accuracy'] - early_metrics['accuracy']
print(f"Accuracy improvement: {improvement:+.1%}")
```

### 4. Pattern Analysis

Regularly analyze patterns to identify systematic issues:

```python
# Discover patterns
fp_patterns = pattern_detector.discover_false_positive_patterns()
fn_patterns = pattern_detector.discover_missed_violation_patterns()

# Review high-impact patterns
high_impact = [p for p in fp_patterns if p.impact_score > 0.1]
for pattern in high_impact:
    print(f"Pattern: {pattern.description}")
    print(f"Impact: {pattern.impact_score:.1%}")
    print(f"Suggested Rule: {pattern.suggested_rule}")
```

## Troubleshooting

### Slow Processing

If processing time exceeds 1 second:

1. Check database file size (should be < 10MB)
2. Reduce pattern detection frequency
3. Use batch processing for historical data

### Inaccurate Adjustments

If confidence adjustments seem wrong:

1. Check feedback quality (are corrections accurate?)
2. Increase minimum occurrences for patterns
3. Review calibration metrics by check type

### Missing Patterns

If patterns aren't being detected:

1. Ensure minimum occurrences threshold is met
2. Check that pattern detector is initialized
3. Verify feedback records have correction types set

## API Reference

### FeedbackIntegration

#### `__init__(feedback_interface, confidence_calibrator, pattern_detector=None, audit_logger=None)`
Initialize feedback integration system.

#### `process_review_decision(feedback_record) -> FeedbackProcessingResult`
Process a single review decision and update learning models.

#### `get_accuracy_metrics(check_type=None) -> Dict`
Calculate accuracy metrics (precision, recall, F1, accuracy).

#### `get_processing_stats() -> Dict`
Get statistics about feedback processing performance.

#### `process_batch_feedback(check_type=None, days=30)`
Process historical feedback for pattern analysis.

#### `print_integration_report()`
Print comprehensive integration report to console.

#### `export_integration_metrics(filepath)`
Export integration metrics to JSON file.

## Examples

See:
- `feedback_integration.py` - Main implementation with test example
- `test_review_feedback_integration.py` - Integration test with ReviewManager
- `feedback_loop.py` - Feedback interface examples
- `confidence_calibrator.py` - Calibration examples

## Requirements Met

This implementation satisfies all requirements from task 5:

✅ **3.1**: Real-time feedback triggers immediate calibration (< 1s)  
✅ **3.2**: Similar violations get learned adjustments applied  
✅ **3.3**: Pattern detector analyzes false positives/negatives  
✅ **3.4**: Feedback history maintained for audit  
✅ **3.5**: Real-time confidence adjustment based on corrections  

## Next Steps

1. Integrate with CLI review interface (review.py)
2. Add audit logger for compliance reporting
3. Implement rule suggestion engine
4. Create web dashboard for metrics visualization
