# Batch Review Operations Guide

## Overview

Batch review operations allow you to efficiently process multiple similar review items at once, significantly reducing the time needed to review large numbers of flagged violations.

## Features

- **Multiple Selection Criteria**: Select items by check type, document, severity, confidence range, or similarity
- **Bulk Actions**: Approve or reject multiple items with a single command
- **Individual Feedback**: Each item in a batch receives individual feedback records for audit purposes
- **Safety Confirmations**: All batch operations require explicit confirmation before execution
- **Flexible Limits**: Control the number of items processed with optional limits

## Selection Criteria

### By Check Type

Select all items of a specific check type (STRUCTURE, PERFORMANCE, ESG, VALUES):

```bash
python review.py batch --check-type=STRUCTURE --action=approve
```

**Use Case**: When you've verified that all violations of a certain type are valid.

### By Document

Select all items from a specific document:

```bash
python review.py batch --document=test_doc_001 --action=reject --notes "Test data"
```

**Use Case**: When an entire document is test data or needs to be excluded.

### By Severity

Select all items with a specific severity level:

```bash
python review.py batch --severity=LOW --action=approve
```

**Use Case**: When low-severity items can be batch-approved after spot-checking.

### By Confidence Range

Select items within a specific confidence range:

```bash
python review.py batch --min-confidence=60 --max-confidence=70 --action=reject --notes "Too uncertain"
```

**Use Case**: When items in a certain confidence range consistently turn out to be false positives.

### By Similarity

Select items similar to the current review (requires an active review in interactive mode):

```bash
# In interactive mode
review> next
review> batch --similar --action=approve --notes "Same pattern"
```

**Use Case**: When you encounter a pattern that repeats across multiple items.

## Actions

### Approve

Marks all selected items as approved (violation confirmed):

```bash
python review.py batch --check-type=STRUCTURE --action=approve
```

Notes are optional for approval.

### Reject

Marks all selected items as rejected (false positive):

```bash
python review.py batch --document=test_doc --action=reject --notes "Explanation required"
```

**Important**: Notes are required for rejection to document why items were rejected.

## Optional Parameters

### Limit

Limit the number of items processed:

```bash
python review.py batch --check-type=STRUCTURE --action=approve --limit=10
```

**Use Case**: Process items in smaller batches for better control.

### Threshold

Set similarity threshold for `--similar` selection (0-1, default: 0.85):

```bash
python review.py batch --similar --threshold=0.75 --action=approve
```

Lower threshold = more items included (less strict similarity).

## Interactive Mode

Batch operations can also be used in interactive review sessions:

```bash
python review.py

review> status
review> batch --check-type=STRUCTURE --action=approve
review> status
```

## Examples

### Example 1: Clean Up Test Data

Remove all items from test documents:

```bash
python review.py batch --document=test_doc_001 --action=reject --notes "Test data - not real violations"
```

### Example 2: Approve Low-Risk Items

Approve all low-severity items after verification:

```bash
python review.py batch --severity=LOW --action=approve --limit=20
```

### Example 3: Reject Uncertain Predictions

Reject items with low confidence scores:

```bash
python review.py batch --min-confidence=50 --max-confidence=60 --action=reject --notes "Confidence too low"
```

### Example 4: Process by Check Type

Approve all STRUCTURE violations in batches:

```bash
# First batch
python review.py batch --check-type=STRUCTURE --action=approve --limit=5

# Review results, then continue
python review.py batch --check-type=STRUCTURE --action=approve --limit=5
```

### Example 5: Similar Items

In interactive mode, find and process similar items:

```bash
python review.py

review> next
# Review the item
review> batch --similar --threshold=0.8 --action=approve --notes "Same issue pattern"
```

## Safety Features

### Confirmation Required

All batch operations display a summary and require explicit confirmation:

```
Selection Criteria:
  check-type: STRUCTURE

Items Found: 5

Sample Items:
  1. review_001 - STRUCTURE (confidence: 65%, severity: HIGH)
  2. review_002 - STRUCTURE (confidence: 68%, severity: MEDIUM)
  3. review_003 - STRUCTURE (confidence: 62%, severity: HIGH)

Action: APPROVE all 5 items

Confirm batch operation? (yes/no):
```

Type `yes` to proceed or `no` to cancel.

### Individual Feedback Records

Each item in a batch receives its own feedback record with:
- Individual review ID
- Reviewer ID
- Decision (APPROVE/REJECT)
- Timestamp
- Notes
- Tags (including `batch_approve` or `batch_reject`)

This ensures complete audit trail for compliance.

## Best Practices

1. **Start Small**: Use `--limit` to process items in smaller batches initially
2. **Verify First**: Review a few items manually before batch processing similar ones
3. **Use Descriptive Notes**: Especially for rejections, provide clear reasoning
4. **Check Status**: Use `status` command before and after batch operations
5. **Document Patterns**: When using `--similar`, document the pattern in notes
6. **Test Data First**: Process test/demo data separately from production data

## Performance

Batch operations are optimized for efficiency:
- Process hundreds of items in seconds
- Thread-safe queue operations
- Atomic updates with rollback on failure
- Minimal memory footprint

## Troubleshooting

### No Items Found

If batch selection returns no items:
- Verify the selection criteria match existing items
- Check queue status with `status` command
- Ensure items are in PENDING state (not already reviewed)

### Batch Operation Failed

If some items fail to process:
- Check the error messages for specific items
- Verify queue file permissions
- Ensure no concurrent modifications to the queue

### Confirmation Not Working

If confirmation prompt doesn't appear:
- Ensure you're running in an interactive terminal
- Check that stdin is not redirected
- Try running without piping or redirection

## Integration with Feedback System

All batch operations integrate with the feedback system:
- Confidence calibration is updated based on batch decisions
- Pattern detection analyzes batch feedback
- Accuracy metrics include batch-reviewed items
- Audit logs maintain complete history

## Command Reference

```bash
# General format
python review.py batch --<selection> --action=<approve|reject> [options]

# Selection options (choose one)
--check-type=TYPE          # Select by check type
--document=DOC_ID          # Select by document
--severity=LEVEL           # Select by severity
--min-confidence=N         # Select by confidence range (with --max-confidence)
--max-confidence=N         # Select by confidence range (with --min-confidence)
--similar                  # Select similar to current review

# Required
--action=<approve|reject>  # Action to perform

# Optional
--notes "text"             # Reviewer notes (required for reject)
--limit=N                  # Limit number of items
--threshold=0.85           # Similarity threshold (for --similar)
--queue-file=path          # Custom queue file path
--reviewer-id=ID           # Custom reviewer ID
```

## See Also

- [HITL Integration Guide](HITL_INTEGRATION.md)
- [Review CLI Documentation](review.py)
- [Review Manager API](review_manager.py)
