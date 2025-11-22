# Review Mode Integration - Usage Guide

## Overview

The compliance checker now integrates with the Human-in-the-Loop (HITL) review system, allowing you to identify and review low-confidence violations interactively.

## New Command-Line Options

### `--review-mode`
Enter interactive review mode after checking completes. This allows you to review flagged violations immediately.

```bash
python check.py exemple.json --review-mode
```

### `--review-threshold=N`
Set the confidence threshold for flagging violations for review. Violations with confidence below this threshold will be flagged.

Default: 70%

```bash
python check.py exemple.json --review-threshold=60
```

## Usage Examples

### Basic Check with Review Summary
```bash
python check.py exemple.json
```

If violations with low confidence are detected, you'll see a summary like:

```
======================================================================
👤 HUMAN REVIEW REQUIRED
======================================================================

3 violation(s) flagged for human review
(confidence below 70%)

By check type:
  PERFORMANCE: 1
  SECURITIES/VALUES: 2

Confidence range: 45% - 65%
Average confidence: 55.0%

======================================================================

💡 Tip: Use --review-mode flag to enter interactive review
   Or run 'python review.py' to review flagged violations
```

### Check with Custom Review Threshold
```bash
python check.py exemple.json --review-threshold=60
```

This will flag violations with confidence below 60% for review.

### Check with Interactive Review Mode
```bash
python check.py exemple.json --review-mode
```

After the check completes, if there are violations requiring review, you'll automatically enter the interactive review interface.

### Combined Options
```bash
python check.py exemple.json --hybrid-mode=on --review-threshold=65 --review-mode
```

This enables:
- AI+Rules hybrid mode
- Custom review threshold of 65%
- Interactive review mode after checking

## Configuration

The review threshold can also be configured in `hybrid_config.json`:

```json
{
  "hitl": {
    "enabled": false,
    "review_threshold": 70,
    "auto_queue_low_confidence": true,
    "queue_max_size": 10000,
    "batch_similarity_threshold": 0.85,
    "interactive_mode_default": false,
    "audit_log_path": "./audit_logs/",
    "export_formats": ["json", "csv"]
  }
}
```

### Configuration Options

- `enabled`: Enable/disable HITL system (default: false)
- `review_threshold`: Confidence threshold for flagging violations (default: 70)
- `auto_queue_low_confidence`: Automatically queue low-confidence violations (default: true)
- `queue_max_size`: Maximum number of items in review queue (default: 10000)
- `batch_similarity_threshold`: Similarity threshold for batch operations (default: 0.85)
- `interactive_mode_default`: Start in interactive mode by default (default: false)
- `audit_log_path`: Path for audit logs (default: "./audit_logs/")
- `export_formats`: Supported export formats (default: ["json", "csv"])

## Review Summary Display

The review summary shows:

1. **Total violations flagged**: Number of violations requiring human review
2. **Threshold**: The confidence threshold used
3. **By check type**: Breakdown of violations by category
4. **Confidence range**: Min and max confidence scores
5. **Average confidence**: Mean confidence of flagged violations

## Notes

- Violations with confidence = 0% are rule-based and not flagged for review
- Violations with confidence >= threshold are considered high-confidence and not flagged
- Only violations with 0 < confidence < threshold are flagged for review
- The review summary only appears when there are violations requiring review

## Backward Compatibility

All existing workflows remain fully compatible:

```bash
# Traditional usage (no changes)
python check.py exemple.json

# With hybrid mode (no changes)
python check.py exemple.json --hybrid-mode=on

# With metrics (no changes)
python check.py exemple.json --show-metrics
```

The review features are opt-in and don't affect existing functionality.

## Next Steps

After seeing the review summary, you can:

1. Use `--review-mode` flag to enter interactive review immediately
2. Run `python review.py` separately to review flagged violations
3. Review the JSON output file for detailed violation information
4. Adjust the review threshold if needed

## Related Documentation

- `HITL_INTEGRATION.md` - Complete HITL system documentation
- `API_DOCUMENTATION.md` - API reference
- `QUICK_START.md` - Getting started guide
