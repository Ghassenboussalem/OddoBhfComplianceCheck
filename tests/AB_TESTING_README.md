# A/B Testing Framework

## Overview

The A/B testing framework (`tests/ab_testing.py`) provides comprehensive comparison between the old compliance system (`check.py`) and the new multi-agent system (`check_multiagent.py`).

## Features

- **Parallel Execution**: Runs both systems simultaneously for fair comparison
- **Violation Comparison**: Detailed comparison of detected violations
- **Performance Metrics**: Execution time comparison and speedup calculation
- **Discrepancy Detection**: Identifies and categorizes differences between systems
- **Comprehensive Reports**: Generates detailed text and JSON reports

## Usage

### Basic Usage

Test with the default file (exemple.json):
```bash
python tests/ab_testing.py
```

### Test Specific File

```bash
python tests/ab_testing.py --file exemple.json
```

### Test Multiple Files

```bash
python tests/ab_testing.py --file file1.json file2.json file3.json
```

## Output Files

The framework generates three types of output:

1. **Individual Test Reports**: `tests/ab_test_report_<filename>.txt`
   - Detailed comparison for each test file
   - Violation matching analysis
   - Performance comparison
   - Discrepancy details

2. **Summary Report**: `tests/ab_test_summary.txt`
   - Aggregated statistics across all tests
   - Overall assessment
   - Per-file results summary

3. **JSON Results**: `tests/ab_test_results.json`
   - Structured data for programmatic analysis
   - Violation counts, execution times, match statistics

## Report Sections

### Execution Status
- Success/failure status for both systems
- Error messages if applicable

### Violation Comparison
- Violation counts from both systems
- Matched violations (exact and partial)
- Unmatched violations in each system

### Performance Comparison
- Execution times for both systems
- Speedup/slowdown percentage
- Multi-agent specific metrics (agent timings, workflow status)

### Discrepancies
- Missing violations in new system
- Extra violations in new system
- Field differences in matched violations

### Overall Assessment
- Pass/fail determination
- Performance summary
- Recommendations

## Discrepancy Categories

1. **missing_in_new**: Violations detected by old system but not by new system
2. **extra_in_new**: Violations detected by new system but not by old system
3. **field_difference**: Matched violations with different field values

## Example Output

```
======================================================================
A/B TESTING COMPARISON REPORT
======================================================================

Test file: exemple.json
Date: 2025-11-23 10:49:40

----------------------------------------------------------------------
VIOLATION COMPARISON
----------------------------------------------------------------------
Old system violations: 6
New system violations: 6
Count match: ✓

Matched violations: 6
  Exact matches: 6
  Partial matches: 0

----------------------------------------------------------------------
PERFORMANCE COMPARISON
----------------------------------------------------------------------
Old system execution time: 196.44s
New system execution time: 61.91s
Performance improvement: 68.5% faster

----------------------------------------------------------------------
OVERALL ASSESSMENT
----------------------------------------------------------------------

✅ SYSTEMS MATCH PERFECTLY

All checks passed:
  ✓ Violation counts match
  ✓ All violations match exactly
  ✓ No discrepancies detected

✓ New system is 68.5% faster
======================================================================
```

## Integration with CI/CD

The framework returns appropriate exit codes:
- `0`: All tests passed successfully
- `1`: Some tests failed or had issues

This allows integration into automated testing pipelines:

```bash
python tests/ab_testing.py --file exemple.json
if [ $? -eq 0 ]; then
    echo "A/B tests passed"
else
    echo "A/B tests failed"
    exit 1
fi
```

## Requirements

- Python 3.7+
- Both `check.py` and `check_multiagent.py` must be present
- Test JSON files must exist in the workspace

## Implementation Details

### Parallel Execution
Uses `ThreadPoolExecutor` to run both systems simultaneously, ensuring fair performance comparison.

### Violation Matching
Implements intelligent matching algorithm that:
- Matches on type, rule, and slide
- Calculates similarity scores for partial matches
- Handles field differences gracefully

### Performance Tracking
Measures:
- Total execution time
- Individual agent execution times (for multi-agent system)
- Speedup/slowdown percentages

## Troubleshooting

### Both systems show 0 violations
This is expected if the document is compliant. The framework still validates that both systems agree.

### Timeout errors
Default timeout is 300 seconds (5 minutes). Adjust in code if needed for large documents.

### File not found errors
Ensure test files exist in the workspace root directory.

## Related Files

- `tests/test_validation.py`: Single-system validation test
- `tests/test_workflow.py`: Workflow integration tests
- `tests/test_agent_interactions.py`: Agent interaction tests
