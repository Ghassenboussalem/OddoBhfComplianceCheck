# Scalability Tests for Multi-Agent System

## Overview

The scalability test suite (`test_scalability.py`) measures how the multi-agent compliance system performs under increasing load and identifies performance bottlenecks.

## Test Coverage

### Test 1: Scale to 10 Documents
- Processes 10 medium-complexity documents
- Measures execution time and resource usage
- Validates basic scalability

### Test 2: Scale to 50 Documents
- Processes 50 documents (mix of simple and medium complexity)
- Checks for memory leaks
- Analyzes performance degradation over time

### Test 3: Scale to 100 Documents
- Processes 100 documents (40% simple, 50% medium, 10% complex)
- Tests system stability under sustained load
- Measures memory growth per document
- Validates long-running performance

### Test 4: Concurrent Workflow Executions
- Runs 5 workflows concurrently
- Each workflow processes 3 documents
- Tests thread safety and concurrent resource usage
- Validates no race conditions or deadlocks

### Test 5: Identify Performance Bottlenecks
- Analyzes results from all previous tests
- Identifies memory scaling issues
- Detects performance degradation patterns
- Analyzes CPU utilization
- Provides optimization recommendations

## Resource Monitoring

The test suite uses `psutil` to monitor:
- **Memory Usage**: RSS (Resident Set Size) and VMS (Virtual Memory Size)
- **CPU Usage**: Percentage utilization
- **Thread Count**: Number of active threads
- **Sampling**: Collects samples every 0.5 seconds during execution

## Performance Thresholds

- **Max Memory per Document**: 50MB
- **Max Memory Growth**: 20%
- **Max CPU Usage**: 90%

## Running the Tests

### Prerequisites

Install required dependencies:
```bash
pip install psutil
```

### Run All Tests

```bash
python tests/test_scalability.py
```

**Note**: The complete test suite may take 10-20 minutes to complete, as it processes 165+ documents.

### Run Individual Tests

You can import and run specific tests:

```python
from tests.test_scalability import ScalabilityTests

tests = ScalabilityTests()
tests.test_1_scale_to_10_documents()
```

## Output

### Console Output
- Real-time progress updates
- Detailed statistics for each test
- Resource usage metrics
- Bottleneck analysis
- Optimization recommendations

### JSON Results
Results are saved to `tests/scalability_test_results.json`:
```json
{
  "summary": {
    "total_tests": 5,
    "passed": 5,
    "failed": 0,
    "duration": 1234.56,
    "timestamp": "2024-01-15T10:30:00"
  },
  "test_results": [...],
  "scalability_results": {
    "10_documents": {...},
    "50_documents": {...},
    "100_documents": {...},
    "concurrent_execution": {...},
    "bottleneck_analysis": {...}
  }
}
```

## Interpreting Results

### Memory Usage
- **RSS (Resident Set Size)**: Actual physical memory used
- **VMS (Virtual Memory Size)**: Total virtual memory allocated
- **Growth per Document**: Should be < 50MB/doc

### CPU Usage
- **Average < 50%**: System may be I/O bound
- **Average 50-80%**: Balanced utilization
- **Average > 80%**: CPU bound, may need optimization

### Performance Degradation
- **< 15%**: Acceptable
- **15-20%**: Monitor for memory leaks
- **> 20%**: Investigate memory leaks or resource accumulation

### Throughput
- Measures documents processed per second
- Should remain consistent across batch sizes
- Variance > 1.5x indicates batch size optimization opportunities

## Common Bottlenecks

### Memory Scaling Issues
- **Symptom**: Memory grows > 50MB per document
- **Impact**: System may run out of memory with large batches
- **Solution**: Implement document streaming or batch processing with memory cleanup

### Performance Degradation
- **Symptom**: Processing slows down over time (> 20% degradation)
- **Impact**: Later documents take longer to process
- **Solution**: Investigate memory leaks, implement periodic garbage collection

### CPU Underutilization
- **Symptom**: Average CPU < 50%
- **Impact**: System waiting on I/O or external services
- **Solution**: Profile I/O operations, consider increasing parallelism

### CPU Saturation
- **Symptom**: Peak CPU > 90%
- **Impact**: System becomes unresponsive under load
- **Solution**: Optimize CPU-intensive operations, implement rate limiting

## Integration with CI/CD

Add to your CI pipeline:

```yaml
- name: Run Scalability Tests
  run: |
    pip install psutil
    python tests/test_scalability.py
  timeout-minutes: 30
```

## Troubleshooting

### Test Fails to Import
```bash
# Ensure psutil is installed
pip install psutil

# Verify Python path
python -c "import sys; print(sys.path)"
```

### Memory Errors
```bash
# Reduce test size for limited memory systems
# Edit test_scalability.py and reduce document counts
```

### Timeout Issues
```bash
# Increase timeout for slower systems
# Tests are designed to complete within 20 minutes
```

## Requirements

- Python 3.8+
- psutil >= 5.9.0
- Multi-agent system dependencies (langgraph, langchain, etc.)
- Minimum 4GB RAM recommended
- Minimum 2 CPU cores recommended

## Related Tests

- `test_performance.py`: Performance benchmarks and comparisons
- `test_workflow.py`: Workflow integration tests
- `test_agent_interactions.py`: Agent communication tests

## Maintenance

### Adding New Tests

1. Create a new test method following the pattern:
```python
def test_N_your_test_name(self) -> bool:
    """Test description"""
    logger.info("\n" + "="*70)
    logger.info("TEST N: Your Test Name")
    logger.info("="*70)
    
    try:
        # Test implementation
        self.log_test_result("Your Test Name", passed, details)
        return passed
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        self.log_test_result("Your Test Name", False, str(e))
        return False
```

2. Add to `run_all_tests()` method

### Updating Thresholds

Edit class constants in `ScalabilityTests`:
```python
self.MAX_MEMORY_PER_DOC_MB = 50
self.MAX_MEMORY_GROWTH_PERCENT = 20
self.MAX_CPU_PERCENT = 90
```

## License

Same as parent project.
