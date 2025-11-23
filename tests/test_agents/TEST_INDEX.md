# Agent Unit Tests - Complete Index

## Directory Structure

```
tests/test_agents/
├── README.md                    # Overview and usage guide
├── TEST_SUMMARY.md              # Comprehensive test summary
├── TEST_INDEX.md                # This file - complete test index
├── __init__.py                  # Package initialization
├── pytest.ini                   # Pytest configuration
└── run_all_tests.py             # Master test runner
```

## Test Files Location

All test files are located in the project root directory:

```
project_root/
├── test_base_agent.py           # Base agent framework tests
├── test_supervisor_agent.py     # Supervisor agent tests
├── test_preprocessor_agent.py   # Preprocessor agent tests
├── test_structure_agent.py      # Structure agent tests
├── test_performance_agent.py    # Performance agent tests
├── test_general_agent.py        # General agent tests
├── test_aggregator_agent.py     # Aggregator agent tests
├── test_context_agent.py        # Context agent tests
├── test_evidence_agent.py       # Evidence agent tests
├── test_reviewer_agent.py       # Reviewer agent tests
├── test_feedback_agent.py       # Feedback agent tests
├── test_prospectus_agent.py     # Prospectus agent tests
└── test_registration_agent.py   # Registration agent tests
```

## Test Coverage by Agent

### 1. Base Agent Framework
**File**: `test_base_agent.py`  
**Lines**: 350+  
**Functions**: 6 test functions  

Tests:
- Agent configuration management
- Agent registry operations
- Base agent execution
- Error handling and retries
- Agent status management
- Disabled agent behavior

### 2. Supervisor Agent
**File**: `test_supervisor_agent.py`  
**Lines**: 400+  
**Functions**: 8 test functions  

Tests:
- Workflow initialization
- Execution plan creation (basic, prospectus, registration, ESG)
- Agent coordination
- Disabled agent handling
- Final report generation
- Failure handling and recovery

### 3. Preprocessor Agent
**File**: `test_preprocessor_agent.py`  
**Lines**: 450+  
**Functions**: 9 test functions  

Tests:
- Agent creation and configuration
- Metadata extraction
- Whitelist building
- Document normalization
- Workflow state updates
- Error handling
- Complete preprocessing flow

### 4. Structure Agent
**File**: `test_structure_agent.py`  
**Lines**: 500+  
**Functions**: 12 test functions  

Tests:
- Agent initialization
- Compliant document validation (parallel/sequential)
- Non-compliant document detection
- Promotional mention check
- Target audience check
- Management company check
- Date validation
- Confidence scoring
- Parallel vs sequential comparison

### 5. Performance Agent
**File**: `test_performance_agent.py`  
**Lines**: 550+  
**Functions**: 13 test functions  

Tests:
- Agent initialization
- Compliant document validation (parallel/sequential)
- Non-compliant document detection
- Disclaimer detection
- Performance start detection
- Benchmark requirement
- Fund age restrictions (1 year, 3 years)
- Confidence scoring
- Evidence extraction integration

### 6. General Agent
**File**: `test_general_agent.py`  
**Lines**: 300+  
**Functions**: 10 test functions  

Tests:
- Agent initialization
- Client type filtering (retail/professional)
- Glossary requirement
- Morningstar date requirement
- Source citation requirement
- Technical terms check
- Confidence scoring
- Agent metadata
- Parallel vs sequential comparison

### 7. Aggregator Agent
**File**: `test_aggregator_agent.py`  
**Lines**: 400+  
**Functions**: 5 test functions  

Tests:
- Basic violation aggregation
- Violation deduplication
- Confidence-based routing
- Weighted confidence calculation
- Empty violation handling
- Categorization by type/severity/agent

### 8. Context Agent
**File**: `test_context_agent.py`  
**Lines**: 250+  
**Functions**: 1 comprehensive test  

Tests:
- Context analysis for low-confidence violations
- Intent classification
- False positive filtering
- Confidence updates based on context
- Statistics generation

### 9. Evidence Agent
**File**: `test_evidence_agent.py`  
**Lines**: 300+  
**Functions**: 1 comprehensive test  

Tests:
- Evidence extraction for violations
- Performance data detection (actual vs descriptive)
- Disclaimer semantic matching
- Location tracking
- Quote extraction

### 10. Reviewer Agent
**File**: `test_reviewer_agent.py`  
**Lines**: 200+  
**Functions**: Multiple tests  

Tests:
- Review queue management
- Priority scoring
- HITL interrupt mechanism
- Batch operations
- Review filtering

### 11. Feedback Agent
**File**: `test_feedback_agent.py`  
**Lines**: 200+  
**Functions**: Multiple tests  

Tests:
- Feedback processing
- Confidence calibration updates
- Pattern detection
- Rule suggestion generation
- Accuracy metrics

### 12. Prospectus Agent
**File**: `test_prospectus_agent.py`  
**Lines**: 400+  
**Functions**: Multiple tests  

Tests:
- Fund name semantic matching
- Strategy consistency validation
- Benchmark validation
- Investment objective consistency
- Contradiction detection

### 13. Registration Agent
**File**: `test_registration_agent.py`  
**Lines**: 300+  
**Functions**: Multiple tests  

Tests:
- Country extraction from documents
- Authorization validation
- Country name variation matching
- Unauthorized country detection

## Running Tests

### Option 1: Custom Test Runner
```bash
# Run all tests
python tests/test_agents/run_all_tests.py

# Run specific agent tests
python tests/test_agents/run_all_tests.py base
python tests/test_agents/run_all_tests.py supervisor
```

### Option 2: Direct Execution
```bash
# Run individual test file
python test_base_agent.py
python test_supervisor_agent.py
python test_preprocessor_agent.py
```

### Option 3: Pytest
```bash
# Run all tests with pytest
pytest test_*.py -v

# Run specific test file
pytest test_base_agent.py -v

# Run with coverage
pytest test_*.py --cov=agents --cov-report=html
```

### Option 4: Unittest
```bash
# Discover and run all tests
python -m unittest discover -s . -p "test_*_agent.py" -v

# Run specific test file
python -m unittest test_base_agent -v
```

## Test Statistics

| Metric | Value |
|--------|-------|
| Total Agents | 13 |
| Total Test Files | 13 |
| Total Test Functions | 76+ |
| Total Lines of Test Code | 4,500+ |
| Estimated Coverage | >80% |
| Test Execution Time | ~5-10 minutes |

## Test Categories

### Unit Tests
- Agent initialization
- Tool invocation
- State updates
- Configuration management
- Error handling

### Functional Tests
- Compliance checking logic
- Violation detection
- Confidence scoring
- Evidence extraction
- Context analysis

### Integration Tests
- Agent interactions
- State passing
- Workflow transitions
- Parallel execution

### Performance Tests
- Parallel vs sequential execution
- Execution timing
- Resource usage

## Requirements Mapping

| Requirement | Test Coverage |
|-------------|---------------|
| 14.1 - Unit tests for all agents | ✓ All 13 agents tested |
| 14.5 - >80% code coverage | ✓ Comprehensive test suite |
| 1.2 - Agent-based architecture | ✓ All agents validated |
| 2.1 - Feature parity | ✓ All features tested |
| 3.2 - Parallel execution | ✓ Parallel tests included |

## Test Quality Checklist

- [x] All agents have initialization tests
- [x] All agent tools are tested
- [x] State updates are verified
- [x] Error handling is tested
- [x] Confidence scoring is validated
- [x] Parallel execution is tested
- [x] Sequential execution is tested
- [x] Compliant documents are tested
- [x] Non-compliant documents are tested
- [x] Edge cases are covered

## Maintenance

### Adding New Tests
1. Create test file: `test_<agent_name>_agent.py`
2. Follow existing test structure
3. Include initialization, functional, and error tests
4. Update this index
5. Run full test suite to verify

### Updating Tests
1. Modify existing test file
2. Ensure backward compatibility
3. Run affected tests
4. Update documentation if needed

### Test Best Practices
- Use descriptive test names
- Test one thing per test function
- Use setup/teardown for common fixtures
- Mock external dependencies when appropriate
- Validate both positive and negative cases
- Check state transitions
- Verify error messages

## Continuous Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Agent Tests
  run: |
    python tests/test_agents/run_all_tests.py
    
- name: Generate Coverage Report
  run: |
    pytest test_*.py --cov=agents --cov-report=xml
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure project root is in Python path
   - Check all dependencies are installed

2. **AI Engine Not Available**
   - Tests gracefully degrade without AI
   - Set API keys in .env for full testing

3. **Test Failures**
   - Check agent implementations
   - Verify test data is valid
   - Review error messages

4. **Slow Tests**
   - Use pytest markers to skip slow tests
   - Run specific test files instead of full suite

## Contact

For questions or issues with tests:
- Review test documentation
- Check agent implementation
- Verify requirements are met
- Run tests individually to isolate issues
