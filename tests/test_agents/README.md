# Agent Unit Tests

This directory contains comprehensive unit tests for all agents in the multi-agent compliance system.

## Test Coverage

All agents have unit tests covering:
- Agent initialization
- Tool invocation
- State updates
- Error handling
- Confidence scoring
- Parallel vs sequential execution

## Agents Tested

1. **Base Agent** (`test_base_agent.py`) - Framework and base functionality
2. **Supervisor Agent** (`test_supervisor_agent.py`) - Workflow orchestration
3. **Preprocessor Agent** (`test_preprocessor_agent.py`) - Document preprocessing
4. **Structure Agent** (`test_structure_agent.py`) - Structure compliance checks
5. **Performance Agent** (`test_performance_agent.py`) - Performance compliance checks
6. **Securities Agent** - Securities and investment advice checks
7. **General Agent** (`test_general_agent.py`) - General compliance checks
8. **Prospectus Agent** (`test_prospectus_agent.py`) - Prospectus consistency checks
9. **Registration Agent** (`test_registration_agent.py`) - Country authorization checks
10. **ESG Agent** - ESG classification checks
11. **Aggregator Agent** (`test_aggregator_agent.py`) - Violation aggregation
12. **Context Agent** (`test_context_agent.py`) - Context analysis
13. **Evidence Agent** (`test_evidence_agent.py`) - Evidence extraction
14. **Reviewer Agent** (`test_reviewer_agent.py`) - HITL review management
15. **Feedback Agent** (`test_feedback_agent.py`) - Feedback processing

## Running Tests

### Run all agent tests:
```bash
python tests/test_agents/run_all_tests.py
```

### Run individual agent tests:
```bash
python test_base_agent.py
python test_supervisor_agent.py
python test_preprocessor_agent.py
# etc.
```

### Run with pytest:
```bash
pytest tests/test_agents/ -v
pytest tests/test_agents/test_base_agent.py -v
```

### Run with unittest:
```bash
python -m unittest discover tests/test_agents -v
```

## Test Requirements

- Python 3.8+
- All dependencies from requirements.txt
- Valid API keys in .env file (for AI-enabled tests)

## Test Structure

Each test file follows this structure:
1. **Initialization Tests** - Verify agent creates correctly
2. **Functional Tests** - Test core agent functionality
3. **Tool Tests** - Test individual tool invocations
4. **State Tests** - Verify state updates
5. **Error Tests** - Test error handling
6. **Integration Tests** - Test agent interactions

## Code Coverage

Target: >80% code coverage for all agents

Run coverage report:
```bash
pytest tests/test_agents/ --cov=agents --cov-report=html
```

## Requirements Addressed

- **Requirement 14.1**: Unit tests for all agents
- **Requirement 14.5**: >80% code coverage
- **Requirement 1.2**: Agent-based architecture testing
- **Requirement 2.1**: Feature parity validation
- **Requirement 3.2**: Parallel execution testing
