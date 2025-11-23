# Quick Start Guide - Agent Unit Tests

## 🚀 Quick Commands

### Run All Tests
```bash
python tests/test_agents/run_all_tests.py
```

### Run Specific Agent Test
```bash
python test_base_agent.py
python test_supervisor_agent.py
python test_structure_agent.py
```

### Run with Pytest
```bash
pytest test_*_agent.py -v
```

### Generate Coverage Report
```bash
pytest test_*_agent.py --cov=agents --cov-report=html
open htmlcov/index.html
```

## 📁 Test Files

All test files are in the project root:

| Agent | Test File | Tests |
|-------|-----------|-------|
| Base | test_base_agent.py | 6 |
| Supervisor | test_supervisor_agent.py | 8 |
| Preprocessor | test_preprocessor_agent.py | 9 |
| Structure | test_structure_agent.py | 12 |
| Performance | test_performance_agent.py | 13 |
| General | test_general_agent.py | 10 |
| Aggregator | test_aggregator_agent.py | 5 |
| Context | test_context_agent.py | 1 |
| Evidence | test_evidence_agent.py | 1 |
| Reviewer | test_reviewer_agent.py | Multiple |
| Feedback | test_feedback_agent.py | Multiple |
| Prospectus | test_prospectus_agent.py | Multiple |
| Registration | test_registration_agent.py | Multiple |

**Total**: 76+ tests across 13 agents

## ✅ What's Tested

- ✅ Agent initialization
- ✅ Tool invocation
- ✅ State updates
- ✅ Error handling
- ✅ Confidence scoring
- ✅ Parallel execution
- ✅ Sequential execution

## 📊 Coverage Target

**Target**: >80% code coverage  
**Status**: ✅ Achievable with current test suite

## 📚 Documentation

- `README.md` - Overview and usage
- `TEST_SUMMARY.md` - Detailed test summary
- `TEST_INDEX.md` - Complete test index
- `COMPLETION_SUMMARY.md` - Task completion details
- `QUICK_START.md` - This file

## 🔧 Requirements

- Python 3.8+
- All dependencies from requirements.txt
- Optional: API keys in .env for AI tests

## 💡 Tips

1. **Run tests individually** during development
2. **Use pytest markers** to filter tests
3. **Check coverage** regularly
4. **Review test output** for failures
5. **Update tests** when agents change

## 🐛 Troubleshooting

**Import errors?**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**AI tests failing?**
- Tests gracefully degrade without AI
- Set API keys in .env for full testing

**Slow tests?**
```bash
pytest test_base_agent.py -v  # Run one file
```

## 📞 Need Help?

1. Check `README.md` for detailed instructions
2. Review `TEST_INDEX.md` for test details
3. See `COMPLETION_SUMMARY.md` for task info
4. Run individual tests to isolate issues

---

**Task 49 Status**: ✅ COMPLETED  
**Requirements**: 14.1, 14.5  
**Date**: November 23, 2025
