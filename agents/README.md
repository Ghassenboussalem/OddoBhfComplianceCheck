# Multi-Agent Compliance System - Agents

This directory contains all specialized agents for the multi-agent compliance checking system.

## Overview

Each agent is responsible for a specific aspect of compliance checking. All agents inherit from `BaseAgent` and follow a standard interface for processing the shared `ComplianceState`.

## Available Agents

### 1. Base Agent (`base_agent.py`)
- **Purpose**: Abstract base class for all agents
- **Features**:
  - Standard interface (`process()` method)
  - Timing and error handling decorators
  - Agent registry for dynamic loading
  - Configuration management
  - Execution statistics tracking

### 2. Supervisor Agent (`supervisor_agent.py`)
- **Purpose**: Orchestrate the entire compliance workflow
- **Responsibilities**:
  - Initialize workflow
  - Create execution plan based on document type
  - Coordinate agent execution
  - Handle failures and retries
  - Generate final compliance report
- **Status**: ✅ Implemented

### 3. Preprocessor Agent (`preprocessor_agent.py`)
- **Purpose**: Prepare documents for compliance checking
- **Responsibilities**:
  - Extract metadata (fund ISIN, client type, etc.)
  - Build whitelist of allowed terms
  - Normalize document structure
  - Validate document completeness
- **Tools Used**:
  - `extract_metadata`
  - `build_whitelist`
  - `normalize_document`
  - `validate_document`
- **Status**: ✅ Implemented
- **Requirements**: 1.2, 2.3, 6.1

### 4. Structure Agent (Planned)
- **Purpose**: Check structural compliance requirements
- **Checks**:
  - Promotional document mention
  - Target audience specification
  - Management company legal mention
  - Fund name presence
  - Date validation

### 5. Performance Agent (Planned)
- **Purpose**: Check performance-related compliance
- **Checks**:
  - Performance disclaimers
  - Document starts with performance
  - Benchmark comparisons
  - Fund age restrictions

### 6. Securities Agent (Planned)
- **Purpose**: Check securities-related compliance
- **Checks**:
  - Prohibited phrases (context-aware)
  - Repeated securities (whitelist-aware)
  - Investment advice detection

### 7. General Agent (Planned)
- **Purpose**: Check general compliance requirements
- **Checks**:
  - Glossary requirements
  - Morningstar date
  - Source citations
  - Technical terms

### 8. Prospectus Agent (Planned)
- **Purpose**: Check prospectus consistency
- **Checks**:
  - Fund name matching
  - Strategy consistency
  - Benchmark validation
  - Investment objective alignment

### 9. Registration Agent (Planned)
- **Purpose**: Check registration compliance
- **Checks**:
  - Country authorization
  - Fund registration validation

### 10. ESG Agent (Planned)
- **Purpose**: Check ESG compliance
- **Checks**:
  - ESG classification
  - Content distribution
  - SFDR compliance

### 11. Aggregator Agent (Planned)
- **Purpose**: Aggregate results from all specialist agents
- **Responsibilities**:
  - Collect violations from all agents
  - Calculate confidence scores
  - Deduplicate violations
  - Determine next action (context analysis or complete)

### 12. Context Agent (Planned)
- **Purpose**: Analyze context to eliminate false positives
- **Responsibilities**:
  - Analyze text context
  - Classify intent (advice vs description)
  - Update violation confidence based on context
  - Filter false positives

### 13. Evidence Agent (Planned)
- **Purpose**: Extract evidence for violations
- **Responsibilities**:
  - Extract specific quotes
  - Find performance data
  - Find disclaimers
  - Track locations and context

### 14. Reviewer Agent (Planned)
- **Purpose**: Manage human-in-the-loop review
- **Responsibilities**:
  - Queue low-confidence violations for review
  - Calculate priority scores
  - Implement HITL interrupts
  - Manage review workflow

### 15. Feedback Agent (Planned)
- **Purpose**: Process human feedback
- **Responsibilities**:
  - Process review decisions
  - Update confidence calibration
  - Detect patterns in false positives
  - Suggest rule modifications

## Agent Interface

All agents must implement the `BaseAgent` abstract class:

```python
from agents.base_agent import BaseAgent, AgentConfig
from data_models_multiagent import ComplianceState

class MyAgent(BaseAgent):
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        super().__init__(config=config, **kwargs)
        # Initialize agent-specific resources
    
    def process(self, state: ComplianceState) -> ComplianceState:
        """
        Process the compliance state.
        
        Args:
            state: Current compliance state
        
        Returns:
            Updated compliance state
        """
        # Perform compliance checks
        # Add violations to state
        # Update state fields
        return state
```

## Usage Example

```python
from agents.preprocessor_agent import create_preprocessor_agent
from data_models_multiagent import initialize_compliance_state

# Create agent
agent = create_preprocessor_agent()

# Initialize state
state = initialize_compliance_state(
    document=my_document,
    document_id="doc_001",
    config=my_config
)

# Execute agent
result = agent(state)

# Access results
print(f"Metadata: {result['metadata']}")
print(f"Whitelist size: {len(result['whitelist'])}")
print(f"Violations: {len(result['violations'])}")
```

## Testing

Each agent has a corresponding test file:
- `test_base_agent.py` - Tests for base agent framework
- `test_supervisor_agent.py` - Tests for supervisor agent
- `test_preprocessor_agent.py` - Tests for preprocessor agent

Run tests:
```bash
python test_preprocessor_agent.py
```

## Configuration

Agents are configured through `AgentConfig`:

```python
from agents.base_agent import AgentConfig

config = AgentConfig(
    name="my_agent",
    enabled=True,
    timeout_seconds=30.0,
    retry_attempts=3,
    log_level="INFO",
    custom_settings={
        "custom_option": "value"
    }
)
```

Or through configuration files:

```json
{
  "agents": {
    "preprocessor": {
      "enabled": true,
      "timeout_seconds": 30.0,
      "log_level": "INFO"
    }
  }
}
```

## Development Guidelines

1. **Inherit from BaseAgent**: All agents must extend `BaseAgent`
2. **Implement process()**: Core logic goes in the `process()` method
3. **Use tools**: Delegate specific operations to tools (see `tools/` directory)
4. **Update state**: Always return an updated `ComplianceState`
5. **Add violations**: Use `self.add_violation()` or `self.add_violations()`
6. **Handle errors**: Errors are automatically logged by base class
7. **Log appropriately**: Use `self.logger` for agent-specific logging
8. **Write tests**: Create comprehensive tests for each agent

## State Flow

```
Initialize State
      ↓
Supervisor Agent (orchestrate)
      ↓
Preprocessor Agent (prepare document)
      ↓
[Parallel Execution]
├─ Structure Agent
├─ Performance Agent
├─ Securities Agent
└─ General Agent
      ↓
Aggregator Agent (combine results)
      ↓
Context Agent (if low confidence)
      ↓
Evidence Agent (extract evidence)
      ↓
Reviewer Agent (if still low confidence)
      ↓
Final Report
```

## Requirements Mapping

- **Requirement 1.2**: Agent-based architecture with specialized agents
- **Requirement 2.3**: Preserve AI-enhanced features (context, whitelist, etc.)
- **Requirement 6.1**: Standard state interface for agent communication
- **Requirement 7.2**: Tool-based agent implementation

## Next Steps

1. Implement remaining specialist agents (Structure, Performance, etc.)
2. Implement context and evidence agents
3. Implement review and feedback agents
4. Integrate all agents into LangGraph workflow
5. Add parallel execution support
6. Implement conditional routing based on confidence
7. Add HITL interrupts for review workflow
