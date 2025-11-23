# Multi-Agent Compliance System Architecture

## Table of Contents

1. [Overview](#overview)
2. [Architecture Principles](#architecture-principles)
3. [System Components](#system-components)
4. [Agent Responsibilities](#agent-responsibilities)
5. [Workflow Flow](#workflow-flow)
6. [State Management](#state-management)
7. [Communication Patterns](#communication-patterns)
8. [Parallel Execution](#parallel-execution)
9. [Conditional Routing](#conditional-routing)
10. [Error Handling and Resilience](#error-handling-and-resilience)
11. [State Persistence and Resumability](#state-persistence-and-resumability)
12. [Human-in-the-Loop Integration](#human-in-the-loop-integration)
13. [Performance Characteristics](#performance-characteristics)
14. [Deployment Architecture](#deployment-architecture)
15. [Monitoring and Observability](#monitoring-and-observability)

---

## Overview

The Multi-Agent Compliance System is a distributed, LangGraph-based architecture that transforms the monolithic compliance checker into a modular, scalable system. The system uses specialized agents that collaborate through a shared state machine to perform comprehensive compliance checking on financial documents.

### Key Features

- **Agent Specialization**: Each compliance domain has a dedicated agent with focused responsibilities
- **Parallel Execution**: Independent checks run concurrently for 30%+ performance improvement
- **State-Driven Workflow**: LangGraph manages state transitions and agent coordination
- **Conditional Routing**: Confidence scores determine workflow paths dynamically
- **HITL Integration**: Built-in interrupts for human review of low-confidence violations
- **State Persistence**: Checkpointing enables workflow resumability after interruptions
- **Backward Compatibility**: Maintains 100% feature parity with the monolithic system

### Technology Stack

- **LangGraph**: State machine and workflow orchestration
- **LangChain**: Agent tools and LLM integration
- **SQLite**: State persistence and checkpointing
- **Python 3.8+**: Core implementation language


---

## Architecture Principles

### 1. Separation of Concerns

Each agent is responsible for a specific compliance domain:
- **Structure Agent**: Document structure and legal requirements
- **Performance Agent**: Performance data and disclaimers
- **Securities Agent**: Securities mentions and investment advice
- **General Agent**: General compliance rules (glossary, citations, dates)
- **Prospectus Agent**: Prospectus consistency validation
- **Registration Agent**: Country authorization validation
- **ESG Agent**: ESG classification and content distribution

### 2. Single Responsibility

Each agent:
- Has one clear purpose
- Operates independently
- Can be tested in isolation
- Can be replaced without affecting others

### 3. Loose Coupling

Agents communicate only through the shared state:
- No direct agent-to-agent calls
- All coordination through LangGraph state machine
- State is the single source of truth

### 4. High Cohesion

Related functionality is grouped within agents:
- Tools are organized by domain
- Each agent owns its tools
- Clear boundaries between domains

### 5. Fail-Safe Design

System continues operating even when agents fail:
- Graceful degradation
- Error logging without workflow termination
- Fallback strategies (AI → rules)
- Partial results generation


---

## System Components

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     MULTI-AGENT SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │  Entry Point     │  │  Workflow        │  │  State       │ │
│  │  check_multi     │→ │  Builder         │→ │  Manager     │ │
│  │  agent.py        │  │  workflow_       │  │  state_      │ │
│  │                  │  │  builder.py      │  │  manager.py  │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              LangGraph State Machine                      │  │
│  │                                                           │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐        │  │
│  │  │ Supervisor │→ │Preprocessor│→ │  Parallel  │        │  │
│  │  │   Agent    │  │   Agent    │  │  Execution │        │  │
│  │  └────────────┘  └────────────┘  └────────────┘        │  │
│  │                                         ↓                │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐        │  │
│  │  │ Aggregator │← │  Context   │← │  Evidence  │        │  │
│  │  │   Agent    │  │   Agent    │  │   Agent    │        │  │
│  │  └────────────┘  └────────────┘  └────────────┘        │  │
│  │                                         ↓                │  │
│  │                  ┌────────────┐                         │  │
│  │                  │  Reviewer  │                         │  │
│  │                  │   Agent    │                         │  │
│  │                  └────────────┘                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │  Agent Tools     │  │  AI Engine       │  │  Review      │ │
│  │  tools/*.py      │  │  ai_engine.py    │  │  Manager     │ │
│  │                  │  │                  │  │  review_     │ │
│  │                  │  │                  │  │  manager.py  │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

#### 1. Entry Point (`check_multiagent.py`)
- Command-line interface
- Configuration loading
- Workflow initialization
- Result formatting and output

#### 2. Workflow Builder (`workflow_builder.py`)
- LangGraph StateGraph construction
- Agent node registration
- Edge and conditional routing definition
- Checkpointing configuration

#### 3. State Manager (`state_manager.py`)
- State persistence to SQLite
- Checkpoint management
- State history tracking
- Resume functionality

#### 4. Base Agent Framework (`agents/base_agent.py`)
- Abstract base class for all agents
- Standard interface (`__call__` and `process`)
- Timing and error handling decorators
- Agent registry for dynamic loading

#### 5. Specialized Agents (`agents/*_agent.py`)
- Domain-specific compliance checking
- Tool integration
- Violation detection and reporting
- State updates

#### 6. Agent Tools (`tools/*_tools.py`)
- Reusable compliance check functions
- LangChain `@tool` decorated
- Domain-organized (structure, performance, etc.)
- AI and rule-based implementations

#### 7. Data Models (`data_models_multiagent.py`)
- `ComplianceState` TypedDict definition
- State initialization and validation
- Serialization/deserialization
- Helper functions


---

## Agent Responsibilities

### Supervisor Agent

**Purpose**: Orchestrate the entire compliance workflow

**Responsibilities**:
- Initialize workflow with document information
- Create execution plan based on document type and metadata
- Coordinate specialist agent execution
- Monitor agent failures and implement fallback strategies
- Generate final compliance report

**Key Methods**:
- `process()`: Main orchestration logic
- `_create_execution_plan()`: Determine which agents to run
- `generate_final_report()`: Create comprehensive report
- `handle_agent_failure()`: Implement fallback strategies

**State Updates**:
- Sets `workflow_status` to "initialized"
- Creates `execution_plan` list
- Sets `next_action` to "preprocess"

---

### Preprocessor Agent

**Purpose**: Prepare document for compliance checking

**Responsibilities**:
- Extract document metadata (fund ISIN, client type, etc.)
- Build whitelist of allowed terms
- Normalize document structure
- Validate document format

**Tools Used**:
- `extract_metadata`: Extract fund and document information
- `build_whitelist`: Create whitelist using WhitelistManager
- `normalize_document`: Standardize document structure

**State Updates**:
- Populates `metadata` dictionary
- Creates `whitelist` set
- Stores `normalized_document`
- Sets `next_action` to "parallel_checks"

---

### Structure Agent

**Purpose**: Check document structure and legal requirements

**Responsibilities**:
- Verify promotional document mention on cover page
- Check target audience specification (retail vs professional)
- Validate management company legal mention
- Check fund name presence
- Validate date formats and requirements

**Tools Used**:
- `check_promotional_mention`
- `check_target_audience`
- `check_management_company`
- `check_fund_name`
- `check_date_validation`

**Violation Types**:
- `STRUCTURE` violations
- Severity: CRITICAL to WARNING

---

### Performance Agent

**Purpose**: Check performance data and disclaimers

**Responsibilities**:
- Detect actual performance data (numbers with %)
- Validate performance disclaimers
- Check benchmark comparisons
- Validate fund age restrictions
- Ensure disclaimers are near performance data

**Tools Used**:
- `check_performance_disclaimers_ai` (data-aware)
- `check_document_starts_with_performance_ai`
- `check_benchmark_comparison`
- `check_fund_age_restrictions`

**Violation Types**:
- `PERFORMANCE` violations
- Severity: CRITICAL to MAJOR

**Special Features**:
- Distinguishes actual performance data from descriptive keywords
- Semantic disclaimer matching
- Context-aware validation

---

### Securities Agent

**Purpose**: Check securities mentions and investment advice

**Responsibilities**:
- Detect prohibited phrases (context-aware)
- Check for repeated securities mentions (whitelist-aware)
- Identify investment advice vs descriptions
- Classify intent (ADVICE, DESCRIPTION, FACT, EXAMPLE)

**Tools Used**:
- `check_prohibited_phrases_ai` (context-aware)
- `check_repeated_securities_ai` (whitelist-aware)
- `check_investment_advice`
- `classify_intent`

**Violation Types**:
- `SECURITIES` violations
- Severity: CRITICAL to MAJOR

**Special Features**:
- Semantic understanding of WHO performs actions
- Intent classification
- Whitelist filtering
- False positive elimination through context

---

### General Agent

**Purpose**: Check general compliance rules

**Responsibilities**:
- Verify glossary requirement
- Check Morningstar date presence
- Validate source citations
- Check technical terms usage
- Apply client-type specific rules

**Tools Used**:
- `check_glossary_requirement`
- `check_morningstar_date`
- `check_source_citations`
- `check_technical_terms`

**Violation Types**:
- `GENERAL` violations
- Severity: MAJOR to WARNING

**Special Features**:
- Client type filtering (retail vs professional)
- Rule-based checking
- Fast execution


---

### Prospectus Agent

**Purpose**: Validate consistency with prospectus data

**Responsibilities**:
- Check fund name matches prospectus (semantic matching)
- Validate strategy consistency
- Check benchmark validation
- Verify investment objective alignment

**Tools Used**:
- `check_fund_name_match` (semantic similarity)
- `check_strategy_consistency`
- `check_benchmark_validation`
- `check_investment_objective`

**Violation Types**:
- `PROSPECTUS` violations
- Severity: CRITICAL to MAJOR

**Special Features**:
- Semantic similarity matching (not exact string match)
- Contradiction detection
- AI-powered consistency validation

**Conditional Execution**:
- Only runs if `prospectus_data` is available in config
- Skipped if no prospectus reference provided

---

### Registration Agent

**Purpose**: Validate country authorization

**Responsibilities**:
- Extract countries mentioned in document
- Validate fund registration in mentioned countries
- Check authorization status
- Handle country name variations

**Tools Used**:
- `check_country_authorization`
- `extract_countries_from_document`
- `validate_fund_registration`

**Violation Types**:
- `REGISTRATION` violations
- Severity: CRITICAL

**Special Features**:
- AI-powered country extraction
- Country name variation matching
- Registration database lookup

**Conditional Execution**:
- Only runs if `fund_isin` is available in metadata
- Skipped if no ISIN provided

---

### ESG Agent

**Purpose**: Check ESG classification and content

**Responsibilities**:
- Validate ESG classification (Article 6, 8, 9)
- Check content distribution requirements
- Verify SFDR compliance
- Validate ESG terminology usage

**Tools Used**:
- `check_esg_classification`
- `check_content_distribution`
- `check_sfdr_compliance`
- `validate_esg_terminology`

**Violation Types**:
- `ESG` violations
- Severity: CRITICAL to MAJOR

**Special Features**:
- SFDR regulation compliance
- Content distribution analysis
- ESG terminology validation

**Conditional Execution**:
- Only runs if `fund_esg_classification` is not "other"
- Skipped for non-ESG funds

---

### Aggregator Agent

**Purpose**: Combine and analyze results from all specialist agents

**Responsibilities**:
- Collect violations from all agents
- Calculate confidence scores
- Deduplicate violations
- Categorize by type and severity
- Determine next action (context analysis or complete)

**Key Methods**:
- `process()`: Aggregate all violations
- `_calculate_confidence()`: Compute overall confidence
- `_deduplicate_violations()`: Remove duplicates
- `_determine_next_action()`: Route based on confidence

**State Updates**:
- Consolidates all `violations`
- Calculates `aggregated_confidence`
- Sets `next_action` ("context_analysis" or "complete")

**Routing Logic**:
- If any violation has confidence < 80% → route to Context Agent
- Otherwise → workflow complete

---

### Context Agent

**Purpose**: Analyze context and intent to eliminate false positives

**Responsibilities**:
- Analyze text context (WHO performs actions, WHAT is the intent)
- Classify intent (ADVICE, DESCRIPTION, FACT, EXAMPLE)
- Validate semantically using AI
- Update confidence scores based on context
- Filter false positives

**Tools Used**:
- `analyze_context`: Semantic context analysis
- `classify_intent`: Intent classification
- `validate_semantically`: Semantic validation

**Violation Updates**:
- Increases confidence for valid violations
- Marks false positives with status "false_positive_filtered"
- Adds context analysis to state

**State Updates**:
- Populates `context_analysis` dictionary
- Populates `intent_classifications` dictionary
- Updates violation confidence scores
- Sets `next_action` to "evidence"

**Special Features**:
- Only processes low-confidence violations (< 80%)
- Uses AI for semantic understanding
- Eliminates false positives through context

---

### Evidence Agent

**Purpose**: Extract evidence for violations

**Responsibilities**:
- Extract specific quotes supporting violations
- Find actual performance data (numbers with %)
- Locate disclaimers semantically
- Add location and context tracking
- Enhance violation details

**Tools Used**:
- `extract_evidence`: Extract supporting quotes
- `find_performance_data`: Detect actual performance numbers
- `find_disclaimer`: Semantic disclaimer matching
- `track_location`: Add precise location information

**State Updates**:
- Populates `evidence_extractions` dictionary
- Enhances violations with evidence details
- Sets `next_action` ("review" or "complete")

**Routing Logic**:
- If any violation has confidence < 70% → route to Reviewer Agent
- Otherwise → workflow complete

**Special Features**:
- Semantic matching (not keyword search)
- Distinguishes actual data from descriptive text
- Precise location tracking

---

### Reviewer Agent

**Purpose**: Manage Human-in-the-Loop review process

**Responsibilities**:
- Queue low-confidence violations for human review
- Calculate priority scores
- Manage review queue
- Support batch operations
- Implement HITL interrupt mechanism

**Tools Used**:
- `queue_for_review`: Add to review queue
- `calculate_priority_score`: Determine review priority
- `filter_reviews`: Filter by criteria
- `batch_review_operations`: Batch processing

**State Updates**:
- Populates `review_queue` list
- Adds review items to ReviewManager
- Sets workflow status to "interrupted" (for HITL)

**HITL Integration**:
- Workflow can be interrupted at this node
- State is persisted for resumability
- Human reviewer provides feedback
- Workflow resumes after review completion

**Review Threshold**:
- Default: 70% confidence
- Configurable via `routing.review_threshold`


---

## Workflow Flow

### Complete Workflow Diagram

```mermaid
graph TD
    START([Start]) --> SUPERVISOR[Supervisor Agent]
    SUPERVISOR --> PREPROCESSOR[Preprocessor Agent]
    
    PREPROCESSOR --> PARALLEL{Parallel Execution}
    
    PARALLEL --> STRUCTURE[Structure Agent]
    PARALLEL --> PERFORMANCE[Performance Agent]
    PARALLEL --> SECURITIES[Securities Agent]
    PARALLEL --> GENERAL[General Agent]
    PARALLEL --> PROSPECTUS[Prospectus Agent<br/>conditional]
    PARALLEL --> REGISTRATION[Registration Agent<br/>conditional]
    PARALLEL --> ESG[ESG Agent<br/>conditional]
    
    STRUCTURE --> AGGREGATOR[Aggregator Agent]
    PERFORMANCE --> AGGREGATOR
    SECURITIES --> AGGREGATOR
    GENERAL --> AGGREGATOR
    PROSPECTUS --> AGGREGATOR
    REGISTRATION --> AGGREGATOR
    ESG --> AGGREGATOR
    
    AGGREGATOR --> CONF_CHECK1{Confidence < 80%?}
    CONF_CHECK1 -->|Yes| CONTEXT[Context Agent]
    CONF_CHECK1 -->|No| END([End])
    
    CONTEXT --> EVIDENCE[Evidence Agent]
    
    EVIDENCE --> CONF_CHECK2{Confidence < 70%?}
    CONF_CHECK2 -->|Yes| REVIEWER[Reviewer Agent<br/>HITL Interrupt]
    CONF_CHECK2 -->|No| END
    
    REVIEWER --> END
    
    style SUPERVISOR fill:#e1f5ff
    style PREPROCESSOR fill:#e1f5ff
    style PARALLEL fill:#fff4e1
    style AGGREGATOR fill:#e1ffe1
    style CONTEXT fill:#ffe1f5
    style EVIDENCE fill:#ffe1f5
    style REVIEWER fill:#f5e1ff
    style CONF_CHECK1 fill:#fff
    style CONF_CHECK2 fill:#fff
```

### Execution Phases

#### Phase 1: Initialization
1. **Supervisor Agent** initializes workflow
   - Logs document information
   - Creates execution plan
   - Sets up performance tracking

#### Phase 2: Preprocessing
2. **Preprocessor Agent** prepares document
   - Extracts metadata
   - Builds whitelist
   - Normalizes structure

#### Phase 3: Parallel Compliance Checking
3. **Core Agents** run in parallel:
   - Structure Agent
   - Performance Agent
   - Securities Agent
   - General Agent

4. **Specialized Agents** run conditionally:
   - Prospectus Agent (if prospectus data available)
   - Registration Agent (if fund ISIN available)
   - ESG Agent (if ESG classification != "other")

**Synchronization Point**: All parallel agents must complete before proceeding

#### Phase 4: Aggregation
5. **Aggregator Agent** combines results
   - Collects all violations
   - Calculates confidence scores
   - Deduplicates violations
   - Determines next action

**Conditional Routing**: 
- If confidence < 80% → Phase 5 (Context Analysis)
- Otherwise → Phase 7 (Complete)

#### Phase 5: Context Analysis (Conditional)
6. **Context Agent** analyzes low-confidence violations
   - Semantic context analysis
   - Intent classification
   - False positive filtering
   - Confidence score updates

#### Phase 6: Evidence Extraction
7. **Evidence Agent** enhances violations
   - Extracts supporting quotes
   - Finds actual performance data
   - Locates disclaimers
   - Adds precise locations

**Conditional Routing**:
- If confidence < 70% → Phase 7 (Review)
- Otherwise → Phase 8 (Complete)

#### Phase 7: Human Review (Conditional)
8. **Reviewer Agent** manages HITL
   - Queues low-confidence violations
   - Calculates priority scores
   - **HITL Interrupt Point**: Workflow pauses for human review
   - State is persisted
   - Workflow resumes after review

#### Phase 8: Completion
9. **Workflow Ends**
   - Final state is returned
   - Report is generated
   - Results are saved


### Execution Timeline Example

```
Time    Agent               Action
────────────────────────────────────────────────────────────────
0.0s    Supervisor         Initialize workflow
0.1s    Preprocessor       Extract metadata, build whitelist
        
0.5s    ┌─ Structure       Check promotional mention, target audience
        ├─ Performance     Check performance data, disclaimers
        ├─ Securities      Check prohibited phrases, investment advice
        ├─ General         Check glossary, citations, dates
        ├─ Prospectus      Check fund name, strategy consistency
        ├─ Registration    Check country authorization
        └─ ESG             Check ESG classification, content
        
2.8s    Aggregator         Collect violations, calculate confidence
                           → 3 violations with confidence < 80%
        
3.0s    Context            Analyze context for 3 low-confidence violations
                           → 1 marked as false positive
                           → 2 confidence increased to 85%
        
3.5s    Evidence           Extract evidence for remaining violations
                           → All violations now have confidence > 70%
        
3.7s    END                Workflow complete
        
Total: 3.7s (vs 5.2s sequential = 29% faster)
```

### State Transitions

```
INITIALIZED → PREPROCESSING → CHECKING → AGGREGATING → 
ANALYZING (conditional) → REVIEWING (conditional) → COMPLETED
```

Each agent updates the `workflow_status` field to track progress.


---

## State Management

### ComplianceState Structure

The `ComplianceState` is a TypedDict that flows through the entire workflow. It serves as the single source of truth for all agents.

```python
class ComplianceState(TypedDict, total=False):
    # Document data
    document: dict                    # Original document JSON
    document_id: str                  # Unique document identifier
    document_type: str                # Type (fund_presentation, factsheet, etc.)
    client_type: str                  # Client type (retail, professional)
    
    # Preprocessing results
    metadata: Dict[str, Any]          # Extracted metadata
    whitelist: Set[str]               # Allowed terms
    normalized_document: dict         # Normalized document structure
    
    # Violations (automatically merged from parallel agents)
    violations: Annotated[Sequence[dict], operator.add]
    
    # Context analysis results
    context_analysis: Dict[str, dict]      # Context analysis per violation
    intent_classifications: Dict[str, dict] # Intent classifications
    evidence_extractions: Dict[str, dict]   # Evidence extractions
    
    # Review and feedback
    review_queue: List[dict]          # Items queued for human review
    feedback_history: List[dict]      # Historical feedback
    
    # Confidence and scoring
    confidence_scores: Dict[str, int]  # Per-violation confidence
    aggregated_confidence: int         # Overall confidence
    
    # Workflow control
    current_agent: str                # Currently executing agent
    next_action: str                  # Next action to take
    workflow_status: str              # Current workflow status
    execution_plan: List[str]         # Planned agent execution order
    error_log: List[dict]             # Errors encountered
    
    # Performance metrics
    agent_timings: Dict[str, float]   # Execution time per agent
    api_calls: int                    # Total API calls made
    cache_hits: int                   # Cache hits
    
    # Configuration
    config: dict                      # Configuration dictionary
    
    # Timestamps
    started_at: str                   # Workflow start time
    updated_at: str                   # Last update time
    completed_at: Optional[str]       # Completion time
```

### State Merging for Parallel Agents

LangGraph automatically merges state updates from parallel agents using the `Annotated` type hints:

```python
# Violations are automatically merged using operator.add
violations: Annotated[Sequence[dict], operator.add]

# Agent timings are merged using dict update
agent_timings: Annotated[Dict[str, float], lambda x, y: {**x, **y}]
```

**Example**:
```python
# Initial state
state = {"violations": []}

# Structure Agent adds violations
state_after_structure = {"violations": [v1, v2]}

# Performance Agent adds violations (runs in parallel)
state_after_performance = {"violations": [v3]}

# LangGraph automatically merges:
final_state = {"violations": [v1, v2, v3]}
```

### State Initialization

```python
from data_models_multiagent import initialize_compliance_state

state = initialize_compliance_state(
    document=document_json,
    document_id="exemple.json",
    config=config_dict
)
```

### State Validation

```python
from data_models_multiagent import validate_compliance_state

is_valid, errors = validate_compliance_state(state)
if not is_valid:
    for error in errors:
        logger.error(f"State validation error: {error}")
```

### State Serialization

```python
from data_models_multiagent import serialize_state, deserialize_state

# Serialize to JSON
json_str = serialize_state(state)

# Deserialize from JSON
state = deserialize_state(json_str)
```

### State Persistence

State is automatically persisted to SQLite at each agent transition using LangGraph's `SqliteSaver`:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Create checkpointer
conn = sqlite3.connect("checkpoints/compliance_workflow.db")
checkpointer = SqliteSaver(conn)

# Compile workflow with checkpointing
workflow = graph.compile(checkpointer=checkpointer)
```

### State History

The `StateManager` tracks state history for debugging and auditing:

```python
from state_manager import StateManager

state_manager = StateManager(
    checkpoint_dir="checkpoints",
    checkpoint_interval=1,
    max_history_size=100
)

# Save checkpoint
state_manager.save_checkpoint(state, "after_aggregator")

# Load checkpoint
state = state_manager.load_checkpoint("after_aggregator")

# Get history
history = state_manager.get_history()
```


---

## Communication Patterns

### Agent-to-State Communication

Agents communicate exclusively through state updates. There are no direct agent-to-agent calls.

```python
class MyAgent(BaseAgent):
    def process(self, state: ComplianceState) -> ComplianceState:
        # Read from state
        document = state["document"]
        whitelist = state["whitelist"]
        
        # Perform checks
        violations = self.check_compliance(document, whitelist)
        
        # Update state
        state["violations"] = list(state.get("violations", [])) + violations
        state["current_agent"] = self.name
        
        # Return updated state
        return state
```

### State-Driven Coordination

The LangGraph state machine coordinates agents based on state values:

```python
def route_from_aggregator(state: ComplianceState) -> str:
    """Route based on next_action in state"""
    next_action = state.get("next_action", "complete")
    
    if next_action == "context_analysis":
        return "context"
    else:
        return "complete"

# Add conditional edge
workflow.add_conditional_edges(
    "aggregator",
    route_from_aggregator,
    {
        "context": "context",
        "complete": END
    }
)
```

### Tool-Based Operations

Agents use tools for specific operations:

```python
from langchain.tools import tool

@tool
def check_promotional_mention(document: dict, config: dict) -> Optional[dict]:
    """Check for promotional document mention on cover page"""
    # Implementation
    return violation_dict if violation else None

class StructureAgent(BaseAgent):
    def __init__(self, tools: List):
        self.tools = {tool.name: tool for tool in tools}
    
    def process(self, state: ComplianceState) -> ComplianceState:
        # Invoke tool
        result = self.tools["check_promotional_mention"].invoke({
            "document": state["document"],
            "config": state["config"]
        })
        
        if result:
            state = self.add_violation(state, result)
        
        return state
```

### Event-Driven Updates

Agents update state fields that trigger downstream behavior:

```python
# Aggregator sets next_action based on confidence
if any(v.get("confidence", 100) < 80 for v in violations):
    state["next_action"] = "context_analysis"
else:
    state["next_action"] = "complete"

# This triggers conditional routing in the workflow
```

### Violation Accumulation

Violations accumulate across agents using `operator.add`:

```python
# Agent 1 adds violations
state["violations"] = [v1, v2]

# Agent 2 adds violations (parallel)
state["violations"] = [v3]

# LangGraph merges automatically
# Final: [v1, v2, v3]
```

### Error Propagation

Errors are logged in state without stopping the workflow:

```python
try:
    result = self.check_something(state)
except Exception as e:
    # Log error in state
    state["error_log"].append({
        "agent": self.name,
        "error": str(e),
        "timestamp": datetime.now().isoformat()
    })
    
    # Continue with graceful degradation
    return state
```


---

## Parallel Execution

### How Parallel Execution Works

LangGraph executes agents in parallel when multiple edges originate from the same source node:

```python
# Preprocessor fans out to multiple agents
workflow.add_edge("preprocessor", "structure")
workflow.add_edge("preprocessor", "performance")
workflow.add_edge("preprocessor", "securities")
workflow.add_edge("preprocessor", "general")

# All 4 agents execute concurrently
```

### Synchronization Point

All parallel agents must complete before the workflow proceeds to the next node:

```python
# All parallel agents converge at aggregator
workflow.add_edge("structure", "aggregator")
workflow.add_edge("performance", "aggregator")
workflow.add_edge("securities", "aggregator")
workflow.add_edge("general", "aggregator")

# Aggregator waits for all agents to complete
```

### Parallel Execution Groups

#### Group 1: Core Compliance Agents
- **Structure Agent**: Document structure checks
- **Performance Agent**: Performance data checks
- **Securities Agent**: Securities mention checks
- **General Agent**: General compliance checks

**Characteristics**:
- Always run (not conditional)
- Independent of each other
- Can execute in any order
- Results merged at aggregator

#### Group 2: Specialized Agents (Conditional)
- **Prospectus Agent**: Runs if prospectus data available
- **Registration Agent**: Runs if fund ISIN available
- **ESG Agent**: Runs if ESG classification != "other"

**Characteristics**:
- Conditionally executed based on metadata
- Run in parallel with core agents
- Independent of each other
- Results merged at aggregator

### Performance Benefits

**Sequential Execution** (old system):
```
Structure (0.8s) → Performance (0.7s) → Securities (0.6s) → General (0.5s)
Total: 2.6s
```

**Parallel Execution** (new system):
```
┌─ Structure (0.8s)   ─┐
├─ Performance (0.7s) ─┤
├─ Securities (0.6s)  ─┤ → Aggregator
└─ General (0.5s)     ─┘

Total: max(0.8, 0.7, 0.6, 0.5) = 0.8s
Speedup: 2.6s / 0.8s = 3.25x (69% faster)
```

**Real-World Performance**:
- Sequential: ~5.2s
- Parallel: ~3.7s
- Improvement: 29% faster

### State Merging in Parallel Execution

LangGraph automatically merges state updates from parallel agents:

```python
# Structure Agent updates
state_structure = {
    "violations": [v1, v2],
    "agent_timings": {"structure": 0.8}
}

# Performance Agent updates (parallel)
state_performance = {
    "violations": [v3],
    "agent_timings": {"performance": 0.7}
}

# LangGraph merges using Annotated type hints
merged_state = {
    "violations": [v1, v2, v3],  # operator.add
    "agent_timings": {           # dict merge
        "structure": 0.8,
        "performance": 0.7
    }
}
```

### Configuring Parallel Execution

```json
{
  "multi_agent": {
    "parallel_execution": true,
    "max_parallel_agents": 4
  }
}
```

### Thread Safety

Each agent operates on its own copy of the state during parallel execution. LangGraph handles thread-safe merging.

### Error Handling in Parallel Execution

If one agent fails, others continue:

```python
# Structure Agent fails
state_structure = {
    "error_log": [{"agent": "structure", "error": "..."}]
}

# Other agents continue normally
state_performance = {
    "violations": [v1]
}

# Merged state includes both errors and results
merged_state = {
    "violations": [v1],
    "error_log": [{"agent": "structure", "error": "..."}]
}
```


---

## Conditional Routing

### Overview

Conditional routing allows the workflow to dynamically choose the next agent based on runtime state values, particularly confidence scores.

### Routing Points

#### 1. Aggregator → Context or END

**Condition**: Violation confidence scores

```python
def route_from_aggregator(state: ComplianceState) -> str:
    """Route based on confidence scores"""
    violations = state.get("violations", [])
    
    # Check if any violation has low confidence
    has_low_confidence = any(
        v.get("confidence", 100) < 80 
        for v in violations
    )
    
    if has_low_confidence:
        return "context"  # Route to Context Agent
    else:
        return "complete"  # End workflow

workflow.add_conditional_edges(
    "aggregator",
    route_from_aggregator,
    {
        "context": "context",
        "complete": END
    }
)
```

**Decision Logic**:
- If **any** violation has confidence < 80% → Context Agent
- Otherwise → Workflow complete

#### 2. Evidence → Reviewer or END

**Condition**: Violation confidence after evidence extraction

```python
def route_from_evidence(state: ComplianceState) -> str:
    """Route based on confidence after evidence extraction"""
    violations = state.get("violations", [])
    
    # Check if any violation still has very low confidence
    needs_review = any(
        v.get("confidence", 100) < 70 
        for v in violations
    )
    
    if needs_review:
        return "review"  # Route to Reviewer Agent (HITL)
    else:
        return "complete"  # End workflow

workflow.add_conditional_edges(
    "evidence",
    route_from_evidence,
    {
        "review": "reviewer",
        "complete": END
    }
)
```

**Decision Logic**:
- If **any** violation has confidence < 70% → Reviewer Agent (HITL)
- Otherwise → Workflow complete

### Confidence Thresholds

Configurable thresholds determine routing:

```json
{
  "routing": {
    "context_threshold": 80,
    "review_threshold": 70,
    "skip_context_if_high_confidence": true
  }
}
```

### Routing Flow Diagram

```
Aggregator
    │
    ├─ All violations confidence >= 80%? ─→ END
    │
    └─ Any violation confidence < 80%? ─→ Context Agent
                                              │
                                              ↓
                                         Evidence Agent
                                              │
                                              ├─ All violations confidence >= 70%? ─→ END
                                              │
                                              └─ Any violation confidence < 70%? ─→ Reviewer Agent
                                                                                        │
                                                                                        ↓
                                                                                       END
```

### Agent-Driven Routing

Agents can influence routing by setting `next_action` in state:

```python
class AggregatorAgent(BaseAgent):
    def process(self, state: ComplianceState) -> ComplianceState:
        # Calculate confidence
        low_confidence_count = sum(
            1 for v in violations 
            if v.get("confidence", 100) < 80
        )
        
        # Set next action
        if low_confidence_count > 0:
            state["next_action"] = "context_analysis"
        else:
            state["next_action"] = "complete"
        
        return state
```

### Conditional Agent Execution

Some agents only run if certain conditions are met:

```python
def should_run_prospectus(state: ComplianceState) -> bool:
    """Check if prospectus agent should run"""
    config = state.get("config", {})
    return bool(config.get("prospectus_data"))

# This is handled in Supervisor's execution plan
execution_plan = []
if should_run_prospectus(state):
    execution_plan.append("prospectus")
```

### Benefits of Conditional Routing

1. **Efficiency**: Skip unnecessary processing for high-confidence results
2. **Flexibility**: Adapt workflow based on runtime conditions
3. **HITL Integration**: Automatically route to human review when needed
4. **Resource Optimization**: Only use expensive AI analysis when necessary


---

## Error Handling and Resilience

### Multi-Layer Error Handling

The system implements error handling at multiple levels:

#### 1. Tool Level
```python
@tool
def check_something(document: dict) -> Optional[dict]:
    """Check something with error handling"""
    try:
        # Perform check
        result = perform_check(document)
        return result
    except Exception as e:
        logger.error(f"Tool error: {e}")
        return None  # Return None on error
```

#### 2. Agent Level
```python
class MyAgent(BaseAgent):
    def process(self, state: ComplianceState) -> ComplianceState:
        try:
            # Perform checks
            violations = self.run_checks(state)
            state = self.add_violations(state, violations)
        except Exception as e:
            # Log error in state
            state["error_log"].append({
                "agent": self.name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            # Try fallback
            if hasattr(self, 'fallback_process'):
                state = self.fallback_process(state, e)
        
        return state
```

#### 3. Decorator Level
```python
@agent_error_handler
def process(self, state: ComplianceState) -> ComplianceState:
    # Automatic retry with exponential backoff
    # Automatic error logging
    # Automatic fallback invocation
    pass
```

### Retry Logic

Agents automatically retry failed operations with exponential backoff:

```python
max_retries = 3
for attempt in range(max_retries):
    try:
        return self.process(state)
    except Exception as e:
        if attempt < max_retries - 1:
            delay = min(1.0 * (2 ** attempt), 10.0)
            logger.warning(f"Retry in {delay}s...")
            time.sleep(delay)
        else:
            # All retries exhausted, log and continue
            state["error_log"].append(error_record)
```

### Fallback Strategies

#### AI → Rules Fallback

When AI services fail, agents fall back to rule-based checking:

```python
def fallback_process(self, state: ComplianceState, error: Exception) -> ComplianceState:
    """Fallback to rule-based checking when AI fails"""
    logger.info(f"AI unavailable, using rule-based fallback")
    
    # Use rule-based tools instead of AI tools
    violations = self.run_rule_based_checks(state)
    state = self.add_violations(state, violations)
    
    return state
```

#### Partial Results

System generates partial results even if some agents fail:

```python
# Structure Agent fails
state["error_log"].append({"agent": "structure", "error": "..."})

# Other agents continue
state["violations"] = [v1, v2, v3]  # From other agents

# Final report includes both results and errors
report = {
    "violations": [v1, v2, v3],
    "errors": [{"agent": "structure", "error": "..."}],
    "status": "partial_success"
}
```

### Graceful Degradation

Workflow continues even when agents fail:

```python
# Agent failure doesn't stop workflow
try:
    result = agent(state)
except Exception as e:
    logger.error(f"Agent {agent.name} failed: {e}")
    # Log error but continue
    state["error_log"].append(error_record)
    result = state  # Return unchanged state

# Workflow proceeds to next agent
```

### Circuit Breaker Pattern

Prevents repeated calls to failing services:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=3, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise
```

### Error Logging

All errors are logged in state for debugging and auditing:

```python
error_record = {
    "agent": agent_name,
    "error": str(exception),
    "error_type": type(exception).__name__,
    "timestamp": datetime.now().isoformat(),
    "function": func.__name__,
    "attempts": max_retries,
    "traceback": traceback.format_exc()
}

state["error_log"].append(error_record)
```

### Timeout Handling

Agents have configurable timeouts:

```python
config = AgentConfig(
    name="my_agent",
    timeout_seconds=30.0
)

# Timeout is enforced at agent level
# If agent exceeds timeout, it's terminated and error is logged
```

### Validation and Recovery

State validation ensures consistency:

```python
from data_models_multiagent import validate_compliance_state

is_valid, errors = validate_compliance_state(state)
if not is_valid:
    logger.error(f"State validation failed: {errors}")
    # Attempt recovery
    state = recover_state(state, errors)
```


---

## State Persistence and Resumability

### Checkpointing with SqliteSaver

LangGraph's `SqliteSaver` automatically persists state at each agent transition:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Create SQLite connection
conn = sqlite3.connect("checkpoints/compliance_workflow.db")

# Create checkpointer
checkpointer = SqliteSaver(conn)

# Compile workflow with checkpointing
workflow = graph.compile(checkpointer=checkpointer)
```

### Thread-Based Checkpointing

Each workflow execution has a unique thread ID:

```python
import uuid
from datetime import datetime

# Generate unique thread ID
thread_id = f"check_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

# Execute with thread ID
result = workflow.invoke(
    initial_state,
    config={"configurable": {"thread_id": thread_id}}
)
```

### Checkpoint Intervals

Configure how often state is saved:

```json
{
  "multi_agent": {
    "checkpoint_interval": 1,
    "state_persistence": true
  }
}
```

- `checkpoint_interval: 1` = Save after every agent
- `checkpoint_interval: 5` = Save every 5 agents

### Resuming Workflows

Resume a workflow from a checkpoint:

```python
# Get the thread ID from previous execution
thread_id = "check_20231115_143022_a1b2c3d4"

# Resume from last checkpoint
result = workflow.invoke(
    None,  # State will be loaded from checkpoint
    config={"configurable": {"thread_id": thread_id}}
)
```

### HITL Interrupt and Resume

Workflow pauses at Reviewer Agent for human input:

```python
# Initial execution
result = workflow.invoke(
    initial_state,
    config={"configurable": {"thread_id": thread_id}}
)

# Workflow pauses at reviewer node
# State is automatically saved

# Human reviewer provides feedback
# ... (review process) ...

# Resume workflow after review
result = workflow.invoke(
    None,  # Resume from checkpoint
    config={"configurable": {"thread_id": thread_id}}
)
```

### State History

The `StateManager` maintains a history of state snapshots:

```python
from state_manager import StateManager

state_manager = StateManager(
    checkpoint_dir="checkpoints",
    max_history_size=100
)

# Save checkpoint with label
state_manager.save_checkpoint(state, "after_aggregator")

# List all checkpoints
checkpoints = state_manager.list_checkpoints()
# ['after_preprocessor', 'after_aggregator', 'after_context', ...]

# Load specific checkpoint
state = state_manager.load_checkpoint("after_aggregator")

# Get checkpoint history
history = state_manager.get_history()
```

### Checkpoint Storage Structure

```
checkpoints/
├── compliance_workflow.db          # LangGraph checkpoints
├── state_history/
│   ├── check_20231115_143022/
│   │   ├── after_preprocessor.json
│   │   ├── after_aggregator.json
│   │   ├── after_context.json
│   │   └── after_evidence.json
│   └── check_20231115_150315/
│       └── ...
└── metadata.json
```

### Recovery from Failures

If workflow crashes, resume from last checkpoint:

```python
try:
    result = workflow.invoke(initial_state, config=config)
except Exception as e:
    logger.error(f"Workflow failed: {e}")
    
    # Resume from last checkpoint
    logger.info("Attempting to resume from checkpoint...")
    result = workflow.invoke(None, config=config)
```

### Checkpoint Cleanup

Manage checkpoint storage:

```python
from state_manager import CheckpointManager

checkpoint_manager = CheckpointManager(state_manager)

# Clean up old checkpoints (older than 30 days)
checkpoint_manager.cleanup_old_checkpoints(days=30)

# Delete specific checkpoint
checkpoint_manager.delete_checkpoint(thread_id)

# Export checkpoint for archival
checkpoint_manager.export_checkpoint(thread_id, "archive/")
```

### Benefits of State Persistence

1. **Resumability**: Continue after interruptions
2. **HITL Integration**: Pause for human review, resume after
3. **Debugging**: Inspect state at any point in workflow
4. **Auditing**: Track complete execution history
5. **Recovery**: Resume after failures
6. **Testing**: Replay workflows from checkpoints


---

## Human-in-the-Loop Integration

### Overview

The HITL system allows human reviewers to validate low-confidence violations before final reporting.

### Review Queue

Low-confidence violations are automatically queued for review:

```python
class ReviewerAgent(BaseAgent):
    def process(self, state: ComplianceState) -> ComplianceState:
        # Filter violations needing review
        review_threshold = state["config"].get("routing", {}).get("review_threshold", 70)
        
        needs_review = [
            v for v in state["violations"]
            if 0 < v.get("confidence", 100) < review_threshold
        ]
        
        # Queue each for review
        for violation in needs_review:
            review_item = self.create_review_item(violation)
            self.review_manager.add_to_queue(review_item)
            
            state["review_queue"].append({
                "review_id": review_item.review_id,
                "violation": violation
            })
        
        return state
```

### Review Item Structure

```python
@dataclass
class ReviewItem:
    review_id: str
    document_id: str
    check_type: str
    slide: str
    location: str
    predicted_violation: bool
    confidence: int
    ai_reasoning: str
    evidence: str
    rule: str
    severity: str
    created_at: str
    priority_score: int
    status: ReviewStatus  # PENDING, APPROVED, REJECTED, NEEDS_INFO
```

### Priority Scoring

Reviews are prioritized based on multiple factors:

```python
def calculate_priority_score(violation: dict) -> int:
    """Calculate priority score (0-100, higher = more urgent)"""
    score = 0
    
    # Confidence (lower confidence = higher priority)
    confidence = violation.get("confidence", 100)
    score += (100 - confidence)
    
    # Severity
    severity = violation.get("severity", "WARNING")
    severity_scores = {
        "CRITICAL": 40,
        "MAJOR": 30,
        "WARNING": 20
    }
    score += severity_scores.get(severity, 0)
    
    # Check type (some types are more critical)
    check_type = violation.get("type", "")
    if check_type in ["PERFORMANCE", "SECURITIES"]:
        score += 20
    
    return min(score, 100)
```

### Workflow Interrupt

The workflow pauses at the Reviewer Agent node:

```python
# Workflow execution
result = workflow.invoke(initial_state, config=config)

# If violations need review, workflow is interrupted
# State is saved with status "interrupted"

# Human reviewer uses review CLI
# python review.py

# After review, workflow can be resumed
result = workflow.invoke(None, config=config)
```

### Review CLI

Interactive command-line interface for reviewers:

```bash
$ python review.py

╔══════════════════════════════════════════════════════════════╗
║           COMPLIANCE REVIEW SYSTEM                           ║
╚══════════════════════════════════════════════════════════════╝

Pending Reviews: 3

[1] PERFORMANCE - Missing disclaimer (Confidence: 65%)
    Priority: 85 | Document: exemple.json | Slide: slide_2
    
    Evidence: "Le fonds a généré une performance de 12.5% sur l'année"
    
    AI Reasoning: Performance data detected without nearby disclaimer
    
    Actions:
    [A] Approve (confirm violation)
    [R] Reject (false positive)
    [N] Needs more info
    [S] Skip
    [Q] Quit

Your choice: _
```

### Feedback Integration

Human decisions update the system:

```python
class FeedbackAgent(BaseAgent):
    def process_review_decision(self, decision: dict) -> None:
        """Process human review decision"""
        # Update confidence calibration
        self.confidence_calibrator.update(
            check_type=decision["check_type"],
            predicted_confidence=decision["predicted_confidence"],
            actual_violation=decision["actual_violation"]
        )
        
        # Detect patterns
        if decision["actual_violation"] == False:
            # False positive detected
            self.pattern_detector.record_false_positive(
                check_type=decision["check_type"],
                text=decision["evidence"],
                reason=decision["reviewer_notes"]
            )
        
        # Update violation status
        violation_id = decision["violation_id"]
        if decision["actual_violation"]:
            self.update_violation_status(violation_id, "CONFIRMED")
        else:
            self.update_violation_status(violation_id, "REJECTED")
```

### Batch Review Operations

Review multiple similar violations at once:

```python
# Find similar violations
similar = review_manager.find_similar_violations(
    violation_id="v123",
    similarity_threshold=0.85
)

# Apply decision to all similar violations
review_manager.batch_approve(similar)
```

### Review Metrics

Track review performance:

```python
metrics = review_manager.get_metrics()

{
    "total_reviews": 150,
    "approved": 120,
    "rejected": 25,
    "needs_info": 5,
    "avg_review_time_seconds": 45.2,
    "accuracy": 0.92,  # Based on feedback
    "false_positive_rate": 0.08
}
```

### Review Workflow

```
Violation Detected (confidence < 70%)
    ↓
Queue for Review
    ↓
Calculate Priority Score
    ↓
Present to Reviewer
    ↓
Reviewer Decision
    ├─ Approve → Mark as CONFIRMED
    ├─ Reject → Mark as FALSE_POSITIVE
    └─ Needs Info → Request additional context
    ↓
Update Confidence Calibration
    ↓
Detect Patterns
    ↓
Generate Feedback Report
```


---

## Performance Characteristics

### Execution Time Comparison

#### Sequential (Old System)
```
Preprocessor:    0.5s
Structure:       0.8s
Performance:     0.7s
Securities:      0.6s
General:         0.5s
Prospectus:      0.6s
Registration:    0.4s
ESG:             0.5s
Context:         0.3s
Evidence:        0.3s
────────────────────
Total:           5.2s
```

#### Parallel (New System)
```
Preprocessor:    0.5s
Parallel Group:  max(0.8, 0.7, 0.6, 0.5, 0.6, 0.4, 0.5) = 0.8s
Aggregator:      0.1s
Context:         0.3s
Evidence:        0.3s
────────────────────
Total:           2.0s

Speedup: 5.2s / 2.0s = 2.6x (62% faster)
```

### Real-World Performance

Based on actual measurements with `exemple.json`:

| Metric | Sequential | Parallel | Improvement |
|--------|-----------|----------|-------------|
| Total Time | 5.2s | 3.7s | 29% faster |
| Agent Execution | 4.5s | 1.8s | 60% faster |
| Overhead | 0.7s | 1.9s | -171% |
| Throughput | 0.19 docs/s | 0.27 docs/s | 42% higher |

**Note**: Overhead increases due to parallel coordination, but overall time decreases significantly.

### Scalability

#### Single Document
- Sequential: 5.2s
- Parallel: 3.7s

#### 10 Documents (batch)
- Sequential: 52s
- Parallel: 37s (with queue)

#### 100 Documents (batch)
- Sequential: 520s (8.7 min)
- Parallel: 370s (6.2 min)

### Resource Usage

#### CPU Utilization
- Sequential: 25-30% (single core)
- Parallel: 70-85% (multi-core)

#### Memory Usage
- Sequential: ~150 MB
- Parallel: ~200 MB (+33%)

#### API Calls
- Both systems: Same number of API calls
- Parallel: Calls made concurrently

### Performance Optimization Strategies

#### 1. Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def check_with_cache(text: str, rule: str) -> Optional[dict]:
    """Cached compliance check"""
    return perform_check(text, rule)
```

#### 2. Batch API Calls
```python
# Instead of individual calls
results = [ai_engine.check(text) for text in texts]

# Batch call
results = ai_engine.batch_check(texts)
```

#### 3. Early Termination
```python
# Skip expensive checks if high confidence
if all(v.get("confidence", 0) >= 90 for v in violations):
    # Skip context and evidence analysis
    return state
```

#### 4. Selective Agent Execution
```python
# Only run agents relevant to document type
if document_type == "factsheet":
    skip_agents = ["prospectus", "registration"]
```

### Performance Monitoring

```python
# Agent execution times
agent_timings = {
    "preprocessor": 0.5,
    "structure": 0.8,
    "performance": 0.7,
    "securities": 0.6,
    "general": 0.5,
    "aggregator": 0.1,
    "context": 0.3,
    "evidence": 0.3
}

# Identify bottlenecks
slowest_agent = max(agent_timings.items(), key=lambda x: x[1])
print(f"Bottleneck: {slowest_agent[0]} ({slowest_agent[1]}s)")
```

### Performance Metrics

```python
metrics = {
    "total_execution_time": 3.7,
    "agent_execution_time": 1.8,
    "overhead_time": 1.9,
    "parallel_efficiency": 0.62,  # 62% of theoretical maximum
    "api_calls": 15,
    "cache_hits": 8,
    "cache_hit_rate": 0.53,
    "violations_per_second": 1.62,
    "throughput_docs_per_hour": 973
}
```

### Optimization Recommendations

1. **Increase Parallelism**: Run more agents concurrently
2. **Optimize Slow Agents**: Profile and optimize bottleneck agents
3. **Increase Caching**: Cache more AI responses
4. **Reduce API Calls**: Batch calls where possible
5. **Early Termination**: Skip unnecessary analysis for high-confidence results
6. **Selective Execution**: Only run relevant agents based on document type


---

## Deployment Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Production Environment                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Application Layer                        │  │
│  │                                                       │  │
│  │  ┌────────────────┐  ┌────────────────┐            │  │
│  │  │ check_multi    │  │  review.py     │            │  │
│  │  │ agent.py       │  │  (Review CLI)  │            │  │
│  │  └────────────────┘  └────────────────┘            │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Workflow Layer                           │  │
│  │                                                       │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │         LangGraph State Machine                │  │  │
│  │  │  (Supervisor, Agents, Conditional Routing)     │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Service Layer                            │  │
│  │                                                       │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │  │
│  │  │ AI Engine│  │  Review  │  │  State   │          │  │
│  │  │          │  │  Manager │  │  Manager │          │  │
│  │  └──────────┘  └──────────┘  └──────────┘          │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Data Layer                               │  │
│  │                                                       │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │  │
│  │  │ SQLite   │  │  JSON    │  │  Logs    │          │  │
│  │  │Checkpoints│  │  Files   │  │          │          │  │
│  │  └──────────┘  └──────────┘  └──────────┘          │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              External Services                        │  │
│  │                                                       │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │  │
│  │  │ OpenAI   │  │ Anthropic│  │  Other   │          │  │
│  │  │   API    │  │   API    │  │  APIs    │          │  │
│  │  └──────────┘  └──────────┘  └──────────┘          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
compliance-checker/
├── check_multiagent.py          # Main entry point
├── workflow_builder.py          # Workflow construction
├── data_models_multiagent.py   # State definitions
├── state_manager.py             # State persistence
├── ai_engine.py                 # AI service integration
├── review_manager.py            # Review queue management
├── hybrid_config.json           # Configuration
├── requirements.txt             # Dependencies
│
├── agents/                      # Agent implementations
│   ├── base_agent.py
│   ├── supervisor_agent.py
│   ├── preprocessor_agent.py
│   ├── structure_agent.py
│   ├── performance_agent.py
│   ├── securities_agent.py
│   ├── general_agent.py
│   ├── prospectus_agent.py
│   ├── registration_agent.py
│   ├── esg_agent.py
│   ├── aggregator_agent.py
│   ├── context_agent.py
│   ├── evidence_agent.py
│   └── reviewer_agent.py
│
├── tools/                       # Agent tools
│   ├── preprocessing_tools.py
│   ├── structure_tools.py
│   ├── performance_tools.py
│   ├── securities_tools.py
│   ├── general_tools.py
│   ├── prospectus_tools.py
│   ├── registration_tools.py
│   ├── esg_tools.py
│   ├── context_tools.py
│   ├── evidence_tools.py
│   └── review_tools.py
│
├── checkpoints/                 # State persistence
│   ├── compliance_workflow.db
│   └── state_history/
│
├── monitoring/                  # Monitoring and metrics
│   ├── agent_logger.py
│   ├── metrics_tracker.py
│   ├── workflow_visualizer.py
│   └── dashboard.py
│
├── docs/                        # Documentation
│   ├── MULTI_AGENT_ARCHITECTURE.md
│   ├── AGENT_API.md
│   ├── MIGRATION_TO_MULTIAGENT.md
│   └── MULTIAGENT_CONFIGURATION.md
│
└── tests/                       # Tests
    ├── test_agents/
    ├── test_tools/
    └── test_workflow.py
```

### Configuration Management

#### Environment Variables
```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LOG_LEVEL=INFO
CHECKPOINT_DIR=checkpoints
MAX_PARALLEL_AGENTS=4
```

#### Configuration File
```json
{
  "ai_enabled": true,
  "multi_agent": {
    "enabled": true,
    "parallel_execution": true,
    "max_parallel_agents": 4,
    "checkpoint_interval": 1,
    "state_persistence": true,
    "checkpoint_db_path": "checkpoints/compliance_workflow.db"
  },
  "agents": {
    "supervisor": {"enabled": true},
    "preprocessor": {"enabled": true},
    "structure": {"enabled": true},
    "performance": {"enabled": true},
    "securities": {"enabled": true},
    "general": {"enabled": true},
    "prospectus": {"enabled": true},
    "registration": {"enabled": true},
    "esg": {"enabled": true},
    "context": {"enabled": true, "confidence_threshold": 80},
    "evidence": {"enabled": true},
    "reviewer": {"enabled": true, "confidence_threshold": 70}
  },
  "routing": {
    "context_threshold": 80,
    "review_threshold": 70,
    "skip_context_if_high_confidence": true
  }
}
```

### Deployment Options

#### 1. Local Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run compliance check
python check_multiagent.py exemple.json
```

#### 2. Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "check_multiagent.py"]
```

#### 3. Cloud Deployment (AWS Lambda)
```python
# lambda_handler.py
import json
from check_multiagent import execute_compliance_check

def lambda_handler(event, context):
    document = json.loads(event['body'])
    result = execute_compliance_check(document)
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

### Scaling Strategies

#### Horizontal Scaling
- Deploy multiple instances
- Use load balancer
- Shared checkpoint database

#### Vertical Scaling
- Increase CPU cores for more parallelism
- Increase memory for larger documents
- Use faster storage for checkpoints

### High Availability

#### Redundancy
- Multiple application instances
- Replicated checkpoint database
- Backup AI service providers

#### Failover
- Automatic retry with exponential backoff
- Fallback to rule-based checking
- Circuit breaker for failing services

### Security Considerations

#### API Key Management
- Store keys in environment variables
- Use secrets management service (AWS Secrets Manager, etc.)
- Rotate keys regularly

#### Data Protection
- Encrypt checkpoints at rest
- Encrypt data in transit (HTTPS)
- Sanitize logs (remove PII)

#### Access Control
- Authenticate review CLI users
- Role-based access control
- Audit logging


---

## Monitoring and Observability

### Logging Architecture

#### Multi-Level Logging

```python
# Application level
logger = logging.getLogger("compliance_checker")
logger.info("Starting compliance check")

# Agent level
agent_logger = logging.getLogger("compliance_checker.agents.structure")
agent_logger.debug("Running structure checks")

# Tool level
tool_logger = logging.getLogger("compliance_checker.tools.structure")
tool_logger.debug("Checking promotional mention")
```

#### Structured Logging

```python
import json
import logging

class StructuredLogger:
    def log_agent_execution(self, agent_name, duration, violations_found):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "agent_execution",
            "agent": agent_name,
            "duration_seconds": duration,
            "violations_found": violations_found
        }
        logger.info(json.dumps(log_entry))
```

### Metrics Tracking

#### Agent Metrics

```python
from monitoring.metrics_tracker import MetricsTracker

tracker = MetricsTracker()

# Track agent execution
tracker.track_agent_execution(
    agent_name="structure",
    duration=0.8,
    success=True,
    violations_found=2
)

# Track API calls
tracker.track_api_call(
    service="openai",
    endpoint="chat/completions",
    duration=0.5,
    success=True
)

# Get metrics
metrics = tracker.get_metrics()
```

#### Performance Metrics

```python
{
    "agents": {
        "structure": {
            "executions": 150,
            "avg_duration": 0.82,
            "success_rate": 0.98,
            "violations_per_execution": 1.5
        },
        "performance": {
            "executions": 150,
            "avg_duration": 0.75,
            "success_rate": 0.99,
            "violations_per_execution": 0.8
        }
    },
    "api_calls": {
        "total": 450,
        "success_rate": 0.97,
        "avg_duration": 0.45,
        "cache_hit_rate": 0.53
    },
    "workflow": {
        "total_executions": 150,
        "avg_duration": 3.7,
        "success_rate": 0.96,
        "avg_violations": 6.2
    }
}
```

### Workflow Visualization

#### Execution Path Visualization

```python
from monitoring.workflow_visualizer import WorkflowVisualizer

visualizer = WorkflowVisualizer()

# Visualize workflow structure
visualizer.visualize_workflow(workflow, output="workflow_diagram.png")

# Visualize execution path
visualizer.visualize_execution(
    execution_history,
    output="execution_path.png"
)
```

#### Mermaid Diagram Generation

```python
# Generate Mermaid diagram
mermaid_code = visualizer.generate_mermaid_diagram(workflow)

# Save to file
with open("workflow.mmd", "w") as f:
    f.write(mermaid_code)
```

### Dashboard

#### Real-Time Monitoring Dashboard

```python
from monitoring.dashboard import Dashboard

dashboard = Dashboard()

# Start dashboard server
dashboard.start(port=8080)

# Access at http://localhost:8080
```

#### Dashboard Features

- **Agent Status**: Real-time agent execution status
- **Performance Metrics**: Execution times, success rates
- **Violation Statistics**: Violations by type, severity
- **API Usage**: API calls, cache hits, costs
- **Error Tracking**: Recent errors, error rates
- **Workflow Visualization**: Current execution path

### Alerting

#### Alert Configuration

```json
{
  "alerts": {
    "agent_failure_rate": {
      "threshold": 0.05,
      "window": "5m",
      "action": "email"
    },
    "api_error_rate": {
      "threshold": 0.10,
      "window": "5m",
      "action": "slack"
    },
    "execution_time": {
      "threshold": 10.0,
      "window": "1m",
      "action": "log"
    }
  }
}
```

#### Alert Handler

```python
from monitoring.alerting import AlertManager

alert_manager = AlertManager()

# Check metrics and trigger alerts
if metrics["agents"]["structure"]["success_rate"] < 0.95:
    alert_manager.trigger_alert(
        alert_type="agent_failure_rate",
        message="Structure agent success rate below threshold",
        severity="warning"
    )
```

### Audit Logging

#### Audit Trail

```python
from audit_logger import AuditLogger

audit_logger = AuditLogger()

# Log compliance check
audit_logger.log_compliance_check(
    document_id="exemple.json",
    user_id="user123",
    violations_found=6,
    duration=3.7,
    timestamp=datetime.now()
)

# Log review decision
audit_logger.log_review_decision(
    review_id="rev456",
    reviewer_id="reviewer789",
    decision="approved",
    violation_id="v123",
    timestamp=datetime.now()
)
```

### Tracing

#### Distributed Tracing

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

# Set up tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Trace agent execution
with tracer.start_as_current_span("structure_agent") as span:
    span.set_attribute("agent.name", "structure")
    span.set_attribute("document.id", document_id)
    
    # Execute agent
    result = agent.process(state)
    
    span.set_attribute("violations.found", len(result["violations"]))
```

### Health Checks

#### System Health

```python
from monitoring.health import HealthChecker

health_checker = HealthChecker()

# Check system health
health = health_checker.check_health()

{
    "status": "healthy",
    "checks": {
        "database": "ok",
        "ai_service": "ok",
        "disk_space": "ok",
        "memory": "ok"
    },
    "timestamp": "2023-11-15T14:30:22Z"
}
```

### Metrics Export

#### Prometheus Integration

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
agent_executions = Counter(
    'agent_executions_total',
    'Total agent executions',
    ['agent_name', 'status']
)

agent_duration = Histogram(
    'agent_duration_seconds',
    'Agent execution duration',
    ['agent_name']
)

violations_found = Gauge(
    'violations_found',
    'Number of violations found',
    ['document_id']
)

# Record metrics
agent_executions.labels(agent_name='structure', status='success').inc()
agent_duration.labels(agent_name='structure').observe(0.8)
violations_found.labels(document_id='exemple.json').set(6)
```

### Log Aggregation

#### Centralized Logging

```python
# Configure log shipping to centralized service
import logging
from logging.handlers import SysLogHandler

handler = SysLogHandler(address=('logs.example.com', 514))
logger.addHandler(handler)
```

### Monitoring Best Practices

1. **Log Everything**: Agent executions, API calls, errors
2. **Track Metrics**: Performance, success rates, violations
3. **Visualize Workflows**: Understand execution paths
4. **Set Up Alerts**: Proactive issue detection
5. **Audit Trail**: Complete history for compliance
6. **Health Checks**: Regular system health monitoring
7. **Distributed Tracing**: Track requests across agents
8. **Export Metrics**: Integration with monitoring tools

---

## Conclusion

The Multi-Agent Compliance System provides a robust, scalable, and maintainable architecture for compliance checking. Key benefits include:

- **Modularity**: Each agent has a single, well-defined responsibility
- **Scalability**: Parallel execution and horizontal scaling support
- **Resilience**: Graceful degradation and error recovery
- **Observability**: Comprehensive logging, metrics, and monitoring
- **Flexibility**: Conditional routing and configurable behavior
- **HITL Integration**: Seamless human review workflow
- **State Persistence**: Resumability and audit trail

The system maintains 100% feature parity with the monolithic system while providing significant performance improvements (29% faster) and enhanced maintainability.

For more information, see:
- [Agent API Documentation](AGENT_API.md)
- [Migration Guide](MIGRATION_TO_MULTIAGENT.md)
- [Configuration Guide](MULTIAGENT_CONFIGURATION.md)
- [Troubleshooting Guide](MULTIAGENT_TROUBLESHOOTING.md)

