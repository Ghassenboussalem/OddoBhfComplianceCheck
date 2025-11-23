# Multi-Agent System API Documentation

## Overview

This document provides comprehensive API documentation for all agents and tools in the Multi-Agent Compliance System. The system uses LangGraph to orchestrate specialized agents that collaborate to perform compliance checking on financial documents.

## Table of Contents

1. [State Structure](#state-structure)
2. [Base Agent Framework](#base-agent-framework)
3. [Specialized Agents](#specialized-agents)
4. [Tools](#tools)
5. [Usage Examples](#usage-examples)

---

## State Structure

### ComplianceState

The `ComplianceState` is a TypedDict that flows through all agents in the LangGraph workflow. Each agent receives this state, performs operations, and returns an updated state.

```python
class ComplianceState(TypedDict, total=False):
    # Document data
    document: dict                    # Original document
    document_id: str                  # Unique document identifier
    document_type: str                # Type (e.g., "fund_presentation")
    client_type: str                  # "retail" or "professional"
    
    # Preprocessing results
    metadata: Dict[str, Any]          # Extracted metadata
    whitelist: Set[str]               # Allowed terms
    normalized_document: dict         # Normalized document structure
    
    # Violations (automatically merged from parallel agents)
    violations: Annotated[Sequence[dict], operator.add]
    
    # Context analysis
    context_analysis: Dict[str, dict]
    intent_classifications: Dict[str, dict]
    evidence_extractions: Dict[str, dict]
    
    # Review and feedback
    review_queue: List[dict]
    feedback_history: List[dict]
    
    # Confidence scoring
    confidence_scores: Dict[str, int]
    aggregated_confidence: int
    
    # Workflow control
    current_agent: str
    next_action: str
    workflow_status: str
    execution_plan: List[str]
    error_log: List[dict]
    
    # Performance metrics
    agent_timings: Dict[str, float]
    api_calls: int
    cache_hits: int
    
    # Configuration
    config: dict
    
    # Timestamps
    started_at: str
    updated_at: str
    completed_at: Optional[str]
```


### Helper Functions

#### initialize_compliance_state

```python
def initialize_compliance_state(
    document: dict,
    document_id: str,
    config: dict
) -> ComplianceState
```

Initialize a new ComplianceState with default values.

**Parameters:**
- `document`: The document to check for compliance
- `document_id`: Unique identifier for the document
- `config`: Configuration dictionary

**Returns:** Initialized ComplianceState ready for workflow execution

---

## Base Agent Framework

### BaseAgent

Abstract base class that all specialized agents inherit from. Provides standard interface, error handling, timing, and logging.

#### Constructor

```python
def __init__(
    self,
    config: Optional[AgentConfig] = None,
    **kwargs
)
```

**Parameters:**
- `config`: AgentConfig instance with agent settings
- `**kwargs`: Additional configuration options
  - `fail_fast`: If True, raise exceptions immediately (default: False)

#### Main Interface

```python
def __call__(self, state: ComplianceState) -> ComplianceState
```

Main entry point called by LangGraph. Wraps `process()` with timing and error handling.

**Parameters:**
- `state`: Current compliance state

**Returns:** Updated compliance state

#### Abstract Method

```python
@abstractmethod
def process(self, state: ComplianceState) -> ComplianceState
```

Must be implemented by all subclasses. Contains the agent's core logic.

#### Utility Methods

```python
def add_violation(self, state: ComplianceState, violation: Dict[str, Any]) -> ComplianceState
```
Add a single violation to the state.

```python
def add_violations(self, state: ComplianceState, violations: List[Dict[str, Any]]) -> ComplianceState
```
Add multiple violations to the state.

```python
def get_config_value(self, key: str, default: Any = None) -> Any
```
Get a configuration value with fallback.

```python
def log_execution_stats(self)
```
Log execution statistics for the agent.

```python
def fallback_process(self, state: ComplianceState, error: Exception) -> ComplianceState
```
Fallback processing when main process fails. Can be overridden by subclasses.

### AgentConfig

Configuration dataclass for agents.

```python
@dataclass
class AgentConfig:
    name: str
    enabled: bool = True
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    log_level: str = "INFO"
    custom_settings: Dict[str, Any] = field(default_factory=dict)
```

### Decorators

#### @agent_timing

Measures and logs agent execution time.

```python
@agent_timing
def process(self, state: ComplianceState) -> ComplianceState:
    # Agent logic here
    pass
```

#### @agent_error_handler

Handles errors gracefully with retry logic and fallback.

```python
@agent_error_handler
def process(self, state: ComplianceState) -> ComplianceState:
    # Agent logic here
    pass
```

---

## Specialized Agents


### 1. SupervisorAgent

Orchestrates the entire compliance workflow, coordinates agents, handles failures, and generates final reports.

#### Constructor

```python
def __init__(
    self,
    config: Optional[AgentConfig] = None,
    **kwargs
)
```

**Additional kwargs:**
- `enable_parallel_execution`: Enable parallel agent execution (default: True)
- `max_parallel_agents`: Maximum concurrent agents (default: 4)
- `enable_conditional_routing`: Enable confidence-based routing (default: True)

#### Methods

```python
def process(self, state: ComplianceState) -> ComplianceState
```
Initialize workflow and create execution plan.

**Returns:** State with execution plan and initialized workflow

```python
def generate_final_report(self, state: ComplianceState) -> Dict[str, Any]
```
Generate comprehensive compliance report.

**Returns:** Dictionary with complete compliance report including:
- Summary statistics
- All violations categorized by status and severity
- Performance metrics
- Execution information

```python
def handle_agent_failure(
    self,
    state: ComplianceState,
    failed_agent: str,
    error: Exception
) -> ComplianceState
```
Handle failure of a specialist agent with graceful degradation.

**Example:**

```python
from agents.supervisor_agent import SupervisorAgent
from agents.base_agent import AgentConfig

config = AgentConfig(name="supervisor", timeout_seconds=60.0)
supervisor = SupervisorAgent(config=config)

# Initialize workflow
state = supervisor(initial_state)
print(f"Execution plan: {state['execution_plan']}")
```

---

### 2. PreprocessorAgent

Handles document preprocessing: metadata extraction, whitelist building, normalization, and validation.

#### Constructor

```python
def __init__(
    self,
    config: Optional[AgentConfig] = None,
    **kwargs
)
```

**Additional kwargs:**
- `validate_before_processing`: Run validation before processing (default: True)
- `fail_on_validation_errors`: Fail if validation errors found (default: False)

#### Methods

```python
def process(self, state: ComplianceState) -> ComplianceState
```
Preprocess document through validation, metadata extraction, normalization, and whitelist building.

**Returns:** State with:
- `metadata`: Extracted metadata
- `normalized_document`: Normalized document structure
- `whitelist`: Set of allowed terms
- `next_action`: "parallel_checks"

**Example:**

```python
from agents.preprocessor_agent import PreprocessorAgent

agent = PreprocessorAgent()
result = agent(state)

print(f"Metadata: {result['metadata']}")
print(f"Whitelist size: {len(result['whitelist'])}")
print(f"Document sections: {result['normalized_document'].keys()}")
```

---

### 3. StructureAgent

Handles structure compliance checks: promotional mention, target audience, management company, fund name, dates.

#### Constructor

```python
def __init__(
    self,
    config: Optional[AgentConfig] = None,
    **kwargs
)
```

**Additional kwargs:**
- `parallel_execution`: Execute checks in parallel (default: True)
- `max_workers`: Maximum parallel workers (default: 5)

#### Methods

```python
def process(self, state: ComplianceState) -> ComplianceState
```
Execute all structure checks and return violations.

**Returns:** Partial state update with:
- `violations`: List of structure violations found
- `confidence_scores`: {"structure": confidence_score}

**Tools Used:**
- `check_promotional_mention`
- `check_target_audience`
- `check_management_company`
- `check_fund_name`
- `check_date_validation`

**Example:**

```python
from agents.structure_agent import StructureAgent

agent = StructureAgent(parallel_execution=True)
result = agent(state)

print(f"Structure violations: {len(result['violations'])}")
print(f"Confidence: {result['confidence_scores']['structure']}%")
```

---

### 4. PerformanceAgent

Handles performance compliance checks: disclaimers, document start, benchmark comparison, fund age restrictions.

#### Constructor

```python
def __init__(
    self,
    config: Optional[AgentConfig] = None,
    **kwargs
)
```

**Additional kwargs:**
- `parallel_execution`: Execute checks in parallel (default: True)
- `max_workers`: Maximum parallel workers (default: 4)

#### Methods

```python
def process(self, state: ComplianceState) -> ComplianceState
```
Execute all performance checks with evidence extraction.

**Returns:** Partial state update with:
- `violations`: List of performance violations (may be multiple per check)
- `confidence_scores`: {"performance": confidence_score}

**Tools Used:**
- `check_performance_disclaimers` (returns list)
- `check_document_starts_with_performance` (returns list)
- `check_benchmark_comparison`
- `check_fund_age_restrictions`

**Example:**

```python
from agents.performance_agent import PerformanceAgent

agent = PerformanceAgent()
result = agent(state)

for violation in result['violations']:
    print(f"Rule: {violation['rule']}")
    print(f"Evidence: {violation['evidence']}")
```

---


### 5. AggregatorAgent

Collects and consolidates violations from all specialist agents, calculates confidence scores, deduplicates results, and determines next workflow action.

#### Constructor

```python
def __init__(
    self,
    config: Optional[AgentConfig] = None,
    **kwargs
)
```

**Additional kwargs:**
- `context_threshold`: Confidence threshold for context analysis (default: 80)
- `review_threshold`: Confidence threshold for human review (default: 70)
- `deduplication_enabled`: Enable violation deduplication (default: True)
- `similarity_threshold`: Similarity threshold for deduplication (default: 0.85)

#### Methods

```python
def process(self, state: ComplianceState) -> ComplianceState
```
Aggregate violations, deduplicate, calculate confidence, and determine next action.

**Returns:** State with:
- `violations`: Deduplicated violations
- `aggregated_confidence`: Overall confidence score
- `next_action`: "context_analysis", "review", or "complete"
- `violation_categorization`: Violations grouped by type, severity, status

```python
def get_aggregation_statistics(self, state: ComplianceState) -> Dict[str, Any]
```
Get detailed aggregation statistics.

**Example:**

```python
from agents.aggregator_agent import AggregatorAgent

agent = AggregatorAgent(
    context_threshold=80,
    review_threshold=70,
    deduplication_enabled=True
)

result = agent(state)
print(f"Total violations: {len(result['violations'])}")
print(f"Aggregated confidence: {result['aggregated_confidence']}%")
print(f"Next action: {result['next_action']}")
```

---

### 6. ContextAgent

Analyzes text context and intent to eliminate false positives by understanding semantic meaning.

#### Constructor

```python
def __init__(
    self,
    config: Optional[AgentConfig] = None,
    ai_engine: Optional[AIEngine] = None,
    **kwargs
)
```

**Parameters:**
- `ai_engine`: AIEngine instance for LLM calls

**Additional kwargs:**
- `confidence_boost_threshold`: Minimum confidence for boosting (default: 70)
- `false_positive_threshold`: Threshold for filtering false positives (default: 85)
- `analyze_all_violations`: Analyze all violations regardless of confidence (default: False)

#### Methods

```python
def process(self, state: ComplianceState) -> ComplianceState
```
Analyze context for low-confidence violations and update confidence scores.

**Returns:** State with:
- `context_analysis`: Context analysis results by violation
- `intent_classifications`: Intent classifications by violation
- Updated `violations` with adjusted confidence scores

```python
def get_context_statistics(self, state: ComplianceState) -> Dict[str, Any]
```
Get context analysis statistics.

**Example:**

```python
from agents.context_agent import ContextAgent
from ai_engine import AIEngine

ai_engine = AIEngine(config)
agent = ContextAgent(ai_engine=ai_engine)

result = agent(state)
print(f"Analyzed: {len(result['context_analysis'])} violations")
print(f"False positives filtered: {sum(1 for v in result['violations'] if v['status'] == 'false_positive_filtered')}")
```

---

### 7. EvidenceAgent

Extracts and tracks evidence supporting compliance violations: quotes, performance data, disclaimers, locations.

#### Constructor

```python
def __init__(
    self,
    config: Optional[AgentConfig] = None,
    ai_engine: Optional[AIEngine] = None,
    **kwargs
)
```

**Parameters:**
- `ai_engine`: Optional AIEngine for semantic analysis

**Additional kwargs:**
- `min_confidence_for_evidence`: Minimum confidence to extract evidence (default: 0)
- `max_violations_to_process`: Maximum violations to process (default: 50)
- `enhance_all_violations`: Enhance all violations with evidence (default: False)

#### Methods

```python
def process(self, state: ComplianceState) -> ComplianceState
```
Extract evidence for violations needing enhancement.

**Returns:** State with:
- `evidence_extractions`: Evidence extraction results by violation
- Updated `violations` with evidence fields:
  - `evidence_quotes`: List of relevant quotes
  - `evidence_locations`: List of locations
  - `evidence_context`: Context description
  - `evidence_confidence`: Confidence in evidence

**Example:**

```python
from agents.evidence_agent import EvidenceAgent

agent = EvidenceAgent(ai_engine=ai_engine)
result = agent(state)

for violation in result['violations']:
    if 'evidence_quotes' in violation:
        print(f"Rule: {violation['rule']}")
        print(f"Quotes: {violation['evidence_quotes']}")
        print(f"Confidence: {violation['evidence_confidence']}%")
```

---

### 8. ReviewerAgent

Manages Human-in-the-Loop review process: queuing violations, priority scoring, filtering, batch operations.

#### Constructor

```python
def __init__(
    self,
    config: Optional[AgentConfig] = None,
    review_manager: Optional[ReviewManager] = None,
    **kwargs
)
```

**Parameters:**
- `review_manager`: ReviewManager instance for queue management

**Additional kwargs:**
- `review_threshold`: Confidence threshold for review (default: 70)
- `auto_queue_enabled`: Automatically queue violations (default: True)
- `batch_operations_enabled`: Enable batch operations (default: True)
- `hitl_interrupt_enabled`: Enable HITL interrupt (default: True)

#### Methods

```python
def process(self, state: ComplianceState) -> ComplianceState
```
Queue low-confidence violations for human review.

**Returns:** State with:
- `review_queue`: List of queued review items
- `batch_opportunities`: Identified batch processing opportunities
- `queue_statistics`: Current queue statistics
- `hitl_interrupt_required`: Flag for HITL interrupt

```python
def get_next_review_item(self, reviewer_id: str) -> Optional[Dict[str, Any]]
```
Get next highest-priority review item for a reviewer.

```python
def filter_reviews_by_criteria(
    self,
    check_type: Optional[str] = None,
    severity: Optional[str] = None,
    min_confidence: Optional[int] = None,
    max_confidence: Optional[int] = None,
    document_id: Optional[str] = None
) -> List[Dict[str, Any]]
```
Filter pending reviews by various criteria.

**Example:**

```python
from agents.reviewer_agent import ReviewerAgent
from review_manager import ReviewManager

review_manager = ReviewManager()
agent = ReviewerAgent(review_manager=review_manager)

result = agent(state)
print(f"Queued for review: {len(result['review_queue'])}")
print(f"Batch opportunities: {len(result.get('batch_opportunities', []))}")

# Get next review
next_item = agent.get_next_review_item("reviewer_001")
```

---


### 9. FeedbackAgent

Processes human feedback to improve system accuracy: confidence calibration, pattern detection, rule suggestions.

#### Constructor

```python
def __init__(
    self,
    config: Optional[AgentConfig] = None,
    feedback_interface: Optional[FeedbackInterface] = None,
    confidence_calibrator: Optional[ConfidenceCalibrator] = None,
    pattern_analyzer: Optional[PatternAnalyzer] = None,
    ai_engine: Optional[Any] = None,
    **kwargs
)
```

**Parameters:**
- `feedback_interface`: FeedbackInterface for accessing feedback data
- `confidence_calibrator`: ConfidenceCalibrator for calibration updates
- `pattern_analyzer`: PatternAnalyzer for pattern detection
- `ai_engine`: Optional AIEngine for advanced pattern analysis

**Additional kwargs:**
- `min_pattern_occurrences`: Minimum occurrences for pattern detection (default: 3)
- `min_impact_threshold`: Minimum impact for rule suggestions (default: 0.05)
- `auto_calibrate`: Automatically update calibration (default: True)
- `pattern_detection_enabled`: Enable pattern detection (default: True)

#### Methods

```python
def process(self, state: ComplianceState) -> ComplianceState
```
Process feedback and update learning models.

**Returns:** State with:
- `feedback_processing`: Processing summary
- `calibration_metrics`: Updated calibration metrics
- `discovered_patterns`: Newly discovered patterns
- `rule_suggestions`: Generated rule suggestions
- `learning_metrics`: Accuracy and learning metrics

```python
def get_learning_report(self) -> Dict[str, Any]
```
Get comprehensive learning report.

```python
def export_learning_data(self, base_path: str = ".")
```
Export all learning data for analysis.

**Example:**

```python
from agents.feedback_agent import FeedbackAgent
from feedback_loop import FeedbackInterface
from confidence_calibrator import ConfidenceCalibrator

feedback_interface = FeedbackInterface()
calibrator = ConfidenceCalibrator()

agent = FeedbackAgent(
    feedback_interface=feedback_interface,
    confidence_calibrator=calibrator,
    ai_engine=ai_engine
)

result = agent(state)
print(f"Feedback processed: {result['feedback_processing']['processed']}")
print(f"Patterns discovered: {result['feedback_processing']['patterns_discovered']}")
print(f"Rules suggested: {result['feedback_processing']['rules_suggested']}")

# Get learning report
report = agent.get_learning_report()
print(f"Accuracy improvement: {report['accuracy_improvement']:.1%}")
```

---

## Tools

### Preprocessing Tools

#### extract_metadata

```python
@tool
def extract_metadata(document: dict) -> dict
```

Extract metadata from document for compliance checking.

**Returns:** Dictionary with:
- `fund_isin`: Fund ISIN code
- `client_type`: "retail" or "professional"
- `document_type`: Document type
- `fund_name`: Fund name
- `esg_classification`: ESG classification
- `country_code`: Country code
- `fund_age_years`: Fund age in years
- `fund_status`: Fund status

#### build_whitelist

```python
@tool
def build_whitelist(document: dict, metadata: dict) -> Set[str]
```

Build whitelist of allowed terms to prevent false positives.

**Returns:** Set of whitelisted terms

#### normalize_document

```python
@tool
def normalize_document(document: dict) -> dict
```

Normalize document structure for consistent processing.

**Returns:** Normalized document with standard sections

#### validate_document

```python
@tool
def validate_document(document: dict) -> dict
```

Validate document structure and content.

**Returns:** Validation result with:
- `valid`: Boolean indicating if valid
- `errors`: List of validation errors
- `warnings`: List of warnings
- `sections_present`: Dictionary of section presence

---

### Structure Tools

#### check_promotional_mention

```python
@tool
def check_promotional_mention(document: dict, config: dict) -> Optional[dict]
```

Check for promotional document mention on cover page.

**Returns:** Violation dict if missing, None otherwise

#### check_target_audience

```python
@tool
def check_target_audience(document: dict, client_type: str) -> Optional[dict]
```

Check target audience specification.

**Returns:** Violation dict if missing, None otherwise

#### check_management_company

```python
@tool
def check_management_company(document: dict) -> Optional[dict]
```

Check management company legal mention.

**Returns:** Violation dict if missing, None otherwise

#### check_fund_name

```python
@tool
def check_fund_name(document: dict, metadata: dict) -> Optional[dict]
```

Check fund name presence and consistency.

**Returns:** Violation dict if issues found, None otherwise

#### check_date_validation

```python
@tool
def check_date_validation(document: dict) -> Optional[dict]
```

Validate document date.

**Returns:** Violation dict if invalid, None otherwise

---

### Performance Tools

#### check_performance_disclaimers

```python
@tool
def check_performance_disclaimers(document: dict, config: dict) -> List[dict]
```

Check that actual performance data has disclaimers (data-aware version).

**Returns:** List of violations (one per slide with performance data but no disclaimer)

#### check_document_starts_with_performance

```python
@tool
def check_document_starts_with_performance(document: dict, config: dict) -> List[dict]
```

Check if document starts with performance data.

**Returns:** List of violations if document starts with performance

#### check_benchmark_comparison

```python
@tool
def check_benchmark_comparison(document: dict, config: dict) -> Optional[dict]
```

Validate benchmark comparison when performance is shown.

**Returns:** Violation dict if benchmark missing, None otherwise

#### check_fund_age_restrictions

```python
@tool
def check_fund_age_restrictions(document: dict, metadata: dict) -> Optional[dict]
```

Check fund age restrictions for performance display.

**Returns:** Violation dict if fund too young, None otherwise

---


### Context Tools

#### analyze_context

```python
@tool
def analyze_context(text: str, check_type: str, ai_engine) -> ContextAnalysis
```

Analyze text context using AI to understand semantic meaning and intent.

**Parameters:**
- `text`: Text to analyze
- `check_type`: Type of check (e.g., "investment_advice", "general")
- `ai_engine`: AIEngine instance for LLM calls

**Returns:** ContextAnalysis with:
- `subject`: WHO performs the action ("fund", "client", "general")
- `intent`: WHAT the intent is ("describe", "advise", "state_fact")
- `is_fund_description`: Boolean
- `is_client_advice`: Boolean
- `confidence`: Confidence score (0-100)
- `reasoning`: AI reasoning
- `evidence`: Supporting evidence

#### classify_intent

```python
@tool
def classify_intent(text: str, ai_engine) -> IntentClassification
```

Classify text intent using AI.

**Returns:** IntentClassification with:
- `intent_type`: "ADVICE", "DESCRIPTION", "FACT", or "EXAMPLE"
- `confidence`: Confidence score (0-100)
- `subject`: Subject of the text
- `reasoning`: AI reasoning
- `evidence`: Supporting evidence

#### extract_subject

```python
@tool
def extract_subject(text: str) -> str
```

Extract the subject (WHO) from text.

**Returns:** Subject string ("fund", "client", "general", etc.)

#### is_fund_strategy_description

```python
@tool
def is_fund_strategy_description(text: str, ai_engine) -> bool
```

Determine if text describes fund strategy (ALLOWED).

**Returns:** True if fund strategy description

#### is_investment_advice

```python
@tool
def is_investment_advice(text: str, ai_engine) -> bool
```

Determine if text provides client investment advice (PROHIBITED).

**Returns:** True if investment advice

---

### Evidence Tools

#### extract_evidence

```python
@tool
def extract_evidence(text: str, violation_type: str, location: str = "") -> Evidence
```

Extract specific evidence for a compliance violation.

**Parameters:**
- `text`: Text to analyze
- `violation_type`: Type of violation (e.g., "performance_data", "missing_disclaimer")
- `location`: Location in document

**Returns:** Evidence with:
- `quotes`: List of relevant quotes
- `locations`: List of locations
- `context`: Context description
- `confidence`: Confidence score

#### find_performance_data

```python
@tool
def find_performance_data(text: str, ai_engine=None) -> List[PerformanceData]
```

Find actual performance data (numbers with percentages).

**Returns:** List of PerformanceData with:
- `value`: Performance value (e.g., "+15.5%")
- `context`: Surrounding context
- `location`: Location in text
- `confidence`: Confidence score

#### find_disclaimer

```python
@tool
def find_disclaimer(text: str, required_disclaimer: str, ai_engine=None) -> DisclaimerMatch
```

Find disclaimer using semantic matching.

**Parameters:**
- `text`: Text to search
- `required_disclaimer`: Required disclaimer text
- `ai_engine`: Optional AIEngine for semantic matching

**Returns:** DisclaimerMatch with:
- `found`: Boolean indicating if found
- `text`: Matched disclaimer text
- `location`: Location in text
- `similarity_score`: Similarity percentage
- `confidence`: Confidence score

#### track_location

```python
@tool
def track_location(text: str, search_text: str) -> List[str]
```

Track locations of text within document.

**Returns:** List of location strings

#### extract_quotes

```python
@tool
def extract_quotes(text: str, max_quotes: int = 3) -> List[str]
```

Extract relevant quotes from text.

**Returns:** List of quote strings

---

### Review Tools

#### queue_for_review

```python
@tool
def queue_for_review(
    violation: Dict[str, Any],
    review_manager: Any,
    document_id: str = "",
    confidence: int = 0,
    severity: str = "MEDIUM"
) -> str
```

Add a violation to the review queue for human review.

**Returns:** Review ID string

#### calculate_priority_score

```python
@tool
def calculate_priority_score(
    confidence: int,
    severity: str,
    age_hours: float
) -> float
```

Calculate priority score for review item.

**Returns:** Priority score (higher = more urgent)

#### filter_reviews

```python
@tool
def filter_reviews(
    review_manager: Any,
    check_type: Optional[str] = None,
    severity: Optional[str] = None,
    min_confidence: Optional[int] = None,
    max_confidence: Optional[int] = None,
    document_id: Optional[str] = None
) -> List[Dict[str, Any]]
```

Filter pending reviews by criteria.

**Returns:** List of filtered review items

#### batch_review_by_check_type

```python
@tool
def batch_review_by_check_type(
    review_manager: Any,
    check_type: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]
```

Get batch of reviews for a specific check type.

**Returns:** List of review items

#### get_queue_statistics

```python
@tool
def get_queue_statistics(review_manager: Any) -> Dict[str, Any]
```

Get current review queue statistics.

**Returns:** Statistics dictionary with:
- `total_pending`: Number of pending reviews
- `total_in_review`: Number currently in review
- `total_reviewed`: Number completed
- `avg_confidence`: Average confidence score
- `by_check_type`: Breakdown by check type
- `by_severity`: Breakdown by severity

#### get_next_review

```python
@tool
def get_next_review(review_manager: Any, reviewer_id: str) -> Optional[Dict[str, Any]]
```

Get next highest-priority review item.

**Returns:** Review item dict or None

---

### Feedback Tools

#### process_feedback

```python
@tool
def process_feedback(
    feedback_record: Dict[str, Any],
    confidence_calibrator: Any,
    pattern_analyzer: Any
) -> bool
```

Process a single feedback record.

**Returns:** True if successful

#### process_feedback_batch

```python
@tool
def process_feedback_batch(
    feedback_records: List[Dict[str, Any]],
    confidence_calibrator: Any,
    pattern_analyzer: Any
) -> Dict[str, Any]
```

Process a batch of feedback records.

**Returns:** Processing results with:
- `total_processed`: Total records processed
- `successful`: Number successful
- `failed`: Number failed
- `patterns_discovered`: Number of patterns found

#### update_confidence_calibration

```python
@tool
def update_confidence_calibration(
    check_type: str,
    predicted_confidence: int,
    predicted_violation: bool,
    actual_violation: bool,
    confidence_calibrator: Any
) -> Dict[str, Any]
```

Update confidence calibration based on prediction outcome.

**Returns:** Calibration update results

#### detect_patterns

```python
@tool
def detect_patterns(
    pattern_analyzer: Any,
    check_type: Optional[str] = None,
    min_occurrences: int = 3
) -> Dict[str, Any]
```

Detect patterns in false positives and false negatives.

**Returns:** Pattern detection results with:
- `false_positive_patterns`: List of FP patterns
- `false_negative_patterns`: List of FN patterns
- `total_patterns`: Total patterns found
- `high_impact_patterns`: High-impact patterns

#### suggest_rule_modifications

```python
@tool
def suggest_rule_modifications(
    pattern_analyzer: Any,
    min_impact: float = 0.05
) -> List[Dict[str, Any]]
```

Generate rule modification suggestions.

**Returns:** List of rule suggestions

#### get_learning_metrics

```python
@tool
def get_learning_metrics(
    feedback_interface: Any,
    pattern_analyzer: Any
) -> Dict[str, Any]
```

Calculate accuracy and learning metrics.

**Returns:** Metrics dictionary with:
- `total_feedback`: Total feedback records
- `false_positives`: Number of false positives
- `false_negatives`: Number of false negatives
- `patterns_discovered`: Patterns discovered
- `rules_suggested`: Rules suggested
- `accuracy_improvement`: Accuracy improvement percentage

---


## Usage Examples

### Example 1: Basic Workflow Execution

```python
from data_models_multiagent import initialize_compliance_state
from agents.supervisor_agent import SupervisorAgent
from agents.preprocessor_agent import PreprocessorAgent
from agents.structure_agent import StructureAgent
from agents.aggregator_agent import AggregatorAgent

# Load document
document = {
    "document_metadata": {
        "fund_isin": "FR0010135103",
        "client_type": "retail",
        "document_type": "fund_presentation"
    },
    "page_de_garde": {...},
    "slide_2": {...},
    "pages_suivantes": [...],
    "page_de_fin": {...}
}

# Initialize state
config = {
    "agents": {
        "supervisor": {"enabled": True},
        "preprocessor": {"enabled": True},
        "structure": {"enabled": True}
    }
}

state = initialize_compliance_state(
    document=document,
    document_id="doc_001",
    config=config
)

# Execute workflow
supervisor = SupervisorAgent()
state = supervisor(state)

preprocessor = PreprocessorAgent()
state = preprocessor(state)

structure = StructureAgent()
state = structure(state)

aggregator = AggregatorAgent()
state = aggregator(state)

# Get results
print(f"Total violations: {len(state['violations'])}")
print(f"Confidence: {state['aggregated_confidence']}%")
print(f"Next action: {state['next_action']}")
```

---

### Example 2: Using LangGraph Workflow

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from data_models_multiagent import ComplianceState
from agents.supervisor_agent import SupervisorAgent
from agents.preprocessor_agent import PreprocessorAgent
from agents.structure_agent import StructureAgent
from agents.aggregator_agent import AggregatorAgent

# Create workflow
workflow = StateGraph(ComplianceState)

# Add nodes
workflow.add_node("supervisor", SupervisorAgent())
workflow.add_node("preprocessor", PreprocessorAgent())
workflow.add_node("structure", StructureAgent())
workflow.add_node("aggregator", AggregatorAgent())

# Define edges
workflow.add_edge("supervisor", "preprocessor")
workflow.add_edge("preprocessor", "structure")
workflow.add_edge("structure", "aggregator")
workflow.add_edge("aggregator", END)

# Set entry point
workflow.set_entry_point("supervisor")

# Add checkpointing
memory = SqliteSaver.from_conn_string(":memory:")

# Compile
app = workflow.compile(checkpointer=memory)

# Execute
config = {"configurable": {"thread_id": "doc_001"}}
result = app.invoke(initial_state, config)

print(f"Workflow complete: {result['workflow_status']}")
```

---

### Example 3: Context Analysis for False Positive Elimination

```python
from agents.context_agent import ContextAgent
from agents.evidence_agent import EvidenceAgent
from ai_engine import AIEngine

# Initialize AI engine
ai_engine = AIEngine(config)

# Create agents
context_agent = ContextAgent(ai_engine=ai_engine)
evidence_agent = EvidenceAgent(ai_engine=ai_engine)

# State with low-confidence violations
state = {
    "violations": [
        {
            "rule": "PROHIBITED_INVESTMENT_ADVICE",
            "type": "ADVICE",
            "confidence": 65,
            "evidence": "Le fonds investit dans des actions européennes",
            "slide": "Slide 2"
        }
    ],
    "document": document,
    "config": config
}

# Analyze context
state = context_agent(state)

# Check if filtered as false positive
for violation in state['violations']:
    if violation['status'] == 'false_positive_filtered':
        print(f"Filtered: {violation['rule']}")
        print(f"Reason: {violation['context_reasoning']}")

# Extract evidence
state = evidence_agent(state)

# Check evidence
for violation in state['violations']:
    if 'evidence_quotes' in violation:
        print(f"Evidence quotes: {violation['evidence_quotes']}")
        print(f"Evidence confidence: {violation['evidence_confidence']}%")
```

---

### Example 4: Human-in-the-Loop Review

```python
from agents.reviewer_agent import ReviewerAgent
from review_manager import ReviewManager

# Initialize review manager
review_manager = ReviewManager()

# Create reviewer agent
reviewer = ReviewerAgent(
    review_manager=review_manager,
    review_threshold=70,
    auto_queue_enabled=True
)

# Process violations
state = reviewer(state)

print(f"Queued for review: {len(state['review_queue'])}")
print(f"Queue statistics: {state['queue_statistics']}")

# Get next review item
next_item = reviewer.get_next_review_item("reviewer_001")

if next_item:
    print(f"Review ID: {next_item['review_id']}")
    print(f"Check type: {next_item['check_type']}")
    print(f"Confidence: {next_item['confidence']}%")
    print(f"Evidence: {next_item['evidence']}")
    
    # Human reviews and provides feedback
    # ... (human review process)
    
    # Update review status
    review_manager.update_review_status(
        review_id=next_item['review_id'],
        status="CONFIRMED",
        reviewer_id="reviewer_001",
        reviewer_notes="Confirmed violation"
    )
```

---

### Example 5: Feedback Processing and Learning

```python
from agents.feedback_agent import FeedbackAgent
from feedback_loop import FeedbackInterface
from confidence_calibrator import ConfidenceCalibrator
from pattern_detector import PatternAnalyzer

# Initialize components
feedback_interface = FeedbackInterface()
calibrator = ConfidenceCalibrator()
pattern_analyzer = PatternAnalyzer(feedback_interface)

# Create feedback agent
feedback_agent = FeedbackAgent(
    feedback_interface=feedback_interface,
    confidence_calibrator=calibrator,
    pattern_analyzer=pattern_analyzer,
    ai_engine=ai_engine,
    auto_calibrate=True,
    pattern_detection_enabled=True
)

# Process feedback
state = feedback_agent(state)

# Get results
processing = state['feedback_processing']
print(f"Feedback processed: {processing['processed']}")
print(f"Patterns discovered: {processing['patterns_discovered']}")
print(f"Rules suggested: {processing['rules_suggested']}")

# Get learning report
report = feedback_agent.get_learning_report()
print(f"Total feedback: {report['total_feedback']}")
print(f"False positives: {report['false_positives']}")
print(f"False negatives: {report['false_negatives']}")
print(f"Accuracy improvement: {report['accuracy_improvement']:.1%}")

# Export learning data
feedback_agent.export_learning_data("./learning_exports")
```

---

### Example 6: Parallel Agent Execution

```python
from langgraph.graph import StateGraph
from agents.structure_agent import StructureAgent
from agents.performance_agent import PerformanceAgent
from agents.securities_agent import SecuritiesAgent
from agents.general_agent import GeneralAgent

# Create workflow with parallel execution
workflow = StateGraph(ComplianceState)

# Add specialist agents
workflow.add_node("structure", StructureAgent())
workflow.add_node("performance", PerformanceAgent())
workflow.add_node("securities", SecuritiesAgent())
workflow.add_node("general", GeneralAgent())
workflow.add_node("aggregator", AggregatorAgent())

# Parallel execution from preprocessor
workflow.add_edge("preprocessor", "structure")
workflow.add_edge("preprocessor", "performance")
workflow.add_edge("preprocessor", "securities")
workflow.add_edge("preprocessor", "general")

# All converge to aggregator
workflow.add_edge("structure", "aggregator")
workflow.add_edge("performance", "aggregator")
workflow.add_edge("securities", "aggregator")
workflow.add_edge("general", "aggregator")

# Compile and execute
app = workflow.compile()
result = app.invoke(state)

# Violations are automatically merged from parallel agents
print(f"Total violations from parallel agents: {len(result['violations'])}")
```

---

### Example 7: Conditional Routing Based on Confidence

```python
from langgraph.graph import StateGraph, END

# Create workflow with conditional routing
workflow = StateGraph(ComplianceState)

# Add agents
workflow.add_node("aggregator", AggregatorAgent())
workflow.add_node("context", ContextAgent(ai_engine=ai_engine))
workflow.add_node("evidence", EvidenceAgent(ai_engine=ai_engine))
workflow.add_node("reviewer", ReviewerAgent(review_manager=review_manager))

# Conditional routing from aggregator
def route_from_aggregator(state):
    next_action = state.get("next_action", "complete")
    if next_action == "context_analysis":
        return "context"
    elif next_action == "review":
        return "reviewer"
    else:
        return END

workflow.add_conditional_edges(
    "aggregator",
    route_from_aggregator,
    {
        "context": "context",
        "reviewer": "reviewer",
        END: END
    }
)

# Context to evidence
workflow.add_edge("context", "evidence")

# Conditional routing from evidence
def route_from_evidence(state):
    # Check if still low confidence
    low_conf = any(
        v.get("confidence", 100) < 70
        for v in state.get("violations", [])
    )
    return "reviewer" if low_conf else END

workflow.add_conditional_edges(
    "evidence",
    route_from_evidence,
    {
        "reviewer": "reviewer",
        END: END
    }
)

workflow.add_edge("reviewer", END)

# Execute
app = workflow.compile()
result = app.invoke(state)
```

---

### Example 8: Error Handling and Fallback

```python
from agents.base_agent import BaseAgent, AgentConfig

class CustomAgent(BaseAgent):
    def process(self, state: ComplianceState) -> ComplianceState:
        # Main processing logic
        # This may raise exceptions
        result = self.risky_operation(state)
        return result
    
    def fallback_process(self, state: ComplianceState, error: Exception) -> ComplianceState:
        """Fallback when main process fails"""
        self.logger.warning(f"Main process failed: {error}")
        self.logger.info("Using rule-based fallback")
        
        # Implement fallback logic (e.g., rule-based checking)
        violations = self.rule_based_check(state)
        
        return self.add_violations(state, violations)

# Create agent with fallback
config = AgentConfig(
    name="custom",
    retry_attempts=3,
    fail_fast=False  # Enable graceful degradation
)

agent = CustomAgent(config=config)

# Execute - will use fallback if main process fails
result = agent(state)
```

---

## Best Practices

### 1. Agent Configuration

```python
# Always provide explicit configuration
config = AgentConfig(
    name="my_agent",
    enabled=True,
    timeout_seconds=30.0,
    retry_attempts=3,
    log_level="INFO",
    custom_settings={
        "parallel_execution": True,
        "max_workers": 4
    }
)

agent = MyAgent(config=config)
```

### 2. State Management

```python
# Always use helper functions to initialize state
state = initialize_compliance_state(document, document_id, config)

# Update timestamps when modifying state
from data_models_multiagent import update_state_timestamp
state = update_state_timestamp(state)

# Mark state as completed
from data_models_multiagent import mark_state_completed
state = mark_state_completed(state)
```

### 3. Error Handling

```python
# Use try-except in agent process methods
def process(self, state: ComplianceState) -> ComplianceState:
    try:
        # Agent logic
        result = self.do_work(state)
        return result
    except Exception as e:
        self.logger.error(f"Error in {self.name}: {e}")
        # Add to error log
        if "error_log" not in state:
            state["error_log"] = []
        state["error_log"].append({
            "agent": self.name,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        return state
```

### 4. Logging

```python
# Use agent logger for consistent logging
self.logger.info("Starting processing")
self.logger.debug(f"Processing {len(items)} items")
self.logger.warning("Low confidence detected")
self.logger.error("Processing failed", exc_info=True)
```

### 5. Testing

```python
# Test agents independently
def test_structure_agent():
    # Create test state
    state = initialize_compliance_state(test_document, "test_001", {})
    state["metadata"] = {"client_type": "retail"}
    state["normalized_document"] = test_document
    
    # Execute agent
    agent = StructureAgent()
    result = agent(state)
    
    # Verify results
    assert "violations" in result
    assert "confidence_scores" in result
    assert result["confidence_scores"]["structure"] >= 0
```

---

## Additional Resources

- **Architecture Documentation**: See `docs/MULTI_AGENT_ARCHITECTURE.md`
- **Migration Guide**: See `docs/MIGRATION_TO_MULTIAGENT.md`
- **Configuration Guide**: See `docs/MULTIAGENT_CONFIGURATION.md`
- **Troubleshooting**: See `docs/MULTIAGENT_TROUBLESHOOTING.md`

---

## Version History

- **v1.0.0** (2024-01-15): Initial multi-agent system implementation
  - All 11 agents implemented
  - 40+ tools available
  - Full LangGraph integration
  - State persistence and resumability
  - HITL interrupt support
  - Parallel execution
  - Conditional routing

---

## Support

For questions or issues:
1. Check the troubleshooting guide
2. Review example usage above
3. Consult the architecture documentation
4. Check agent and tool docstrings in source code

---

*Last Updated: 2024-01-15*
