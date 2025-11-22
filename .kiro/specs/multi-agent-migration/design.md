# Design Document - Multi-Agent System Migration

## Overview

This design document outlines the complete migration of the AI-Enhanced Compliance Checker from a monolithic architecture to a distributed multi-agent system using LangGraph. The system will maintain 100% feature parity while gaining modularity, parallel processing, and improved maintainability.

### Design Philosophy

- **Agent Specialization**: Each compliance domain has a dedicated agent
- **State-Driven Workflow**: LangGraph manages state transitions and routing
- **Parallel Execution**: Independent checks run concurrently
- **Conditional Routing**: Confidence scores determine workflow path
- **HITL Integration**: Built-in interrupts for human review
- **Backward Compatible**: Seamless migration from current system

## Architecture

### High-Level Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ENTRY POINT                                  │
│                    check_multiagent.py                               │
│  - Parse command-line arguments                                      │
│  - Load configuration                                                │
│  - Initialize LangGraph workflow                                     │
│  - Execute compliance checking                                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH STATE MACHINE                           │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    SUPERVISOR AGENT                          │   │
│  │  - Initialize workflow                                       │   │
│  │  - Coordinate agent execution                                │   │
│  │  - Handle failures and retries                               │   │
│  │  - Generate final report                                     │   │
│  └────────────────────┬────────────────────────────────────────┘   │
│                       │                                              │
│                       ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              PREPROCESSOR AGENT                              │   │
│  │  Tools: extract_metadata, build_whitelist, normalize_doc    │   │
│  └────────────────────┬────────────────────────────────────────┘   │
│                       │                                              │
│         ┌─────────────┴─────────────┬──────────────┬────────────┐  │
│         │                           │              │            │  │
│         ▼                           ▼              ▼            ▼  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐   │
│  │  STRUCTURE  │  │ PERFORMANCE  │  │SECURITIES│  │ GENERAL  │   │
│  │   AGENT     │  │    AGENT     │  │  AGENT   │  │  AGENT   │   │
│  │             │  │              │  │          │  │          │   │
│  │ - Promo     │  │ - Perf data  │  │ - Intent │  │ - Glossary│  │
│  │ - Target    │  │ - Disclaimers│  │ - Advice │  │ - Sources │  │
│  │ - Legal     │  │ - Benchmark  │  │ - Whitelist│ │ - Dates  │  │
│  └─────────────┘  └──────────────┘  └──────────┘  └──────────┘   │
│         │                           │              │            │  │
│         └─────────────┬─────────────┴──────────────┴────────────┘  │
│                       │                                              │
│                       ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              PROSPECTUS AGENT                                │   │
│  │  Tools: check_fund_name, check_strategy, check_benchmark    │   │
│  └────────────────────┬────────────────────────────────────────┘   │
│                       │                                              │
│                       ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              REGISTRATION AGENT                              │   │
│  │  Tools: extract_countries, validate_authorization            │   │
│  └────────────────────┬────────────────────────────────────────┘   │
│                       │                                              │
│                       ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  ESG AGENT                                   │   │
│  │  Tools: check_classification, validate_content_distribution │   │
│  └────────────────────┬────────────────────────────────────────┘   │
│                       │                                              │
│                       ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              AGGREGATOR AGENT                                │   │
│  │  - Combine all violations                                    │   │
│  │  - Calculate confidence scores                               │   │
│  │  - Determine next action                                     │   │
│  └────────────────────┬────────────────────────────────────────┘   │
│                       │                                              │
│                       ▼                                              │
│              ┌────────────────┐                                     │
│              │ Low Confidence?│                                     │
│              └────┬───────┬───┘                                     │
│                   │ Yes   │ No                                      │
│                   ▼       ▼                                         │
│  ┌──────────────────┐  ┌──────────────────┐                       │
│  │  CONTEXT AGENT   │  │  FINAL REPORT    │                       │
│  │  - Analyze       │  │  - Generate JSON │                       │
│  │  - Classify      │  │  - Save output   │                       │
│  │  - Validate      │  └──────────────────┘                       │
│  └────────┬─────────┘                                              │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────┐                                              │
│  │  EVIDENCE AGENT  │                                              │
│  │  - Extract quotes│                                              │
│  │  - Find data     │                                              │
│  │  - Locate        │                                              │
│  └────────┬─────────┘                                              │
│           │                                                         │
│           ▼                                                         │
│  ┌────────────────┐                                                │
│  │ Still Low Conf?│                                                │
│  └────┬───────┬───┘                                                │
│       │ Yes   │ No                                                 │
│       ▼       ▼                                                    │
│  ┌──────────────────┐  ┌──────────────────┐                       │
│  │  REVIEWER AGENT  │  │  FINAL REPORT    │                       │
│  │  - Queue review  │  │                  │                       │
│  │  - HITL interrupt│  └──────────────────┘                       │
│  │  - Feedback loop │                                              │
│  └──────────────────┘                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### State Definition

```python
from typing import TypedDict, Annotated, Sequence, Optional, Dict, List, Set
import operator
from datetime import datetime

class ComplianceState(TypedDict):
    """Shared state passed between all agents"""
    
    # Document data
    document: dict
    document_id: str
    document_type: str
    client_type: str
    
    # Preprocessing results
    metadata: dict
    whitelist: Set[str]
    normalized_document: dict
    
    # Violations from all agents
    violations: Annotated[Sequence[dict], operator.add]
    
    # Context analysis results
    context_analysis: Dict[str, dict]
    intent_classifications: Dict[str, dict]
    evidence_extractions: Dict[str, dict]
    
    # Review and feedback
    review_queue: List[dict]
    feedback_history: List[dict]
    
    # Confidence and scoring
    confidence_scores: Dict[str, int]
    aggregated_confidence: int
    
    # Workflow control
    current_agent: str
    next_action: str
    workflow_status: str
    error_log: List[dict]
    
    # Performance metrics
    agent_timings: Dict[str, float]
    api_calls: int
    cache_hits: int
    
    # Configuration
    config: dict
    
    # Timestamps
    started_at: datetime
    updated_at: datetime
```

## Agent Implementations

### 1. Supervisor Agent

**Purpose**: Orchestrate entire workflow, coordinate agents, handle failures

**Implementation**:
```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

class SupervisorAgent:
    def __init__(self, llm: ChatOpenAI, config: dict):
        self.llm = llm
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, state: ComplianceState) -> ComplianceState:
        """
        Initialize workflow and determine execution plan
        """
        self.logger.info(f"Supervisor: Starting compliance check for {state['document_id']}")
        
        # Update state
        state["current_agent"] = "supervisor"
        state["workflow_status"] = "initialized"
        state["started_at"] = datetime.now()
        
        # Determine which agents to invoke based on document type
        execution_plan = self._create_execution_plan(state)
        state["execution_plan"] = execution_plan
        
        # Set next action
        state["next_action"] = "preprocess"
        
        return state
    
    def _create_execution_plan(self, state: ComplianceState) -> List[str]:
        """Determine which agents to run based on document type and config"""
        plan = ["preprocessor"]
        
        # Always run core checks
        plan.extend(["structure", "performance", "securities", "general"])
        
        # Conditional checks
        if state.get("metadata", {}).get("fund_isin"):
            plan.append("registration")
        
        if state.get("config", {}).get("prospectus_data"):
            plan.append("prospectus")
        
        if state.get("config", {}).get("esg_classification") != "other":
            plan.append("esg")
        
        # Always aggregate and analyze
        plan.extend(["aggregator", "context", "evidence"])
        
        return plan
```

### 2. Preprocessor Agent

**Purpose**: Extract metadata, build whitelist, normalize document

**Tools**:
```python
from langchain.tools import tool

@tool
def extract_metadata(document: dict) -> dict:
    """Extract metadata from document"""
    return {
        "fund_isin": document.get("document_metadata", {}).get("fund_isin"),
        "client_type": document.get("document_metadata", {}).get("client_type", "retail"),
        "document_type": document.get("document_metadata", {}).get("document_type", "fund_presentation"),
        "fund_name": document.get("document_metadata", {}).get("fund_name"),
        "esg_classification": document.get("document_metadata", {}).get("fund_esg_classification", "other")
    }

@tool
def build_whitelist(document: dict, metadata: dict) -> Set[str]:
    """Build whitelist of allowed terms"""
    from whitelist_manager import WhitelistManager
    manager = WhitelistManager()
    return manager.build_whitelist(document)

@tool
def normalize_document(document: dict) -> dict:
    """Normalize document structure"""
    # Ensure all expected fields exist
    normalized = {
        "page_de_garde": document.get("page_de_garde", {}),
        "slide_2": document.get("slide_2", {}),
        "pages_suivantes": document.get("pages_suivantes", []),
        "page_de_fin": document.get("page_de_fin", {}),
        "document_metadata": document.get("document_metadata", {})
    }
    return normalized

class PreprocessorAgent:
    def __init__(self, tools: List):
        self.tools = {tool.name: tool for tool in tools}
    
    def __call__(self, state: ComplianceState) -> ComplianceState:
        """Preprocess document"""
        state["current_agent"] = "preprocessor"
        
        # Extract metadata
        state["metadata"] = self.tools["extract_metadata"].invoke({"document": state["document"]})
        
        # Build whitelist
        state["whitelist"] = self.tools["build_whitelist"].invoke({
            "document": state["document"],
            "metadata": state["metadata"]
        })
        
        # Normalize document
        state["normalized_document"] = self.tools["normalize_document"].invoke({
            "document": state["document"]
        })
        
        state["next_action"] = "parallel_checks"
        return state
```

### 3. Structure Agent

**Purpose**: Check structure compliance (promotional mention, target audience, etc.)

**Tools**:
```python
@tool
def check_promotional_mention(document: dict, config: dict) -> Optional[dict]:
    """Check for promotional document mention on cover page"""
    from check_functions_ai import check_promotional_mention_enhanced
    return check_promotional_mention_enhanced(document, config)

@tool
def check_target_audience(document: dict, client_type: str) -> Optional[dict]:
    """Check target audience specification"""
    from check_functions_ai import check_target_audience_enhanced
    return check_target_audience_enhanced(document, client_type)

@tool
def check_management_company(document: dict) -> Optional[dict]:
    """Check management company legal mention"""
    from check_functions_ai import check_management_company_enhanced
    return check_management_company_enhanced(document)

class StructureAgent:
    def __init__(self, tools: List):
        self.tools = {tool.name: tool for tool in tools}
        self.check_type = "STRUCTURE"
    
    def __call__(self, state: ComplianceState) -> ComplianceState:
        """Run all structure checks"""
        state["current_agent"] = "structure"
        violations = []
        
        # Run each structure check
        for tool_name, tool in self.tools.items():
            try:
                result = tool.invoke({
                    "document": state["normalized_document"],
                    "config": state["config"],
                    "client_type": state["metadata"]["client_type"]
                })
                if result:
                    violations.append(result)
            except Exception as e:
                state["error_log"].append({
                    "agent": "structure",
                    "tool": tool_name,
                    "error": str(e)
                })
        
        # Add violations to state
        state["violations"] = state.get("violations", []) + violations
        
        return state
```

### 4. Context Agent

**Purpose**: Analyze context and intent to eliminate false positives

**Tools**:
```python
@tool
def analyze_context(text: str, check_type: str, ai_engine) -> dict:
    """Analyze text context using AI"""
    from context_analyzer import ContextAnalyzer
    analyzer = ContextAnalyzer(ai_engine)
    return analyzer.analyze_context(text, check_type)

@tool
def classify_intent(text: str, ai_engine) -> dict:
    """Classify text intent"""
    from intent_classifier import IntentClassifier
    classifier = IntentClassifier(ai_engine)
    return classifier.classify_intent(text)

@tool
def validate_semantically(text: str, whitelist: Set[str], ai_engine) -> dict:
    """Validate using semantic understanding"""
    from semantic_validator import SemanticValidator
    from context_analyzer import ContextAnalyzer
    from intent_classifier import IntentClassifier
    
    context_analyzer = ContextAnalyzer(ai_engine)
    intent_classifier = IntentClassifier(ai_engine)
    validator = SemanticValidator(ai_engine, context_analyzer, intent_classifier)
    
    return validator.validate_securities_mention(text, whitelist)

class ContextAgent:
    def __init__(self, tools: List, ai_engine):
        self.tools = {tool.name: tool for tool in tools}
        self.ai_engine = ai_engine
    
    def __call__(self, state: ComplianceState) -> ComplianceState:
        """Analyze context for low-confidence violations"""
        state["current_agent"] = "context"
        
        # Filter low-confidence violations
        low_conf_violations = [
            v for v in state["violations"]
            if v.get("confidence", 100) < 80
        ]
        
        # Analyze each
        for violation in low_conf_violations:
            context = self.tools["analyze_context"].invoke({
                "text": violation.get("evidence", ""),
                "check_type": violation.get("type", ""),
                "ai_engine": self.ai_engine
            })
            
            intent = self.tools["classify_intent"].invoke({
                "text": violation.get("evidence", ""),
                "ai_engine": self.ai_engine
            })
            
            # Store analysis
            state["context_analysis"][violation.get("rule", "")] = context
            state["intent_classifications"][violation.get("rule", "")] = intent
            
            # Update confidence based on context
            if context.get("is_fund_description") and not context.get("is_client_advice"):
                violation["confidence"] = max(violation.get("confidence", 0), 85)
                violation["status"] = "FALSE_POSITIVE_FILTERED"
        
        return state
```

### 5. Reviewer Agent

**Purpose**: Manage HITL review queue and feedback

**Tools**:
```python
@tool
def queue_for_review(violation: dict, review_manager) -> str:
    """Add violation to review queue"""
    from review_manager import ReviewItem, ReviewStatus
    from datetime import datetime
    import uuid
    
    review_item = ReviewItem(
        review_id=str(uuid.uuid4()),
        document_id=violation.get("document_id", ""),
        check_type=violation.get("type", ""),
        slide=violation.get("slide", ""),
        location=violation.get("location", ""),
        predicted_violation=True,
        confidence=violation.get("confidence", 0),
        ai_reasoning=violation.get("ai_reasoning", ""),
        evidence=violation.get("evidence", ""),
        rule=violation.get("rule", ""),
        severity=violation.get("severity", ""),
        created_at=datetime.now().isoformat(),
        priority_score=100 - violation.get("confidence", 0),
        status=ReviewStatus.PENDING
    )
    
    return review_manager.add_to_queue(review_item)

@tool
def process_feedback(review_decision: dict, feedback_interface) -> bool:
    """Process human feedback"""
    return feedback_interface.provide_correction(
        check_type=review_decision.get("check_type"),
        document_id=review_decision.get("document_id"),
        predicted_violation=review_decision.get("predicted_violation"),
        actual_violation=review_decision.get("actual_violation"),
        predicted_confidence=review_decision.get("predicted_confidence"),
        reviewer_notes=review_decision.get("reviewer_notes")
    )

class ReviewerAgent:
    def __init__(self, tools: List, review_manager, feedback_interface):
        self.tools = {tool.name: tool for tool in tools}
        self.review_manager = review_manager
        self.feedback_interface = feedback_interface
    
    def __call__(self, state: ComplianceState) -> ComplianceState:
        """Queue low-confidence violations for review"""
        state["current_agent"] = "reviewer"
        
        # Filter violations needing review
        review_threshold = state["config"].get("hitl", {}).get("review_threshold", 70)
        needs_review = [
            v for v in state["violations"]
            if 0 < v.get("confidence", 100) < review_threshold
        ]
        
        # Queue each
        for violation in needs_review:
            review_id = self.tools["queue_for_review"].invoke({
                "violation": violation,
                "review_manager": self.review_manager
            })
            state["review_queue"].append({
                "review_id": review_id,
                "violation": violation
            })
        
        return state
```

## LangGraph Workflow Definition

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

def create_compliance_workflow(config: dict) -> StateGraph:
    """Create the complete multi-agent compliance workflow"""
    
    # Initialize components
    ai_engine = AIEngine(config)
    review_manager = ReviewManager()
    feedback_interface = FeedbackInterface()
    
    # Create agents
    supervisor = SupervisorAgent(llm, config)
    preprocessor = PreprocessorAgent(preprocessor_tools)
    structure = StructureAgent(structure_tools)
    performance = PerformanceAgent(performance_tools)
    securities = SecuritiesAgent(securities_tools)
    general = GeneralAgent(general_tools)
    prospectus = ProspectusAgent(prospectus_tools)
    registration = RegistrationAgent(registration_tools)
    esg = ESGAgent(esg_tools)
    aggregator = AggregatorAgent()
    context = ContextAgent(context_tools, ai_engine)
    evidence = EvidenceAgent(evidence_tools, ai_engine)
    reviewer = ReviewerAgent(reviewer_tools, review_manager, feedback_interface)
    
    # Create graph
    workflow = StateGraph(ComplianceState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("preprocessor", preprocessor)
    workflow.add_node("structure", structure)
    workflow.add_node("performance", performance)
    workflow.add_node("securities", securities)
    workflow.add_node("general", general)
    workflow.add_node("prospectus", prospectus)
    workflow.add_node("registration", registration)
    workflow.add_node("esg", esg)
    workflow.add_node("aggregator", aggregator)
    workflow.add_node("context", context)
    workflow.add_node("evidence", evidence)
    workflow.add_node("reviewer", reviewer)
    
    # Define edges
    workflow.add_edge("supervisor", "preprocessor")
    workflow.add_edge("preprocessor", "structure")
    workflow.add_edge("preprocessor", "performance")
    workflow.add_edge("preprocessor", "securities")
    workflow.add_edge("preprocessor", "general")
    
    # Conditional edges based on document type
    workflow.add_conditional_edges(
        "preprocessor",
        lambda state: "prospectus" if state.get("config", {}).get("prospectus_data") else "aggregator",
        {
            "prospectus": "prospectus",
            "aggregator": "aggregator"
        }
    )
    
    # All specialist agents flow to aggregator
    workflow.add_edge("structure", "aggregator")
    workflow.add_edge("performance", "aggregator")
    workflow.add_edge("securities", "aggregator")
    workflow.add_edge("general", "aggregator")
    workflow.add_edge("prospectus", "aggregator")
    workflow.add_edge("registration", "aggregator")
    workflow.add_edge("esg", "aggregator")
    
    # Conditional routing based on confidence
    workflow.add_conditional_edges(
        "aggregator",
        lambda state: "context" if any(v.get("confidence", 100) < 80 for v in state["violations"]) else END,
        {
            "context": "context",
            END: END
        }
    )
    
    workflow.add_edge("context", "evidence")
    
    # Route to reviewer if still low confidence
    workflow.add_conditional_edges(
        "evidence",
        lambda state: "reviewer" if any(v.get("confidence", 100) < 70 for v in state["violations"]) else END,
        {
            "reviewer": "reviewer",
            END: END
        }
    )
    
    workflow.add_edge("reviewer", END)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add checkpointing for state persistence
    memory = SqliteSaver.from_conn_string(":memory:")
    
    # Compile
    app = workflow.compile(checkpointer=memory)
    
    return app
```

## Migration Strategy

### Phase 1: Foundation (Week 1-2)
- Set up LangGraph infrastructure
- Define ComplianceState
- Implement Supervisor and Preprocessor agents
- Create basic workflow with 2-3 agents

### Phase 2: Core Agents (Week 3-4)
- Migrate Structure, Performance, Securities, General agents
- Convert existing check functions to tools
- Implement parallel execution
- Test with exemple.json

### Phase 3: Advanced Agents (Week 5-6)
- Implement Context, Evidence, Reviewer agents
- Migrate false-positive elimination logic
- Add conditional routing
- Implement HITL interrupts

### Phase 4: Integration (Week 7-8)
- Add remaining agents (Prospectus, Registration, ESG)
- Implement state persistence
- Add monitoring and metrics
- Full integration testing

### Phase 5: Validation (Week 9-10)
- A/B testing with current system
- Performance optimization
- Documentation
- Migration guide

## Testing Strategy

### Unit Tests
- Test each agent independently
- Test each tool function
- Test state transitions
- Test conditional routing logic

### Integration Tests
- Test complete workflow end-to-end
- Test parallel agent execution
- Test HITL interrupts and resume
- Test error handling and fallbacks

### Validation Tests
- Compare results with current system on exemple.json
- Verify 6 violations, 0 false positives
- Verify all existing features work
- Performance benchmarking

## Configuration

```json
{
  "multi_agent": {
    "enabled": true,
    "parallel_execution": true,
    "max_parallel_agents": 4,
    "agent_timeout_seconds": 30,
    "checkpoint_interval": 5,
    "state_persistence": true
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

This design provides a complete blueprint for migrating to a multi-agent system while preserving all existing functionality and adding new capabilities.
