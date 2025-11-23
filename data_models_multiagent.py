"""
Multi-Agent Compliance State Data Models

This module defines the shared state structure used across all agents in the
LangGraph-based multi-agent compliance checking system.
"""

from typing import TypedDict, Annotated, Sequence, Optional, Dict, List, Set, Any
from datetime import datetime
from enum import Enum
import operator
import json
from dataclasses import dataclass, asdict, field


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    INITIALIZED = "initialized"
    PREPROCESSING = "preprocessing"
    CHECKING = "checking"
    AGGREGATING = "aggregating"
    ANALYZING = "analyzing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ViolationStatus(str, Enum):
    """Status of a compliance violation"""
    DETECTED = "detected"
    VALIDATED = "validated"
    FALSE_POSITIVE_FILTERED = "false_positive_filtered"
    PENDING_REVIEW = "pending_review"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class ComplianceState(TypedDict, total=False):
    """
    Shared state passed between all agents in the compliance workflow.
    
    This TypedDict defines the complete state structure that flows through
    the LangGraph state machine. Each agent receives this state, performs
    its operations, and returns an updated state.
    
    The 'violations' field uses Annotated with operator.add to automatically
    merge violation lists from parallel agents.
    """
    
    # Document data
    document: dict
    document_id: str
    document_type: str
    client_type: str
    
    # Preprocessing results
    metadata: Dict[str, Any]
    whitelist: Set[str]
    normalized_document: dict
    
    # Violations from all agents (automatically merged with operator.add)
    violations: Annotated[Sequence[dict], operator.add]
    
    # Context analysis results
    context_analysis: Dict[str, dict]
    intent_classifications: Dict[str, dict]
    evidence_extractions: Dict[str, dict]
    
    # Review and feedback
    review_queue: List[dict]
    feedback_history: List[dict]
    
    # Confidence and scoring (merged with dict update for parallel agents)
    confidence_scores: Annotated[Dict[str, int], lambda x, y: {**x, **y}]
    aggregated_confidence: int
    
    # Workflow control
    current_agent: str
    next_action: str
    workflow_status: str
    execution_plan: List[str]
    error_log: Annotated[Sequence[dict], operator.add]
    
    # Performance metrics (merged with dict update for parallel agents)
    agent_timings: Annotated[Dict[str, float], lambda x, y: {**x, **y}]
    api_calls: int
    cache_hits: int
    
    # Configuration
    config: dict
    
    # Timestamps
    started_at: str
    updated_at: str
    completed_at: Optional[str]


@dataclass
class ViolationRecord:
    """
    Structured representation of a compliance violation.
    
    This dataclass provides a typed structure for violations detected
    by agents, making it easier to work with violation data.
    """
    rule: str
    type: str
    severity: str
    slide: str
    location: str
    evidence: str
    ai_reasoning: str
    confidence: int
    status: str = ViolationStatus.DETECTED.value
    agent: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ViolationRecord':
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class AgentExecutionRecord:
    """Record of an agent's execution"""
    agent_name: str
    started_at: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    violations_found: int = 0
    status: str = "running"
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return asdict(self)


# Helper Functions

def initialize_compliance_state(
    document: dict,
    document_id: str,
    config: dict
) -> ComplianceState:
    """
    Initialize a new ComplianceState with default values.
    
    Args:
        document: The document to check for compliance
        document_id: Unique identifier for the document
        config: Configuration dictionary
    
    Returns:
        Initialized ComplianceState ready for workflow execution
    """
    metadata = document.get("document_metadata", {})
    
    state: ComplianceState = {
        # Document data
        "document": document,
        "document_id": document_id,
        "document_type": metadata.get("document_type", "fund_presentation"),
        "client_type": metadata.get("client_type", "retail"),
        
        # Preprocessing results
        "metadata": {},
        "whitelist": set(),
        "normalized_document": {},
        
        # Violations (empty list, will be populated by agents)
        "violations": [],
        
        # Context analysis results
        "context_analysis": {},
        "intent_classifications": {},
        "evidence_extractions": {},
        
        # Review and feedback
        "review_queue": [],
        "feedback_history": [],
        
        # Confidence and scoring
        "confidence_scores": {},
        "aggregated_confidence": 100,
        
        # Workflow control
        "current_agent": "initializing",
        "next_action": "start",
        "workflow_status": WorkflowStatus.INITIALIZED.value,
        "execution_plan": [],
        "error_log": [],
        
        # Performance metrics
        "agent_timings": {},
        "api_calls": 0,
        "cache_hits": 0,
        
        # Configuration
        "config": config,
        
        # Timestamps
        "started_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "completed_at": None
    }
    
    return state


def validate_compliance_state(state: ComplianceState) -> tuple[bool, List[str]]:
    """
    Validate that a ComplianceState has all required fields and correct types.
    
    Args:
        state: The state to validate
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required fields
    required_fields = [
        "document", "document_id", "document_type", "client_type",
        "violations", "workflow_status", "config"
    ]
    
    for field in required_fields:
        if field not in state:
            errors.append(f"Missing required field: {field}")
    
    # Validate field types
    if "document" in state and not isinstance(state["document"], dict):
        errors.append("Field 'document' must be a dict")
    
    if "document_id" in state and not isinstance(state["document_id"], str):
        errors.append("Field 'document_id' must be a string")
    
    if "violations" in state and not isinstance(state["violations"], (list, tuple)):
        errors.append("Field 'violations' must be a list or tuple")
    
    if "whitelist" in state and not isinstance(state["whitelist"], set):
        errors.append("Field 'whitelist' must be a set")
    
    if "config" in state and not isinstance(state["config"], dict):
        errors.append("Field 'config' must be a dict")
    
    # Validate workflow status
    if "workflow_status" in state:
        valid_statuses = [s.value for s in WorkflowStatus]
        if state["workflow_status"] not in valid_statuses:
            errors.append(f"Invalid workflow_status: {state['workflow_status']}")
    
    # Validate violations structure
    if "violations" in state:
        for i, violation in enumerate(state["violations"]):
            if not isinstance(violation, dict):
                errors.append(f"Violation at index {i} must be a dict")
                continue
            
            required_violation_fields = ["rule", "type", "severity", "confidence"]
            for field in required_violation_fields:
                if field not in violation:
                    errors.append(f"Violation at index {i} missing field: {field}")
    
    return len(errors) == 0, errors


def serialize_state(state: ComplianceState) -> str:
    """
    Serialize ComplianceState to JSON string.
    
    Handles special types like sets and datetime objects.
    
    Args:
        state: The state to serialize
    
    Returns:
        JSON string representation of the state
    """
    def default_serializer(obj):
        """Custom serializer for non-JSON types"""
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    return json.dumps(state, default=default_serializer, indent=2)


def deserialize_state(json_str: str) -> ComplianceState:
    """
    Deserialize JSON string to ComplianceState.
    
    Reconstructs special types like sets.
    
    Args:
        json_str: JSON string representation of state
    
    Returns:
        Reconstructed ComplianceState
    """
    state_dict = json.loads(json_str)
    
    # Convert whitelist back to set
    if "whitelist" in state_dict and isinstance(state_dict["whitelist"], list):
        state_dict["whitelist"] = set(state_dict["whitelist"])
    
    # Ensure violations is a list
    if "violations" not in state_dict:
        state_dict["violations"] = []
    
    return state_dict


def merge_violations(
    existing_violations: Sequence[dict],
    new_violations: Sequence[dict]
) -> List[dict]:
    """
    Merge violation lists, removing duplicates based on rule and location.
    
    Args:
        existing_violations: Current violations in state
        new_violations: New violations to add
    
    Returns:
        Merged list of violations without duplicates
    """
    # Create a set of unique keys for existing violations
    existing_keys = {
        (v.get("rule", ""), v.get("slide", ""), v.get("location", ""))
        for v in existing_violations
    }
    
    # Start with existing violations
    merged = list(existing_violations)
    
    # Add new violations that don't duplicate existing ones
    for violation in new_violations:
        key = (
            violation.get("rule", ""),
            violation.get("slide", ""),
            violation.get("location", "")
        )
        if key not in existing_keys:
            merged.append(violation)
            existing_keys.add(key)
    
    return merged


def update_state_timestamp(state: ComplianceState) -> ComplianceState:
    """
    Update the 'updated_at' timestamp in the state.
    
    Args:
        state: The state to update
    
    Returns:
        State with updated timestamp
    """
    state["updated_at"] = datetime.now().isoformat()
    return state


def mark_state_completed(state: ComplianceState) -> ComplianceState:
    """
    Mark the state as completed with final timestamp.
    
    Args:
        state: The state to mark as completed
    
    Returns:
        State marked as completed
    """
    now = datetime.now().isoformat()
    state["workflow_status"] = WorkflowStatus.COMPLETED.value
    state["updated_at"] = now
    state["completed_at"] = now
    return state


def get_state_summary(state: ComplianceState) -> dict:
    """
    Get a summary of the current state for logging/monitoring.
    
    Args:
        state: The state to summarize
    
    Returns:
        Dictionary with key state metrics
    """
    return {
        "document_id": state.get("document_id", "unknown"),
        "workflow_status": state.get("workflow_status", "unknown"),
        "current_agent": state.get("current_agent", "none"),
        "violations_count": len(state.get("violations", [])),
        "agents_executed": len(state.get("agent_timings", {})),
        "api_calls": state.get("api_calls", 0),
        "cache_hits": state.get("cache_hits", 0),
        "errors": len(state.get("error_log", [])),
        "started_at": state.get("started_at", "unknown"),
        "updated_at": state.get("updated_at", "unknown")
    }


# Export all public symbols
__all__ = [
    "ComplianceState",
    "WorkflowStatus",
    "ViolationStatus",
    "ViolationRecord",
    "AgentExecutionRecord",
    "initialize_compliance_state",
    "validate_compliance_state",
    "serialize_state",
    "deserialize_state",
    "merge_violations",
    "update_state_timestamp",
    "mark_state_completed",
    "get_state_summary"
]
