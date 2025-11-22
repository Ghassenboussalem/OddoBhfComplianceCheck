# Design Document - Human-in-the-Loop Integration

## Overview

This design document outlines the architecture and implementation approach for integrating a comprehensive Human-in-the-Loop (HITL) system into the AI-Enhanced Compliance Checker. The system will leverage existing components (`feedback_loop.py`, `confidence_scorer.py`, `performance_monitor.py`) while adding new interactive review capabilities.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    check.py (Entry Point)                    │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│           HybridComplianceChecker (Core Engine)              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Layer 1: Rule Pre-filtering                       │     │
│  │  Layer 2: AI Analysis                              │     │
│  │  Layer 3: Confidence Scoring                       │     │
│  └────────────────────────────────────────────────────┘     │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ├─────► Low Confidence? ────► Review Queue
                   │                                    │
                   │                                    ▼
                   │                          ┌─────────────────┐
                   │                          │  Review Manager │
                   │                          │  - Queue Mgmt   │
                   │                          │  - Prioritize   │
                   │                          │  - Batch Ops    │
                   │                          └────────┬────────┘
                   │                                   │
                   │                                   ▼
                   │                          ┌─────────────────┐
                   │                          │ Review Interface│
                   │                          │  - CLI          │
                   │                          │  - Interactive  │
                   │                          └────────┬────────┘
                   │                                   │
                   │                                   ▼
                   │                          ┌─────────────────┐
                   │                          │ Feedback System │
                   │                          │  - Record       │
                   │                          │  - Learn        │
                   │                          │  - Calibrate    │
                   │                          └────────┬────────┘
                   │                                   │
                   └───────────────────────────────────┘
                                    │
                                    ▼
                          ┌──────────────────┐
                          │  Audit Logger    │
                          │  - Immutable Log │
                          │  - Export        │
                          └──────────────────┘
```

### Component Interactions

1. **HybridComplianceChecker** detects violations and flags low-confidence results
2. **ReviewManager** queues flagged items and manages review workflow
3. **ReviewInterface** presents items to human reviewers (CLI or programmatic)
4. **FeedbackSystem** records corrections and updates calibration models
5. **AuditLogger** maintains immutable records for compliance

## Components and Interfaces

### 1. ReviewManager

**Purpose:** Manages the review queue, prioritization, and batch operations.

**Key Methods:**
```python
class ReviewManager:
    def add_to_queue(self, review_item: ReviewItem) -> str
    def get_next_review(self, reviewer_id: str) -> Optional[ReviewItem]
    def get_pending_reviews(self, filters: Dict) -> List[ReviewItem]
    def mark_reviewed(self, review_id: str, decision: ReviewDecision) -> bool
    def get_similar_items(self, review_id: str, threshold: float) -> List[ReviewItem]
    def batch_review(self, review_ids: List[str], decision: ReviewDecision) -> int
    def get_queue_stats(self) -> QueueStats
```

**Data Model:**
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
    created_at: datetime
    priority_score: float
    status: ReviewStatus  # PENDING, IN_REVIEW, REVIEWED
    assigned_to: Optional[str]
```

### 2. ReviewInterface (CLI)

**Purpose:** Provides command-line interface for interactive review.

**Key Commands:**
```bash
# View next pending review
python review.py next

# Approve current review
python review.py approve --notes "Correct detection"

# Reject current review
python review.py reject --notes "False positive - example only"

# Skip to next review
python review.py skip

# View queue status
python review.py status

# Batch review similar items
python review.py batch --check-type=STRUCTURE --action=approve

# Export audit log
python review.py export --output=audit_log.json
```

**Interactive Mode:**
```python
class ReviewCLI:
    def start_interactive_session(self, reviewer_id: str)
    def display_review_item(self, item: ReviewItem)
    def prompt_for_decision(self) -> ReviewDecision
    def show_progress(self, completed: int, total: int)
    def display_queue_summary(self, stats: QueueStats)
```

### 3. ReviewDecision

**Purpose:** Captures human reviewer decisions and feedback.

**Data Model:**
```python
@dataclass
class ReviewDecision:
    review_id: str
    reviewer_id: str
    decision: str  # APPROVE, REJECT, MODIFY
    actual_violation: bool
    corrected_confidence: Optional[int]
    reviewer_notes: str
    tags: List[str]
    reviewed_at: datetime
    review_duration_seconds: int
```

### 4. FeedbackIntegration

**Purpose:** Integrates human feedback into the learning system.

**Key Methods:**
```python
class FeedbackIntegration:
    def process_review_decision(self, decision: ReviewDecision) -> bool
    def update_confidence_calibration(self, decision: ReviewDecision)
    def detect_patterns(self, decisions: List[ReviewDecision]) -> List[Pattern]
    def suggest_rule_modifications(self, pattern: Pattern) -> List[RuleSuggestion]
    def get_accuracy_metrics(self, check_type: Optional[str]) -> AccuracyMetrics
```

**Integration Points:**
- `confidence_calibrator.py` - Update calibration models
- `pattern_detector.py` - Analyze feedback patterns
- `feedback_loop.py` - Store and retrieve feedback
- `performance_monitor.py` - Track accuracy metrics

### 5. AuditLogger

**Purpose:** Maintains immutable audit trail for compliance.

**Key Methods:**
```python
class AuditLogger:
    def log_review(self, decision: ReviewDecision)
    def log_queue_action(self, action: str, details: Dict)
    def export_audit_log(self, filepath: str, format: str)
    def get_audit_trail(self, filters: Dict) -> List[AuditEntry]
    def generate_compliance_report(self, start_date: date, end_date: date) -> Report
```

**Audit Entry Format:**
```json
{
  "audit_id": "uuid",
  "timestamp": "2025-01-18T10:30:00Z",
  "action": "REVIEW_COMPLETED",
  "reviewer_id": "reviewer_001",
  "review_id": "review_123",
  "original_prediction": {
    "violation": true,
    "confidence": 65,
    "check_type": "STRUCTURE"
  },
  "human_decision": {
    "decision": "REJECT",
    "actual_violation": false,
    "notes": "False positive - example only"
  },
  "immutable_hash": "sha256_hash"
}
```

## Data Models

### ReviewStatus Enum
```python
class ReviewStatus(Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    REVIEWED = "reviewed"
    SKIPPED = "skipped"
```

### QueueStats
```python
@dataclass
class QueueStats:
    total_pending: int
    total_in_review: int
    total_reviewed: int
    avg_confidence: float
    by_check_type: Dict[str, int]
    by_severity: Dict[str, int]
    oldest_pending_age_hours: float
```

### AccuracyMetrics
```python
@dataclass
class AccuracyMetrics:
    total_reviews: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    avg_confidence_correct: float
    avg_confidence_incorrect: float
```

## Error Handling

### Review Queue Errors
- **Queue Full:** Implement queue size limits with oldest-first eviction
- **Concurrent Access:** Use locking mechanisms for queue operations
- **Invalid Review ID:** Return clear error messages

### Feedback Integration Errors
- **Calibration Failure:** Log error and continue without calibration update
- **Pattern Detection Failure:** Gracefully skip pattern analysis
- **Storage Failure:** Retry with exponential backoff

### Audit Logging Errors
- **Write Failure:** Buffer entries and retry
- **Corruption Detection:** Validate hash chains on read
- **Export Failure:** Provide partial export with error indication

## Testing Strategy

### Unit Tests
- ReviewManager queue operations
- ReviewDecision validation
- FeedbackIntegration calibration updates
- AuditLogger hash verification

### Integration Tests
- End-to-end review workflow
- Feedback loop integration
- CLI command execution
- Batch operations

### Performance Tests
- Queue operations with 1000+ items
- Concurrent reviewer access
- Audit log export with large datasets
- Pattern detection on large feedback sets

### User Acceptance Tests
- Interactive CLI review session
- Batch review workflow
- Audit report generation
- Accuracy metrics calculation

## Configuration

### hybrid_config.json Extensions
```json
{
  "hitl": {
    "enabled": true,
    "review_threshold": 70,
    "auto_queue_low_confidence": true,
    "queue_max_size": 10000,
    "batch_similarity_threshold": 0.85,
    "interactive_mode_default": false,
    "audit_log_path": "./audit_logs/",
    "export_formats": ["json", "csv"]
  },
  "review_priorities": {
    "critical_severity_boost": 20,
    "low_confidence_boost": 10,
    "age_penalty_per_hour": 0.5
  }
}
```

## Migration Path

### Phase 1: Core Components (Week 1)
1. Implement ReviewManager with queue operations
2. Create ReviewItem and ReviewDecision data models
3. Add review queue integration to HybridComplianceChecker
4. Basic unit tests

### Phase 2: CLI Interface (Week 2)
1. Implement ReviewCLI with interactive mode
2. Add command-line commands for review operations
3. Create progress indicators and status displays
4. Integration tests for CLI

### Phase 3: Feedback Integration (Week 3)
1. Connect ReviewManager to FeedbackLoop
2. Implement confidence calibration updates
3. Add pattern detection integration
4. Accuracy metrics calculation

### Phase 4: Audit & Reporting (Week 4)
1. Implement AuditLogger with immutable records
2. Add export functionality
3. Create compliance report generation
4. End-to-end testing

### Phase 5: Polish & Documentation (Week 5)
1. Performance optimization
2. User documentation
3. API documentation
4. Deployment guide

## Security Considerations

- **Reviewer Authentication:** Require reviewer ID for all operations
- **Audit Log Integrity:** Use cryptographic hashes to prevent tampering
- **Access Control:** Implement role-based access for sensitive operations
- **Data Privacy:** Sanitize PII from audit logs and exports
- **Secure Storage:** Encrypt sensitive feedback data at rest

## Performance Targets

- **Queue Operations:** < 10ms for add/retrieve operations
- **Review Display:** < 100ms to display review item
- **Feedback Processing:** < 1s to update calibration models
- **Audit Export:** < 5s for 10,000 entries
- **Pattern Detection:** < 30s for 1,000 feedback items

## Future Enhancements

- Web-based review dashboard
- Mobile review interface
- Real-time collaboration features
- Advanced analytics and visualizations
- Machine learning model retraining pipeline
- Integration with external review systems
