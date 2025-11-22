# Implementation Plan - Human-in-the-Loop Integration

## Status: Ready for Implementation

This implementation plan converts the HITL design into actionable coding tasks. The system will build on existing components (`feedback_loop.py`, `confidence_scorer.py`, `hybrid_compliance_checker.py`) to create a complete interactive review workflow.

---

## Core Components

- [x] 1. Create ReviewManager for queue management





  - Implement ReviewManager class with queue operations (add, retrieve, prioritize)
  - Create ReviewItem and ReviewDecision data models with all required fields
  - Implement priority scoring algorithm (confidence ascending, severity descending)
  - Add filtering capabilities (by check type, severity, confidence range)
  - Implement batch operations for similar items
  - Add queue statistics tracking (pending, in_review, reviewed counts)
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 2. Integrate ReviewManager with HybridComplianceChecker





  - Modify HybridComplianceChecker to automatically queue low-confidence violations
  - Update check_compliance method to call ReviewManager.add_to_queue when confidence < threshold
  - Pass all required context (document, AI reasoning, evidence, rule) to queue
  - Ensure backward compatibility with existing check.py workflow
  - _Requirements: 1.1, 7.1, 7.3_

- [x] 3. Build CLI review interface (review.py)





  - Create ReviewCLI class with interactive session management
  - Implement `next` command to display pending reviews with full context
  - Implement `approve` command with optional notes parameter
  - Implement `reject` command requiring explanatory notes
  - Implement `skip` command for navigation
  - Implement `status` command showing queue summary and progress
  - Add progress indicators (X of Y reviews, completion percentage)
  - Format output for readability (colors, sections, clear prompts)
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 4. Implement batch review operations





  - Add similarity detection algorithm to group related violations
  - Implement batch selection by similarity score, check type, or document
  - Create bulk approve/reject functionality
  - Ensure individual feedback is recorded for each item in batch
  - Add batch operation commands to CLI (`batch --check-type=X --action=approve`)
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 5. Connect review feedback to learning system





  - Update FeedbackInterface.provide_correction to trigger immediate calibration
  - Integrate with ConfidenceCalibrator to update models within 1 second
  - Connect to PatternDetector for false positive/negative analysis
  - Ensure feedback history is maintained for audit purposes
  - Add real-time confidence adjustment based on reviewer corrections
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 6. Add review metrics and reporting





  - Implement metrics tracking (pending, completed, average review time)
  - Calculate AI accuracy metrics (precision, recall, F1 score) from feedback
  - Create performance report generation showing accuracy trends
  - Identify check types with lowest confidence for improvement
  - Add reviewer productivity metrics (reviews per day, average time)
  - Implement `metrics` command in CLI to display current statistics
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 7. Implement audit trail system





  - Create AuditLogger class with immutable record storage
  - Record all review actions with timestamp, reviewer ID, and decision
  - Store original AI prediction alongside human decision and reasoning
  - Implement JSON and CSV export functionality
  - Add cryptographic hashing for tamper detection
  - Create compliance report generation (review coverage, accuracy metrics)
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 8. Integrate with check.py workflow





  - Add `--review-mode` flag to enter interactive review after checking
  - Display summary of violations requiring review after check.py completes
  - Add review threshold configuration to hybrid_config.json
  - Ensure 100% backward compatibility with existing JSON output
  - Update help text and documentation for new flags
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 9. Create review data persistence





  - Implement JSON-based storage for review queue (review_queue.json)
  - Add load/save functionality for queue state persistence
  - Implement audit log storage with rotation (audit_logs/)
  - Add data migration utilities for schema updates
  - Ensure thread-safe file operations with locking
  - _Requirements: 8.1, 8.2, 8.4_

- [x] 10. Add configuration and setup





  - Extend hybrid_config.json with HITL settings (thresholds, paths, behavior)
  - Add review threshold configuration (default: 70)
  - Configure queue size limits and eviction policies
  - Add batch similarity threshold configuration
  - Create default configuration template with documentation
  - _Requirements: 7.5_

---

## Testing & Validation

- [ ]* 11. Write integration tests
  - Test end-to-end review workflow (submit → review → feedback → learning)
  - Test CLI command execution and output formatting
  - Test batch operations with multiple similar items
  - Test audit trail integrity and export functionality
  - Test backward compatibility with existing check.py workflow
  - _Requirements: All_

- [ ]* 12. Create example review scenarios
  - Create sample documents with low-confidence violations
  - Document example CLI review sessions
  - Create test data for batch review operations
  - Provide example audit reports
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

---

## Documentation

- [ ]* 13. Write user documentation
  - Create HITL user guide with CLI command reference
  - Document review workflow and best practices
  - Add configuration guide for hybrid_config.json HITL settings
  - Create troubleshooting guide for common issues
  - Document audit trail and compliance reporting
  - _Requirements: All_

---

## Notes

- All core components (1-10) must be implemented for a functional HITL system
- Testing tasks (11-12) are optional but recommended for production readiness
- Documentation (13) is optional but highly recommended for user adoption
- Each task builds incrementally on previous tasks
- The system integrates with existing components: FeedbackInterface, ConfidenceCalibrator, PatternDetector, PerformanceMonitor
- Backward compatibility with existing check.py workflow is critical
