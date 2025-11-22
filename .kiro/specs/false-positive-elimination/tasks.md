# Implementation Plan - False Positive Elimination

## Overview

This plan eliminates 34 false positives (85% error rate) by shifting from keyword matching to AI-driven context understanding. Each task is focused and builds incrementally.

**Target**: Reduce violations from 40 → 6 on `exemple.json` (0 false positives)

---

## Phase 1: Core Infrastructure (Foundation)

- [x] 1. Create WhitelistManager for term management



  - Implement WhitelistManager class with whitelist building logic
  - Add methods: `build_whitelist()`, `is_whitelisted()`, `get_whitelist_reason()`
  - Define default whitelists: strategy terms, regulatory terms, benchmark terms, generic financial terms
  - Add fund name extraction from document metadata
  - Create unit tests for whitelist functionality
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  - _File: `whitelist_manager.py` (NEW)_

- [x] 2. Create ContextAnalyzer for semantic understanding





  - Implement ContextAnalyzer class with AI-powered context analysis
  - Add method: `analyze_context(text, check_type)` returning ContextAnalysis dataclass
  - Add method: `is_fund_strategy_description(text)` for fund description detection
  - Add method: `is_investment_advice(text)` for client advice detection
  - Add method: `extract_subject(text)` to identify WHO performs action
  - Implement fallback rule-based analysis for AI failures
  - Create AI prompt templates for context analysis
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - _File: `context_analyzer.py` (NEW)_

- [x] 3. Create IntentClassifier for intent detection





  - Implement IntentClassifier class with intent classification logic
  - Add method: `classify_intent(text)` returning IntentClassification dataclass
  - Add method: `is_client_advice(text)` for advice detection
  - Add method: `is_fund_description(text)` for description detection
  - Create AI prompt template for intent classification (ADVICE|DESCRIPTION|FACT|EXAMPLE)
  - Handle edge cases and ambiguous text
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - _File: `intent_classifier.py` (NEW)_

- [x] 4. Create EvidenceExtractor for evidence identification





  - Implement EvidenceExtractor class with evidence extraction logic
  - Add method: `extract_evidence(text, violation_type)` returning Evidence dataclass
  - Add method: `find_performance_data(text)` to detect actual performance numbers
  - Add method: `find_disclaimer(text, required_disclaimer)` for semantic disclaimer matching
  - Implement location tracking (slide, section)
  - Create AI prompts for evidence extraction
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5_
  - _File: `evidence_extractor.py` (NEW)_

- [x] 5. Create SemanticValidator for meaning-based validation





  - Implement SemanticValidator class integrating all components
  - Add method: `validate_securities_mention(text, whitelist)` for securities validation
  - Add method: `validate_performance_claim(text)` for performance data detection
  - Add method: `validate_prospectus_consistency(doc_text, prospectus_text)` for contradiction detection
  - Implement confidence scoring based on AI + rules agreement
  - Add error handling and graceful degradation
  - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 5.2, 5.3, 5.4, 5.5_
  - _File: `semantic_validator.py` (NEW)_

---

## Phase 2: Fix Securities/Values Detection (Eliminates 25 False Positives)

- [x] 6. Replace check_prohibited_phrases with AI context-aware version





  - Locate current `check_prohibited_phrases()` function in `agent.py`
  - Create new `check_prohibited_phrases_ai()` function using ContextAnalyzer and IntentClassifier
  - Implement logic: Only flag if intent=ADVICE AND subject=client
  - Add AI reasoning and evidence to violation output
  - Test with "Tirer parti du momentum" → should NOT flag
  - Test with "Vous devriez investir" → should flag
  - Update function calls in `check.py` to use new version
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - _File: `agent.py` (MODIFY)_
  - _Impact: Eliminates 25 false positives_

---

## Phase 3: Fix Repeated Mentions Detection (Eliminates 16 False Positives)

- [x] 7. Replace check_repeated_securities with whitelist-aware version





  - Locate current `check_repeated_securities()` function in `agent.py`
  - Create new `check_repeated_securities_ai()` function using WhitelistManager
  - Build whitelist from document (fund name, strategy terms, regulatory terms)
  - Extract capitalized words as potential company names
  - Skip whitelisted terms before counting
  - Use SemanticValidator to verify external company names (3+ mentions)
  - Test with "ODDO BHF" (31 mentions) → should NOT flag
  - Test with "momentum" (2 mentions) → should NOT flag
  - Test with "SRI" (2 mentions) → should NOT flag
  - Update function calls in `check.py` to use new version
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  - _File: `agent.py` (MODIFY)_
  - _Impact: Eliminates 16 false positives_

---

## Phase 4: Fix Performance Detection (Eliminates 3 False Positives)

- [x] 8. Replace check_performance_disclaimers with data-aware version





  - Locate current `check_performance_disclaimers_ai()` function in `check_functions_ai.py`
  - Modify to use EvidenceExtractor.find_performance_data()
  - Only check disclaimers if ACTUAL performance data present (numbers with %)
  - Use semantic matching for disclaimer detection (not keyword matching)
  - Verify disclaimer is on SAME slide as performance data
  - Test with "attractive performance" → should NOT flag (no numbers)
  - Test with "15% return" without disclaimer → should flag
  - Test with "performance objective" → should NOT flag (descriptive)
  - Update function calls in `check.py` to use new version
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  - _File: `check_functions_ai.py` (MODIFY)_
  - _Impact: Eliminates 3 false positives_

- [x] 9. Fix check_document_starts_with_performance





  - Locate current function in `check_functions_ai.py` or `agent.py`
  - Modify to use EvidenceExtractor.find_performance_data() on cover page only
  - Only flag if ACTUAL performance numbers on cover (not keywords)
  - Test with exemple.json cover page → should NOT flag (no performance data)
  - _Requirements: 3.1, 3.2, 4.1, 4.2, 4.3_
  - _File: `check_functions_ai.py` or `agent.py` (MODIFY)_
  - _Impact: Part of 3 false positive elimination_

---

## Phase 5: Add Missing Checks (Catches 2 Missed Violations)

- [x] 10. Add check_risk_profile_consistency for cross-slide validation





  - Create new function `check_risk_profile_consistency(doc)` in `check_functions_ai.py`
  - Extract risks from Slide 2 using regex patterns
  - Extract risks from final page (page_de_fin)
  - Compare risk counts: Slide 2 should have >= risks from final page
  - If Slide 2 has fewer risks, flag as MAJOR violation
  - Include evidence: list missing risks
  - Test with exemple.json → should detect Slide 2 has 4 risks vs 11+ elsewhere
  - Add to check workflow in `check.py`
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  - _File: `check_functions_ai.py` (NEW FUNCTION)_
  - _Impact: Catches 1 missed violation_

- [x] 11. Add check_anglicisms_retail for English terms in retail docs





  - Create new function `check_anglicisms_retail(doc, client_type)` in `check_functions_ai.py`
  - Only apply to retail documents (skip professional)
  - Define list of common English terms in French financial docs
  - Check if terms are used in document
  - Check if glossary exists ("glossaire" or "glossary")
  - If terms found and no glossary, flag as MINOR violation
  - Test with exemple.json → should detect "momentum" without glossary
  - Add to check workflow in `check.py`
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  - _File: `check_functions_ai.py` (NEW FUNCTION)_
  - _Impact: Catches 1 missed violation_

---

## Phase 6: Integration and Testing

- [x] 12. Update check.py to use new functions





  - Replace old function calls with new AI-enhanced versions
  - Ensure backward compatibility (feature flag for gradual rollout)
  - Add configuration option: `USE_AI_CONTEXT_AWARE = True`
  - Update error handling for new components
  - Maintain existing JSON output format
  - Add AI reasoning fields to violation output
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 10.1, 10.2, 10.3, 10.4, 10.5_
  - _File: `check.py` (MODIFY)_

- [x] 13. Create data models for new components





  - Define ContextAnalysis dataclass
  - Define IntentClassification dataclass
  - Define ValidationResult dataclass
  - Define Evidence dataclass
  - Define PerformanceData dataclass
  - Define DisclaimerMatch dataclass
  - Add type hints throughout
  - _Requirements: All_
  - _File: `data_models.py` (NEW)_

- [x] 14. Add configuration for AI context-aware checking





  - Extend `hybrid_config.json` with context_analysis settings
  - Add whitelist configuration options
  - Add performance optimization settings (batch_slides, timeout)
  - Add confidence threshold configuration
  - Create default configuration template
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 10.1, 10.2, 10.3, 10.4, 10.5_
  - _File: `hybrid_config.json` (MODIFY)_

---

## Phase 7: Validation and Metrics

- [x] 15. Run full test on exemple.json and validate results





  - Execute: `python check.py exemple.json`
  - Verify total violations = 6 (down from 40)
  - Verify 0 false positives
  - Verify all 6 actual violations still caught:
    - Missing "Document promotionnel" (STRUCT_003)
    - Missing target audience (STRUCT_004)
    - Missing management company mention (STRUCT_011)
    - Incomplete risk profile Slide 2 (STRUCT_009)
    - Missing glossary (GEN_005)
    - Morningstar date missing (GEN_021)
  - Compare output to Kiro analysis
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 16. Create unit tests for new components






  - Test WhitelistManager: fund name extraction, term whitelisting
  - Test ContextAnalyzer: fund descriptions vs advice classification
  - Test IntentClassifier: all intent types correctly identified
  - Test EvidenceExtractor: performance data detection, disclaimer matching
  - Test SemanticValidator: whitelist filtering, semantic validation
  - Create test fixtures with known inputs/outputs
  - Achieve >80% code coverage
  - _Requirements: All_
  - _File: `test_false_positive_elimination.py` (NEW)_

- [x] 17. Create integration tests for end-to-end validation






  - Test: Fund strategy descriptions not flagged (25 cases)
  - Test: Fund name repetition not flagged (16 cases)
  - Test: Performance keywords without data not flagged (3 cases)
  - Test: Actual violations still caught (6 cases)
  - Test: AI fallback works when service unavailable
  - Test: Confidence scores are appropriate
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  - _File: `test_integration_false_positives.py` (NEW)_

- [x] 18. Add performance monitoring and metrics






  - Implement ComplianceMetrics class for tracking
  - Record: false positive rate, false negative rate, precision, recall
  - Record: AI API calls, cache hits, fallback rate
  - Record: processing time per document
  - Create metrics dashboard/summary output
  - Add metrics to check.py output (optional --show-metrics flag)
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
  - _File: `compliance_metrics.py` (NEW)_

---

## Phase 8: Documentation and Deployment

- [x] 19. Update documentation for new AI-enhanced checking






  - Document new components: ContextAnalyzer, IntentClassifier, etc.
  - Update README with new features and configuration
  - Create migration guide from old to new system
  - Document AI prompt templates and customization
  - Add troubleshooting guide for common issues
  - Document whitelist management and customization
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_
  - _File: `README.md`, `MIGRATION_GUIDE.md` (MODIFY/NEW)_

- [x] 20. Create comparison report showing improvements






  - Generate before/after comparison on exemple.json
  - Show: 40 violations → 6 violations
  - Show: 34 false positives eliminated
  - Show: 0 false negatives introduced
  - Include confidence scores and AI reasoning examples
  - Document processing time and performance metrics
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  - _File: `IMPROVEMENT_REPORT.md` (NEW)_

---

## Notes

- **Core tasks (1-15)**: Must be implemented for functional system
- **Testing tasks (16-18)**: Optional but highly recommended for production
- **Documentation tasks (19-20)**: Optional but recommended for team adoption
- Each task builds incrementally on previous tasks
- Tasks marked with `*` are optional (testing/documentation)
- Focus on Phase 1-5 first (core functionality)
- Phase 6-8 can be done in parallel or after core is working

## Task Execution Order

**Recommended sequence**:
1. Phase 1 (Tasks 1-5): Build foundation components
2. Phase 2 (Task 6): Fix biggest issue (25 false positives)
3. Phase 3 (Task 7): Fix second biggest issue (16 false positives)
4. Phase 4 (Tasks 8-9): Fix remaining false positives (3)
5. Phase 5 (Tasks 10-11): Add missing checks (2 violations)
6. Phase 6 (Tasks 12-14): Integration
7. Phase 7 (Task 15): Validation on exemple.json
8. Phase 8 (Tasks 16-20): Testing and documentation (optional)

## Success Criteria

After completing core tasks (1-15):
- ✅ `python check.py exemple.json` produces 6 violations (not 40)
- ✅ 0 false positives
- ✅ All 6 actual violations still detected
- ✅ AI reasoning included in violation output
- ✅ System gracefully handles AI failures (fallback to rules)
- ✅ Processing time < 60 seconds per document

