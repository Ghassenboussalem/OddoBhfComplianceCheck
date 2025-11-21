# Implementation Plan

- [x] 1. Set up core hybrid architecture infrastructure







  - Create base HybridComplianceChecker class with three-layer architecture
  - Implement AIEngine class with LLM client abstraction and caching
  - Build ConfidenceScorer class for result combination logic
  - Add error handling and fallback mechanisms for AI service failures
  - _Requirements: 1.5, 4.1, 4.2, 5.1, 5.2, 6.6_

- [x] 1.1 Create HybridComplianceChecker base class


  - Implement main orchestration logic for three-layer processing
  - Add method signatures for all eight compliance check types
  - Create configuration system for AI/rule balance settings
  - _Requirements: 1.5, 6.1, 6.2_

- [x] 1.2 Implement AIEngine with LLM abstraction






  - Create unified interface for Token Factory and Gemini APIs
  - Add prompt template system for different check types
  - Implement response caching to reduce redundant API calls
  - Add JSON response parsing and validation
  - _Requirements: 5.2, 5.4, 6.2_

- [x] 1.3 Build ConfidenceScorer for result combination


  - Implement logic to combine AI and rule-based results
  - Create confidence calculation algorithms (0-100 scale)
  - Add status categorization (VERIFIED_BY_BOTH, AI_DETECTED_VARIATION, etc.)
  - Build threshold management for human review flagging
  - _Requirements: 4.1, 4.3, 4.4, 7.4_

- [x] 1.4 Add comprehensive error handling


  - Implement fallback to rule-only processing when AI fails
  - Add retry logic with exponential backoff for API calls
  - Create graceful degradation for partial AI service availability
  - Add logging and monitoring for AI service health
  - _Requirements: 5.5, 6.6_

- [x] 2. Enhance critical compliance checks with AI





  - Convert promotional document detection to hybrid approach
  - Implement semantic performance claims analysis
  - Add context-aware fund name matching
  - Enhance disclaimer validation with semantic understanding
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 3.1, 3.2_

- [x] 2.1 Enhance promotional document detection


  - Create AI prompt template for promotional mention analysis
  - Implement multi-language detection (French/English)
  - Add OCR error and typo handling through semantic matching
  - Build rule-based pre-filtering for keyword hints
  - Combine results with confidence scoring
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2.2 Implement semantic performance claims analysis


  - Create AI prompts to distinguish historical vs predictive claims
  - Add context understanding for performance disclaimers
  - Implement same-slide disclaimer validation logic
  - Build evidence extraction for performance violations
  - _Requirements: 2.1, 2.2, 2.3, 2.5_

- [x] 2.3 Add semantic fund name matching


  - Implement similarity scoring algorithm for fund names
  - Handle abbreviations, reordering, and missing words
  - Create reasoning output for match/no-match decisions
  - Add threshold-based flagging for manual review
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 2.4 Enhance disclaimer validation


  - Create semantic similarity matching for required disclaimers
  - Implement context-aware disclaimer detection
  - Add location tracking for disclaimer placement
  - Build confidence scoring based on semantic match quality
  - _Requirements: 4.2, 4.5, 6.2_

- [x] 2.5 Write unit tests for critical checks






  - Create test cases for promotional document detection edge cases
  - Add performance claims analysis test scenarios
  - Build fund name matching test suite with various name formats
  - Test disclaimer validation with different phrasings
  - _Requirements: 1.1, 2.1, 3.1_

- [x] 3. Convert remaining compliance checks to hybrid approach






  - Enhance registration rules with country extraction AI
  - Convert structure rules to semantic validation
  - Improve general rules with context understanding
  - Add semantic analysis to values/securities checks
  - _Requirements: 6.1, 6.2, 6.3_

- [x] 3.1 Enhance registration compliance checking


  - Implement AI-powered country extraction from documents
  - Add semantic matching for country name variations
  - Create validation logic against authorized countries database
  - Build confidence scoring for country identification
  - _Requirements: 6.1, 6.2_


- [x] 3.2 Convert structure rules to semantic validation



  - Enhance target audience detection with context understanding
  - Improve management company mention validation
  - Add semantic date and fund name validation
  - Create flexible structure validation with AI reasoning
  - _Requirements: 6.1, 6.2, 4.2_





- [x] 3.3 Improve general rules with AI context


  - Add semantic glossary term detection for technical language
  - Enhance Morningstar date validation with context awareness
  - Implement intelligent technical term identification
  - Create context-aware rule application logic
  - _Requirements: 6.1, 6.2, 4.2_

- [x] 3.4 Add semantic analysis to values/securities checks

  - Implement context understanding for company mentions
  - Distinguish between examples and recommendations using AI
  - Add semantic disclaimer detection for securities content
  - Create intent analysis for investment advice detection
  - _Requirements: 6.1, 6.2, 2.3_

- [x] 3.5 Write integration tests for remaining checks






  - Test registration country extraction accuracy
  - Validate structure rule semantic understanding
  - Test general rules context awareness
  - Verify values/securities intent detection
  - _Requirements: 6.1, 6.2_

- [-] 4. Implement performance optimization and caching



  - Add intelligent caching system for AI responses
  - Implement batch processing for multiple documents
  - Create async processing for improved throughput
  - Add performance monitoring and metrics collection
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 4.1 Create intelligent caching system


  - Implement content-based cache keys for AI responses
  - Add cache invalidation logic for rule updates
  - Create cache size management and LRU eviction
  - Build cache hit rate monitoring and reporting
  - _Requirements: 5.2, 5.4_

- [x] 4.2 Implement batch processing capabilities


  - Create document queue management system
  - Add parallel processing for multiple AI calls
  - Implement rate limiting for API compliance
  - Build progress tracking for batch operations
  - _Requirements: 5.2, 5.3_


- [x] 4.3 Add async processing architecture







  - Convert synchronous AI calls to async operations
  - Implement concurrent processing for different check types
  - Add timeout handling for long-running AI requests
  - Create async result aggregation logic
  - _Requirements: 5.3_

- [x] 4.4 Build performance monitoring system






  - Add timing metrics for each processing layer
  - Create API usage tracking and cost monitoring
  - Implement accuracy tracking over time
  - Build performance dashboard and alerting
  - _Requirements: 5.1, 5.3, 7.1_

- [x] 5. Add learning and feedback capabilities







  - Implement confidence calibration system
  - Create feedback loop for human corrections
  - Add pattern detection for rule enhancement
  - Build accuracy tracking and threshold adjustment
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_


- [x] 5.1 Implement confidence calibration system

  - Create accuracy tracking database for confidence scores
  - Add calibration algorithms based on historical performance
  - Implement dynamic threshold adjustment logic
  - Build confidence score reliability metrics
  - _Requirements: 7.1, 7.4_



- [x] 5.2 Create feedback loop for human corrections





  - Add interface for human reviewers to provide corrections
  - Implement feedback storage and pattern analysis
  - Create learning algorithms to improve future predictions
  - Build correction impact tracking and validation
  - _Requirements: 7.2, 7.5_

- [x] 5.3 Add pattern detection for rule enhancement


  - Implement AI pattern discovery for missed violations
  - Create rule suggestion system based on AI findings
  - Add false positive pattern detection and filtering
  - Build automated rule update recommendations
  - _Requirements: 7.3, 7.5_

- [x] 5.4 Build comprehensive testing framework






  - Create end-to-end testing with real document samples
  - Add accuracy benchmarking against current system
  - Implement regression testing for AI model changes
  - Build performance testing for scalability validation
  - _Requirements: 7.1, 7.2_

- [x] 6. Integration and backward compatibility




  - Ensure seamless integration with existing check.py workflow
  - Maintain JSON output format compatibility
  - Add configuration options for AI/rule balance
  - Create migration path from current system
  - _Requirements: 6.3, 6.4_

- [x] 6.1 Ensure seamless integration with existing workflow


  - Modify check.py to use HybridComplianceChecker
  - Maintain all existing command-line interfaces
  - Preserve current error handling and reporting
  - Add backward compatibility flags for gradual migration
  - _Requirements: 6.3_

- [x] 6.2 Maintain JSON output format compatibility


  - Ensure enhanced violations match existing structure
  - Add new fields (confidence, reasoning) as optional extensions
  - Preserve existing violation categorization and severity levels
  - Create output format validation tests
  - _Requirements: 6.3, 4.1_

- [x] 6.3 Add configuration system for AI/rule balance


  - Create configuration file for AI service settings
  - Add runtime switches for AI enhancement levels
  - Implement feature flags for gradual rollout
  - Build configuration validation and error handling
  - _Requirements: 6.4, 5.5_

- [x] 6.4 Create comprehensive documentation






  - Write API documentation for new hybrid classes
  - Create configuration guide for AI service setup
  - Add troubleshooting guide for common issues
  - Build migration guide from current system
  - _Requirements: 6.1, 6.2_