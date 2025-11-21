# AI-Enhanced Compliance Checker Design

## Overview

The enhanced compliance checker implements a three-layer hybrid architecture that combines AI semantic understanding with rule-based validation. This design addresses the current system's limitations with keyword matching fragility, regex specificity, and lack of context awareness while maintaining performance and reliability.

### Core Philosophy
- **AI does the heavy lifting** (understanding, extraction, reasoning)
- **Rules provide guardrails** (validation, confidence scoring, edge cases)  
- **Combine both for robustness** (AI + Rules > AI alone or Rules alone)

## Architecture

### Layer 1: Rule-Based Pre-Filtering (Fast Screening)
```
Purpose: Quick elimination of obvious cases
- If keyword found → Flag for AI review with hints
- If keyword NOT found → Still send to AI (might be phrased differently)
- Rules act as "hints" to AI, not replacements
- Processing time: < 1ms per check
```

### Layer 2: AI Analysis (Deep Understanding)
```
Purpose: Semantic understanding and context analysis
- Extract entities with context using LLM prompts
- Understand intent and meaning beyond keywords
- Handle variations, typos, multiple languages
- Provide reasoning for decisions
- Processing time: 500-2000ms per check
```

### Layer 3: Rule-Based Validation (Confidence Scoring)
```
Purpose: Validate AI output and boost confidence
- If AI says "compliant" AND rules confirm → High confidence (95-100%)
- If AI says "violation" BUT rules don't see it → Medium confidence (70-84%), flag for review
- If both agree on violation → Critical violation (90-100%)
- If AI finds new patterns rules missed → Log for rule enhancement
```

## Components and Interfaces

### Enhanced Checker Classes

#### HybridComplianceChecker
```python
class HybridComplianceChecker:
    def __init__(self, ai_engine, rule_engine, confidence_scorer):
        self.ai_engine = ai_engine
        self.rule_engine = rule_engine
        self.confidence_scorer = confidence_scorer
    
    def check_compliance(self, document, check_type):
        # Layer 1: Rule pre-filtering
        rule_result = self.rule_engine.quick_scan(document, check_type)
        
        # Layer 2: AI analysis with rule hints
        ai_result = self.ai_engine.analyze(document, check_type, rule_result)
        
        # Layer 3: Confidence scoring and validation
        final_result = self.confidence_scorer.combine_results(rule_result, ai_result)
        
        return final_result
```

#### AIEngine
```python
class AIEngine:
    def __init__(self, llm_client, prompt_templates):
        self.llm_client = llm_client
        self.prompt_templates = prompt_templates
        self.cache = {}
    
    def analyze(self, document, check_type, rule_hints):
        # Get appropriate prompt template
        prompt = self.prompt_templates.get_prompt(check_type, document, rule_hints)
        
        # Check cache first
        cache_key = self._generate_cache_key(prompt)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Call LLM
        result = self.llm_client.call(prompt)
        
        # Cache result
        self.cache[cache_key] = result
        return result
```

#### ConfidenceScorer
```python
class ConfidenceScorer:
    def combine_results(self, rule_result, ai_result):
        base_confidence = ai_result.get('confidence', 50)
        
        # Boost confidence if both agree
        if self._results_agree(rule_result, ai_result):
            base_confidence = min(100, base_confidence + 15)
            status = "VERIFIED_BY_BOTH"
        elif rule_result['found'] and not ai_result['violation']:
            status = "FALSE_POSITIVE_FILTERED"
        elif not rule_result['found'] and ai_result['violation']:
            status = "AI_DETECTED_VARIATION"
        else:
            status = "VIOLATION_CONFIRMED"
            base_confidence = 95
        
        return {
            'violation': ai_result['violation'],
            'confidence': base_confidence,
            'status': status,
            'evidence': ai_result['evidence'],
            'reasoning': ai_result['reasoning']
        }
```

### Enhanced Check Functions

#### Promotional Document Detection
```python
def check_promotional_mention_enhanced(document):
    # Layer 1: Rule-based quick scan
    rule_result = {
        'found_keywords': [],
        'suspicious_phrases': [],
        'confidence': 0
    }
    
    promotional_keywords = [
        'document promotionnel', 'promotional document',
        'document à caractère promotionnel', 'marketing material'
    ]
    
    cover_text = extract_cover_text(document)
    for keyword in promotional_keywords:
        if keyword in cover_text.lower():
            rule_result['found_keywords'].append(keyword)
            rule_result['confidence'] += 20
    
    # Layer 2: AI Analysis
    ai_prompt = f"""
    Analyze this cover page for promotional document indication:
    
    COVER PAGE TEXT: {cover_text}
    
    TASK:
    1. Is there a clear indication this is a promotional/marketing document?
    2. What specific phrases indicate this?
    3. Are there any variations or indirect mentions?
    4. Consider: "document promotionnel", "à caractère promotionnel", 
       "marketing material", "promotional", etc.
    
    REGULATORY CONTEXT:
    - French/EU regulations require explicit promotional document labeling
    - Must be visible and unambiguous on cover page
    
    Respond with JSON:
    {{
      "has_promotional_mention": true/false,
      "confidence": 0-100,
      "found_phrases": ["phrase1", "phrase2"],
      "reasoning": "explanation",
      "location": "where on page",
      "compliance_status": "COMPLIANT" / "NON_COMPLIANT" / "UNCLEAR",
      "variations_detected": ["any non-standard phrasings"]
    }}
    
    RULE-BASED HINTS: Keywords found: {rule_result['found_keywords']}
    """
    
    ai_result = call_llm(ai_prompt)
    
    # Layer 3: Combine and validate
    return confidence_scorer.combine_results(rule_result, ai_result)
```

#### Performance Claims Analysis
```python
def check_performance_claims_enhanced(document):
    # AI prompt for context understanding
    ai_prompt = f"""
    Analyze this text for performance claims and context:
    
    TEXT: {slide_text}
    
    DISTINGUISH:
    1. Historical fact: "The fund generated 5% returns in 2023"
    2. Predictive claim: "The fund will deliver strong performance"  
    3. Capability statement: "The fund aims to generate returns"
    4. Example/illustration: "For example, a 5% return would mean..."
    
    TASK:
    - Identify the type of performance statement
    - Determine if disclaimers are required
    - Check if appropriate disclaimers are present
    - Assess compliance risk level
    
    Respond with JSON:
    {{
      "performance_type": "historical|predictive|capability|example",
      "requires_disclaimer": true/false,
      "disclaimer_present": true/false,
      "disclaimer_location": "same_slide|different_slide|not_found",
      "compliance_status": "COMPLIANT|NON_COMPLIANT|NEEDS_REVIEW",
      "confidence": 0-100,
      "evidence": "specific text found",
      "reasoning": "explanation of decision"
    }}
    """
    
    return process_ai_analysis(ai_prompt, document)
```

#### Semantic Fund Name Matching
```python
def check_fund_name_match_enhanced(doc_fund_name, prospectus_fund_name):
    ai_prompt = f"""
    Compare these two fund names semantically:
    
    PROSPECTUS FUND: {prospectus_fund_name}
    DOCUMENT FUND: {doc_fund_name}
    
    Do they refer to the same fund? Consider:
    - Abbreviations (ODDO BHF vs Oddo Bank)
    - Word order (Algo Trend US vs US Algo Trend)
    - Missing/extra words (Fund, SICAV, etc.)
    - Different naming conventions
    - Legal entity variations
    
    Respond with JSON:
    {{
      "is_same_fund": true/false,
      "confidence": 0-100,
      "similarity_score": 0-100,
      "reasoning": "explanation",
      "differences_noted": ["list of differences"],
      "match_factors": ["what makes them similar"]
    }}
    """
    
    return process_ai_analysis(ai_prompt)
```

## Data Models

### Enhanced Violation Structure
```python
@dataclass
class EnhancedViolation:
    type: str  # STRUCTURE, PERFORMANCE, etc.
    severity: str  # CRITICAL, MAJOR, WARNING
    slide: str
    location: str
    rule: str
    message: str
    evidence: str
    confidence: int  # 0-100
    ai_reasoning: str  # AI explanation
    rule_validation: str  # Rule engine confirmation
    status: str  # VERIFIED_BY_BOTH, AI_DETECTED_VARIATION, etc.
    suggestions: List[str]  # Remediation suggestions
```

### AI Analysis Result
```python
@dataclass
class AIAnalysisResult:
    violation_detected: bool
    confidence: int
    evidence: List[str]
    reasoning: str
    context_understanding: str
    variations_found: List[str]
    location_details: str
```

### Rule Validation Result
```python
@dataclass
class RuleValidationResult:
    keywords_found: List[str]
    patterns_matched: List[str]
    confidence_boost: int
    validation_status: str  # CONFIRMS, CONTRADICTS, NEUTRAL
```

## Error Handling

### AI Service Failures
```python
class AIServiceHandler:
    def __init__(self, primary_ai, fallback_ai, rule_engine):
        self.primary_ai = primary_ai
        self.fallback_ai = fallback_ai
        self.rule_engine = rule_engine
    
    def analyze_with_fallback(self, prompt):
        try:
            return self.primary_ai.call(prompt)
        except Exception as e:
            logger.warning(f"Primary AI failed: {e}")
            try:
                return self.fallback_ai.call(prompt)
            except Exception as e2:
                logger.error(f"Fallback AI failed: {e2}")
                return self.rule_engine.fallback_analysis()
```

### Confidence Calibration
```python
class ConfidenceCalibrator:
    def __init__(self):
        self.accuracy_history = []
        self.threshold_adjustments = {}
    
    def calibrate_confidence(self, predicted_confidence, actual_result):
        # Track accuracy over time
        self.accuracy_history.append({
            'predicted': predicted_confidence,
            'actual': actual_result,
            'timestamp': datetime.now()
        })
        
        # Adjust thresholds based on performance
        if len(self.accuracy_history) > 100:
            self._recalibrate_thresholds()
    
    def _recalibrate_thresholds(self):
        # Analyze recent accuracy and adjust confidence thresholds
        recent_data = self.accuracy_history[-100:]
        # Implementation for threshold adjustment logic
```

## Testing Strategy

### Unit Testing Approach
- **Rule Engine Tests**: Verify keyword matching and pattern detection
- **AI Engine Tests**: Mock AI responses and test prompt generation
- **Confidence Scorer Tests**: Validate scoring logic with various input combinations
- **Integration Tests**: Test full pipeline with sample documents

### AI Testing Methodology
```python
class AITestFramework:
    def __init__(self):
        self.test_cases = self._load_test_cases()
        self.mock_responses = self._load_mock_responses()
    
    def test_ai_accuracy(self, check_type):
        results = []
        for test_case in self.test_cases[check_type]:
            # Use mock AI response for consistent testing
            mock_response = self.mock_responses[test_case['id']]
            result = self.process_with_mock(test_case, mock_response)
            results.append(result)
        
        return self._calculate_accuracy_metrics(results)
```

### Performance Testing
- **Latency Tests**: Measure response times for each layer
- **Throughput Tests**: Test batch processing capabilities  
- **Cost Analysis**: Monitor AI API usage and token consumption
- **Degradation Tests**: Verify graceful fallback when AI services fail

### Validation Testing
- **False Positive Detection**: Test with documents that should pass
- **False Negative Detection**: Test with documents that should fail
- **Edge Case Handling**: Test with malformed or unusual documents
- **Multi-language Testing**: Verify French/English handling

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)
- Implement HybridComplianceChecker base class
- Create AIEngine with prompt templates
- Build ConfidenceScorer logic
- Set up error handling and fallback mechanisms

### Phase 2: Critical Checks Enhancement (Week 3-4)
- Enhance promotional document detection
- Improve performance claims analysis
- Implement semantic fund name matching
- Add comprehensive disclaimer validation

### Phase 3: Remaining Checks (Week 5-6)
- Convert registration, structure, and general rules
- Enhance ESG and values/securities checks
- Implement prospectus semantic matching
- Add batch processing optimization

### Phase 4: Optimization & Learning (Week 7-8)
- Implement caching and performance optimization
- Add confidence calibration system
- Create feedback loop for continuous improvement
- Comprehensive testing and validation

This design provides a robust foundation for transforming your brittle rule-based system into an intelligent, context-aware compliance checker that maintains reliability while dramatically improving accuracy and flexibility.