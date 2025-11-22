# Design Document - False Positive Elimination

## Overview

This design transforms the compliance checker from brittle keyword matching to AI-driven contextual understanding. The core insight: an LLM naturally understands that "The fund invests in momentum strategies" is a fund description (ALLOWED), not investment advice (PROHIBITED). We eliminate 34 false positives (85% error rate) by leveraging AI's semantic understanding instead of regex patterns.

### Current Problem

The existing system generates 40 violations on `exemple.json`, but 34 are false positives:
- **25 false positives**: Fund strategy descriptions flagged as investment advice
- **16 false positives**: Fund name/strategy terms flagged as repeated securities
- **3 false positives**: Performance keywords flagged without actual data

### Target State

After implementation: **6 violations, 0 false positives** (matches Kiro analysis)

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Document Input (JSON)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Context-Aware Compliance Checker                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  1. Pre-Processing: Extract metadata & whitelist      │  │
│  │     - Fund name → whitelist                           │  │
│  │     - Strategy terms → whitelist                      │  │
│  │     - Regulatory terms → whitelist                    │  │
│  └───────────────────────────────────────────────────────┘  │
│                       │                                      │
│                       ▼                                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  2. AI Context Analysis (Primary)                     │  │
│  │     - Intent Classification                           │  │
│  │     - Semantic Understanding                          │  │
│  │     - Evidence Extraction                             │  │
│  └───────────────────────────────────────────────────────┘  │
│                       │                                      │
│                       ▼                                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  3. Rule-Based Validation (Secondary)                 │  │
│  │     - Quick screening                                 │  │
│  │     - Confidence boosting                             │  │
│  │     - Fallback when AI unavailable                    │  │
│  └───────────────────────────────────────────────────────┘  │
│                       │                                      │
│                       ▼                                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  4. Result Combination & Confidence Scoring           │  │
│  │     - AI + Rules agreement → High confidence          │  │
│  │     - AI only → Medium confidence                     │  │
│  │     - Disagreement → Flag for review                  │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Violations Output (with confidence & reasoning)      │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
compliance_checker/
├── context_analyzer.py          # NEW: AI-powered context understanding
├── intent_classifier.py         # NEW: Classify text intent (advice vs description)
├── semantic_validator.py        # NEW: Semantic validation logic
├── evidence_extractor.py        # NEW: Extract and quote evidence
├── whitelist_manager.py         # NEW: Manage fund names, strategy terms
├── check_functions_ai.py        # MODIFY: Update to use new components
├── agent.py                     # MODIFY: Update check functions
├── ai_engine.py                 # EXISTING: LLM abstraction (reuse)
└── hybrid_compliance_checker.py # EXISTING: Orchestration (reuse)
```

## Components

### 1. Context Analyzer (NEW)

**Purpose**: Understand the semantic meaning and context of text passages

**File**: `context_analyzer.py`

**Key Methods**:
```python
class ContextAnalyzer:
    def __init__(self, ai_engine):
        self.ai_engine = ai_engine
        self.prompt_templates = PromptTemplateLibrary()
    
    def analyze_context(self, text: str, check_type: str) -> ContextAnalysis:
        """
        Analyze text context using AI
        
        Returns:
            ContextAnalysis with:
            - subject: WHO is performing action (fund vs client)
            - intent: WHAT is the purpose (describe vs advise)
            - confidence: 0-100
            - reasoning: Explanation
        """
    
    def is_fund_strategy_description(self, text: str) -> bool:
        """Check if text describes fund strategy (ALLOWED)"""
    
    def is_investment_advice(self, text: str) -> bool:
        """Check if text advises clients (PROHIBITED)"""
    
    def extract_subject(self, text: str) -> str:
        """Extract WHO is performing the action"""
```

**Integration**: Used by all check functions to understand context before flagging violations


### 2. Intent Classifier (NEW)

**Purpose**: Determine whether text is advice, description, or factual statement

**File**: `intent_classifier.py`

**Key Methods**:
```python
class IntentClassifier:
    def __init__(self, ai_engine):
        self.ai_engine = ai_engine
    
    def classify_intent(self, text: str) -> IntentClassification:
        """
        Classify text intent
        
        Returns:
            IntentClassification with:
            - intent_type: ADVICE | DESCRIPTION | FACT | EXAMPLE
            - confidence: 0-100
            - reasoning: Explanation
            - evidence: Supporting quotes
        """
    
    def is_client_advice(self, text: str) -> bool:
        """
        Check if text advises clients what to do
        
        Examples:
        - "Vous devriez investir" → TRUE (advice)
        - "Le fonds investit" → FALSE (description)
        """
    
    def is_fund_description(self, text: str) -> bool:
        """
        Check if text describes fund characteristics
        
        Examples:
        - "Tirer parti du momentum" → TRUE (strategy goal)
        - "Nous recommandons d'acheter" → FALSE (advice)
        """
```

**AI Prompt Template**:
```python
INTENT_CLASSIFICATION_PROMPT = """
Analyze this text and classify its intent.

TEXT: {text}

CLASSIFICATION TYPES:
1. ADVICE: Tells clients what they should do
   - "Vous devriez investir"
   - "Nous recommandons d'acheter"
   - "Bon moment pour investir"

2. DESCRIPTION: Describes what the fund does
   - "Le fonds investit dans..."
   - "La stratégie vise à..."
   - "Tirer parti du momentum"

3. FACT: States objective information
   - "Le fonds a généré 5% en 2023"
   - "Le SRRI est de 4/7"

4. EXAMPLE: Illustrative scenario
   - "Par exemple, un investissement de 1000€..."

Respond with JSON:
{{
  "intent_type": "ADVICE|DESCRIPTION|FACT|EXAMPLE",
  "confidence": 0-100,
  "subject": "who performs action (fund|client|general)",
  "reasoning": "detailed explanation",
  "evidence": "key phrases that support classification"
}}
"""
```

### 3. Semantic Validator (NEW)

**Purpose**: Validate compliance based on meaning, not keywords

**File**: `semantic_validator.py`

**Key Methods**:
```python
class SemanticValidator:
    def __init__(self, ai_engine, context_analyzer, intent_classifier):
        self.ai_engine = ai_engine
        self.context_analyzer = context_analyzer
        self.intent_classifier = intent_classifier
    
    def validate_securities_mention(self, text: str, whitelist: Set[str]) -> ValidationResult:
        """
        Validate if text mentions prohibited securities
        
        Args:
            text: Text to analyze
            whitelist: Allowed terms (fund name, strategy terms, etc.)
        
        Returns:
            ValidationResult with violation flag and reasoning
        """
    
    def validate_performance_claim(self, text: str) -> ValidationResult:
        """
        Validate if text contains actual performance data
        
        Distinguishes:
        - "15% return in 2024" → ACTUAL DATA (needs disclaimer)
        - "attractive performance" → DESCRIPTIVE (no disclaimer needed)
        """
    
    def validate_prospectus_consistency(self, doc_text: str, prospectus_text: str) -> ValidationResult:
        """
        Check for contradictions (not missing details)
        
        VIOLATION: "invests in European stocks" vs "US stocks only"
        NOT VIOLATION: "invests in S&P 500" vs "at least 70% in S&P 500"
        """
```

### 4. Evidence Extractor (NEW)

**Purpose**: Identify and quote specific text supporting findings

**File**: `evidence_extractor.py`

**Key Methods**:
```python
class EvidenceExtractor:
    def __init__(self, ai_engine):
        self.ai_engine = ai_engine
    
    def extract_evidence(self, text: str, violation_type: str) -> Evidence:
        """
        Extract specific evidence for a violation
        
        Returns:
            Evidence with:
            - quotes: List of relevant text passages
            - locations: Where in document (slide, section)
            - context: Surrounding text for clarity
        """
    
    def find_performance_data(self, text: str) -> List[PerformanceData]:
        """
        Find actual performance numbers/charts
        
        Returns:
            List of PerformanceData with:
            - value: "15%", "+20%", etc.
            - context: Surrounding text
            - location: Slide/section
        """
    
    def find_disclaimer(self, text: str, required_disclaimer: str) -> Optional[DisclaimerMatch]:
        """
        Find disclaimer using semantic similarity
        
        Matches:
        - "Les performances passées ne préjugent pas..."
        - "Past performance is not indicative..."
        - Variations and paraphrases
        """
```

### 5. Whitelist Manager (NEW)

**Purpose**: Manage terms that are allowed to repeat (fund names, strategy terms, regulatory terms)

**File**: `whitelist_manager.py`

**Key Methods**:
```python
class WhitelistManager:
    def __init__(self):
        self.fund_name_terms = set()
        self.strategy_terms = set()
        self.regulatory_terms = set()
        self.benchmark_terms = set()
    
    def build_whitelist(self, doc: Dict) -> Set[str]:
        """
        Build comprehensive whitelist from document
        
        Extracts:
        - Fund name components
        - Strategy terminology
        - Regulatory terms
        - Benchmark names
        """
    
    def is_whitelisted(self, term: str) -> bool:
        """Check if term is in any whitelist"""
    
    def get_whitelist_reason(self, term: str) -> str:
        """Explain why term is whitelisted"""
```

**Default Whitelists**:
```python
STRATEGY_TERMS = {
    'momentum', 'quantitative', 'quantitatif', 'systematic', 'systématique',
    'algorithmic', 'algorithmique', 'smart', 'trend', 'behavioral',
    'value', 'growth', 'blend', 'core', 'satellite'
}

REGULATORY_TERMS = {
    'sri', 'srri', 'sfdr', 'ucits', 'mifid', 'amf', 'esma',
    'kiid', 'kid', 'priips', 'esg', 'article', 'regulation'
}

BENCHMARK_TERMS = {
    's&p', '500', 'msci', 'stoxx', 'eurostoxx', 'cac', 'dax',
    'ftse', 'russell', 'dow', 'jones', 'nasdaq', 'index', 'indice'
}

GENERIC_FINANCIAL_TERMS = {
    'actions', 'equities', 'bonds', 'obligations', 'fund', 'fonds',
    'portfolio', 'portefeuille', 'investment', 'investissement',
    'asset', 'actif', 'allocation', 'diversification'
}
```

## Data Models

### ContextAnalysis
```python
@dataclass
class ContextAnalysis:
    subject: str  # "fund", "client", "general"
    intent: str   # "describe", "advise", "state_fact"
    confidence: int  # 0-100
    reasoning: str
    evidence: List[str]
    is_fund_description: bool
    is_client_advice: bool
```

### IntentClassification
```python
@dataclass
class IntentClassification:
    intent_type: str  # "ADVICE", "DESCRIPTION", "FACT", "EXAMPLE"
    confidence: int
    subject: str
    reasoning: str
    evidence: str
```

### ValidationResult
```python
@dataclass
class ValidationResult:
    is_violation: bool
    confidence: int
    reasoning: str
    evidence: List[str]
    method: str  # "AI_ONLY", "RULES_ONLY", "AI_AND_RULES"
    rule_hints: Optional[str]
```

### Evidence
```python
@dataclass
class Evidence:
    quotes: List[str]
    locations: List[str]  # ["Slide 2", "Cover Page"]
    context: str
    confidence: int
```

## Integration with Existing Code

### Modified: `check_functions_ai.py`

**Current Function** (generates false positives):
```python
def check_prohibited_phrases(doc, rule):
    # Current: Keyword matching
    prohibited_phrases = ["recommend", "suggest", "should buy"]
    
    if any(phrase in text for phrase in prohibited_phrases):
        return violation  # FALSE POSITIVE!
```

**New Function** (context-aware):
```python
def check_prohibited_phrases_ai(doc, rule):
    """
    Check for investment advice using AI context understanding
    
    Eliminates 25 false positives by distinguishing:
    - Fund strategy descriptions (ALLOWED)
    - Investment advice to clients (PROHIBITED)
    """
    # Initialize components
    context_analyzer = ContextAnalyzer(ai_engine)
    intent_classifier = IntentClassifier(ai_engine)
    
    # Extract text
    all_text = extract_all_text_from_doc(doc)
    
    # Analyze context
    context = context_analyzer.analyze_context(all_text, "investment_advice")
    
    # Classify intent
    intent = intent_classifier.classify_intent(all_text)
    
    # Only flag if it's actual client advice
    if intent.intent_type == "ADVICE" and context.subject == "client":
        return {
            'violation': True,
            'confidence': min(context.confidence, intent.confidence),
            'rule': rule['rule_id'],
            'message': 'Investment advice to clients detected',
            'evidence': intent.evidence,
            'ai_reasoning': f"Context: {context.reasoning}. Intent: {intent.reasoning}",
            'method': 'AI_CONTEXT_AWARE'
        }
    
    # Not a violation if it's fund description
    return None
```

### Modified: `agent.py`

**Current Function** (generates false positives):
```python
def check_repeated_securities(doc):
    # Current: Count ALL mentions
    security_mentions = Counter()
    for word in text.split():
        security_mentions[word] += 1
    
    # Problem: Flags fund name, strategy terms
    if security_mentions[word] > 2:
        return violation  # FALSE POSITIVE!
```

**New Function** (whitelist-aware):
```python
def check_repeated_securities_ai(doc):
    """
    Check for repeated external company mentions
    
    Eliminates 16 false positives by whitelisting:
    - Fund name components
    - Strategy terminology
    - Regulatory terms
    """
    # Build whitelist
    whitelist_mgr = WhitelistManager()
    whitelist = whitelist_mgr.build_whitelist(doc)
    
    # Extract all text
    all_text = extract_all_text_from_doc(doc)
    
    # Find capitalized words (potential company names)
    words = re.findall(r'\b[A-Z][a-z]+\b', all_text)
    word_counts = Counter(w.lower() for w in words)
    
    violations = []
    for word, count in word_counts.items():
        # Skip if whitelisted
        if whitelist_mgr.is_whitelisted(word):
            continue
        
        # Only flag if mentioned 3+ times
        if count >= 3:
            # Use AI to verify it's actually a company name
            semantic_validator = SemanticValidator(ai_engine, None, None)
            result = semantic_validator.validate_securities_mention(
                text=all_text,
                term=word,
                whitelist=whitelist
            )
            
            if result.is_violation:
                violations.append({
                    'violation': True,
                    'confidence': result.confidence,
                    'rule': 'VAL_005',
                    'message': f'External company "{word}" mentioned {count} times',
                    'evidence': result.evidence,
                    'ai_reasoning': result.reasoning
                })
    
    return violations
```


### Modified: Performance Detection

**Current Function** (generates false positives):
```python
def check_performance_disclaimers(doc):
    # Current: Keyword matching
    if 'performance' in text.lower():
        # Problem: Flags "performance objective", "attractive performance"
        return violation  # FALSE POSITIVE!
```

**New Function** (data-aware):
```python
def check_performance_disclaimers_ai(doc):
    """
    Check performance disclaimers only when actual data present
    
    Eliminates 3 false positives by detecting:
    - ACTUAL DATA: "15% return", "+20% in 2024" → needs disclaimer
    - KEYWORDS: "performance objective" → no disclaimer needed
    """
    evidence_extractor = EvidenceExtractor(ai_engine)
    semantic_validator = SemanticValidator(ai_engine, None, None)
    
    violations = []
    
    for slide_name, slide_data in get_all_slides(doc):
        slide_text = json.dumps(slide_data)
        
        # Find actual performance data
        perf_data = evidence_extractor.find_performance_data(slide_text)
        
        # Only check if actual data present
        if not perf_data:
            continue  # No actual performance data, skip
        
        # Check for disclaimer on same slide
        disclaimer = evidence_extractor.find_disclaimer(
            slide_text,
            required_disclaimer="performances passées ne préjugent pas"
        )
        
        if not disclaimer:
            violations.append({
                'violation': True,
                'confidence': 95,
                'rule': 'PERF_001',
                'slide': slide_name,
                'message': 'Performance data without disclaimer',
                'evidence': f'Found: {perf_data[0].value} at {perf_data[0].location}',
                'ai_reasoning': 'Actual performance numbers detected without accompanying disclaimer'
            })
    
    return violations
```

## Error Handling

### AI Service Failures

**Graceful Degradation**:
```python
class ContextAnalyzer:
    def analyze_context(self, text: str, check_type: str) -> ContextAnalysis:
        try:
            # Try AI analysis
            ai_result = self.ai_engine.call_with_cache(prompt)
            return self._parse_ai_result(ai_result)
        
        except AIServiceError as e:
            logger.warning(f"AI service failed: {e}")
            # Fall back to rule-based heuristics
            return self._fallback_rule_based_analysis(text, check_type)
```

**Fallback Logic**:
- AI unavailable → Use enhanced rule-based checks
- AI timeout → Return low-confidence result, flag for review
- AI parse error → Retry once, then fallback

### Confidence Thresholds

**Decision Matrix**:
```
AI Confidence | Rules Agreement | Action
-------------|-----------------|------------------
> 90%        | Yes             | High confidence violation
> 90%        | No              | Flag for review
70-90%       | Yes             | Medium confidence violation
70-90%       | No              | Flag for review
< 70%        | Yes             | Low confidence, queue for HITL
< 70%        | No              | No violation (likely false positive)
```

## Testing Strategy

### Unit Tests

**Test Coverage**:
1. **Context Analyzer**
   - Fund descriptions correctly classified
   - Investment advice correctly classified
   - Edge cases (ambiguous text)

2. **Intent Classifier**
   - All intent types correctly identified
   - Confidence scores appropriate
   - Evidence extraction accurate

3. **Semantic Validator**
   - Whitelisted terms not flagged
   - External companies correctly flagged
   - Performance data vs keywords distinguished

4. **Evidence Extractor**
   - Performance numbers correctly extracted
   - Disclaimers found with semantic matching
   - Locations accurately identified

### Integration Tests

**Test Scenarios**:
```python
def test_fund_strategy_not_flagged():
    """Test that fund strategy descriptions are not flagged as advice"""
    doc = load_test_doc("exemple.json")
    violations = check_prohibited_phrases_ai(doc, rule)
    
    # Should NOT flag "Tirer parti du momentum"
    assert len(violations) == 0

def test_fund_name_not_flagged():
    """Test that fund name repetition is not flagged"""
    doc = load_test_doc("exemple.json")
    violations = check_repeated_securities_ai(doc)
    
    # Should NOT flag "ODDO BHF" (31 mentions)
    oddo_violations = [v for v in violations if 'oddo' in v['message'].lower()]
    assert len(oddo_violations) == 0

def test_performance_keyword_not_flagged():
    """Test that performance keywords without data are not flagged"""
    doc = load_test_doc("exemple.json")
    violations = check_performance_disclaimers_ai(doc)
    
    # Should NOT flag "attractive performance" (no numbers)
    assert len(violations) == 0

def test_actual_violations_still_caught():
    """Test that real violations are still detected"""
    doc = load_test_doc("exemple.json")
    all_violations = run_all_checks(doc)
    
    # Should catch the 6 actual violations
    assert len(all_violations) == 6
    assert any('promotional' in v['message'].lower() for v in all_violations)
    assert any('target audience' in v['message'].lower() for v in all_violations)
```

### Accuracy Validation

**Benchmark Against Kiro Analysis**:
```python
def test_matches_kiro_analysis():
    """Validate that results match Kiro's analysis"""
    doc = load_test_doc("exemple.json")
    our_violations = run_all_checks(doc)
    kiro_violations = load_kiro_analysis("kiro_results.json")
    
    # Should have same number of violations
    assert len(our_violations) == len(kiro_violations)
    
    # Should have same violation types
    our_types = {v['rule'] for v in our_violations}
    kiro_types = {v['rule'] for v in kiro_violations}
    assert our_types == kiro_types
```

## Performance Considerations

### Caching Strategy

**Cache AI Responses**:
- Cache key: Hash of (prompt + document excerpt)
- TTL: 24 hours
- Max size: 1000 entries
- Eviction: LRU

**Expected Cache Hit Rate**: 60-70% for similar documents

### Batch Processing

**Optimize AI Calls**:
```python
def check_all_slides_batch(doc):
    """Process multiple slides in single AI call"""
    slides = get_all_slides(doc)
    
    # Batch slides into single prompt
    batch_prompt = create_batch_prompt(slides)
    
    # Single AI call for all slides
    result = ai_engine.call_with_cache(batch_prompt)
    
    # Parse results for each slide
    return parse_batch_results(result, slides)
```

### Performance Targets

**Metrics**:
- Single check: < 3 seconds (AI call)
- Full document: < 60 seconds (all checks)
- Cache hit: < 10ms
- Fallback (rules only): < 1 second

## Migration Strategy

### Phase 1: Parallel Execution (Week 1)

**Run both old and new logic**:
```python
def check_with_comparison(doc, rule):
    """Run both old and new logic, compare results"""
    
    # Old logic
    old_violations = check_prohibited_phrases_old(doc, rule)
    
    # New logic
    new_violations = check_prohibited_phrases_ai(doc, rule)
    
    # Log differences
    log_comparison(old_violations, new_violations)
    
    # Return new results
    return new_violations
```

**Validation**:
- Compare results on 100 test documents
- Identify any regressions
- Tune confidence thresholds

### Phase 2: Gradual Rollout (Week 2)

**Feature Flag**:
```python
# In config
USE_AI_CONTEXT_AWARE = True  # Enable new logic

if USE_AI_CONTEXT_AWARE:
    violations = check_prohibited_phrases_ai(doc, rule)
else:
    violations = check_prohibited_phrases_old(doc, rule)
```

**Rollout Plan**:
1. Enable for 10% of checks
2. Monitor accuracy and performance
3. Increase to 50%
4. Full rollout after validation

### Phase 3: Deprecation (Week 3)

**Remove old logic**:
- Archive old functions
- Update documentation
- Remove feature flags

## Monitoring and Metrics

### Key Metrics

**Accuracy Metrics**:
- False positive rate: Target < 5%
- False negative rate: Target < 5%
- Precision: Target > 95%
- Recall: Target > 95%

**Performance Metrics**:
- Average processing time per document
- AI API call count
- Cache hit rate
- Fallback rate (AI failures)

**Cost Metrics**:
- AI API costs per document
- Token usage
- Cache savings

### Dashboards

**Real-Time Monitoring**:
```python
class ComplianceMetrics:
    def __init__(self):
        self.total_checks = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.ai_calls = 0
        self.cache_hits = 0
        self.fallbacks = 0
    
    def record_check(self, result: CheckResult):
        """Record metrics for a check"""
        self.total_checks += 1
        
        if result.was_cached:
            self.cache_hits += 1
        else:
            self.ai_calls += 1
        
        if result.used_fallback:
            self.fallbacks += 1
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        return {
            'total_checks': self.total_checks,
            'false_positive_rate': self.false_positives / self.total_checks,
            'cache_hit_rate': self.cache_hits / self.total_checks,
            'fallback_rate': self.fallbacks / self.total_checks,
            'avg_ai_calls_per_doc': self.ai_calls / self.total_checks
        }
```

## Configuration

### AI Engine Configuration

**File**: `hybrid_config.json`

```json
{
  "ai_engine": {
    "primary_provider": "gemini",
    "fallback_provider": "token_factory",
    "confidence_threshold": 70,
    "cache_enabled": true,
    "cache_size": 1000,
    "cache_ttl_hours": 24
  },
  "context_analysis": {
    "enabled": true,
    "min_confidence": 60,
    "use_fallback_rules": true
  },
  "whitelist": {
    "auto_extract_fund_name": true,
    "include_strategy_terms": true,
    "include_regulatory_terms": true,
    "custom_terms": []
  },
  "performance": {
    "batch_slides": true,
    "max_batch_size": 5,
    "timeout_seconds": 30
  }
}
```

### Prompt Templates Configuration

**File**: `prompt_templates.json`

```json
{
  "investment_advice_detection": {
    "system_message": "You are a financial compliance expert...",
    "user_prompt_template": "Analyze this text for investment advice...",
    "required_vars": ["text", "rule_hints"],
    "max_tokens": 1000
  },
  "performance_data_detection": {
    "system_message": "You are a financial document analyzer...",
    "user_prompt_template": "Find actual performance data...",
    "required_vars": ["slide_text"],
    "max_tokens": 800
  }
}
```

## Dependencies

### New Dependencies

```python
# requirements.txt additions
# (All existing dependencies remain)

# No new external dependencies required
# Uses existing:
# - google-generativeai (Gemini)
# - openai (Token Factory)
# - Standard library (re, json, hashlib, etc.)
```

### Internal Dependencies

**Reuse Existing Components**:
- `ai_engine.py`: LLM abstraction layer
- `hybrid_compliance_checker.py`: Orchestration
- `confidence_scorer.py`: Confidence calculation
- `performance_monitor.py`: Metrics tracking

## Security Considerations

### Data Privacy

**Sensitive Data Handling**:
- Fund names: Public information, safe to send to AI
- Performance data: Public information, safe to send to AI
- Client information: Should be redacted before AI analysis

**Redaction Strategy**:
```python
def redact_sensitive_data(text: str) -> str:
    """Redact sensitive information before AI analysis"""
    # Redact email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Redact phone numbers
    text = re.sub(r'\b\d{10,}\b', '[PHONE]', text)
    
    # Redact specific client names (if configured)
    for client_name in SENSITIVE_CLIENT_NAMES:
        text = text.replace(client_name, '[CLIENT]')
    
    return text
```

### API Key Management

**Secure Storage**:
- API keys in `.env` file (not committed)
- Environment variables for production
- Key rotation support

## Rollback Plan

### Quick Rollback

**If issues detected**:
1. Set feature flag `USE_AI_CONTEXT_AWARE = False`
2. System reverts to old logic immediately
3. No data loss, no downtime

### Rollback Triggers

**Automatic rollback if**:
- False positive rate > 20%
- AI service downtime > 50%
- Processing time > 2x baseline

## Success Criteria

### Quantitative Metrics

**Must Achieve**:
- ✅ Reduce violations from 40 → 6 on `exemple.json`
- ✅ Eliminate all 34 false positives
- ✅ Maintain detection of all 6 actual violations
- ✅ False positive rate < 5% on test set
- ✅ Processing time < 60 seconds per document

### Qualitative Metrics

**Must Achieve**:
- ✅ AI reasoning is understandable and auditable
- ✅ Confidence scores accurately reflect reliability
- ✅ System gracefully handles AI failures
- ✅ Results match Kiro analysis quality

## Future Enhancements

### Phase 2 Features (Post-MVP)

1. **Cross-Slide Analysis**
   - Detect incomplete risk profiles (Slide 2 vs Slide 6)
   - Validate consistency across document

2. **Anglicism Detection**
   - Identify English terms in French retail docs
   - Verify glossary presence

3. **Learning from Corrections**
   - Track human review decisions
   - Adjust confidence thresholds
   - Improve prompt templates

4. **Multi-Language Support**
   - Extend beyond French/English
   - Language-specific compliance rules

## Appendix

### Example AI Prompts

**Investment Advice Detection**:
```
Analyze this text and determine if it contains investment advice to clients.

TEXT: "Tirer parti du momentum des marchés américains grâce à une stratégie quantitative"

CONTEXT:
- This is from a fund marketing document
- French/EU regulations prohibit direct investment advice in marketing materials

DISTINGUISH:
1. Fund strategy description (ALLOWED): "The fund invests in...", "The strategy aims to..."
2. Client advice (PROHIBITED): "You should invest...", "We recommend buying..."

WHO is performing the action?
- If "the fund" or "the strategy" → ALLOWED
- If "you" or "investors" → PROHIBITED

Respond with JSON:
{
  "is_investment_advice": false,
  "confidence": 95,
  "subject": "fund",
  "reasoning": "Text describes the fund's strategy goal ('tirer parti'), not advice to clients",
  "evidence": "Subject is implicit 'the fund', verb is 'tirer parti' (take advantage), describes strategy approach"
}
```

### Glossary

- **False Positive**: System flags a violation that doesn't actually exist
- **False Negative**: System misses an actual violation
- **Context Analysis**: Understanding the semantic meaning and intent of text
- **Intent Classification**: Determining whether text is advice, description, or fact
- **Semantic Validation**: Validating compliance based on meaning, not keywords
- **Whitelist**: Terms allowed to repeat (fund names, strategy terms, etc.)
- **Evidence Extraction**: Identifying and quoting specific text supporting findings
- **Graceful Degradation**: Falling back to rule-based checks when AI unavailable

