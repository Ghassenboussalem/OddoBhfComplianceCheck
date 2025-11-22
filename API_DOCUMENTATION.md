# API Documentation - AI-Enhanced Compliance Checker

## Overview

This document provides comprehensive API documentation for the hybrid AI-enhanced compliance checking system. The system combines AI semantic understanding with rule-based validation to provide robust compliance checking for fund documents.

## Core Classes

### Context-Aware Components (NEW)

The system now includes five new components that provide semantic understanding and eliminate false positives:

#### ContextAnalyzer

Analyzes text context to understand WHO performs actions and WHAT the intent is.

**Constructor:**
```python
ContextAnalyzer(
    ai_engine: AIEngine,
    prompt_template_file: str = None
)
```

**Methods:**

##### analyze_context()

```python
analyze_context(
    text: str,
    check_type: str
) -> ContextAnalysis
```

Analyzes text context using AI to determine subject and intent.

**Parameters:**
- `text`: Text to analyze
- `check_type`: Type of analysis ("investment_advice", "performance_claim", etc.)

**Returns:**
`ContextAnalysis` dataclass with:
- `subject`: str - WHO performs action ("fund", "client", "general")
- `intent`: str - WHAT is the purpose ("describe", "advise", "state_fact")
- `confidence`: int - Confidence score 0-100
- `reasoning`: str - Explanation of analysis
- `evidence`: List[str] - Supporting quotes
- `is_fund_description`: bool - True if describing fund strategy
- `is_client_advice`: bool - True if advising clients

**Example:**
```python
from context_analyzer import ContextAnalyzer
from ai_engine import AIEngine

analyzer = ContextAnalyzer(AIEngine())
context = analyzer.analyze_context(
    "Le fonds tire parti du momentum",
    check_type="investment_advice"
)

print(f"Subject: {context.subject}")  # "fund"
print(f"Intent: {context.intent}")    # "describe"
print(f"Is fund description: {context.is_fund_description}")  # True
```

##### is_fund_strategy_description()

```python
is_fund_strategy_description(text: str) -> bool
```

Quick check if text describes fund strategy (ALLOWED).

##### is_investment_advice()

```python
is_investment_advice(text: str) -> bool
```

Quick check if text advises clients (PROHIBITED).

---

#### IntentClassifier

Classifies text intent into ADVICE, DESCRIPTION, FACT, or EXAMPLE.

**Constructor:**
```python
IntentClassifier(ai_engine: AIEngine)
```

**Methods:**

##### classify_intent()

```python
classify_intent(text: str) -> IntentClassification
```

Classifies the intent of text.

**Parameters:**
- `text`: Text to classify

**Returns:**
`IntentClassification` dataclass with:
- `intent_type`: str - "ADVICE", "DESCRIPTION", "FACT", or "EXAMPLE"
- `confidence`: int - Confidence score 0-100
- `subject`: str - WHO performs action
- `reasoning`: str - Explanation
- `evidence`: str - Key phrases supporting classification

**Example:**
```python
from intent_classifier import IntentClassifier

classifier = IntentClassifier(AIEngine())

# Client advice (PROHIBITED)
intent1 = classifier.classify_intent("Vous devriez investir maintenant")
print(intent1.intent_type)  # "ADVICE"
print(intent1.subject)      # "client"

# Fund description (ALLOWED)
intent2 = classifier.classify_intent("Le fonds investit dans des actions")
print(intent2.intent_type)  # "DESCRIPTION"
print(intent2.subject)      # "fund"
```

##### is_client_advice()

```python
is_client_advice(text: str) -> bool
```

Quick check if text is advice to clients.

##### is_fund_description()

```python
is_fund_description(text: str) -> bool
```

Quick check if text describes fund characteristics.

---

#### SemanticValidator

Validates compliance based on meaning rather than keywords.

**Constructor:**
```python
SemanticValidator(
    ai_engine: AIEngine,
    context_analyzer: ContextAnalyzer,
    intent_classifier: IntentClassifier
)
```

**Methods:**

##### validate_securities_mention()

```python
validate_securities_mention(
    text: str,
    whitelist: Set[str]
) -> ValidationResult
```

Validates if text mentions prohibited securities, respecting whitelist.

**Parameters:**
- `text`: Text to analyze
- `whitelist`: Set of allowed terms (fund names, strategy terms, etc.)

**Returns:**
`ValidationResult` dataclass with:
- `is_violation`: bool - True if violation found
- `confidence`: int - Confidence score 0-100
- `reasoning`: str - Explanation
- `evidence`: List[str] - Supporting quotes
- `method`: str - "AI_ONLY", "RULES_ONLY", or "AI_AND_RULES"

**Example:**
```python
from semantic_validator import SemanticValidator

validator = SemanticValidator(ai_engine, context_analyzer, intent_classifier)

result = validator.validate_securities_mention(
    text="ODDO BHF momentum strategy",
    whitelist={"oddo", "bhf", "momentum"}
)

print(result.is_violation)  # False (all terms whitelisted)
```

##### validate_performance_claim()

```python
validate_performance_claim(text: str) -> ValidationResult
```

Validates if text contains actual performance data requiring disclaimers.

**Example:**
```python
# Descriptive keyword (no violation)
result1 = validator.validate_performance_claim("attractive performance")
print(result1.is_violation)  # False

# Actual data (violation if no disclaimer)
result2 = validator.validate_performance_claim("15% return in 2024")
print(result2.is_violation)  # True
```

##### validate_prospectus_consistency()

```python
validate_prospectus_consistency(
    doc_text: str,
    prospectus_text: str
) -> ValidationResult
```

Checks for contradictions (not missing details).

---

#### EvidenceExtractor

Identifies and quotes specific evidence supporting findings.

**Constructor:**
```python
EvidenceExtractor(ai_engine: AIEngine)
```

**Methods:**

##### extract_evidence()

```python
extract_evidence(
    text: str,
    violation_type: str
) -> Evidence
```

Extracts specific evidence for a violation.

**Parameters:**
- `text`: Text to analyze
- `violation_type`: Type of violation to find evidence for

**Returns:**
`Evidence` dataclass with:
- `quotes`: List[str] - Relevant text passages
- `locations`: List[str] - Where in document (slide, section)
- `context`: str - Surrounding text for clarity
- `confidence`: int - Confidence score 0-100

**Example:**
```python
from evidence_extractor import EvidenceExtractor

extractor = EvidenceExtractor(AIEngine())

evidence = extractor.extract_evidence(
    text=slide_text,
    violation_type="missing_disclaimer"
)

print(f"Quotes: {evidence.quotes}")
print(f"Locations: {evidence.locations}")
```

##### find_performance_data()

```python
find_performance_data(text: str) -> List[PerformanceData]
```

Finds actual performance numbers/charts (not keywords).

**Returns:**
List of `PerformanceData` dataclass with:
- `value`: str - Performance value ("15%", "+20%", etc.)
- `context`: str - Surrounding text
- `location`: str - Slide/section

**Example:**
```python
# No actual data
perf1 = extractor.find_performance_data("attractive performance")
print(len(perf1))  # 0

# Actual data
perf2 = extractor.find_performance_data("The fund returned 15% in 2024")
print(len(perf2))  # 1
print(perf2[0].value)  # "15%"
```

##### find_disclaimer()

```python
find_disclaimer(
    text: str,
    required_disclaimer: str
) -> Optional[DisclaimerMatch]
```

Finds disclaimer using semantic similarity (not exact matching).

**Returns:**
`DisclaimerMatch` dataclass with:
- `text`: str - Matched disclaimer text
- `location`: str - Where found
- `confidence`: int - Match confidence 0-100

**Example:**
```python
disclaimer = extractor.find_disclaimer(
    text=slide_text,
    required_disclaimer="performances passées ne préjugent pas"
)

if disclaimer:
    print(f"Found: {disclaimer.text}")
    print(f"Confidence: {disclaimer.confidence}%")
```

---

#### WhitelistManager

Manages terms that are allowed to repeat without triggering violations.

**Constructor:**
```python
WhitelistManager()
```

**Methods:**

##### build_whitelist()

```python
build_whitelist(doc: Dict) -> Set[str]
```

Builds comprehensive whitelist from document.

**Parameters:**
- `doc`: Document dictionary with metadata

**Returns:**
Set of whitelisted terms (lowercase)

**Example:**
```python
from whitelist_manager import WhitelistManager

manager = WhitelistManager()
whitelist = manager.build_whitelist(document)

print(f"Total terms: {len(whitelist)}")
print(f"Sample: {list(whitelist)[:10]}")
```

##### is_whitelisted()

```python
is_whitelisted(term: str) -> bool
```

Checks if term is in any whitelist.

##### get_whitelist_reason()

```python
get_whitelist_reason(term: str) -> str
```

Explains why term is whitelisted.

**Example:**
```python
if manager.is_whitelisted("ODDO"):
    reason = manager.get_whitelist_reason("ODDO")
    print(reason)  # "Fund name component"

if manager.is_whitelisted("momentum"):
    reason = manager.get_whitelist_reason("momentum")
    print(reason)  # "Strategy terminology"
```

##### add_custom_terms()

```python
add_custom_terms(terms: List[str]) -> None
```

Adds custom terms to whitelist.

**Example:**
```python
manager.add_custom_terms(["proprietary", "alpha", "beta"])
```

---

### HybridComplianceChecker

The main orchestrator class that coordinates the three-layer hybrid architecture.

#### Constructor

```python
HybridComplianceChecker(
    ai_engine: AIEngine,
    rule_engine: RuleEngine,
    confidence_scorer: ConfidenceScorer,
    config: dict = None
)
```

**Parameters:**
- `ai_engine`: Instance of AIEngine for semantic analysis
- `rule_engine`: Instance of RuleEngine for rule-based validation
- `confidence_scorer`: Instance of ConfidenceScorer for result combination
- `config`: Optional configuration dictionary

**Example:**
```python
from hybrid_compliance_checker import HybridComplianceChecker
from ai_engine import AIEngine
from confidence_scorer import ConfidenceScorer

checker = HybridComplianceChecker(
    ai_engine=AIEngine(api_key="your_key"),
    rule_engine=RuleEngine(),
    confidence_scorer=ConfidenceScorer()
)
```

#### Methods

##### check_compliance()

```python
check_compliance(
    document: dict,
    check_type: str,
    options: dict = None
) -> dict
```

Performs comprehensive compliance checking using the three-layer hybrid approach.

**Parameters:**
- `document`: Document dictionary containing text, metadata, and structure
- `check_type`: Type of check to perform (see Check Types section)
- `options`: Optional parameters for customizing the check

**Returns:**
Dictionary containing:
- `violation`: Boolean indicating if violation was found
- `confidence`: Integer 0-100 indicating confidence level
- `status`: String indicating result status (see Status Types)
- `evidence`: List of evidence supporting the finding
- `reasoning`: AI-generated explanation
- `location`: Specific location in document
- `suggestions`: List of remediation suggestions

**Example:**
```python
result = checker.check_compliance(
    document=parsed_document,
    check_type="promotional_mention",
    options={"language": "french"}
)

if result['violation']:
    print(f"Violation found with {result['confidence']}% confidence")
    print(f"Reasoning: {result['reasoning']}")
```

##### check_all_compliance()

```python
check_all_compliance(
    document: dict,
    check_types: list = None
) -> list
```

Performs multiple compliance checks on a document.

**Parameters:**
- `document`: Document dictionary
- `check_types`: Optional list of specific check types (defaults to all)

**Returns:**
List of result dictionaries, one per check type

**Example:**
```python
results = checker.check_all_compliance(
    document=parsed_document,
    check_types=["promotional_mention", "performance_claims", "fund_name"]
)

for result in results:
    if result['violation']:
        print(f"{result['check_type']}: {result['message']}")
```

### AIEngine

Handles all AI-powered semantic analysis and understanding.

#### Constructor

```python
AIEngine(
    api_key: str = None,
    model: str = "gemini-pro",
    cache_enabled: bool = True,
    timeout: int = 30
)
```

**Parameters:**
- `api_key`: API key for AI service (reads from environment if not provided)
- `model`: Model identifier to use
- `cache_enabled`: Enable response caching
- `timeout`: Request timeout in seconds

#### Methods

##### analyze()

```python
analyze(
    document: dict,
    check_type: str,
    rule_hints: dict = None
) -> dict
```

Performs AI-powered semantic analysis on document content.

**Parameters:**
- `document`: Document to analyze
- `check_type`: Type of analysis to perform
- `rule_hints`: Optional hints from rule-based pre-filtering

**Returns:**
Dictionary containing AI analysis results with confidence, evidence, and reasoning

**Example:**
```python
ai_engine = AIEngine(api_key="your_key")

result = ai_engine.analyze(
    document=doc,
    check_type="performance_claims",
    rule_hints={"keywords_found": ["performance", "returns"]}
)
```

##### batch_analyze()

```python
batch_analyze(
    documents: list,
    check_type: str
) -> list
```

Performs batch analysis on multiple documents for efficiency.

**Parameters:**
- `documents`: List of documents to analyze
- `check_type`: Type of analysis to perform

**Returns:**
List of analysis results

### ConfidenceScorer

Combines AI and rule-based results to produce final confidence scores.

#### Constructor

```python
ConfidenceScorer(
    thresholds: dict = None,
    calibration_enabled: bool = True
)
```

**Parameters:**
- `thresholds`: Custom confidence thresholds
- `calibration_enabled`: Enable automatic threshold calibration

#### Methods

##### combine_results()

```python
combine_results(
    rule_result: dict,
    ai_result: dict
) -> dict
```

Combines rule-based and AI results into final assessment.

**Parameters:**
- `rule_result`: Result from rule-based analysis
- `ai_result`: Result from AI analysis

**Returns:**
Combined result with final confidence score and status

**Example:**
```python
scorer = ConfidenceScorer()

final_result = scorer.combine_results(
    rule_result={"found": True, "confidence": 60},
    ai_result={"violation": True, "confidence": 85}
)

print(f"Final confidence: {final_result['confidence']}%")
print(f"Status: {final_result['status']}")
```

## Check Types

The system supports the following check types:

### promotional_mention
Detects promotional document mentions on cover pages.

**Key Features:**
- Multi-language support (French/English)
- OCR error tolerance
- Variation detection

### performance_claims
Analyzes performance claims and validates disclaimers.

**Key Features:**
- Distinguishes historical vs predictive claims
- Context-aware disclaimer validation
- Same-slide disclaimer checking

### fund_name_match
Validates fund name consistency using semantic matching.

**Key Features:**
- Handles abbreviations and reordering
- Synonym recognition
- Similarity scoring

### disclaimer_validation
Validates presence and placement of required disclaimers.

**Key Features:**
- Semantic similarity matching
- Location tracking
- Context awareness

### registration_compliance
Checks registration and country authorization requirements.

**Key Features:**
- Country extraction
- Authorization validation
- Multi-country support

### structure_validation
Validates document structure requirements.

**Key Features:**
- Target audience detection
- Management company validation
- Date and metadata checking

### general_rules
Applies general compliance rules with context understanding.

**Key Features:**
- Glossary term detection
- Technical language identification
- Context-aware rule application

### values_securities
Analyzes mentions of securities and investment recommendations.

**Key Features:**
- Intent analysis
- Example vs recommendation distinction
- Disclaimer detection

## Status Types

Results include a status field indicating the nature of the finding:

- `VERIFIED_BY_BOTH`: Both AI and rules agree on the finding (highest confidence)
- `AI_DETECTED_VARIATION`: AI found a variation that rules missed
- `FALSE_POSITIVE_FILTERED`: Rules flagged but AI determined it's not a violation
- `VIOLATION_CONFIRMED`: Clear violation detected
- `NEEDS_REVIEW`: Confidence below threshold, requires human review
- `COMPLIANT`: No violations found

## Configuration Options

### Global Configuration

```python
config = {
    "ai": {
        "enabled": True,
        "model": "gemini-pro",
        "timeout": 30,
        "max_retries": 3,
        "cache_enabled": True
    },
    "rules": {
        "enabled": True,
        "strict_mode": False
    },
    "confidence": {
        "min_threshold": 70,
        "review_threshold": 85,
        "calibration_enabled": True
    },
    "performance": {
        "batch_size": 10,
        "async_enabled": True,
        "max_concurrent": 5
    }
}

checker = HybridComplianceChecker(
    ai_engine=ai_engine,
    rule_engine=rule_engine,
    confidence_scorer=scorer,
    config=config
)
```

### Per-Check Options

```python
options = {
    "language": "french",  # or "english"
    "strict_matching": False,
    "include_suggestions": True,
    "detailed_reasoning": True,
    "min_confidence": 80
}

result = checker.check_compliance(
    document=doc,
    check_type="promotional_mention",
    options=options
)
```

## Error Handling

### Exception Types

#### AIServiceError
Raised when AI service is unavailable or returns errors.

```python
from hybrid_compliance_checker import AIServiceError

try:
    result = checker.check_compliance(doc, "promotional_mention")
except AIServiceError as e:
    print(f"AI service error: {e}")
    # System automatically falls back to rule-only mode
```

#### ConfigurationError
Raised when configuration is invalid.

```python
from hybrid_compliance_checker import ConfigurationError

try:
    checker = HybridComplianceChecker(config=invalid_config)
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

#### DocumentParsingError
Raised when document cannot be parsed.

```python
from hybrid_compliance_checker import DocumentParsingError

try:
    result = checker.check_compliance(malformed_doc, "fund_name")
except DocumentParsingError as e:
    print(f"Document parsing error: {e}")
```

## Performance Monitoring

### Metrics Collection

```python
from performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
checker = HybridComplianceChecker(
    ai_engine=ai_engine,
    rule_engine=rule_engine,
    confidence_scorer=scorer,
    monitor=monitor
)

# Perform checks...

# Get metrics
metrics = monitor.get_metrics()
print(f"Average processing time: {metrics['avg_time']}ms")
print(f"AI cache hit rate: {metrics['cache_hit_rate']}%")
print(f"Total API calls: {metrics['api_calls']}")
```

### Performance Optimization

```python
# Enable batch processing for multiple documents
results = checker.batch_check_compliance(
    documents=document_list,
    check_type="promotional_mention",
    batch_size=10
)

# Enable async processing
import asyncio

async def check_documents_async():
    results = await checker.check_compliance_async(
        documents=document_list,
        check_types=["promotional_mention", "performance_claims"]
    )
    return results

results = asyncio.run(check_documents_async())
```

## Feedback and Learning

### Providing Feedback

```python
from feedback_loop import FeedbackLoop

feedback = FeedbackLoop()

# After human review, provide correction
feedback.submit_correction(
    check_id=result['id'],
    correct_outcome=True,  # or False
    notes="Explanation of correction"
)

# System learns from feedback and adjusts confidence thresholds
```

### Confidence Calibration

```python
from confidence_calibrator import ConfidenceCalibrator

calibrator = ConfidenceCalibrator()

# Track accuracy over time
calibrator.track_result(
    predicted_confidence=85,
    actual_outcome=True
)

# Get calibration recommendations
recommendations = calibrator.get_recommendations()
print(f"Suggested threshold adjustments: {recommendations}")
```

## Integration Examples

### Basic Integration

```python
from hybrid_compliance_checker import HybridComplianceChecker
from ai_engine import AIEngine
from confidence_scorer import ConfidenceScorer

# Initialize components
ai_engine = AIEngine(api_key="your_key")
rule_engine = RuleEngine()
scorer = ConfidenceScorer()

# Create checker
checker = HybridComplianceChecker(ai_engine, rule_engine, scorer)

# Load and parse document
document = parse_document("fund_document.pdf")

# Run compliance checks
results = checker.check_all_compliance(document)

# Process results
for result in results:
    if result['violation']:
        print(f"Violation: {result['message']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Evidence: {result['evidence']}")
```

### Advanced Integration with Monitoring

```python
from hybrid_compliance_checker import HybridComplianceChecker
from performance_monitor import PerformanceMonitor
from feedback_loop import FeedbackLoop

# Setup with monitoring
monitor = PerformanceMonitor()
feedback = FeedbackLoop()

checker = HybridComplianceChecker(
    ai_engine=AIEngine(),
    rule_engine=RuleEngine(),
    confidence_scorer=ConfidenceScorer(),
    monitor=monitor,
    feedback=feedback
)

# Process documents
for doc_path in document_paths:
    document = parse_document(doc_path)
    results = checker.check_all_compliance(document)
    
    # Store results for review
    store_results(doc_path, results)
    
    # Flag low-confidence results for human review
    for result in results:
        if result['confidence'] < 85:
            flag_for_review(result)

# Generate performance report
report = monitor.generate_report()
print(f"Documents processed: {report['total_documents']}")
print(f"Average confidence: {report['avg_confidence']}%")
print(f"API cost: ${report['api_cost']}")
```

## API Reference Summary

### Main Classes
- `HybridComplianceChecker`: Main orchestrator
- `AIEngine`: AI-powered analysis
- `RuleEngine`: Rule-based validation
- `ConfidenceScorer`: Result combination
- `PerformanceMonitor`: Performance tracking
- `FeedbackLoop`: Learning system

### Key Methods
- `check_compliance()`: Single check
- `check_all_compliance()`: Multiple checks
- `batch_check_compliance()`: Batch processing
- `analyze()`: AI analysis
- `combine_results()`: Result combination

### Configuration
- Global config via constructor
- Per-check options via method parameters
- Environment variables for API keys

### Error Handling
- `AIServiceError`: AI service issues
- `ConfigurationError`: Config problems
- `DocumentParsingError`: Document issues

For more detailed examples and use cases, see the MIGRATION_GUIDE.md and CONFIGURATION_GUIDE.md files.
