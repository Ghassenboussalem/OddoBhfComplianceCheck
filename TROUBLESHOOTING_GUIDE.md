# Troubleshooting Guide - AI-Enhanced Compliance Checker

## Overview

This guide helps diagnose and resolve common issues with the AI-enhanced compliance checking system.

## Quick Diagnostics

### System Health Check

```python
from hybrid_compliance_checker import HybridComplianceChecker

checker = HybridComplianceChecker.from_config(config)

# Run health check
health = checker.health_check()

print(f"AI Service: {health['ai_service']['status']}")
print(f"Rule Engine: {health['rule_engine']['status']}")
print(f"Cache: {health['cache']['status']}")
print(f"Overall: {health['overall_status']}")
```

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hybrid_compliance_checker')
logger.setLevel(logging.DEBUG)

# Now run your checks with detailed logging
```

## Common Issues and Solutions

### 0. Context-Aware Component Issues (NEW)

#### Symptom: Fund Descriptions Still Flagged as Investment Advice

**Possible Causes and Solutions**

**A. Context Analysis Not Enabled**

```python
# Check if context analysis is enabled
from context_analyzer import ContextAnalyzer
from ai_engine import AIEngine

try:
    analyzer = ContextAnalyzer(AIEngine())
    print("✓ Context Analyzer initialized")
except Exception as e:
    print(f"✗ Context Analyzer error: {e}")

# Solution: Enable in configuration
config = {
    "context_analysis": {
        "enabled": true,
        "min_confidence": 60,
        "use_fallback_rules": true
    }
}
```

**B. Intent Classification Failing**

```python
# Test intent classifier
from intent_classifier import IntentClassifier
from ai_engine import AIEngine

classifier = IntentClassifier(AIEngine())

# Test with known fund description
test_text = "Le fonds investit dans des stratégies momentum"
intent = classifier.classify_intent(test_text)

print(f"Intent type: {intent.intent_type}")  # Should be "DESCRIPTION"
print(f"Subject: {intent.subject}")  # Should be "fund"
print(f"Confidence: {intent.confidence}%")

if intent.intent_type != "DESCRIPTION":
    print("✗ Intent classifier not working correctly")
    print("Check AI service connection and prompt templates")
```

**C. Prompt Template Issues**

```python
# Verify prompt templates are loaded
analyzer = ContextAnalyzer(AIEngine())
template = analyzer.get_prompt_template("investment_advice_detection")

print(f"Template loaded: {template is not None}")
print(f"Required vars: {template.get('required_vars', [])}")

# Solution: Check prompt_templates.json exists and is valid
```

#### Symptom: Fund Name Repetitions Still Flagged

**Possible Causes and Solutions**

**A. Whitelist Not Building Correctly**

```python
# Debug whitelist building
from whitelist_manager import WhitelistManager

manager = WhitelistManager()
document = load_document("example.json")

# Check fund name extraction
fund_name = document.get('metadata', {}).get('fund_name', '')
print(f"Fund name from metadata: '{fund_name}'")

# Build whitelist
whitelist = manager.build_whitelist(document)
print(f"Total whitelisted terms: {len(whitelist)}")
print(f"Sample terms: {list(whitelist)[:20]}")

# Check if fund name components are whitelisted
fund_parts = fund_name.lower().split()
for part in fund_parts:
    if part in whitelist:
        print(f"✓ '{part}' is whitelisted")
    else:
        print(f"✗ '{part}' NOT whitelisted")
        print(f"  Reason: {manager.get_whitelist_reason(part)}")
```

**B. Whitelist Not Being Used in Checks**

```python
# Verify whitelist is passed to semantic validator
from semantic_validator import SemanticValidator

validator = SemanticValidator(ai_engine, context_analyzer, intent_classifier)

# Test with whitelisted term
result = validator.validate_securities_mention(
    text="ODDO BHF appears multiple times",
    whitelist={"oddo", "bhf"}
)

print(f"Is violation: {result.is_violation}")  # Should be False
print(f"Reasoning: {result.reasoning}")

if result.is_violation:
    print("✗ Whitelist not being applied correctly")
```

**C. Custom Terms Not Added**

```python
# Add custom terms to whitelist
config = {
    "whitelist": {
        "custom_terms": ["your", "custom", "terms"],
        "auto_extract_fund_name": true,
        "include_strategy_terms": true
    }
}

# Or programmatically
manager = WhitelistManager()
manager.add_custom_terms(["proprietary", "alpha", "beta"])
```

#### Symptom: Performance Keywords Flagged Without Data

**Possible Causes and Solutions**

**A. Evidence Extraction Not Working**

```python
# Test evidence extractor
from evidence_extractor import EvidenceExtractor
from ai_engine import AIEngine

extractor = EvidenceExtractor(AIEngine())

# Test 1: No actual data (should return empty)
text1 = "The fund seeks attractive performance"
perf1 = extractor.find_performance_data(text1)
print(f"Test 1 - Found {len(perf1)} performance data items")  # Should be 0

# Test 2: Actual data (should detect)
text2 = "The fund returned 15% in 2024"
perf2 = extractor.find_performance_data(text2)
print(f"Test 2 - Found {len(perf2)} performance data items")  # Should be > 0

if len(perf1) > 0:
    print("✗ Evidence extractor flagging keywords as data")
    print("Check AI prompt templates for performance detection")
```

**B. Evidence Extraction Not Enabled**

```python
# Enable evidence extraction
config = {
    "evidence_extraction": {
        "enabled": true,
        "semantic_disclaimer_matching": true,
        "include_context": true
    }
}
```

**C. Disclaimer Matching Too Strict**

```python
# Test semantic disclaimer matching
extractor = EvidenceExtractor(AIEngine())

slide_text = """
Les performances passées ne sont pas un indicateur fiable 
des performances futures.
"""

disclaimer = extractor.find_disclaimer(
    text=slide_text,
    required_disclaimer="performances passées ne préjugent pas"
)

print(f"Disclaimer found: {disclaimer is not None}")
print(f"Match confidence: {disclaimer.confidence if disclaimer else 0}%")

# Solution: Lower semantic matching threshold
config = {
    "evidence_extraction": {
        "semantic_disclaimer_matching": true,
        "disclaimer_match_threshold": 70  # Lower from 85
    }
}
```

#### Symptom: Low Confidence Scores on Context Analysis

**Possible Causes and Solutions**

**A. AI Service Latency or Errors**

```python
# Check AI service performance
from ai_engine import AIEngine
import time

ai_engine = AIEngine()

start = time.time()
try:
    result = ai_engine.analyze_text(
        "Le fonds investit dans des actions",
        analysis_type="intent_classification"
    )
    latency = time.time() - start
    print(f"✓ AI service responding in {latency:.2f}s")
    print(f"Result confidence: {result.get('confidence', 0)}%")
except Exception as e:
    print(f"✗ AI service error: {e}")
```

**B. Fallback to Rules Too Frequent**

```python
# Check fallback rate
from performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
metrics = monitor.get_metrics()

print(f"AI calls: {metrics['ai_calls']}")
print(f"Fallback calls: {metrics['fallback_calls']}")
print(f"Fallback rate: {metrics['fallback_rate']}%")

if metrics['fallback_rate'] > 20:
    print("⚠ High fallback rate - check AI service reliability")
```

**C. Prompt Templates Need Tuning**

```python
# Review and customize prompt templates
from context_analyzer import ContextAnalyzer

analyzer = ContextAnalyzer(AIEngine())

# Set custom prompt for better results
analyzer.set_prompt_template(
    "investment_advice_detection",
    system_message="You are an expert at analyzing French financial documents...",
    user_prompt="""
    Analyze this text and determine if it's investment advice to clients or 
    a description of the fund's strategy.
    
    TEXT: {text}
    
    Respond with JSON: {{"intent": "ADVICE|DESCRIPTION", "confidence": 0-100, "reasoning": "..."}}
    """
)
```

### 1. AI Service Connection Errors

#### Symptom
```
AIServiceError: Failed to connect to AI service
```

#### Possible Causes and Solutions

**A. Invalid or Missing API Key**

```python
# Check if API key is set
import os
print(f"API Key present: {bool(os.getenv('AI_API_KEY'))}")

# Solution: Set API key in environment
export AI_API_KEY=your_actual_api_key_here

# Or in .env file
AI_API_KEY=your_actual_api_key_here
```

**B. Network Connectivity Issues**

```python
# Test network connectivity
import requests

try:
    response = requests.get("https://generativelanguage.googleapis.com", timeout=5)
    print(f"Network OK: {response.status_code}")
except Exception as e:
    print(f"Network issue: {e}")

# Solution: Check firewall, proxy settings, or VPN
```

**C. API Rate Limiting**

```
Error: 429 Too Many Requests
```

```python
# Solution: Enable rate limiting in config
config = {
    "ai_service": {
        "max_retries": 5,
        "retry_delay": 2,
        "exponential_backoff": True
    },
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 60
    }
}
```

**D. API Quota Exceeded**

```
Error: 403 Quota Exceeded
```

```python
# Solution: Check quota usage and upgrade plan
# Enable fallback to rule-only mode
config = {
    "fallback": {
        "enabled": True,
        "fallback_on_quota_exceeded": True
    }
}
```

### 2. Low Confidence Scores

#### Symptom
```
Most results have confidence scores below 70%
```

#### Possible Causes and Solutions

**A. Insufficient Training Data**

```python
# Check calibration status
from confidence_calibrator import ConfidenceCalibrator

calibrator = ConfidenceCalibrator()
stats = calibrator.get_stats()

print(f"Samples collected: {stats['sample_count']}")
print(f"Calibration active: {stats['is_calibrated']}")

# Solution: Collect more feedback data
# Minimum 100 samples recommended before calibration
```

**B. Mismatched Thresholds**

```python
# Adjust confidence thresholds
config = {
    "confidence": {
        "min_confidence": 60,  # Lower if too strict
        "review_threshold": 75,
        "calibration_enabled": True
    }
}
```

**C. AI and Rules Disagreeing**

```python
# Check disagreement rate
from performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
metrics = monitor.get_metrics()

print(f"Agreement rate: {metrics['agreement_rate']}%")
print(f"AI-only detections: {metrics['ai_only_count']}")
print(f"Rule-only detections: {metrics['rule_only_count']}")

# Solution: Review and update rules to align with AI findings
```

### 3. Slow Performance

#### Symptom
```
Document processing takes > 60 seconds
```

#### Possible Causes and Solutions

**A. Cache Not Enabled**

```python
# Check cache status
health = checker.health_check()
print(f"Cache enabled: {health['cache']['enabled']}")
print(f"Cache hit rate: {health['cache']['hit_rate']}%")

# Solution: Enable caching
config = {
    "ai_service": {
        "cache_enabled": True,
        "cache_ttl": 3600
    }
}
```

**B. Sequential Processing**

```python
# Solution: Enable batch and async processing
config = {
    "performance": {
        "batch_enabled": True,
        "batch_size": 10,
        "async_enabled": True,
        "max_concurrent": 5
    }
}

# Use batch processing
results = checker.batch_check_compliance(documents, check_type)
```

**C. Large Documents**

```python
# Solution: Implement document chunking
config = {
    "document_processing": {
        "max_chunk_size": 4000,  # tokens
        "chunk_overlap": 200,
        "parallel_chunks": True
    }
}
```

**D. Network Latency**

```python
# Measure API latency
import time

start = time.time()
result = ai_engine.analyze(document, "promotional_mention")
latency = time.time() - start

print(f"API latency: {latency:.2f}s")

# Solution: Use regional endpoints or increase timeout
config = {
    "ai_service": {
        "timeout": 60,  # Increase timeout
        "endpoint": "https://api.region.provider.com"  # Use closer region
    }
}
```

### 4. Incorrect Results

#### Symptom
```
System reports violations that don't exist (false positives)
or misses actual violations (false negatives)
```

#### Possible Causes and Solutions

**A. False Positives**

```python
# Analyze false positive patterns
from pattern_detector import PatternDetector

detector = PatternDetector()
patterns = detector.analyze_false_positives()

print(f"Common false positive patterns: {patterns}")

# Solution: Add false positive filters
config = {
    "false_positive_filtering": {
        "enabled": True,
        "patterns": patterns,
        "confidence_penalty": 20
    }
}
```

**B. False Negatives**

```python
# Check if AI is finding variations that rules miss
results = checker.check_compliance(document, "promotional_mention")

if results['status'] == 'AI_DETECTED_VARIATION':
    print(f"AI found variation: {results['evidence']}")
    print(f"Rules missed: {results['rule_result']['found']}")
    
# Solution: Update rules based on AI findings
from feedback_loop import FeedbackLoop

feedback = FeedbackLoop()
feedback.submit_correction(
    check_id=results['id'],
    correct_outcome=True,
    notes="AI correctly detected variation"
)
```

**C. Language or Context Issues**

```python
# Specify language explicitly
result = checker.check_compliance(
    document=doc,
    check_type="promotional_mention",
    options={"language": "french"}  # or "english"
)

# Enable multi-language support
config = {
    "language": {
        "auto_detect": True,
        "supported": ["french", "english"],
        "fallback": "french"
    }
}
```

### 5. Memory Issues

#### Symptom
```
MemoryError or system running out of memory
```

#### Possible Causes and Solutions

**A. Cache Growing Too Large**

```python
# Check cache size
health = checker.health_check()
print(f"Cache entries: {health['cache']['size']}")
print(f"Cache memory: {health['cache']['memory_mb']} MB")

# Solution: Limit cache size
config = {
    "ai_service": {
        "cache_enabled": True,
        "max_cache_size": 500,  # Limit entries
        "eviction_policy": "lru"  # Least recently used
    }
}
```

**B. Processing Too Many Documents Simultaneously**

```python
# Solution: Reduce concurrency
config = {
    "performance": {
        "max_concurrent": 3,  # Reduce from 5
        "batch_size": 5  # Reduce from 10
    }
}
```

**C. Large Document Accumulation**

```python
# Solution: Process documents in smaller batches
def process_documents_in_batches(documents, batch_size=10):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        results = checker.batch_check_compliance(batch)
        yield results
        # Memory is freed between batches

for batch_results in process_documents_in_batches(all_documents):
    process_results(batch_results)
```

### 6. Configuration Errors

#### Symptom
```
ConfigurationError: Invalid configuration
```

#### Possible Causes and Solutions

**A. Invalid JSON Format**

```python
# Validate JSON syntax
import json

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except json.JSONDecodeError as e:
    print(f"JSON error at line {e.lineno}: {e.msg}")

# Solution: Fix JSON syntax errors
```

**B. Missing Required Fields**

```python
# Use configuration validator
from config_validator import ConfigValidator

validator = ConfigValidator()
try:
    validator.validate(config)
except ConfigurationError as e:
    print(f"Missing fields: {e.missing_fields}")
    print(f"Suggestions: {e.suggestions}")

# Solution: Add required fields
config["ai_service"]["api_key"] = os.getenv("AI_API_KEY")
```

**C. Invalid Values**

```python
# Check for invalid values
validator = ConfigValidator()
issues = validator.check_values(config)

for issue in issues:
    print(f"Invalid: {issue['field']} = {issue['value']}")
    print(f"Expected: {issue['expected']}")

# Solution: Correct invalid values
config["confidence"]["min_confidence"] = 70  # Must be 0-100
```

### 7. Fallback Mode Issues

#### Symptom
```
System stuck in rule-only fallback mode
```

#### Possible Causes and Solutions

**A. AI Service Marked as Down**

```python
# Check AI service status
health = checker.health_check()
print(f"AI service status: {health['ai_service']['status']}")
print(f"Last error: {health['ai_service']['last_error']}")
print(f"Fallback active: {health['fallback_active']}")

# Solution: Reset AI service status
checker.reset_ai_service()

# Or restart with fresh connection
checker = HybridComplianceChecker.from_config(config)
```

**B. Persistent Connection Issues**

```python
# Test AI connection directly
from ai_engine import AIEngine

ai_engine = AIEngine(api_key=os.getenv("AI_API_KEY"))
try:
    test_result = ai_engine.test_connection()
    print(f"Connection test: {test_result}")
except Exception as e:
    print(f"Connection failed: {e}")

# Solution: Check network, API key, and service status
```

### 8. Feedback Loop Not Working

#### Symptom
```
Submitted corrections not improving results
```

#### Possible Causes and Solutions

**A. Insufficient Feedback Data**

```python
# Check feedback statistics
from feedback_loop import FeedbackLoop

feedback = FeedbackLoop()
stats = feedback.get_stats()

print(f"Total feedback: {stats['total_count']}")
print(f"By check type: {stats['by_check_type']}")
print(f"Patterns detected: {stats['patterns_detected']}")

# Solution: Collect more feedback (minimum 50 per check type)
```

**B. Calibration Not Enabled**

```python
# Enable calibration
config = {
    "confidence": {
        "calibration_enabled": True,
        "min_samples": 50,
        "recalibration_interval": 100
    }
}
```

**C. Feedback Not Being Applied**

```python
# Force recalibration
from confidence_calibrator import ConfidenceCalibrator

calibrator = ConfidenceCalibrator()
calibrator.force_recalibration()

# Check if improvements applied
improvements = calibrator.get_improvements()
print(f"Threshold adjustments: {improvements}")
```

## Diagnostic Commands

### Check System Status

```bash
# Run diagnostic script
python -m hybrid_compliance_checker.diagnostics

# Output:
# ✓ AI Service: Connected
# ✓ Rule Engine: Loaded
# ✓ Cache: Active (512 entries)
# ✓ Configuration: Valid
# ⚠ Calibration: Needs more data (45/100 samples)
```

### Test Individual Components

```python
# Test AI Engine
from ai_engine import AIEngine

ai = AIEngine()
test_result = ai.self_test()
print(f"AI Engine: {test_result['status']}")

# Test Rule Engine
from rule_engine import RuleEngine

rules = RuleEngine()
test_result = rules.self_test()
print(f"Rule Engine: {test_result['status']}")

# Test Confidence Scorer
from confidence_scorer import ConfidenceScorer

scorer = ConfidenceScorer()
test_result = scorer.self_test()
print(f"Confidence Scorer: {test_result['status']}")

# Test Context-Aware Components (NEW)
from context_analyzer import ContextAnalyzer
from intent_classifier import IntentClassifier
from semantic_validator import SemanticValidator
from evidence_extractor import EvidenceExtractor
from whitelist_manager import WhitelistManager

ai_engine = AIEngine()

# Test Context Analyzer
print("\n=== Context Analyzer ===")
analyzer = ContextAnalyzer(ai_engine)
context = analyzer.analyze_context(
    "Le fonds investit dans des stratégies momentum",
    check_type="investment_advice"
)
print(f"✓ Subject: {context.subject}")
print(f"✓ Intent: {context.intent}")
print(f"✓ Is fund description: {context.is_fund_description}")
print(f"✓ Confidence: {context.confidence}%")

# Test Intent Classifier
print("\n=== Intent Classifier ===")
classifier = IntentClassifier(ai_engine)
intent = classifier.classify_intent("Vous devriez investir maintenant")
print(f"✓ Intent type: {intent.intent_type}")
print(f"✓ Subject: {intent.subject}")
print(f"✓ Confidence: {intent.confidence}%")

# Test Whitelist Manager
print("\n=== Whitelist Manager ===")
manager = WhitelistManager()
test_doc = {"metadata": {"fund_name": "ODDO BHF Algo Trend"}}
whitelist = manager.build_whitelist(test_doc)
print(f"✓ Whitelisted terms: {len(whitelist)}")
print(f"✓ 'ODDO' whitelisted: {manager.is_whitelisted('oddo')}")
print(f"✓ 'momentum' whitelisted: {manager.is_whitelisted('momentum')}")

# Test Evidence Extractor
print("\n=== Evidence Extractor ===")
extractor = EvidenceExtractor(ai_engine)
perf_data = extractor.find_performance_data("The fund returned 15% in 2024")
print(f"✓ Performance data found: {len(perf_data) > 0}")
if perf_data:
    print(f"✓ Value: {perf_data[0].value}")

# Test Semantic Validator
print("\n=== Semantic Validator ===")
validator = SemanticValidator(ai_engine, analyzer, classifier)
result = validator.validate_securities_mention(
    text="ODDO BHF momentum strategy",
    whitelist={"oddo", "bhf", "momentum"}
)
print(f"✓ Is violation: {result.is_violation}")
print(f"✓ Confidence: {result.confidence}%")
```

### Performance Profiling

```python
# Profile a check
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

result = checker.check_compliance(document, "promotional_mention")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 slowest functions
```

## Error Messages Reference

### AIServiceError

| Error Message | Cause | Solution |
|--------------|-------|----------|
| "API key not found" | Missing API key | Set AI_API_KEY environment variable |
| "Connection timeout" | Network issue | Check network, increase timeout |
| "Rate limit exceeded" | Too many requests | Enable rate limiting, reduce concurrency |
| "Invalid response format" | API returned unexpected data | Check API version compatibility |
| "Model not found" | Invalid model name | Use supported model (gemini-pro, gpt-4) |

### ConfigurationError

| Error Message | Cause | Solution |
|--------------|-------|----------|
| "Missing required field" | Config incomplete | Add required configuration fields |
| "Invalid value for field" | Wrong data type/range | Check configuration guide for valid values |
| "Conflicting settings" | Incompatible options | Review configuration for conflicts |

### DocumentParsingError

| Error Message | Cause | Solution |
|--------------|-------|----------|
| "Unable to extract text" | Corrupted document | Verify document integrity |
| "Unsupported format" | Wrong file type | Use supported formats (PDF, DOCX) |
| "Empty document" | No content found | Check document has actual content |

## Getting Help

### Enable Detailed Logging

```python
# Save detailed logs for support
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Run your check
result = checker.check_compliance(document, check_type)

# Send debug.log to support
```

### Collect Diagnostic Information

```python
# Generate diagnostic report
from hybrid_compliance_checker import generate_diagnostic_report

report = generate_diagnostic_report(
    include_config=True,
    include_logs=True,
    include_metrics=True,
    anonymize=True  # Remove sensitive data
)

# Save report
with open('diagnostic_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# Send diagnostic_report.json to support
```

### Common Support Questions

**Q: How do I know if the AI is working correctly?**

A: Check the `status` field in results. `VERIFIED_BY_BOTH` means AI and rules agree. `AI_DETECTED_VARIATION` means AI found something rules missed.

**Q: Why are confidence scores inconsistent?**

A: Confidence scores improve over time with calibration. Ensure calibration is enabled and collect feedback data.

**Q: Can I use the system without AI?**

A: Yes, set `ai_service.enabled = false` in config. System will use rule-only mode.

**Q: How do I reduce API costs?**

A: Enable caching, use batch processing, and set appropriate confidence thresholds to reduce unnecessary AI calls.

**Q: What's the minimum API quota needed?**

A: For 100 documents/day with 8 checks each: ~800 API calls/day. With caching: ~400-500 calls/day.

## Best Practices for Troubleshooting

1. **Start with Health Check**: Always run `health_check()` first
2. **Enable Debug Logging**: Provides detailed information about what's happening
3. **Test Components Individually**: Isolate the problem to specific components
4. **Check Configuration**: Many issues stem from configuration problems
5. **Monitor Metrics**: Track performance over time to identify trends
6. **Collect Feedback**: Helps improve accuracy and identify systematic issues
7. **Review Logs**: Error messages often contain helpful diagnostic information
8. **Test with Simple Cases**: Use known-good documents to verify system works
9. **Update Regularly**: Keep system and dependencies up to date
10. **Document Issues**: Keep track of problems and solutions for future reference

For additional support, see API_DOCUMENTATION.md and CONFIGURATION_GUIDE.md.
