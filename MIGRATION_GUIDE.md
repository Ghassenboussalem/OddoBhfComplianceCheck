# Migration Guide - AI-Enhanced Compliance Checker

## Overview

This guide helps you migrate from the current rule-based compliance checking system to the new AI-enhanced hybrid system with context-aware analysis. The migration is designed to be gradual and backward-compatible, allowing you to adopt new features incrementally.

### What's New in Context-Aware System

The enhanced system introduces five new components that eliminate false positives through semantic understanding:

1. **Context Analyzer** - Understands WHO performs actions and WHAT the intent is
2. **Intent Classifier** - Distinguishes advice from descriptions from facts
3. **Semantic Validator** - Validates based on meaning, not keywords
4. **Evidence Extractor** - Identifies and quotes specific supporting evidence
5. **Whitelist Manager** - Automatically recognizes fund names and strategy terms

**Key Improvements**:
- 85% reduction in false positives (40 → 6 violations on test documents)
- Distinguishes fund strategy descriptions from investment advice
- Automatically whitelists fund names and regulatory terms
- Detects actual performance data vs performance keywords
- Provides evidence and reasoning for each finding

## Migration Strategy

### Recommended Approach: Phased Migration

1. **Phase 1**: Run both systems in parallel (validation)
2. **Phase 2**: Enable AI for specific check types (gradual rollout)
3. **Phase 3**: Full migration with AI-enhanced checks
4. **Phase 4**: Optimize and tune based on feedback

## Pre-Migration Checklist

### System Requirements

- [ ] Python 3.8 or higher
- [ ] Existing compliance checker working correctly
- [ ] AI service API key obtained (Gemini or OpenAI)
- [ ] Sufficient API quota for expected volume
- [ ] Test environment available
- [ ] Backup of current system and data

### Dependencies

```bash
# Install new dependencies
pip install google-generativeai  # For Gemini
# OR
pip install openai  # For OpenAI

# Install additional requirements
pip install redis  # Optional, for distributed caching
pip install prometheus-client  # Optional, for monitoring
```

### Configuration Preparation

```bash
# Create configuration directory
mkdir -p config

# Create environment file
cat > .env << EOF
AI_API_KEY=your_api_key_here
AI_MODEL=gemini-pro
AI_TIMEOUT=30
MIN_CONFIDENCE=70
EOF
```

## Phase 1: Parallel Validation

### Step 1.1: Install Hybrid System

```bash
# Clone or update repository
git pull origin main

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from hybrid_compliance_checker import HybridComplianceChecker; print('OK')"
```

### Step 1.2: Configure for Parallel Mode

```python
# config/parallel_mode.json
{
  "mode": "parallel",
  "ai_service": {
    "provider": "gemini",
    "api_key": "${AI_API_KEY}",
    "enabled": true
  },
  "legacy_system": {
    "enabled": true,
    "compare_results": true
  },
  "output": {
    "save_both_results": true,
    "comparison_report": true
  }
}
```

### Step 1.3: Run Parallel Validation

```python
# parallel_validation.py
from hybrid_compliance_checker import HybridComplianceChecker
from check import check_document as legacy_check
import json

# Initialize hybrid checker
hybrid_checker = HybridComplianceChecker.from_config_file('config/parallel_mode.json')

# Process test documents
test_documents = load_test_documents()

comparison_results = []

for doc in test_documents:
    # Run legacy system
    legacy_result = legacy_check(doc)
    
    # Run hybrid system
    hybrid_result = hybrid_checker.check_all_compliance(doc)
    
    # Compare results
    comparison = compare_results(legacy_result, hybrid_result)
    comparison_results.append(comparison)
    
    print(f"Document: {doc['name']}")
    print(f"  Legacy violations: {len(legacy_result['violations'])}")
    print(f"  Hybrid violations: {len(hybrid_result)}")
    print(f"  Agreement: {comparison['agreement_rate']}%")
    print(f"  New detections: {comparison['new_detections']}")
    print(f"  Missed detections: {comparison['missed_detections']}")

# Generate comparison report
generate_comparison_report(comparison_results, 'validation_report.html')
```

### Step 1.4: Analyze Validation Results

```python
# analyze_validation.py
import json

with open('validation_results.json', 'r') as f:
    results = json.load(f)

# Calculate metrics
total_docs = len(results)
agreement_rate = sum(r['agreement_rate'] for r in results) / total_docs
new_detections = sum(len(r['new_detections']) for r in results)
missed_detections = sum(len(r['missed_detections']) for r in results)

print(f"Validation Summary:")
print(f"  Documents tested: {total_docs}")
print(f"  Agreement rate: {agreement_rate:.1f}%")
print(f"  New detections by AI: {new_detections}")
print(f"  Missed by AI: {missed_detections}")

# Review new detections
print("\nNew Detections (AI found, legacy missed):")
for result in results:
    for detection in result['new_detections']:
        print(f"  - {detection['type']}: {detection['message']}")
        print(f"    Confidence: {detection['confidence']}%")
        print(f"    Evidence: {detection['evidence']}")
```

**Decision Point**: If agreement rate > 90% and new detections look valid, proceed to Phase 2.

## Phase 2: Gradual Rollout

### Step 2.1: Enable AI for Specific Check Types

Start with the most problematic check types that have high false positive/negative rates.

```python
# config/gradual_rollout.json
{
  "mode": "hybrid",
  "feature_flags": {
    "ai_enhanced_checks": {
      "promotional_mention": true,      // Start with this
      "performance_claims": false,      // Enable later
      "fund_name_match": false,
      "disclaimer_validation": false,
      "registration_compliance": false,
      "structure_validation": false,
      "general_rules": false,
      "values_securities": false
    }
  },
  "context_analysis": {
    "enabled": true,
    "min_confidence": 60,
    "use_fallback_rules": true
  },
  "whitelist": {
    "auto_extract_fund_name": true,
    "include_strategy_terms": true,
    "include_regulatory_terms": true
  },
  "fallback": {
    "enabled": true,
    "fallback_on_error": true
  }
}
```

### Step 2.1b: Test Context-Aware Components

Before enabling checks, verify the new components work correctly:

```python
# test_context_aware.py
from context_analyzer import ContextAnalyzer
from intent_classifier import IntentClassifier
from whitelist_manager import WhitelistManager
from ai_engine import AIEngine

# Initialize components
ai_engine = AIEngine()
context_analyzer = ContextAnalyzer(ai_engine)
intent_classifier = IntentClassifier(ai_engine)
whitelist_manager = WhitelistManager()

# Test 1: Context Analysis
print("Test 1: Context Analysis")
context = context_analyzer.analyze_context(
    "Le fonds tire parti du momentum",
    check_type="investment_advice"
)
print(f"  Subject: {context.subject}")
print(f"  Intent: {context.intent}")
print(f"  Is fund description: {context.is_fund_description}")
print(f"  Confidence: {context.confidence}%")
assert context.is_fund_description == True, "Should recognize fund description"

# Test 2: Intent Classification
print("\nTest 2: Intent Classification")
intent = intent_classifier.classify_intent("Vous devriez investir maintenant")
print(f"  Intent type: {intent.intent_type}")
print(f"  Subject: {intent.subject}")
print(f"  Confidence: {intent.confidence}%")
assert intent.intent_type == "ADVICE", "Should classify as advice"

# Test 3: Whitelist Management
print("\nTest 3: Whitelist Management")
document = load_test_document()
whitelist = whitelist_manager.build_whitelist(document)
print(f"  Whitelisted terms: {len(whitelist)}")
print(f"  Sample terms: {list(whitelist)[:10]}")
assert whitelist_manager.is_whitelisted("ODDO"), "Should whitelist fund name"

print("\n✅ All context-aware components working correctly")
```

### Step 2.2: Update Check Script

```python
# check_hybrid.py (new version of check.py)
from hybrid_compliance_checker import HybridComplianceChecker
from check import check_document as legacy_check
import sys

def check_document_hybrid(document_path, config_path='config/gradual_rollout.json'):
    """
    Enhanced check function that uses hybrid system for enabled checks
    and legacy system for others.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize hybrid checker
    hybrid_checker = HybridComplianceChecker.from_config(config)
    
    # Parse document
    document = parse_document(document_path)
    
    # Run checks
    results = []
    
    for check_type in ALL_CHECK_TYPES:
        if config['feature_flags']['ai_enhanced_checks'].get(check_type, False):
            # Use hybrid system
            result = hybrid_checker.check_compliance(document, check_type)
            results.append(result)
        else:
            # Use legacy system
            result = legacy_check_single(document, check_type)
            results.append(result)
    
    return format_results(results)

if __name__ == '__main__':
    document_path = sys.argv[1]
    results = check_document_hybrid(document_path)
    print(json.dumps(results, indent=2))
```

### Step 2.3: Monitor and Validate

```python
# monitor_rollout.py
from performance_monitor import PerformanceMonitor
import time

monitor = PerformanceMonitor()

# Process documents with monitoring
for doc_path in document_paths:
    start_time = time.time()
    
    results = check_document_hybrid(doc_path)
    
    processing_time = time.time() - start_time
    
    # Log metrics
    monitor.log_check(
        document=doc_path,
        processing_time=processing_time,
        results=results,
        ai_enabled_checks=get_ai_enabled_checks()
    )

# Generate rollout report
report = monitor.generate_rollout_report()
print(f"Average processing time: {report['avg_time']}s")
print(f"AI-enhanced checks: {report['ai_check_count']}")
print(f"Confidence distribution: {report['confidence_dist']}")
print(f"Error rate: {report['error_rate']}%")
```

### Step 2.4: Gradually Enable More Checks

After validating each check type, enable the next one:

```python
# Week 1: promotional_mention only
# Week 2: Add performance_claims
# Week 3: Add fund_name_match
# Week 4: Add disclaimer_validation
# etc.

# Update config/gradual_rollout.json after each successful week
{
  "feature_flags": {
    "ai_enhanced_checks": {
      "promotional_mention": true,
      "performance_claims": true,  // Newly enabled
      "fund_name_match": false,
      // ...
    }
  }
}
```

## Phase 3: Full Migration

### Step 3.1: Enable All AI-Enhanced Checks

```python
# config/production.json
{
  "mode": "hybrid",
  "ai_service": {
    "provider": "gemini",
    "api_key": "${AI_API_KEY}",
    "model": "gemini-pro",
    "timeout": 30,
    "cache_enabled": true,
    "batch_enabled": true
  },
  "feature_flags": {
    "ai_enhanced_checks": {
      "promotional_mention": true,
      "performance_claims": true,
      "fund_name_match": true,
      "disclaimer_validation": true,
      "registration_compliance": true,
      "structure_validation": true,
      "general_rules": true,
      "values_securities": true
    }
  },
  "confidence": {
    "min_confidence": 70,
    "review_threshold": 85,
    "calibration_enabled": true
  },
  "performance": {
    "cache_enabled": true,
    "batch_enabled": true,
    "async_enabled": true
  },
  "monitoring": {
    "enabled": true,
    "metrics_endpoint": "/metrics"
  }
}
```

### Step 3.2: Replace Legacy Check Script

```python
# Backup legacy system
cp check.py check_legacy.py

# Update check.py to use hybrid system
cat > check.py << 'EOF'
#!/usr/bin/env python3
"""
AI-Enhanced Compliance Checker
Backward-compatible replacement for legacy check.py
"""

from hybrid_compliance_checker import HybridComplianceChecker
import sys
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: python check.py <document_path>")
        sys.exit(1)
    
    document_path = sys.argv[1]
    
    # Initialize checker
    checker = HybridComplianceChecker.from_config_file('config/production.json')
    
    # Parse document
    document = parse_document(document_path)
    
    # Run all checks
    results = checker.check_all_compliance(document)
    
    # Format output (backward-compatible)
    output = format_legacy_output(results)
    
    # Print results
    print(json.dumps(output, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
EOF
```

### Step 3.3: Update Integration Points

```python
# If you have automated workflows, update them

# Example: CI/CD pipeline
# .github/workflows/compliance_check.yml
name: Compliance Check
on: [push]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run compliance check
        env:
          AI_API_KEY: ${{ secrets.AI_API_KEY }}
        run: python check.py documents/fund_document.pdf
```

## Phase 4: Optimization

### Step 4.1: Enable Feedback Loop

```python
# feedback_integration.py
from feedback_loop import FeedbackLoop
from hybrid_compliance_checker import HybridComplianceChecker

feedback = FeedbackLoop()
checker = HybridComplianceChecker.from_config_file('config/production.json')

# After human review, submit corrections
def submit_review_feedback(check_id, human_decision, notes):
    feedback.submit_correction(
        check_id=check_id,
        correct_outcome=human_decision,
        notes=notes
    )
    
    # System automatically learns and adjusts

# Integrate with review workflow
def review_workflow(results):
    for result in results:
        if result['confidence'] < 85:
            # Flag for human review
            human_decision = get_human_review(result)
            submit_review_feedback(
                result['id'],
                human_decision,
                "Human review feedback"
            )
```

### Step 4.2: Optimize Performance

```python
# Enable advanced caching
config = {
    "performance": {
        "cache": {
            "enabled": true,
            "backend": "redis",  // Upgrade from memory to Redis
            "redis_host": "localhost",
            "redis_port": 6379,
            "max_size": 10000,
            "ttl": 7200
        },
        "batch": {
            "enabled": true,
            "batch_size": 20,  // Increase batch size
            "parallel_batches": 5
        },
        "async": {
            "enabled": true,
            "max_concurrent": 10  // Increase concurrency
        }
    }
}
```

### Step 4.3: Tune Confidence Thresholds

```python
# After collecting sufficient data, analyze and adjust thresholds
from confidence_calibrator import ConfidenceCalibrator

calibrator = ConfidenceCalibrator()

# Analyze historical accuracy
analysis = calibrator.analyze_accuracy()

print(f"Current thresholds:")
print(f"  Min confidence: {analysis['current_thresholds']['min']}")
print(f"  Review threshold: {analysis['current_thresholds']['review']}")

print(f"\nRecommended thresholds:")
print(f"  Min confidence: {analysis['recommended_thresholds']['min']}")
print(f"  Review threshold: {analysis['recommended_thresholds']['review']}")

# Apply recommendations
if analysis['confidence_score'] > 0.8:
    calibrator.apply_recommendations()
    print("Thresholds updated based on historical performance")
```

## Backward Compatibility

### Output Format Compatibility

The hybrid system maintains backward compatibility with the legacy output format:

```python
# Legacy format
{
  "violations": [
    {
      "type": "PROMOTIONAL",
      "severity": "CRITICAL",
      "slide": "1",
      "message": "Missing promotional mention",
      "rule": "promotional_mention"
    }
  ]
}

# Hybrid format (backward-compatible with extensions)
{
  "violations": [
    {
      "type": "PROMOTIONAL",
      "severity": "CRITICAL",
      "slide": "1",
      "message": "Missing promotional mention",
      "rule": "promotional_mention",
      // New fields (optional, ignored by legacy systems)
      "confidence": 95,
      "status": "VERIFIED_BY_BOTH",
      "evidence": ["Analyzed cover page text..."],
      "reasoning": "AI detected absence of required promotional language",
      "suggestions": ["Add 'Document promotionnel' to cover page"]
    }
  ]
}
```

### API Compatibility

```python
# Legacy API still works
from check import check_document

results = check_document("document.pdf")

# New API is backward-compatible
from hybrid_compliance_checker import HybridComplianceChecker

checker = HybridComplianceChecker.from_config_file('config/production.json')
results = checker.check_document_legacy_format("document.pdf")
# Returns same format as legacy system
```

## Rollback Plan

If issues arise, you can quickly rollback:

### Quick Rollback

```bash
# Disable AI, use rule-only mode
export AI_ENABLED=false

# Or update config
{
  "ai_service": {
    "enabled": false
  }
}

# System automatically falls back to rule-only mode
```

### Full Rollback

```bash
# Restore legacy system
cp check_legacy.py check.py

# Uninstall hybrid dependencies (optional)
pip uninstall google-generativeai

# System reverts to original behavior
```

## Migration Checklist

### Pre-Migration
- [ ] Backup current system
- [ ] Install dependencies
- [ ] Obtain AI API key
- [ ] Set up test environment
- [ ] Create configuration files

### Phase 1: Validation
- [ ] Run parallel validation
- [ ] Analyze comparison results
- [ ] Review new detections
- [ ] Verify agreement rate > 90%
- [ ] Get stakeholder approval

### Phase 2: Gradual Rollout
- [ ] Enable first check type (promotional_mention)
- [ ] Monitor for 1 week
- [ ] Enable second check type (performance_claims)
- [ ] Monitor for 1 week
- [ ] Continue for remaining check types
- [ ] Collect feedback from users

### Phase 3: Full Migration
- [ ] Enable all AI-enhanced checks
- [ ] Replace legacy check script
- [ ] Update integration points
- [ ] Update documentation
- [ ] Train users on new features

### Phase 4: Optimization
- [ ] Enable feedback loop
- [ ] Optimize caching and performance
- [ ] Tune confidence thresholds
- [ ] Set up monitoring and alerting
- [ ] Document lessons learned

## Common Migration Issues

### Issue: High API Costs

**Solution**: Enable caching and batch processing
```python
config = {
    "ai_service": {
        "cache_enabled": true,
        "batch_enabled": true,
        "batch_size": 20
    }
}
```

### Issue: Slower Performance

**Solution**: Enable async processing and increase concurrency
```python
config = {
    "performance": {
        "async_enabled": true,
        "max_concurrent": 10,
        "batch_slides": true
    }
}
```

### Issue: Different Results from Legacy

**Solution**: This is expected. Review new detections to verify they're valid improvements.

The context-aware system should produce FEWER violations (fewer false positives):
- Fund strategy descriptions no longer flagged as advice
- Fund name repetitions no longer flagged as securities
- Performance keywords without data no longer flagged

**Validation**:
```python
# Compare results
legacy_violations = 40  # Example
context_aware_violations = 6  # Expected

if context_aware_violations < legacy_violations:
    print("✅ False positive reduction working")
    print(f"Eliminated {legacy_violations - context_aware_violations} false positives")
```

### Issue: Low Confidence Scores

**Solution**: Collect feedback data and enable calibration
```python
config = {
    "confidence": {
        "calibration_enabled": true,
        "min_samples": 100
    }
}
```

### Issue: Whitelist Not Extracting Fund Name

**Problem**: Fund name not being automatically whitelisted

**Solution**: Check document metadata structure
```python
from whitelist_manager import WhitelistManager

manager = WhitelistManager()

# Debug fund name extraction
document = load_document("example.json")
print(f"Document metadata: {document.get('metadata', {})}")

# Manually verify fund name
fund_name = document.get('metadata', {}).get('fund_name', '')
print(f"Fund name: {fund_name}")

# Check if extracted correctly
whitelist = manager.build_whitelist(document)
fund_name_parts = fund_name.lower().split()
for part in fund_name_parts:
    if part in whitelist:
        print(f"✅ '{part}' whitelisted")
    else:
        print(f"❌ '{part}' NOT whitelisted - check extraction logic")
```

### Issue: Context Analysis Not Working

**Problem**: Still getting false positives on fund descriptions

**Solution**: Verify context analysis is enabled and working
```python
from context_analyzer import ContextAnalyzer
from ai_engine import AIEngine

# Test context analysis
ai_engine = AIEngine()
analyzer = ContextAnalyzer(ai_engine)

# Test with known fund description
test_text = "Le fonds investit dans des stratégies momentum"
context = analyzer.analyze_context(test_text, "investment_advice")

print(f"Subject: {context.subject}")  # Should be "fund"
print(f"Intent: {context.intent}")    # Should be "describe"
print(f"Is fund description: {context.is_fund_description}")  # Should be True

if not context.is_fund_description:
    print("❌ Context analysis not working correctly")
    print("Check AI service connection and prompt templates")
```

### Issue: Evidence Extraction Missing Performance Data

**Problem**: Performance disclaimers flagged even when no data present

**Solution**: Test evidence extraction
```python
from evidence_extractor import EvidenceExtractor
from ai_engine import AIEngine

extractor = EvidenceExtractor(AIEngine())

# Test 1: No performance data (should return empty)
text1 = "The fund seeks attractive performance"
perf1 = extractor.find_performance_data(text1)
print(f"Test 1 (no data): {len(perf1)} items found")  # Should be 0

# Test 2: Actual performance data (should detect)
text2 = "The fund returned 15% in 2024"
perf2 = extractor.find_performance_data(text2)
print(f"Test 2 (has data): {len(perf2)} items found")  # Should be > 0
if perf2:
    print(f"  Found: {perf2[0].value}")  # Should be "15%"
```

## Whitelist Customization

### Understanding Whitelists

The whitelist system prevents false positives by recognizing terms that are allowed to appear frequently:

**Automatic Whitelists**:
- Fund name components (extracted from metadata)
- Strategy terms (momentum, quantitative, systematic, etc.)
- Regulatory terms (SRI, SRRI, SFDR, UCITS, etc.)
- Benchmark terms (S&P 500, MSCI, STOXX, etc.)
- Generic financial terms (actions, bonds, portfolio, etc.)

### Adding Custom Whitelist Terms

**Method 1: Configuration File**
```json
{
  "whitelist": {
    "custom_terms": [
      "proprietary",
      "alpha",
      "beta",
      "your-fund-specific-term"
    ]
  }
}
```

**Method 2: Programmatically**
```python
from whitelist_manager import WhitelistManager

manager = WhitelistManager()

# Add custom terms
manager.add_custom_terms(["proprietary", "alpha", "beta"])

# Add category-specific terms
manager.add_strategy_terms(["factor-based", "multi-asset"])
manager.add_regulatory_terms(["custom-regulation"])

# Build whitelist with custom terms
whitelist = manager.build_whitelist(document)
```

### Viewing Current Whitelist

```python
from whitelist_manager import WhitelistManager

manager = WhitelistManager()
whitelist = manager.build_whitelist(document)

# View all whitelisted terms
print(f"Total whitelisted terms: {len(whitelist)}")
print(f"Terms: {sorted(whitelist)}")

# View by category
print("\nFund name terms:")
for term in manager.fund_name_terms:
    print(f"  - {term}")

print("\nStrategy terms:")
for term in manager.strategy_terms:
    print(f"  - {term}")

print("\nRegulatory terms:")
for term in manager.regulatory_terms:
    print(f"  - {term}")
```

### Disabling Automatic Whitelist Categories

If you want more control, disable automatic categories:

```json
{
  "whitelist": {
    "auto_extract_fund_name": false,
    "include_strategy_terms": false,
    "include_regulatory_terms": true,
    "include_benchmark_terms": true,
    "custom_terms": ["only", "these", "terms"]
  }
}
```

## Post-Migration

### Monitoring

Set up ongoing monitoring:

```python
from performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()

# Track key metrics
metrics = monitor.get_metrics()
print(f"Documents processed: {metrics['total_documents']}")
print(f"Average confidence: {metrics['avg_confidence']}%")
print(f"API cost: ${metrics['api_cost']}")
print(f"Cache hit rate: {metrics['cache_hit_rate']}%")
```

### Continuous Improvement

```python
# Regular calibration
from confidence_calibrator import ConfidenceCalibrator

calibrator = ConfidenceCalibrator()

# Monthly recalibration
calibrator.recalibrate()

# Review and apply rule recommendations
from pattern_detector import PatternDetector

detector = PatternDetector()
recommendations = detector.get_rule_recommendations()

for rec in recommendations:
    print(f"Suggested rule: {rec['rule']}")
    print(f"Confidence: {rec['confidence']}")
    print(f"Based on: {rec['pattern_count']} occurrences")
```

## Success Criteria

Migration is successful when:

1. **Accuracy**: Agreement rate with validated results > 95%
2. **Performance**: Processing time < 30 seconds per document
3. **Reliability**: Error rate < 1%
4. **Cost**: API costs within budget
5. **User Satisfaction**: Positive feedback from compliance team

## Support

For migration support:
- Review TROUBLESHOOTING_GUIDE.md for common issues
- Check API_DOCUMENTATION.md for technical details
- See CONFIGURATION_GUIDE.md for configuration options

## Timeline Estimate

- **Phase 1 (Validation)**: 1-2 weeks
- **Phase 2 (Gradual Rollout)**: 4-8 weeks (1 week per check type)
- **Phase 3 (Full Migration)**: 1-2 weeks
- **Phase 4 (Optimization)**: Ongoing

**Total**: 6-12 weeks for complete migration

This gradual approach minimizes risk and allows for validation at each step.
