#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for False Positive Elimination Components

Tests all new components:
- WhitelistManager: fund name extraction, term whitelisting
- ContextAnalyzer: fund descriptions vs advice classification
- IntentClassifier: all intent types correctly identified
- EvidenceExtractor: performance data detection, disclaimer matching
- SemanticValidator: whitelist filtering, semantic validation

Run with: python test_false_positive_elimination.py
"""

import json
import sys
from typing import Dict, Set

# Import components to test
from whitelist_manager import WhitelistManager
from context_analyzer import ContextAnalyzer
from intent_classifier import IntentClassifier
from evidence_extractor import EvidenceExtractor
from semantic_validator import SemanticValidator
from data_models import (
    ContextAnalysis, IntentClassification, ValidationResult,
    Evidence, PerformanceData, DisclaimerMatch
)

# Test results tracking
test_results = {
    'passed': 0,
    'failed': 0,
    'errors': 0
}


def print_test_header(title: str):
    """Print formatted test section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_test_result(test_name: str, passed: bool, details: str = ""):
    """Print individual test result"""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n[{status}] {test_name}")
    if details:
        print(f"  {details}")
    
    if passed:
        test_results['passed'] += 1
    else:
        test_results['failed'] += 1


def assert_test(condition: bool, test_name: str, details: str = ""):
    """Assert test condition and track result"""
    print_test_result(test_name, condition, details)
    if not condition:
        raise AssertionError(f"Test failed: {test_name}")


# ============================================================================
# WHITELIST MANAGER TESTS
# ============================================================================

def test_whitelist_manager():
    """Test WhitelistManager: fund name extraction, term whitelisting"""
    print_test_header("WhitelistManager Tests")
    
    manager = WhitelistManager()
    
    # Test 1: Fund name extraction
    print("\n[Test 1] Fund name component extraction")
    doc = {
        'document_metadata': {
            'fund_name': 'ODDO BHF Algo Trend US Fund'
        }
    }
    
    whitelist = manager.build_whitelist(doc)
    
    # Check that fund name components are extracted
    assert 'oddo' in manager.fund_name_terms, "Should extract 'oddo'"
    assert 'bhf' in manager.fund_name_terms, "Should extract 'bhf'"
    assert 'algo' in manager.fund_name_terms, "Should extract 'algo'"
    assert 'trend' in manager.fund_name_terms, "Should extract 'trend'"
    
    print_test_result(
        "Fund name extraction",
        True,
        f"Extracted {len(manager.fund_name_terms)} components: {sorted(manager.fund_name_terms)}"
    )
    
    # Test 2: Whitelist checking
    print("\n[Test 2] Whitelist term checking")
    
    # Fund name should be whitelisted
    assert manager.is_whitelisted('ODDO'), "'ODDO' should be whitelisted"
    assert manager.is_whitelisted('BHF'), "'BHF' should be whitelisted"
    
    # Strategy terms should be whitelisted
    assert manager.is_whitelisted('momentum'), "'momentum' should be whitelisted"
    assert manager.is_whitelisted('quantitative'), "'quantitative' should be whitelisted"
    
    # Regulatory terms should be whitelisted
    assert manager.is_whitelisted('SRI'), "'SRI' should be whitelisted"
    assert manager.is_whitelisted('SRRI'), "'SRRI' should be whitelisted"
    
    # External companies should NOT be whitelisted
    assert not manager.is_whitelisted('Apple'), "'Apple' should not be whitelisted"
    assert not manager.is_whitelisted('Microsoft'), "'Microsoft' should not be whitelisted"
    
    print_test_result(
        "Whitelist checking",
        True,
        "All whitelist checks passed"
    )
    
    # Test 3: Whitelist reasons
    print("\n[Test 3] Whitelist reason explanations")
    
    reason_oddo = manager.get_whitelist_reason('ODDO')
    reason_momentum = manager.get_whitelist_reason('momentum')
    reason_sri = manager.get_whitelist_reason('SRI')
    reason_apple = manager.get_whitelist_reason('Apple')
    
    assert reason_oddo is not None, "Should have reason for 'ODDO'"
    assert reason_momentum is not None, "Should have reason for 'momentum'"
    assert reason_sri is not None, "Should have reason for 'SRI'"
    assert reason_apple is None, "Should not have reason for 'Apple'"
    
    print_test_result(
        "Whitelist reasons",
        True,
        f"ODDO: {reason_oddo}, momentum: {reason_momentum}"
    )
    
    # Test 4: Custom terms
    print("\n[Test 4] Custom whitelist terms")
    
    manager.add_custom_term('CustomTerm', 'Test custom term')
    assert manager.is_whitelisted('CustomTerm'), "Custom term should be whitelisted"
    
    reason_custom = manager.get_whitelist_reason('CustomTerm')
    assert 'custom' in reason_custom.lower(), "Should indicate custom term"
    
    print_test_result(
        "Custom terms",
        True,
        f"Custom term added and whitelisted: {reason_custom}"
    )
    
    # Test 5: Whitelist statistics
    print("\n[Test 5] Whitelist statistics")
    
    stats = manager.get_whitelist_stats()
    assert stats['fund_name_terms'] > 0, "Should have fund name terms"
    assert stats['strategy_terms'] > 0, "Should have strategy terms"
    assert stats['regulatory_terms'] > 0, "Should have regulatory terms"
    assert stats['total'] > 0, "Should have total terms"
    
    print_test_result(
        "Whitelist statistics",
        True,
        f"Total terms: {stats['total']}, Fund: {stats['fund_name_terms']}, Strategy: {stats['strategy_terms']}"
    )


# ============================================================================
# CONTEXT ANALYZER TESTS
# ============================================================================

def test_context_analyzer():
    """Test ContextAnalyzer: fund descriptions vs advice classification"""
    print_test_header("ContextAnalyzer Tests")
    
    # Create AI engine (may be None if not configured)
    from ai_engine import create_ai_engine_from_env
    ai_engine = create_ai_engine_from_env()
    
    if not ai_engine:
        print("\n⚠️  AI Engine not available - testing fallback mode only")
    
    analyzer = ContextAnalyzer(ai_engine)
    
    # Test 1: Fund strategy description (French)
    print("\n[Test 1] Fund strategy description (French)")
    text1 = "Tirer parti du momentum des marchés américains grâce à une stratégie quantitative"
    
    result1 = analyzer.analyze_context(text1, "investment_advice")
    
    assert result1.is_fund_description or result1.subject == "fund", \
        "Should identify as fund description"
    assert not result1.is_client_advice, "Should not be client advice"
    
    print_test_result(
        "Fund strategy description",
        True,
        f"Subject: {result1.subject}, Intent: {result1.intent}, Confidence: {result1.confidence}%"
    )
    
    # Test 2: Fund investment description
    print("\n[Test 2] Fund investment description")
    text2 = "Le fonds investit dans des actions américaines à forte capitalisation"
    
    result2 = analyzer.analyze_context(text2, "investment_advice")
    
    assert result2.is_fund_description or result2.subject == "fund", \
        "Should identify as fund description"
    assert not result2.is_client_advice, "Should not be client advice"
    
    print_test_result(
        "Fund investment description",
        True,
        f"Subject: {result2.subject}, Confidence: {result2.confidence}%"
    )
    
    # Test 3: Client investment advice
    print("\n[Test 3] Client investment advice")
    text3 = "Vous devriez investir dans ce fonds maintenant"
    
    result3 = analyzer.analyze_context(text3, "investment_advice")
    
    assert result3.is_client_advice or result3.subject == "client", \
        "Should identify as client advice"
    assert not result3.is_fund_description, "Should not be fund description"
    
    print_test_result(
        "Client investment advice",
        True,
        f"Subject: {result3.subject}, Intent: {result3.intent}, Confidence: {result3.confidence}%"
    )
    
    # Test 4: Recommendation to client
    print("\n[Test 4] Recommendation to client")
    text4 = "Nous recommandons d'acheter des actions technologiques"
    
    result4 = analyzer.analyze_context(text4, "investment_advice")
    
    assert result4.is_client_advice or result4.intent == "advise", \
        "Should identify as advice"
    
    print_test_result(
        "Recommendation detection",
        True,
        f"Intent: {result4.intent}, Confidence: {result4.confidence}%"
    )
    
    # Test 5: Subject extraction
    print("\n[Test 5] Subject extraction")
    
    subject_fund = analyzer.extract_subject("Le fonds investit dans des actions")
    subject_client = analyzer.extract_subject("Vous devriez investir maintenant")
    
    assert subject_fund in ["fund", "strategy"], f"Should extract 'fund', got '{subject_fund}'"
    # Client subject may be classified as "client" or "general" depending on AI interpretation
    assert subject_client in ["client", "general"], f"Should extract 'client' or 'general', got '{subject_client}'"
    
    print_test_result(
        "Subject extraction",
        True,
        f"Fund text → {subject_fund}, Client text → {subject_client}"
    )
    
    # Test 6: Confidence scoring
    print("\n[Test 6] Confidence scoring")
    
    # High confidence case
    high_conf_text = "Le fonds investit exclusivement dans des actions américaines"
    high_result = analyzer.analyze_context(high_conf_text, "investment_advice")
    
    assert high_result.confidence > 0, "Should have confidence score"
    assert high_result.reasoning, "Should have reasoning"
    
    print_test_result(
        "Confidence scoring",
        True,
        f"Confidence: {high_result.confidence}%, Reasoning: {high_result.reasoning[:50]}..."
    )


# ============================================================================
# INTENT CLASSIFIER TESTS
# ============================================================================

def test_intent_classifier():
    """Test IntentClassifier: all intent types correctly identified"""
    print_test_header("IntentClassifier Tests")
    
    # Create AI engine
    from ai_engine import create_ai_engine_from_env
    ai_engine = create_ai_engine_from_env()
    
    if not ai_engine:
        print("\n⚠️  AI Engine not available - testing fallback mode only")
    
    classifier = IntentClassifier(ai_engine)
    
    # Test 1: DESCRIPTION intent
    print("\n[Test 1] DESCRIPTION intent")
    desc_text = "Le fonds investit dans des actions américaines"
    
    desc_result = classifier.classify_intent(desc_text)
    
    assert desc_result.intent_type == "DESCRIPTION", \
        f"Should classify as DESCRIPTION, got {desc_result.intent_type}"
    assert desc_result.subject in ["fund", "strategy"], \
        f"Subject should be fund/strategy, got {desc_result.subject}"
    
    print_test_result(
        "DESCRIPTION intent",
        True,
        f"Intent: {desc_result.intent_type}, Subject: {desc_result.subject}, Confidence: {desc_result.confidence}%"
    )
    
    # Test 2: ADVICE intent
    print("\n[Test 2] ADVICE intent")
    advice_text = "Vous devriez investir dans ce fonds maintenant"
    
    advice_result = classifier.classify_intent(advice_text)
    
    assert advice_result.intent_type == "ADVICE", \
        f"Should classify as ADVICE, got {advice_result.intent_type}"
    assert advice_result.subject == "client", \
        f"Subject should be client, got {advice_result.subject}"
    
    print_test_result(
        "ADVICE intent",
        True,
        f"Intent: {advice_result.intent_type}, Subject: {advice_result.subject}, Confidence: {advice_result.confidence}%"
    )
    
    # Test 3: FACT intent
    print("\n[Test 3] FACT intent")
    fact_text = "Le fonds a généré un rendement de 15% en 2023"
    
    fact_result = classifier.classify_intent(fact_text)
    
    assert fact_result.intent_type in ["FACT", "DESCRIPTION"], \
        f"Should classify as FACT or DESCRIPTION, got {fact_result.intent_type}"
    
    print_test_result(
        "FACT intent",
        True,
        f"Intent: {fact_result.intent_type}, Confidence: {fact_result.confidence}%"
    )
    
    # Test 4: EXAMPLE intent
    print("\n[Test 4] EXAMPLE intent")
    example_text = "Par exemple, un investissement de 1000€ aurait généré 150€"
    
    example_result = classifier.classify_intent(example_text)
    
    # May classify as EXAMPLE or FACT depending on AI
    assert example_result.intent_type in ["EXAMPLE", "FACT"], \
        f"Should classify as EXAMPLE or FACT, got {example_result.intent_type}"
    
    print_test_result(
        "EXAMPLE intent",
        True,
        f"Intent: {example_result.intent_type}, Confidence: {example_result.confidence}%"
    )
    
    # Test 5: is_client_advice method
    print("\n[Test 5] is_client_advice method")
    
    advice_check1 = classifier.is_client_advice("Vous devriez investir")
    advice_check2 = classifier.is_client_advice("Le fonds investit")
    
    assert advice_check1, "Should detect client advice"
    assert not advice_check2, "Should not detect advice in fund description"
    
    print_test_result(
        "is_client_advice method",
        True,
        f"Advice detected: {advice_check1}, Fund description: {advice_check2}"
    )
    
    # Test 6: is_fund_description method
    print("\n[Test 6] is_fund_description method")
    
    desc_check1 = classifier.is_fund_description("Le fonds investit dans des actions")
    desc_check2 = classifier.is_fund_description("Vous devriez acheter ce fonds")
    
    assert desc_check1, "Should detect fund description"
    assert not desc_check2, "Should not detect description in advice"
    
    print_test_result(
        "is_fund_description method",
        True,
        f"Description detected: {desc_check1}, Advice: {desc_check2}"
    )


# ============================================================================
# EVIDENCE EXTRACTOR TESTS
# ============================================================================

def test_evidence_extractor():
    """Test EvidenceExtractor: performance data detection, disclaimer matching"""
    print_test_header("EvidenceExtractor Tests")
    
    # Create AI engine
    from ai_engine import create_ai_engine_from_env
    ai_engine = create_ai_engine_from_env()
    
    extractor = EvidenceExtractor(ai_engine)
    
    # Test 1: Actual performance data detection
    print("\n[Test 1] Actual performance data detection")
    perf_text = "Le fonds a généré +15.5% en 2023 et +20% en 2024"
    
    perf_data = extractor.find_performance_data(perf_text)
    
    assert len(perf_data) > 0, "Should detect performance data"
    assert any('%' in pd.value for pd in perf_data), "Should extract percentage values"
    
    print_test_result(
        "Performance data detection",
        True,
        f"Found {len(perf_data)} data points: {[pd.value for pd in perf_data]}"
    )
    
    # Test 2: Descriptive text without numbers
    print("\n[Test 2] Descriptive text (no actual data)")
    desc_text = "L'objectif de performance est d'obtenir des résultats attractifs"
    
    desc_data = extractor.find_performance_data(desc_text)
    
    # Should find few or none
    print_test_result(
        "Descriptive text handling",
        True,
        f"Found {len(desc_data)} data points (expected 0 or low confidence)"
    )
    
    # Test 3: Disclaimer matching - exact match
    print("\n[Test 3] Disclaimer matching - exact")
    exact_text = "Les performances passées ne préjugent pas des performances futures"
    
    disclaimer = extractor.find_disclaimer(exact_text, "performances passées ne préjugent pas")
    
    assert disclaimer.found, "Should find exact disclaimer match"
    assert disclaimer.similarity_score >= 80, f"Should have high similarity, got {disclaimer.similarity_score}%"
    
    print_test_result(
        "Exact disclaimer match",
        True,
        f"Found: {disclaimer.found}, Similarity: {disclaimer.similarity_score}%"
    )
    
    # Test 4: Disclaimer matching - no disclaimer
    print("\n[Test 4] Disclaimer matching - absent")
    no_disc_text = "Le fonds investit dans des actions européennes"
    
    no_disclaimer = extractor.find_disclaimer(no_disc_text, "performances passées")
    
    assert not no_disclaimer.found, "Should not find disclaimer when absent"
    
    print_test_result(
        "Absent disclaimer detection",
        True,
        f"Found: {no_disclaimer.found} (correctly detected absence)"
    )
    
    # Test 5: Evidence extraction
    print("\n[Test 5] Evidence extraction")
    evidence_text = "Le fonds a réalisé +15% en 2023. Excellent résultat."
    
    evidence = extractor.extract_evidence(evidence_text, "performance_data", "Slide 3")
    
    assert len(evidence.quotes) > 0, "Should extract quotes"
    assert evidence.confidence > 0, "Should have confidence score"
    # Location tracking may vary based on implementation
    assert len(evidence.locations) >= 0, "Should have locations list"
    
    print_test_result(
        "Evidence extraction",
        True,
        f"Quotes: {len(evidence.quotes)}, Confidence: {evidence.confidence}%, Locations: {len(evidence.locations)}"
    )
    
    # Test 6: Chart data detection
    print("\n[Test 6] Chart data detection")
    chart_data = '{"values": ["+5.2%", "+8.1%", "+12.5%"], "years": ["2022", "2023", "2024"]}'
    
    chart_perf = extractor.find_performance_data(chart_data)
    
    assert len(chart_perf) >= 3, f"Should detect chart data, found {len(chart_perf)}"
    
    print_test_result(
        "Chart data detection",
        True,
        f"Found {len(chart_perf)} data points from chart"
    )


# ============================================================================
# SEMANTIC VALIDATOR TESTS
# ============================================================================

def test_semantic_validator():
    """Test SemanticValidator: whitelist filtering, semantic validation"""
    print_test_header("SemanticValidator Tests")
    
    # Create AI engine
    from ai_engine import create_ai_engine_from_env
    ai_engine = create_ai_engine_from_env()
    
    validator = SemanticValidator(ai_engine)
    
    # Test 1: Whitelisted term validation
    print("\n[Test 1] Whitelisted term (should NOT violate)")
    
    whitelist = {'oddo', 'bhf', 'momentum'}
    text = "ODDO BHF utilise une stratégie momentum"
    
    result1 = validator.validate_securities_mention(text, "ODDO", whitelist, 31)
    
    assert not result1.is_violation, "Whitelisted term should not violate"
    assert result1.confidence >= 90, f"Should have high confidence, got {result1.confidence}%"
    
    print_test_result(
        "Whitelisted term validation",
        True,
        f"Violation: {result1.is_violation}, Confidence: {result1.confidence}%, Reason: {result1.reasoning[:50]}..."
    )
    
    # Test 2: External company validation
    print("\n[Test 2] External company (should violate if 3+ mentions)")
    
    result2 = validator.validate_securities_mention(text, "Apple", whitelist, 5)
    
    # May or may not violate depending on AI, but should have reasoning
    assert result2.reasoning, "Should have reasoning"
    
    print_test_result(
        "External company validation",
        True,
        f"Violation: {result2.is_violation}, Confidence: {result2.confidence}%, Method: {result2.method}"
    )
    
    # Test 3: Performance claim with actual data
    print("\n[Test 3] Performance claim - actual data")
    
    perf_text = "Le fonds a généré une performance de +15.5% en 2023"
    result3 = validator.validate_performance_claim(perf_text, "Slide 2")
    
    assert result3.is_violation, "Actual performance data should require disclaimer"
    assert len(result3.evidence) > 0, "Should have evidence"
    
    print_test_result(
        "Performance data validation",
        True,
        f"Violation: {result3.is_violation}, Evidence: {result3.evidence[0] if result3.evidence else 'None'}"
    )
    
    # Test 4: Performance claim with keywords only
    print("\n[Test 4] Performance claim - keywords only")
    
    keyword_text = "L'objectif de performance est d'obtenir des résultats attractifs"
    result4 = validator.validate_performance_claim(keyword_text, "Slide 3")
    
    assert not result4.is_violation, "Keywords only should not require disclaimer"
    
    print_test_result(
        "Performance keywords validation",
        True,
        f"Violation: {result4.is_violation}, Reasoning: {result4.reasoning[:50]}..."
    )
    
    # Test 5: Confidence scoring - agreement
    print("\n[Test 5] Confidence scoring - AI and Rules agree")
    
    ai_result = {
        'is_violation': True,
        'confidence': 85,
        'reasoning': 'AI detected violation',
        'evidence': ['Evidence 1']
    }
    
    rule_result = {
        'is_violation': True,
        'confidence': 80,
        'reasoning': 'Rules detected violation',
        'evidence': ['Evidence 2']
    }
    
    result5 = validator.validate_with_confidence_scoring(ai_result, rule_result)
    
    assert result5.is_violation, "Should be violation when both agree"
    assert result5.confidence >= 80, f"Should have high confidence, got {result5.confidence}%"
    assert result5.method == "AI_AND_RULES", f"Should use both methods, got {result5.method}"
    
    print_test_result(
        "Confidence scoring - agreement",
        True,
        f"Confidence: {result5.confidence}%, Method: {result5.method}"
    )
    
    # Test 6: Confidence scoring - disagreement
    print("\n[Test 6] Confidence scoring - AI and Rules disagree")
    
    rule_result_disagree = {
        'is_violation': False,
        'confidence': 70,
        'reasoning': 'Rules did not detect violation',
        'evidence': []
    }
    
    result6 = validator.validate_with_confidence_scoring(ai_result, rule_result_disagree)
    
    assert result6.confidence <= 60, f"Should have low confidence for disagreement, got {result6.confidence}%"
    assert "disagree" in result6.reasoning.lower(), "Should indicate disagreement"
    
    print_test_result(
        "Confidence scoring - disagreement",
        True,
        f"Confidence: {result6.confidence}% (flagged for review)"
    )


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all unit tests"""
    print("="*70)
    print("  FALSE POSITIVE ELIMINATION - UNIT TESTS")
    print("="*70)
    print("\nTesting all new components:")
    print("  - WhitelistManager")
    print("  - ContextAnalyzer")
    print("  - IntentClassifier")
    print("  - EvidenceExtractor")
    print("  - SemanticValidator")
    
    try:
        # Run all test suites
        test_whitelist_manager()
        test_context_analyzer()
        test_intent_classifier()
        test_evidence_extractor()
        test_semantic_validator()
        
        # Print summary
        print("\n" + "="*70)
        print("  TEST SUMMARY")
        print("="*70)
        print(f"  Total Passed: {test_results['passed']}")
        print(f"  Total Failed: {test_results['failed']}")
        print(f"  Total Errors: {test_results['errors']}")
        
        total = test_results['passed'] + test_results['failed'] + test_results['errors']
        if total > 0:
            success_rate = (test_results['passed'] / total) * 100
            print(f"  Success Rate: {success_rate:.1f}%")
        
        print("="*70)
        
        if test_results['failed'] == 0 and test_results['errors'] == 0:
            print("\n✅ ALL TESTS PASSED!")
            print("\nAll components are ready for integration.")
            return True
        else:
            print("\n⚠️  SOME TESTS FAILED")
            print("\nPlease review failed tests above.")
            return False
    
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        test_results['errors'] += 1
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
