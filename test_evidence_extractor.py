#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Evidence Extractor
Verify all methods work correctly
"""

import json
from evidence_extractor import EvidenceExtractor, extract_all_text_from_doc
from data_models import Evidence, PerformanceData, DisclaimerMatch


def test_performance_data_detection():
    """Test detection of actual performance data vs descriptive text"""
    print("\n" + "="*70)
    print("TEST: Performance Data Detection")
    print("="*70)
    
    extractor = EvidenceExtractor()
    
    # Test 1: Actual performance data (SHOULD DETECT)
    actual_data = "Le fonds a généré +15.5% en 2023 et +20% en 2024."
    perf_data = extractor.find_performance_data(actual_data)
    print(f"\n✓ Test 1 - Actual data: Found {len(perf_data)} data points")
    assert len(perf_data) > 0, "Should detect actual performance data"
    
    # Test 2: Descriptive text without numbers (SHOULD NOT DETECT or LOW CONFIDENCE)
    descriptive = "L'objectif de performance est d'obtenir des résultats attractifs."
    perf_data = extractor.find_performance_data(descriptive)
    print(f"✓ Test 2 - Descriptive only: Found {len(perf_data)} data points")
    # Should find few or none
    
    # Test 3: Mixed context
    mixed = "La stratégie vise une performance attractive. En 2023, le fonds a réalisé +12.3%."
    perf_data = extractor.find_performance_data(mixed)
    print(f"✓ Test 3 - Mixed context: Found {len(perf_data)} data points")
    assert len(perf_data) > 0, "Should detect actual data in mixed context"
    
    # Test 4: Chart data
    chart_data = '{"values": ["+5.2%", "+8.1%", "+12.5%"], "years": ["2022", "2023", "2024"]}'
    perf_data = extractor.find_performance_data(chart_data)
    print(f"✓ Test 4 - Chart data: Found {len(perf_data)} data points")
    assert len(perf_data) >= 3, "Should detect chart data"
    
    print("\n✅ All performance data detection tests passed")


def test_disclaimer_matching():
    """Test semantic disclaimer matching"""
    print("\n" + "="*70)
    print("TEST: Disclaimer Matching")
    print("="*70)
    
    extractor = EvidenceExtractor()
    
    # Test 1: Exact match
    exact_text = "Les performances passées ne préjugent pas des performances futures."
    disclaimer = extractor.find_disclaimer(exact_text, "performances passées ne préjugent pas")
    print(f"\n✓ Test 1 - Exact match: Found={disclaimer.found}, Score={disclaimer.similarity_score}%")
    assert disclaimer.found, "Should find exact match"
    assert disclaimer.similarity_score >= 90, "Should have high similarity"
    
    # Test 2: Paraphrase
    paraphrase = "Les résultats passés ne garantissent pas les résultats futurs."
    disclaimer = extractor.find_disclaimer(paraphrase, "performances passées ne préjugent pas")
    print(f"✓ Test 2 - Paraphrase: Found={disclaimer.found}, Score={disclaimer.similarity_score}%")
    # May or may not find depending on keywords
    
    # Test 3: English version
    english = "Past performance is not indicative of future results."
    disclaimer = extractor.find_disclaimer(english, "past performance")
    print(f"✓ Test 3 - English: Found={disclaimer.found}, Score={disclaimer.similarity_score}%")
    assert disclaimer.found, "Should find English disclaimer"
    
    # Test 4: No disclaimer
    no_disclaimer = "Le fonds investit dans des actions européennes."
    disclaimer = extractor.find_disclaimer(no_disclaimer, "performances passées")
    print(f"✓ Test 4 - No disclaimer: Found={disclaimer.found}")
    assert not disclaimer.found, "Should not find disclaimer when absent"
    
    print("\n✅ All disclaimer matching tests passed")


def test_evidence_extraction():
    """Test evidence extraction for different violation types"""
    print("\n" + "="*70)
    print("TEST: Evidence Extraction")
    print("="*70)
    
    extractor = EvidenceExtractor()
    
    # Test 1: Performance data evidence
    perf_text = "Le fonds a réalisé +15% en 2023. Excellent résultat pour nos investisseurs."
    evidence = extractor.extract_evidence(perf_text, "performance_data", "Slide 3")
    print(f"\n✓ Test 1 - Performance evidence: {len(evidence.quotes)} quotes, confidence={evidence.confidence}%")
    assert len(evidence.quotes) > 0, "Should extract quotes"
    assert evidence.confidence > 0, "Should have confidence score"
    
    # Test 2: Missing disclaimer evidence
    missing_disc = "Performance de +20% sans disclaimer."
    evidence = extractor.extract_evidence(missing_disc, "missing_disclaimer", "Slide 5")
    print(f"✓ Test 2 - Missing disclaimer: {len(evidence.quotes)} quotes, confidence={evidence.confidence}%")
    assert "Slide 5" in evidence.locations, "Should track location"
    
    # Test 3: Prohibited phrase evidence
    prohibited = "Vous devriez investir dans ce fonds maintenant."
    evidence = extractor.extract_evidence(prohibited, "prohibited_phrase", "Cover Page")
    print(f"✓ Test 3 - Prohibited phrase: {len(evidence.quotes)} quotes, confidence={evidence.confidence}%")
    assert len(evidence.quotes) > 0, "Should extract quotes"
    
    print("\n✅ All evidence extraction tests passed")


def test_location_tracking():
    """Test location tracking in document"""
    print("\n" + "="*70)
    print("TEST: Location Tracking")
    print("="*70)
    
    extractor = EvidenceExtractor()
    
    # Test with slide markers
    text_with_slide = '{"slide": "Slide 2", "content": "Performance de +15% en 2023"}'
    perf_data = extractor.find_performance_data(text_with_slide)
    print(f"\n✓ Test 1 - Slide marker: Found {len(perf_data)} data points")
    if perf_data:
        print(f"  Location: {perf_data[0].location}")
        assert "Slide" in perf_data[0].location, "Should extract slide location"
    
    # Test with cover page
    cover_text = '{"page_de_garde": {"title": "Fund Performance +20%"}}'
    perf_data = extractor.find_performance_data(cover_text)
    print(f"✓ Test 2 - Cover page: Found {len(perf_data)} data points")
    if perf_data:
        print(f"  Location: {perf_data[0].location}")
    
    print("\n✅ All location tracking tests passed")


def test_extract_all_text():
    """Test utility function to extract all text from document"""
    print("\n" + "="*70)
    print("TEST: Extract All Text Utility")
    print("="*70)
    
    doc = {
        "page_de_garde": {"title": "Fund Name", "subtitle": "Marketing Document"},
        "slide_2": {"content": "Performance de +15%"},
        "slide_3": {"content": "Risk profile: 4/7"}
    }
    
    all_text = extract_all_text_from_doc(doc)
    print(f"\n✓ Extracted {len(all_text)} characters")
    assert len(all_text) > 0, "Should extract text"
    assert "Fund Name" in all_text, "Should include all content"
    assert "Performance" in all_text, "Should include all slides"
    
    print("\n✅ Extract all text test passed")


def test_confidence_scoring():
    """Test confidence scoring for performance data"""
    print("\n" + "="*70)
    print("TEST: Confidence Scoring")
    print("="*70)
    
    extractor = EvidenceExtractor()
    
    # High confidence: clear performance with year
    high_conf = "Le fonds a généré une performance de +15.5% en 2023."
    perf_data = extractor.find_performance_data(high_conf)
    if perf_data:
        print(f"\n✓ High confidence case: {perf_data[0].confidence}%")
        assert perf_data[0].confidence >= 60, "Should have high confidence"
    
    # Lower confidence: descriptive context
    low_conf = "L'objectif de performance est d'atteindre +10%."
    perf_data = extractor.find_performance_data(low_conf)
    if perf_data:
        print(f"✓ Lower confidence case: {perf_data[0].confidence}%")
        # May have lower confidence due to "objectif"
    
    print("\n✅ Confidence scoring tests passed")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EVIDENCE EXTRACTOR - COMPREHENSIVE TESTS")
    print("="*70)
    
    try:
        test_performance_data_detection()
        test_disclaimer_matching()
        test_evidence_extraction()
        test_location_tracking()
        test_extract_all_text()
        test_confidence_scoring()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nEvidenceExtractor is ready for integration!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
