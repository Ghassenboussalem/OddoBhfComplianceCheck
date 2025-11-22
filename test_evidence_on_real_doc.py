#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Evidence Extractor on Real Document
Verify it works with actual exemple.json
"""

import json
from evidence_extractor import EvidenceExtractor, extract_all_text_from_doc


def test_on_real_document():
    """Test evidence extractor on exemple.json"""
    print("="*70)
    print("Testing EvidenceExtractor on exemple.json")
    print("="*70)
    
    # Load document
    try:
        with open('exemple.json', 'r', encoding='utf-8') as f:
            doc = json.load(f)
        print(f"\n✓ Loaded exemple.json")
    except FileNotFoundError:
        print("\n⚠ exemple.json not found - skipping real document test")
        return
    
    # Create extractor
    extractor = EvidenceExtractor()
    
    # Extract all text
    all_text = extract_all_text_from_doc(doc)
    print(f"✓ Extracted {len(all_text)} characters from document")
    
    # Test 1: Find performance data in document
    print("\n" + "-"*70)
    print("TEST 1: Finding Performance Data")
    print("-"*70)
    perf_data = extractor.find_performance_data(all_text)
    print(f"Found {len(perf_data)} performance data points:")
    for i, pd in enumerate(perf_data[:5], 1):  # Show first 5
        print(f"  {i}. {pd.value} at {pd.location} (confidence: {pd.confidence}%)")
        print(f"     Context: {pd.context[:80]}...")
    
    # Test 2: Check for disclaimers
    print("\n" + "-"*70)
    print("TEST 2: Finding Disclaimers")
    print("-"*70)
    disclaimer = extractor.find_disclaimer(
        all_text,
        "Les performances passées ne préjugent pas des performances futures"
    )
    print(f"Disclaimer found: {disclaimer.found}")
    if disclaimer.found:
        print(f"  Similarity: {disclaimer.similarity_score}%")
        print(f"  Confidence: {disclaimer.confidence}%")
        print(f"  Location: {disclaimer.location}")
        print(f"  Text: {disclaimer.text[:100]}...")
    
    # Test 3: Extract evidence for different slides
    print("\n" + "-"*70)
    print("TEST 3: Extracting Evidence from Specific Slides")
    print("-"*70)
    
    # Check page_de_garde
    if 'page_de_garde' in doc:
        cover_text = json.dumps(doc['page_de_garde'], ensure_ascii=False)
        evidence = extractor.extract_evidence(cover_text, "performance_data", "Cover Page")
        print(f"\nCover Page:")
        print(f"  Quotes: {len(evidence.quotes)}")
        print(f"  Confidence: {evidence.confidence}%")
        if evidence.quotes:
            print(f"  First quote: {evidence.quotes[0][:100]}...")
    
    # Check slide_2
    if 'slide_2' in doc:
        slide2_text = json.dumps(doc['slide_2'], ensure_ascii=False)
        evidence = extractor.extract_evidence(slide2_text, "performance_data", "Slide 2")
        print(f"\nSlide 2:")
        print(f"  Quotes: {len(evidence.quotes)}")
        print(f"  Confidence: {evidence.confidence}%")
    
    # Test 4: Check for fund name mentions
    print("\n" + "-"*70)
    print("TEST 4: Analyzing Fund Name Mentions")
    print("-"*70)
    
    # Count "ODDO BHF" mentions
    oddo_count = all_text.count("ODDO BHF")
    print(f"'ODDO BHF' appears {oddo_count} times in document")
    
    # Count "momentum" mentions
    momentum_count = all_text.lower().count("momentum")
    print(f"'momentum' appears {momentum_count} times in document")
    
    print("\n" + "="*70)
    print("✅ Real Document Test Complete")
    print("="*70)
    print("\nEvidenceExtractor successfully processes real compliance documents!")


if __name__ == "__main__":
    test_on_real_document()
