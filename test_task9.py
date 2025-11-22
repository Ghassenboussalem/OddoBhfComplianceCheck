#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Task 9: check_document_starts_with_performance_ai
Verify that the function only flags ACTUAL performance data on cover page
"""

import json
import sys

def test_check_document_starts_with_performance():
    """Test the new check_document_starts_with_performance_ai function"""
    
    print("="*70)
    print("Testing Task 9: check_document_starts_with_performance_ai")
    print("="*70)
    
    # Load exemple.json
    print("\n📄 Loading exemple.json...")
    try:
        with open('exemple.json', 'r', encoding='utf-8') as f:
            doc = json.load(f)
        print("✅ Document loaded successfully")
    except Exception as e:
        print(f"❌ Error loading document: {e}")
        return False
    
    # Import the function
    print("\n🔧 Importing check_document_starts_with_performance_ai...")
    try:
        from check_functions_ai import check_document_starts_with_performance_ai
        print("✅ Function imported successfully")
    except Exception as e:
        print(f"❌ Error importing function: {e}")
        return False
    
    # Check cover page content
    print("\n📋 Cover page content:")
    if 'page_de_garde' in doc:
        cover = doc['page_de_garde']
        print(f"  Title: {cover.get('title', 'N/A')}")
        print(f"  Subtitle: {cover.get('subtitle', 'N/A')}")
        text_preview = cover.get('text_content', '')[:200]
        print(f"  Text preview: {text_preview}...")
        
        # Check for actual performance numbers
        import re
        perf_patterns = [
            r'[+\-]?\d+[.,]\d+\s*%',  # +15.5%, -3.2%
            r'[+\-]?\d+\s*%',  # +15%, -3%
        ]
        has_numbers = any(re.search(pattern, cover.get('text_content', '')) for pattern in perf_patterns)
        print(f"  Contains performance numbers: {has_numbers}")
    else:
        print("  ❌ No cover page found")
        return False
    
    # Run the check
    print("\n🔍 Running check_document_starts_with_performance_ai...")
    try:
        violations = check_document_starts_with_performance_ai(doc)
        print(f"✅ Check completed")
        print(f"  Violations found: {len(violations)}")
    except Exception as e:
        print(f"❌ Error running check: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Analyze results
    print("\n📊 Results Analysis:")
    if len(violations) == 0:
        print("  ✅ PASS: No violations found (expected)")
        print("  ✅ Cover page has NO actual performance data")
        print("  ✅ Descriptive text like 'momentum' and 'potentiel' correctly ignored")
        success = True
    else:
        print("  ❌ FAIL: Violations found (unexpected)")
        print("  ❌ Cover page should NOT be flagged")
        print("\n  Violation details:")
        for i, v in enumerate(violations, 1):
            print(f"\n  Violation {i}:")
            print(f"    Rule: {v.get('rule', 'N/A')}")
            print(f"    Message: {v.get('message', 'N/A')}")
            print(f"    Evidence: {v.get('evidence', 'N/A')}")
            print(f"    Confidence: {v.get('confidence', 'N/A')}")
            print(f"    Method: {v.get('method', 'N/A')}")
        success = False
    
    # Test with a document that SHOULD be flagged
    print("\n" + "="*70)
    print("Testing with document that SHOULD be flagged")
    print("="*70)
    
    # Create a test document with actual performance data on cover
    test_doc = {
        'page_de_garde': {
            'title': 'Test Fund',
            'text_content': 'Le fonds a généré une performance de +15.5% en 2024. Rendement annualisé de 8.2% sur 5 ans.'
        }
    }
    
    print("\n📋 Test cover page content:")
    print(f"  Text: {test_doc['page_de_garde']['text_content']}")
    
    print("\n🔍 Running check on test document...")
    try:
        test_violations = check_document_starts_with_performance_ai(test_doc)
        print(f"✅ Check completed")
        print(f"  Violations found: {len(test_violations)}")
    except Exception as e:
        print(f"❌ Error running check: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n📊 Test Results Analysis:")
    if len(test_violations) > 0:
        print("  ✅ PASS: Violation correctly detected")
        print("  ✅ Actual performance data (+15.5%, 8.2%) correctly flagged")
        print("\n  Violation details:")
        v = test_violations[0]
        print(f"    Rule: {v.get('rule', 'N/A')}")
        print(f"    Message: {v.get('message', 'N/A')}")
        print(f"    Confidence: {v.get('confidence', 'N/A')}")
        print(f"    Method: {v.get('method', 'N/A')}")
    else:
        print("  ❌ FAIL: No violation found (should have been flagged)")
        print("  ❌ Actual performance data should be detected")
        success = False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if success:
        print("✅ Task 9 implementation SUCCESSFUL")
        print("✅ Function correctly distinguishes:")
        print("   - Descriptive text (NOT flagged)")
        print("   - Actual performance data (FLAGGED)")
        print("✅ False positives eliminated")
    else:
        print("❌ Task 9 implementation needs adjustment")
    
    return success


if __name__ == "__main__":
    success = test_check_document_starts_with_performance()
    sys.exit(0 if success else 1)
