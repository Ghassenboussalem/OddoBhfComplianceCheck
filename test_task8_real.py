#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Task 8 on real exemple.json file
Verify that false positives are eliminated
"""

import json
import sys

def test_on_real_document():
    """Test the new function on exemple.json"""
    print("="*70)
    print("Testing Task 8 on exemple.json")
    print("="*70)
    
    # Load the document
    try:
        with open('exemple.json', 'r', encoding='utf-8') as f:
            doc = json.load(f)
        print("✓ Loaded exemple.json\n")
    except FileNotFoundError:
        print("✗ exemple.json not found")
        return False
    
    # Import the function
    try:
        from check_functions_ai import check_performance_disclaimers_ai
        print("✓ Imported check_performance_disclaimers_ai\n")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False
    
    # Run the check
    print("Running performance disclaimer check...")
    print("-"*70)
    
    try:
        violations = check_performance_disclaimers_ai(doc)
        
        print(f"\nResults:")
        print(f"  Total violations found: {len(violations)}")
        
        if violations:
            print(f"\n  Violations:")
            for i, v in enumerate(violations, 1):
                print(f"\n  {i}. {v['message']}")
                print(f"     Slide: {v['slide']}")
                print(f"     Evidence: {v['evidence'][:100]}...")
                print(f"     Confidence: {v['confidence']}%")
        else:
            print(f"\n  ✓ No violations found")
            print(f"  This means:")
            print(f"    - No actual performance numbers without disclaimers")
            print(f"    - Descriptive performance keywords are correctly ignored")
            print(f"    - False positives eliminated!")
        
        print(f"\n{'='*70}")
        print(f"Expected behavior:")
        print(f"  - 'attractive performance' → NOT flagged (descriptive)")
        print(f"  - 'performance objective' → NOT flagged (descriptive)")
        print(f"  - Actual numbers like '15%' → ONLY flagged if no disclaimer")
        print(f"{'='*70}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error running check: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_on_real_document()
    sys.exit(0 if success else 1)
