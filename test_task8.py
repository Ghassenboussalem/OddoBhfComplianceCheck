#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for Task 8: Data-aware performance disclaimer checking
Tests the new check_performance_disclaimers_ai function
"""

import json
import sys

# Test cases based on task requirements
test_cases = [
    {
        "name": "Test 1: Descriptive text without numbers (should NOT flag)",
        "slide_data": {
            "title": "Investment Strategy",
            "content": "The fund aims for attractive performance through momentum strategies. Performance objective is to outperform the benchmark."
        },
        "expected_violation": False,
        "reason": "No actual performance numbers, just descriptive keywords"
    },
    {
        "name": "Test 2: Actual performance data without disclaimer (should flag)",
        "slide_data": {
            "title": "Performance Results",
            "content": "The fund generated a return of 15% in 2023 and 20% in 2024."
        },
        "expected_violation": True,
        "reason": "Actual performance numbers (15%, 20%) without disclaimer"
    },
    {
        "name": "Test 3: Performance objective without numbers (should NOT flag)",
        "slide_data": {
            "title": "Strategy Goals",
            "content": "Performance objective: achieve consistent returns. The strategy focuses on performance potential."
        },
        "expected_violation": False,
        "reason": "Performance keywords but no actual data"
    },
    {
        "name": "Test 4: Performance data WITH disclaimer (should NOT flag)",
        "slide_data": {
            "title": "Historical Performance",
            "content": "The fund achieved 15% return in 2023. Les performances passées ne préjugent pas des performances futures."
        },
        "expected_violation": False,
        "reason": "Performance data present but disclaimer is also present"
    }
]

def run_tests():
    """Run test cases for the new function"""
    print("="*70)
    print("Testing Task 8: Data-Aware Performance Disclaimer Checking")
    print("="*70)
    
    # Import the function
    try:
        from check_functions_ai import check_performance_disclaimers_ai
        print("✓ Successfully imported check_performance_disclaimers_ai\n")
    except ImportError as e:
        print(f"✗ Failed to import function: {e}")
        return False
    
    # Run each test case
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'-'*70}")
        print(f"Test {i}: {test['name']}")
        print(f"{'-'*70}")
        print(f"Reason: {test['reason']}")
        
        # Create test document
        doc = {
            "pages_suivantes": [test['slide_data']]
        }
        
        # Run the check
        try:
            violations = check_performance_disclaimers_ai(doc)
            has_violation = len(violations) > 0
            
            # Check result
            if has_violation == test['expected_violation']:
                print(f"✓ PASSED")
                print(f"  Expected violation: {test['expected_violation']}")
                print(f"  Got violation: {has_violation}")
                if violations:
                    print(f"  Violation message: {violations[0]['message']}")
                passed += 1
            else:
                print(f"✗ FAILED")
                print(f"  Expected violation: {test['expected_violation']}")
                print(f"  Got violation: {has_violation}")
                if violations:
                    print(f"  Violation details: {violations[0]}")
                failed += 1
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Test Summary")
    print(f"{'='*70}")
    print(f"Passed: {passed}/{len(test_cases)}")
    print(f"Failed: {failed}/{len(test_cases)}")
    
    if failed == 0:
        print(f"\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  Some tests failed")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
