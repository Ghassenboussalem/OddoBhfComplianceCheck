#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Task 6: AI Context-Aware Prohibited Phrases Check
Tests the new check_prohibited_phrases_ai function
"""

import json
import sys

# Test cases from task requirements
test_cases = [
    {
        "name": "Fund strategy description (should NOT flag)",
        "text": "Tirer parti du momentum des marchés américains grâce à une stratégie quantitative",
        "expected_violation": False,
        "description": "Fund strategy goal - ALLOWED"
    },
    {
        "name": "Fund investment description (should NOT flag)",
        "text": "Le fonds investit dans des actions américaines à forte capitalisation",
        "expected_violation": False,
        "description": "Fund description - ALLOWED"
    },
    {
        "name": "Client investment advice (should flag)",
        "text": "Vous devriez investir dans ce fonds maintenant pour profiter de cette opportunité",
        "expected_violation": True,
        "description": "Direct client advice - PROHIBITED"
    },
    {
        "name": "Recommendation to client (should flag)",
        "text": "Nous recommandons d'acheter des actions technologiques",
        "expected_violation": True,
        "description": "Recommendation - PROHIBITED"
    }
]

def create_test_doc(text):
    """Create a minimal test document"""
    return {
        'document_metadata': {
            'fund_name': 'Test Fund',
            'client_type': 'retail'
        },
        'page_de_garde': {
            'slide_number': 1,
            'content': text
        }
    }

def create_test_rule():
    """Create a test rule for prohibited phrases"""
    return {
        'rule_id': 'VAL_001',
        'rule_text': 'No investment recommendations',
        'severity': 'critical',
        'category': 'prohibition',
        'prohibited_phrases': [
            'recommend', 'suggest', 'should buy', 'should invest',
            'devriez investir', 'recommandons', 'bon moment'
        ]
    }

def run_tests():
    """Run all test cases"""
    print("="*70)
    print("Task 6: AI Context-Aware Prohibited Phrases Check - Tests")
    print("="*70)
    
    # Load agent
    print("\nLoading agent...")
    try:
        with open('agent.py', 'r', encoding='utf-8') as f:
            exec(f.read(), globals())
        print("[OK] Agent loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load agent: {e}")
        return False
    
    # Check if function exists
    if 'check_prohibited_phrases_ai' not in globals():
        print("[ERROR] check_prohibited_phrases_ai function not found in agent.py")
        return False
    
    print("[OK] check_prohibited_phrases_ai function found")
    
    # Run test cases
    print("\n" + "="*70)
    print("Running Test Cases")
    print("="*70)
    
    passed = 0
    failed = 0
    
    rule = create_test_rule()
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test['name']}")
        print(f"Text: \"{test['text']}\"")
        print(f"Expected: {'VIOLATION' if test['expected_violation'] else 'NO VIOLATION'}")
        
        # Create test document
        doc = create_test_doc(test['text'])
        
        # Run check
        try:
            violations = check_prohibited_phrases_ai(doc, rule)
            
            has_violation = len(violations) > 0
            
            print(f"Result: {'VIOLATION' if has_violation else 'NO VIOLATION'}")
            
            if has_violation:
                print(f"  Violations found: {len(violations)}")
                for v in violations:
                    print(f"    - {v.get('message', 'No message')}")
            
            # Check if result matches expected
            if has_violation == test['expected_violation']:
                print(f"  [PASS]")
                passed += 1
            else:
                print(f"  [FAIL] - Expected {'violation' if test['expected_violation'] else 'no violation'}, got {'violation' if has_violation else 'no violation'}")
                failed += 1
                
        except Exception as e:
            print(f"  [ERROR]: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(test_cases)*100):.1f}%")
    print("="*70)
    
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
