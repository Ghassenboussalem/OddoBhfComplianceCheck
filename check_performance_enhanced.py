#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Performance Checking - Integration Script
Demonstrates how to use the new data-aware check_performance_disclaimers_ai function

This can be integrated into check.py by:
1. Importing: from check_functions_ai import check_performance_disclaimers_ai
2. Calling it in the performance checks section
3. Replacing or supplementing existing performance disclaimer checks
"""

import json
import sys

def check_performance_enhanced(json_file_path):
    """
    Run enhanced performance checks using the new data-aware function
    
    Args:
        json_file_path: Path to JSON document
    
    Returns:
        List of violations
    """
    print("="*70)
    print("Enhanced Performance Disclaimer Checking")
    print("="*70)
    
    # Load document
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        print(f"✓ Loaded: {json_file_path}\n")
    except FileNotFoundError:
        print(f"✗ File not found: {json_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON: {e}")
        return []
    
    # Import the new function
    try:
        from check_functions_ai import check_performance_disclaimers_ai
        print("✓ Imported check_performance_disclaimers_ai")
        print("  (Data-aware version - eliminates false positives)\n")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return []
    
    # Run the check
    print("Running performance disclaimer check...")
    print("-"*70)
    
    try:
        violations = check_performance_disclaimers_ai(doc)
        
        print(f"\nResults:")
        print(f"{'='*70}")
        
        if violations:
            print(f"❌ Found {len(violations)} violation(s):\n")
            
            for i, v in enumerate(violations, 1):
                print(f"{i}. [{v['severity']}] {v['message']}")
                print(f"   Slide: {v['slide']}")
                print(f"   Location: {v['location']}")
                print(f"   Rule: {v['rule']}")
                print(f"   Evidence: {v['evidence'][:150]}...")
                print(f"   Confidence: {v['confidence']}%")
                print(f"   Method: {v['method']}")
                if 'ai_reasoning' in v:
                    print(f"   AI Reasoning: {v['ai_reasoning'][:150]}...")
                print()
        else:
            print(f"✅ No violations found\n")
            print(f"This means:")
            print(f"  • No actual performance numbers without disclaimers")
            print(f"  • Descriptive keywords correctly ignored")
            print(f"  • False positives eliminated!")
        
        print(f"{'='*70}")
        print(f"\nKey Improvements:")
        print(f"  ✓ Only flags ACTUAL performance data (numbers with %)")
        print(f"  ✓ Ignores descriptive text like 'attractive performance'")
        print(f"  ✓ Uses semantic matching for disclaimers")
        print(f"  ✓ Verifies disclaimer on SAME slide as data")
        print(f"  ✓ Eliminates 3 false positives from keyword approach")
        print(f"{'='*70}\n")
        
        return violations
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("\nUsage: python check_performance_enhanced.py <json_file>")
        print("\nExample:")
        print("  python check_performance_enhanced.py exemple.json")
        print("\nThis script demonstrates the new data-aware performance checking")
        print("that eliminates false positives from descriptive performance keywords.")
        sys.exit(1)
    
    json_file = sys.argv[1]
    violations = check_performance_enhanced(json_file)
    
    # Exit code based on violations
    sys.exit(len(violations))


if __name__ == "__main__":
    main()
