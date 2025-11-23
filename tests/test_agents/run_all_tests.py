#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Runner for All Agent Tests

This script runs all agent unit tests and generates a summary report.
"""

import sys
import os
import unittest
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def discover_and_run_tests():
    """Discover and run all agent tests"""
    
    print("="*80)
    print("MULTI-AGENT SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test files to run (in order)
    test_files = [
        'test_base_agent.py',
        'test_supervisor_agent.py',
        'test_preprocessor_agent.py',
        'test_structure_agent.py',
        'test_performance_agent.py',
        'test_general_agent.py',
        'test_aggregator_agent.py',
        'test_context_agent.py',
        'test_evidence_agent.py',
        'test_reviewer_agent.py',
        'test_feedback_agent.py',
        'test_prospectus_agent.py',
        'test_registration_agent.py',
    ]
    
    results = {}
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    
    for test_file in test_files:
        test_path = project_root / test_file
        
        if not test_path.exists():
            print(f"⚠ Skipping {test_file} (not found)")
            continue
        
        print(f"\n{'='*80}")
        print(f"Running: {test_file}")
        print(f"{'='*80}\n")
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        try:
            # Load tests from file
            module_name = test_file[:-3]  # Remove .py
            spec = __import__(module_name)
            
            # Add all tests from module
            suite.addTests(loader.loadTestsFromModule(spec))
            
            # Run tests
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            # Store results
            results[test_file] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped),
                'success': result.wasSuccessful()
            }
            
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            total_skipped += len(result.skipped)
            
        except Exception as e:
            print(f"✗ Error running {test_file}: {e}")
            results[test_file] = {
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0,
                'success': False,
                'error_message': str(e)
            }
            total_errors += 1
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print()
    
    print(f"{'Test File':<40} {'Tests':<8} {'Pass':<8} {'Fail':<8} {'Error':<8} {'Skip':<8}")
    print("-"*80)
    
    for test_file, result in results.items():
        status = "✓" if result['success'] else "✗"
        tests = result['tests_run']
        passed = tests - result['failures'] - result['errors']
        failures = result['failures']
        errors = result['errors']
        skipped = result['skipped']
        
        print(f"{status} {test_file:<38} {tests:<8} {passed:<8} {failures:<8} {errors:<8} {skipped:<8}")
    
    print("-"*80)
    total_passed = total_tests - total_failures - total_errors
    print(f"{'TOTAL':<40} {total_tests:<8} {total_passed:<8} {total_failures:<8} {total_errors:<8} {total_skipped:<8}")
    print()
    
    # Overall result
    all_passed = total_failures == 0 and total_errors == 0
    
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print(f"  {total_tests} tests executed successfully")
    else:
        print("✗ SOME TESTS FAILED")
        print(f"  {total_passed}/{total_tests} tests passed")
        print(f"  {total_failures} failures, {total_errors} errors")
    
    print()
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return 0 if all_passed else 1


def run_specific_agent_test(agent_name):
    """Run tests for a specific agent"""
    test_file = f"test_{agent_name}_agent.py"
    test_path = project_root / test_file
    
    if not test_path.exists():
        print(f"✗ Test file not found: {test_file}")
        return 1
    
    print(f"Running tests for {agent_name} agent...")
    print()
    
    loader = unittest.TestLoader()
    suite = loader.discover(str(project_root), pattern=test_file)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Run specific agent test
        agent_name = sys.argv[1]
        return run_specific_agent_test(agent_name)
    else:
        # Run all tests
        return discover_and_run_tests()


if __name__ == "__main__":
    sys.exit(main())
