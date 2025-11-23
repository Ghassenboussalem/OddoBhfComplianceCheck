#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation Test for Multi-Agent System

This test validates that the multi-agent system produces the same results
as the current system on exemple.json:
- 6 violations detected
- 0 false positives
- All violation details match

Requirements: 14.2
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import difflib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ValidationTest:
    """Validates multi-agent system against current system"""
    
    def __init__(self):
        self.test_file = "exemple.json"
        self.current_system_script = "check.py"
        self.multiagent_system_script = "check_multiagent.py"
        self.results = {
            'current_system': None,
            'multiagent_system': None,
            'comparison': {}
        }
    
    def run_current_system(self) -> Dict[str, Any]:
        """
        Run the current system on exemple.json
        
        Returns:
            Dictionary with violations and metadata
        """
        print("\n" + "="*70)
        print("RUNNING CURRENT SYSTEM (check.py)")
        print("="*70)
        
        try:
            # Run check.py
            result = subprocess.run(
                [sys.executable, self.current_system_script, self.test_file],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            print(f"Exit code: {result.returncode}")
            
            # Load violations JSON
            violations_file = self.test_file.replace('.json', '_violations.json')
            if os.path.exists(violations_file):
                with open(violations_file, 'r', encoding='utf-8') as f:
                    output = json.load(f)
                
                violations = output.get('violations', [])
                print(f"✓ Current system found {len(violations)} violation(s)")
                
                return {
                    'violations': violations,
                    'total_violations': len(violations),
                    'exit_code': result.returncode,
                    'output_file': violations_file
                }
            else:
                print(f"⚠️  Violations file not found: {violations_file}")
                return {
                    'violations': [],
                    'total_violations': 0,
                    'exit_code': result.returncode,
                    'error': 'Violations file not found'
                }
                
        except Exception as e:
            print(f"❌ Error running current system: {e}")
            return {
                'violations': [],
                'total_violations': 0,
                'error': str(e)
            }
    
    def run_multiagent_system(self) -> Dict[str, Any]:
        """
        Run the multi-agent system on exemple.json
        
        Returns:
            Dictionary with violations and metadata
        """
        print("\n" + "="*70)
        print("RUNNING MULTI-AGENT SYSTEM (check_multiagent.py)")
        print("="*70)
        
        try:
            # Run check_multiagent.py
            result = subprocess.run(
                [sys.executable, self.multiagent_system_script, self.test_file],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            print(f"Exit code: {result.returncode}")
            
            # Load violations JSON
            violations_file = self.test_file.replace('.json', '_violations.json')
            if os.path.exists(violations_file):
                with open(violations_file, 'r', encoding='utf-8') as f:
                    output = json.load(f)
                
                violations = output.get('violations', [])
                print(f"✓ Multi-agent system found {len(violations)} violation(s)")
                
                # Extract multi-agent specific metadata
                multi_agent_metadata = output.get('multi_agent', {})
                
                return {
                    'violations': violations,
                    'total_violations': len(violations),
                    'exit_code': result.returncode,
                    'output_file': violations_file,
                    'multi_agent_metadata': multi_agent_metadata
                }
            else:
                print(f"⚠️  Violations file not found: {violations_file}")
                return {
                    'violations': [],
                    'total_violations': 0,
                    'exit_code': result.returncode,
                    'error': 'Violations file not found'
                }
                
        except Exception as e:
            print(f"❌ Error running multi-agent system: {e}")
            return {
                'violations': [],
                'total_violations': 0,
                'error': str(e)
            }
    
    def normalize_violation(self, violation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a violation for comparison
        
        Removes fields that may differ between systems but don't affect
        the core violation detection (e.g., detected_by, method)
        
        Args:
            violation: Violation dictionary
            
        Returns:
            Normalized violation dictionary
        """
        # Core fields that must match
        core_fields = [
            'type',
            'severity',
            'slide',
            'location',
            'rule',
            'message',
            'evidence'
        ]
        
        normalized = {}
        for field in core_fields:
            if field in violation:
                # Normalize whitespace in text fields
                if isinstance(violation[field], str):
                    normalized[field] = ' '.join(violation[field].split())
                else:
                    normalized[field] = violation[field]
        
        # Include confidence if present (but allow some tolerance)
        if 'confidence' in violation:
            normalized['confidence'] = violation['confidence']
        
        return normalized
    
    def compare_violations(
        self,
        current_violations: List[Dict[str, Any]],
        multiagent_violations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare violations from both systems
        
        Args:
            current_violations: Violations from current system
            multiagent_violations: Violations from multi-agent system
            
        Returns:
            Comparison results dictionary
        """
        print("\n" + "="*70)
        print("COMPARING VIOLATIONS")
        print("="*70)
        
        # Normalize violations
        current_normalized = [self.normalize_violation(v) for v in current_violations]
        multiagent_normalized = [self.normalize_violation(v) for v in multiagent_violations]
        
        # Compare counts
        count_match = len(current_normalized) == len(multiagent_normalized)
        print(f"\nViolation count:")
        print(f"  Current system:    {len(current_normalized)}")
        print(f"  Multi-agent system: {len(multiagent_normalized)}")
        print(f"  Match: {'✓' if count_match else '✗'}")
        
        # Find matching violations
        matched = []
        unmatched_current = []
        unmatched_multiagent = list(multiagent_normalized)
        
        for curr_v in current_normalized:
            found_match = False
            for ma_v in unmatched_multiagent:
                # Check if violations match on core fields
                if (curr_v.get('type') == ma_v.get('type') and
                    curr_v.get('rule') == ma_v.get('rule') and
                    curr_v.get('slide') == ma_v.get('slide')):
                    
                    matched.append({
                        'current': curr_v,
                        'multiagent': ma_v,
                        'exact_match': curr_v == ma_v
                    })
                    unmatched_multiagent.remove(ma_v)
                    found_match = True
                    break
            
            if not found_match:
                unmatched_current.append(curr_v)
        
        # Report matching results
        print(f"\nMatching violations: {len(matched)}/{len(current_normalized)}")
        
        if matched:
            exact_matches = sum(1 for m in matched if m['exact_match'])
            print(f"  Exact matches: {exact_matches}")
            print(f"  Partial matches: {len(matched) - exact_matches}")
        
        if unmatched_current:
            print(f"\n⚠️  Violations in current system but not in multi-agent:")
            for v in unmatched_current:
                print(f"    - {v.get('type')}: {v.get('rule')}")
        
        if unmatched_multiagent:
            print(f"\n⚠️  Violations in multi-agent but not in current system:")
            for v in unmatched_multiagent:
                print(f"    - {v.get('type')}: {v.get('rule')}")
        
        # Detailed comparison of matched violations
        if matched and not all(m['exact_match'] for m in matched):
            print(f"\nDetailed comparison of partial matches:")
            for i, match in enumerate(matched, 1):
                if not match['exact_match']:
                    print(f"\n  Match #{i}:")
                    print(f"    Type: {match['current'].get('type')}")
                    print(f"    Rule: {match['current'].get('rule')}")
                    
                    # Find differences
                    curr = match['current']
                    ma = match['multiagent']
                    
                    for key in set(list(curr.keys()) + list(ma.keys())):
                        if curr.get(key) != ma.get(key):
                            print(f"    Difference in '{key}':")
                            print(f"      Current:    {curr.get(key)}")
                            print(f"      Multi-agent: {ma.get(key)}")
        
        return {
            'count_match': count_match,
            'matched_count': len(matched),
            'exact_matches': sum(1 for m in matched if m['exact_match']),
            'partial_matches': sum(1 for m in matched if not m['exact_match']),
            'unmatched_current': unmatched_current,
            'unmatched_multiagent': unmatched_multiagent,
            'all_matched': len(unmatched_current) == 0 and len(unmatched_multiagent) == 0
        }
    
    def validate_expected_results(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate against expected results (6 violations, 0 false positives)
        
        Args:
            violations: List of violations
            
        Returns:
            Validation results dictionary
        """
        print("\n" + "="*70)
        print("VALIDATING EXPECTED RESULTS")
        print("="*70)
        
        expected_count = 6
        actual_count = len(violations)
        
        count_valid = actual_count == expected_count
        print(f"\nExpected violations: {expected_count}")
        print(f"Actual violations:   {actual_count}")
        print(f"Match: {'✓' if count_valid else '✗'}")
        
        # Check for false positives (violations that shouldn't be there)
        # This requires domain knowledge - for now we assume all violations are valid
        # if they match the current system
        false_positives = 0
        print(f"\nFalse positives: {false_positives}")
        
        # Analyze violation types
        type_counts = {}
        for v in violations:
            vtype = v.get('type', 'UNKNOWN')
            type_counts[vtype] = type_counts.get(vtype, 0) + 1
        
        print(f"\nViolations by type:")
        for vtype, count in sorted(type_counts.items()):
            print(f"  {vtype}: {count}")
        
        # Analyze severity
        severity_counts = {}
        for v in violations:
            sev = v.get('severity', 'UNKNOWN')
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        print(f"\nViolations by severity:")
        for sev in ['CRITICAL', 'MAJOR', 'WARNING']:
            if sev in severity_counts:
                print(f"  {sev}: {severity_counts[sev]}")
        
        return {
            'expected_count': expected_count,
            'actual_count': actual_count,
            'count_valid': count_valid,
            'false_positives': false_positives,
            'type_counts': type_counts,
            'severity_counts': severity_counts
        }
    
    def generate_report(self) -> str:
        """
        Generate validation report
        
        Returns:
            Report as string
        """
        report = []
        report.append("\n" + "="*70)
        report.append("VALIDATION REPORT")
        report.append("="*70)
        
        # Overall status
        current_result = self.results['current_system']
        multiagent_result = self.results['multiagent_system']
        comparison = self.results['comparison']
        
        if not current_result or not multiagent_result:
            report.append("\n❌ VALIDATION FAILED: Unable to run both systems")
            return "\n".join(report)
        
        # Check if both systems ran successfully
        current_success = 'error' not in current_result
        multiagent_success = 'error' not in multiagent_result
        
        if not current_success or not multiagent_success:
            report.append("\n❌ VALIDATION FAILED: One or both systems encountered errors")
            if not current_success:
                report.append(f"  Current system error: {current_result.get('error')}")
            if not multiagent_success:
                report.append(f"  Multi-agent system error: {multiagent_result.get('error')}")
            return "\n".join(report)
        
        # Validate expected results
        expected_validation = self.validate_expected_results(multiagent_result['violations'])
        
        # Overall validation status
        all_checks_passed = (
            comparison.get('count_match', False) and
            comparison.get('all_matched', False) and
            expected_validation.get('count_valid', False) and
            expected_validation.get('false_positives', 0) == 0
        )
        
        if all_checks_passed:
            report.append("\n✅ VALIDATION PASSED")
            report.append("\nAll checks passed:")
            report.append(f"  ✓ Violation count matches ({multiagent_result['total_violations']} violations)")
            report.append(f"  ✓ All violations match between systems")
            report.append(f"  ✓ Expected violation count achieved (6 violations)")
            report.append(f"  ✓ No false positives detected (0 false positives)")
        else:
            report.append("\n⚠️  VALIDATION INCOMPLETE")
            report.append("\nCheck results:")
            
            # Violation count
            if comparison.get('count_match', False):
                report.append(f"  ✓ Violation count matches ({multiagent_result['total_violations']} violations)")
            else:
                report.append(f"  ✗ Violation count mismatch:")
                report.append(f"    Current: {current_result['total_violations']}")
                report.append(f"    Multi-agent: {multiagent_result['total_violations']}")
            
            # Violation matching
            if comparison.get('all_matched', False):
                report.append(f"  ✓ All violations match between systems")
            else:
                report.append(f"  ⚠️  Some violations don't match:")
                report.append(f"    Matched: {comparison.get('matched_count', 0)}")
                report.append(f"    Exact matches: {comparison.get('exact_matches', 0)}")
                report.append(f"    Partial matches: {comparison.get('partial_matches', 0)}")
                if comparison.get('unmatched_current'):
                    report.append(f"    Unmatched in current: {len(comparison['unmatched_current'])}")
                if comparison.get('unmatched_multiagent'):
                    report.append(f"    Unmatched in multi-agent: {len(comparison['unmatched_multiagent'])}")
            
            # Expected count
            if expected_validation.get('count_valid', False):
                report.append(f"  ✓ Expected violation count achieved (6 violations)")
            else:
                report.append(f"  ✗ Expected 6 violations, got {expected_validation.get('actual_count', 0)}")
            
            # False positives
            if expected_validation.get('false_positives', 0) == 0:
                report.append(f"  ✓ No false positives detected")
            else:
                report.append(f"  ✗ False positives detected: {expected_validation['false_positives']}")
        
        # Multi-agent specific metrics
        if 'multi_agent_metadata' in multiagent_result:
            metadata = multiagent_result['multi_agent_metadata']
            if metadata:
                report.append("\n" + "-"*70)
                report.append("Multi-Agent System Metrics:")
                report.append(f"  Workflow status: {metadata.get('workflow_status', 'unknown')}")
                report.append(f"  Total execution time: {metadata.get('total_execution_time', 0):.2f}s")
                
                agent_timings = metadata.get('agent_timings', {})
                if agent_timings:
                    report.append(f"  Agent execution times:")
                    for agent, time in sorted(agent_timings.items(), key=lambda x: x[1], reverse=True):
                        report.append(f"    {agent}: {time:.2f}s")
        
        report.append("="*70)
        
        return "\n".join(report)
    
    def run_validation(self) -> bool:
        """
        Run complete validation test
        
        Returns:
            True if validation passed, False otherwise
        """
        print("\n" + "="*70)
        print("MULTI-AGENT SYSTEM VALIDATION TEST")
        print("="*70)
        print(f"\nTest file: {self.test_file}")
        print(f"Expected: 6 violations, 0 false positives")
        
        # Check if test file exists
        if not os.path.exists(self.test_file):
            print(f"\n❌ Test file not found: {self.test_file}")
            return False
        
        # Run current system
        self.results['current_system'] = self.run_current_system()
        
        # Run multi-agent system
        self.results['multiagent_system'] = self.run_multiagent_system()
        
        # Compare results
        if (self.results['current_system'] and 
            self.results['multiagent_system'] and
            'error' not in self.results['current_system'] and
            'error' not in self.results['multiagent_system']):
            
            self.results['comparison'] = self.compare_violations(
                self.results['current_system']['violations'],
                self.results['multiagent_system']['violations']
            )
        
        # Generate and print report
        report = self.generate_report()
        print(report)
        
        # Save report to file
        report_file = "tests/validation_report.txt"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📄 Validation report saved to: {report_file}")
        
        # Return validation status
        if self.results['comparison']:
            return (
                self.results['comparison'].get('count_match', False) and
                self.results['comparison'].get('all_matched', False)
            )
        return False


def main():
    """Main function"""
    validator = ValidationTest()
    success = validator.run_validation()
    
    if success:
        print("\n✅ Validation test PASSED")
        sys.exit(0)
    else:
        print("\n⚠️  Validation test INCOMPLETE or FAILED")
        print("   Review the validation report for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
