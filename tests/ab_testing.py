#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B Testing Framework for Multi-Agent System

This framework runs both the old (check.py) and new (check_multiagent.py) systems
in parallel, compares their results, generates comparison reports, and identifies
any discrepancies.

Features:
- Parallel execution of both systems
- Detailed comparison of violations
- Performance metrics comparison
- Discrepancy identification and analysis
- Comprehensive comparison report generation

Requirements: 14.4
"""

import sys
import os
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ABTestingFramework:
    """A/B testing framework for comparing old and new compliance systems"""
    
    def __init__(self, test_files: Optional[List[str]] = None):
        """
        Initialize A/B testing framework
        
        Args:
            test_files: List of JSON files to test (default: ["exemple.json"])
        """
        self.test_files = test_files or ["exemple.json"]
        self.old_system_script = "check.py"
        self.new_system_script = "check_multiagent.py"
        self.results = {}
        self.comparison_report = []
        
    def run_system(
        self,
        script: str,
        test_file: str,
        system_name: str
    ) -> Dict[str, Any]:
        """
        Run a compliance system on a test file
        
        Args:
            script: Script to run (check.py or check_multiagent.py)
            test_file: JSON file to test
            system_name: Name for logging (old/new)
            
        Returns:
            Dictionary with results and metrics
        """
        print(f"\n{'='*70}")
        print(f"RUNNING {system_name.upper()} SYSTEM: {script}")
        print(f"Test file: {test_file}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        try:
            # Run the system
            result = subprocess.run(
                [sys.executable, script, test_file],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300  # 5 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            print(f"Exit code: {result.returncode}")
            print(f"Execution time: {execution_time:.2f}s")
            
            # Load violations JSON
            violations_file = test_file.replace('.json', '_violations.json')
            
            if os.path.exists(violations_file):
                with open(violations_file, 'r', encoding='utf-8') as f:
                    output = json.load(f)
                
                violations = output.get('violations', [])
                print(f"✓ Found {len(violations)} violation(s)")
                
                # Extract metadata
                metadata = output.get('metadata', {})
                multi_agent_metadata = output.get('multi_agent', {})
                
                return {
                    'system': system_name,
                    'script': script,
                    'test_file': test_file,
                    'violations': violations,
                    'total_violations': len(violations),
                    'execution_time': execution_time,
                    'exit_code': result.returncode,
                    'output_file': violations_file,
                    'metadata': metadata,
                    'multi_agent_metadata': multi_agent_metadata,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'success': True
                }
            else:
                print(f"⚠️  Violations file not found: {violations_file}")
                return {
                    'system': system_name,
                    'script': script,
                    'test_file': test_file,
                    'violations': [],
                    'total_violations': 0,
                    'execution_time': execution_time,
                    'exit_code': result.returncode,
                    'error': 'Violations file not found',
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'success': False
                }
                
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            print(f"❌ Timeout after {execution_time:.2f}s")
            return {
                'system': system_name,
                'script': script,
                'test_file': test_file,
                'violations': [],
                'total_violations': 0,
                'execution_time': execution_time,
                'error': 'Timeout',
                'success': False
            }
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Error: {e}")
            return {
                'system': system_name,
                'script': script,
                'test_file': test_file,
                'violations': [],
                'total_violations': 0,
                'execution_time': execution_time,
                'error': str(e),
                'success': False
            }

    def run_parallel_tests(self, test_file: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run both systems in parallel on a test file
        
        Args:
            test_file: JSON file to test
            
        Returns:
            Tuple of (old_system_result, new_system_result)
        """
        print(f"\n{'='*70}")
        print(f"RUNNING PARALLEL A/B TEST")
        print(f"Test file: {test_file}")
        print(f"{'='*70}")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            old_future = executor.submit(
                self.run_system,
                self.old_system_script,
                test_file,
                "old"
            )
            new_future = executor.submit(
                self.run_system,
                self.new_system_script,
                test_file,
                "new"
            )
            
            # Wait for both to complete
            old_result = old_future.result()
            new_result = new_future.result()
        
        return old_result, new_result
    
    def normalize_violation(self, violation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a violation for comparison
        
        Removes system-specific fields and normalizes text
        
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
        
        # Include confidence if present
        if 'confidence' in violation:
            normalized['confidence'] = violation['confidence']
        
        return normalized
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity ratio between two strings
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity ratio (0.0 to 1.0)
        """
        return difflib.SequenceMatcher(None, str1, str2).ratio()

    def compare_violations(
        self,
        old_violations: List[Dict[str, Any]],
        new_violations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare violations from both systems
        
        Args:
            old_violations: Violations from old system
            new_violations: Violations from new system
            
        Returns:
            Comparison results dictionary
        """
        print(f"\n{'='*70}")
        print("COMPARING VIOLATIONS")
        print(f"{'='*70}")
        
        # Normalize violations
        old_normalized = [self.normalize_violation(v) for v in old_violations]
        new_normalized = [self.normalize_violation(v) for v in new_violations]
        
        # Compare counts
        count_match = len(old_normalized) == len(new_normalized)
        print(f"\nViolation count:")
        print(f"  Old system: {len(old_normalized)}")
        print(f"  New system: {len(new_normalized)}")
        print(f"  Match: {'✓' if count_match else '✗'}")
        
        # Find matching violations
        matched = []
        unmatched_old = []
        unmatched_new = list(new_normalized)
        
        for old_v in old_normalized:
            best_match = None
            best_score = 0.0
            
            for new_v in unmatched_new:
                # Calculate match score based on key fields
                score = 0.0
                
                # Type match (most important)
                if old_v.get('type') == new_v.get('type'):
                    score += 0.3
                
                # Rule match (very important)
                if old_v.get('rule') == new_v.get('rule'):
                    score += 0.3
                
                # Slide match (important)
                if old_v.get('slide') == new_v.get('slide'):
                    score += 0.2
                
                # Evidence similarity (less important)
                old_evidence = old_v.get('evidence', '')
                new_evidence = new_v.get('evidence', '')
                if old_evidence and new_evidence:
                    evidence_sim = self.calculate_similarity(old_evidence, new_evidence)
                    score += 0.2 * evidence_sim
                
                if score > best_score:
                    best_score = score
                    best_match = new_v
            
            # Consider it a match if score > 0.7
            if best_match and best_score > 0.7:
                # Check for exact match
                exact_match = old_v == best_match
                
                matched.append({
                    'old': old_v,
                    'new': best_match,
                    'match_score': best_score,
                    'exact_match': exact_match
                })
                unmatched_new.remove(best_match)
            else:
                unmatched_old.append(old_v)
        
        # Report matching results
        print(f"\nMatching violations: {len(matched)}/{len(old_normalized)}")
        
        if matched:
            exact_matches = sum(1 for m in matched if m['exact_match'])
            partial_matches = len(matched) - exact_matches
            print(f"  Exact matches: {exact_matches}")
            print(f"  Partial matches: {partial_matches}")
            
            if partial_matches > 0:
                avg_score = sum(m['match_score'] for m in matched if not m['exact_match']) / partial_matches
                print(f"  Average match score: {avg_score:.2f}")
        
        # Report discrepancies
        discrepancies = []
        
        if unmatched_old:
            print(f"\n⚠️  Violations in OLD system but not in NEW:")
            for v in unmatched_old:
                print(f"    - {v.get('type')}: {v.get('rule')}")
                discrepancies.append({
                    'type': 'missing_in_new',
                    'violation': v
                })
        
        if unmatched_new:
            print(f"\n⚠️  Violations in NEW system but not in OLD:")
            for v in unmatched_new:
                print(f"    - {v.get('type')}: {v.get('rule')}")
                discrepancies.append({
                    'type': 'extra_in_new',
                    'violation': v
                })
        
        # Detailed comparison of partial matches
        if matched and any(not m['exact_match'] for m in matched):
            print(f"\nDetailed comparison of partial matches:")
            for i, match in enumerate(matched, 1):
                if not match['exact_match']:
                    print(f"\n  Match #{i} (score: {match['match_score']:.2f}):")
                    print(f"    Type: {match['old'].get('type')}")
                    print(f"    Rule: {match['old'].get('rule')}")
                    
                    # Find differences
                    old = match['old']
                    new = match['new']
                    
                    for key in set(list(old.keys()) + list(new.keys())):
                        if old.get(key) != new.get(key):
                            print(f"    Difference in '{key}':")
                            print(f"      Old: {old.get(key)}")
                            print(f"      New: {new.get(key)}")
                            
                            discrepancies.append({
                                'type': 'field_difference',
                                'field': key,
                                'old_value': old.get(key),
                                'new_value': new.get(key),
                                'violation_type': old.get('type'),
                                'rule': old.get('rule')
                            })
        
        return {
            'count_match': count_match,
            'old_count': len(old_normalized),
            'new_count': len(new_normalized),
            'matched_count': len(matched),
            'exact_matches': sum(1 for m in matched if m['exact_match']),
            'partial_matches': sum(1 for m in matched if not m['exact_match']),
            'unmatched_old': unmatched_old,
            'unmatched_new': unmatched_new,
            'all_matched': len(unmatched_old) == 0 and len(unmatched_new) == 0,
            'discrepancies': discrepancies,
            'matched_violations': matched
        }

    def compare_performance(
        self,
        old_result: Dict[str, Any],
        new_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare performance metrics between systems
        
        Args:
            old_result: Result from old system
            new_result: Result from new system
            
        Returns:
            Performance comparison dictionary
        """
        print(f"\n{'='*70}")
        print("COMPARING PERFORMANCE")
        print(f"{'='*70}")
        
        old_time = old_result.get('execution_time', 0)
        new_time = new_result.get('execution_time', 0)
        
        print(f"\nExecution time:")
        print(f"  Old system: {old_time:.2f}s")
        print(f"  New system: {new_time:.2f}s")
        
        if old_time > 0:
            speedup = ((old_time - new_time) / old_time) * 100
            if speedup > 0:
                print(f"  Speedup: {speedup:.1f}% faster")
            else:
                print(f"  Slowdown: {abs(speedup):.1f}% slower")
        
        # Multi-agent specific metrics
        multi_agent_metadata = new_result.get('multi_agent_metadata', {})
        if multi_agent_metadata:
            print(f"\nMulti-agent system metrics:")
            print(f"  Workflow status: {multi_agent_metadata.get('workflow_status', 'unknown')}")
            
            agent_timings = multi_agent_metadata.get('agent_timings', {})
            if agent_timings:
                print(f"  Agent execution times:")
                for agent, time in sorted(agent_timings.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    {agent}: {time:.2f}s")
        
        return {
            'old_execution_time': old_time,
            'new_execution_time': new_time,
            'speedup_percentage': ((old_time - new_time) / old_time * 100) if old_time > 0 else 0,
            'faster': new_time < old_time,
            'multi_agent_metadata': multi_agent_metadata
        }
    
    def identify_discrepancies(
        self,
        comparison: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify and categorize discrepancies
        
        Args:
            comparison: Comparison results
            
        Returns:
            List of categorized discrepancies
        """
        print(f"\n{'='*70}")
        print("IDENTIFYING DISCREPANCIES")
        print(f"{'='*70}")
        
        discrepancies = comparison.get('discrepancies', [])
        
        if not discrepancies:
            print("\n✓ No discrepancies found")
            return []
        
        # Categorize discrepancies
        categories = {
            'missing_in_new': [],
            'extra_in_new': [],
            'field_difference': []
        }
        
        for disc in discrepancies:
            disc_type = disc.get('type')
            if disc_type in categories:
                categories[disc_type].append(disc)
        
        # Report by category
        print(f"\nDiscrepancy summary:")
        print(f"  Missing in new system: {len(categories['missing_in_new'])}")
        print(f"  Extra in new system: {len(categories['extra_in_new'])}")
        print(f"  Field differences: {len(categories['field_difference'])}")
        
        # Detailed analysis
        if categories['missing_in_new']:
            print(f"\n⚠️  Violations missing in new system:")
            for disc in categories['missing_in_new']:
                v = disc['violation']
                print(f"    - {v.get('type')}: {v.get('rule')}")
                print(f"      Slide: {v.get('slide')}, Location: {v.get('location')}")
        
        if categories['extra_in_new']:
            print(f"\n⚠️  Extra violations in new system:")
            for disc in categories['extra_in_new']:
                v = disc['violation']
                print(f"    - {v.get('type')}: {v.get('rule')}")
                print(f"      Slide: {v.get('slide')}, Location: {v.get('location')}")
        
        if categories['field_difference']:
            print(f"\n⚠️  Field differences in matched violations:")
            # Group by field
            by_field = {}
            for disc in categories['field_difference']:
                field = disc.get('field')
                if field not in by_field:
                    by_field[field] = []
                by_field[field].append(disc)
            
            for field, discs in by_field.items():
                print(f"    {field}: {len(discs)} difference(s)")
        
        return discrepancies

    def generate_comparison_report(
        self,
        test_file: str,
        old_result: Dict[str, Any],
        new_result: Dict[str, Any],
        comparison: Dict[str, Any],
        performance: Dict[str, Any],
        discrepancies: List[Dict[str, Any]]
    ) -> str:
        """
        Generate comprehensive comparison report
        
        Args:
            test_file: Test file name
            old_result: Old system result
            new_result: New system result
            comparison: Violation comparison
            performance: Performance comparison
            discrepancies: List of discrepancies
            
        Returns:
            Report as string
        """
        report = []
        report.append("="*70)
        report.append("A/B TESTING COMPARISON REPORT")
        report.append("="*70)
        report.append(f"\nTest file: {test_file}")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # System information
        report.append(f"\n{'-'*70}")
        report.append("SYSTEMS TESTED")
        report.append(f"{'-'*70}")
        report.append(f"Old system: {old_result.get('script')}")
        report.append(f"New system: {new_result.get('script')}")
        
        # Execution status
        report.append(f"\n{'-'*70}")
        report.append("EXECUTION STATUS")
        report.append(f"{'-'*70}")
        report.append(f"Old system: {'✓ Success' if old_result.get('success') else '✗ Failed'}")
        if not old_result.get('success'):
            report.append(f"  Error: {old_result.get('error', 'Unknown')}")
        
        report.append(f"New system: {'✓ Success' if new_result.get('success') else '✗ Failed'}")
        if not new_result.get('success'):
            report.append(f"  Error: {new_result.get('error', 'Unknown')}")
        
        # Only continue if both systems succeeded
        if not (old_result.get('success') and new_result.get('success')):
            report.append(f"\n{'='*70}")
            report.append("❌ COMPARISON INCOMPLETE: One or both systems failed")
            report.append("="*70)
            return "\n".join(report)
        
        # Violation comparison
        report.append(f"\n{'-'*70}")
        report.append("VIOLATION COMPARISON")
        report.append(f"{'-'*70}")
        report.append(f"Old system violations: {comparison.get('old_count', 0)}")
        report.append(f"New system violations: {comparison.get('new_count', 0)}")
        report.append(f"Count match: {'✓' if comparison.get('count_match') else '✗'}")
        report.append(f"\nMatched violations: {comparison.get('matched_count', 0)}")
        report.append(f"  Exact matches: {comparison.get('exact_matches', 0)}")
        report.append(f"  Partial matches: {comparison.get('partial_matches', 0)}")
        report.append(f"\nUnmatched in old: {len(comparison.get('unmatched_old', []))}")
        report.append(f"Unmatched in new: {len(comparison.get('unmatched_new', []))}")
        
        # Performance comparison
        report.append(f"\n{'-'*70}")
        report.append("PERFORMANCE COMPARISON")
        report.append(f"{'-'*70}")
        report.append(f"Old system execution time: {performance.get('old_execution_time', 0):.2f}s")
        report.append(f"New system execution time: {performance.get('new_execution_time', 0):.2f}s")
        
        speedup = performance.get('speedup_percentage', 0)
        if speedup > 0:
            report.append(f"Performance improvement: {speedup:.1f}% faster")
        elif speedup < 0:
            report.append(f"Performance degradation: {abs(speedup):.1f}% slower")
        else:
            report.append(f"Performance: Same")
        
        # Multi-agent metrics
        multi_agent_metadata = performance.get('multi_agent_metadata', {})
        if multi_agent_metadata:
            report.append(f"\nMulti-agent system metrics:")
            report.append(f"  Workflow status: {multi_agent_metadata.get('workflow_status', 'unknown')}")
            report.append(f"  Total execution time: {multi_agent_metadata.get('total_execution_time', 0):.2f}s")
            
            agent_timings = multi_agent_metadata.get('agent_timings', {})
            if agent_timings:
                report.append(f"  Top 5 agent execution times:")
                for agent, time in sorted(agent_timings.items(), key=lambda x: x[1], reverse=True)[:5]:
                    report.append(f"    {agent}: {time:.2f}s")
        
        # Discrepancies
        report.append(f"\n{'-'*70}")
        report.append("DISCREPANCIES")
        report.append(f"{'-'*70}")
        
        if not discrepancies:
            report.append("✓ No discrepancies found")
        else:
            report.append(f"Total discrepancies: {len(discrepancies)}")
            
            # Categorize
            categories = {
                'missing_in_new': [],
                'extra_in_new': [],
                'field_difference': []
            }
            
            for disc in discrepancies:
                disc_type = disc.get('type')
                if disc_type in categories:
                    categories[disc_type].append(disc)
            
            report.append(f"\nBy category:")
            report.append(f"  Missing in new system: {len(categories['missing_in_new'])}")
            report.append(f"  Extra in new system: {len(categories['extra_in_new'])}")
            report.append(f"  Field differences: {len(categories['field_difference'])}")
            
            # Details
            if categories['missing_in_new']:
                report.append(f"\nViolations missing in new system:")
                for disc in categories['missing_in_new']:
                    v = disc['violation']
                    report.append(f"  - {v.get('type')}: {v.get('rule')}")
                    report.append(f"    Slide: {v.get('slide')}, Location: {v.get('location')}")
            
            if categories['extra_in_new']:
                report.append(f"\nExtra violations in new system:")
                for disc in categories['extra_in_new']:
                    v = disc['violation']
                    report.append(f"  - {v.get('type')}: {v.get('rule')}")
                    report.append(f"    Slide: {v.get('slide')}, Location: {v.get('location')}")
            
            if categories['field_difference']:
                report.append(f"\nField differences (first 10):")
                for disc in categories['field_difference'][:10]:
                    report.append(f"  - {disc.get('violation_type')}: {disc.get('rule')}")
                    report.append(f"    Field: {disc.get('field')}")
                    report.append(f"    Old: {disc.get('old_value')}")
                    report.append(f"    New: {disc.get('new_value')}")
        
        # Overall assessment
        report.append(f"\n{'='*70}")
        report.append("OVERALL ASSESSMENT")
        report.append(f"{'='*70}")
        
        # Determine pass/fail
        all_checks_passed = (
            comparison.get('count_match', False) and
            comparison.get('all_matched', False) and
            len(discrepancies) == 0
        )
        
        if all_checks_passed:
            report.append("\n✅ SYSTEMS MATCH PERFECTLY")
            report.append("\nAll checks passed:")
            report.append("  ✓ Violation counts match")
            report.append("  ✓ All violations match exactly")
            report.append("  ✓ No discrepancies detected")
        else:
            report.append("\n⚠️  SYSTEMS HAVE DIFFERENCES")
            report.append("\nCheck results:")
            
            if comparison.get('count_match'):
                report.append("  ✓ Violation counts match")
            else:
                report.append("  ✗ Violation counts differ")
            
            if comparison.get('all_matched'):
                report.append("  ✓ All violations match")
            else:
                report.append("  ⚠️  Some violations don't match")
            
            if len(discrepancies) == 0:
                report.append("  ✓ No discrepancies")
            else:
                report.append(f"  ⚠️  {len(discrepancies)} discrepancy(ies) found")
        
        # Performance assessment
        if performance.get('faster'):
            report.append(f"\n✓ New system is {performance.get('speedup_percentage', 0):.1f}% faster")
        else:
            report.append(f"\n⚠️  New system is {abs(performance.get('speedup_percentage', 0)):.1f}% slower")
        
        report.append("="*70)
        
        return "\n".join(report)

    def run_ab_test(self, test_file: str) -> Dict[str, Any]:
        """
        Run A/B test on a single file
        
        Args:
            test_file: JSON file to test
            
        Returns:
            Test results dictionary
        """
        print(f"\n{'='*70}")
        print(f"A/B TEST: {test_file}")
        print(f"{'='*70}")
        
        # Check if test file exists
        if not os.path.exists(test_file):
            print(f"\n❌ Test file not found: {test_file}")
            return {
                'test_file': test_file,
                'success': False,
                'error': 'Test file not found'
            }
        
        # Run both systems in parallel
        old_result, new_result = self.run_parallel_tests(test_file)
        
        # Check if both succeeded
        if not (old_result.get('success') and new_result.get('success')):
            print(f"\n❌ One or both systems failed")
            return {
                'test_file': test_file,
                'old_result': old_result,
                'new_result': new_result,
                'success': False,
                'error': 'System execution failed'
            }
        
        # Compare violations
        comparison = self.compare_violations(
            old_result['violations'],
            new_result['violations']
        )
        
        # Compare performance
        performance = self.compare_performance(old_result, new_result)
        
        # Identify discrepancies
        discrepancies = self.identify_discrepancies(comparison)
        
        # Generate report
        report = self.generate_comparison_report(
            test_file,
            old_result,
            new_result,
            comparison,
            performance,
            discrepancies
        )
        
        # Print report
        print(f"\n{report}")
        
        # Save report to file
        report_filename = f"tests/ab_test_report_{test_file.replace('.json', '')}.txt"
        os.makedirs(os.path.dirname(report_filename), exist_ok=True)
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {report_filename}")
        
        return {
            'test_file': test_file,
            'old_result': old_result,
            'new_result': new_result,
            'comparison': comparison,
            'performance': performance,
            'discrepancies': discrepancies,
            'report': report,
            'report_file': report_filename,
            'success': True
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run A/B tests on all test files
        
        Returns:
            Aggregated results dictionary
        """
        print(f"\n{'='*70}")
        print("RUNNING A/B TESTS ON ALL FILES")
        print(f"{'='*70}")
        print(f"\nTest files: {', '.join(self.test_files)}")
        
        all_results = []
        
        for test_file in self.test_files:
            result = self.run_ab_test(test_file)
            all_results.append(result)
            self.results[test_file] = result
        
        # Generate summary report
        summary = self.generate_summary_report(all_results)
        
        # Print summary
        print(f"\n{summary}")
        
        # Save summary
        summary_filename = "tests/ab_test_summary.txt"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"\n📄 Summary saved to: {summary_filename}")
        
        # Save detailed results as JSON
        json_filename = "tests/ab_test_results.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            # Prepare JSON-serializable results
            json_results = {}
            for test_file, result in self.results.items():
                json_results[test_file] = {
                    'success': result.get('success', False),
                    'old_violations': result.get('old_result', {}).get('total_violations', 0),
                    'new_violations': result.get('new_result', {}).get('total_violations', 0),
                    'old_execution_time': result.get('old_result', {}).get('execution_time', 0),
                    'new_execution_time': result.get('new_result', {}).get('execution_time', 0),
                    'matched_count': result.get('comparison', {}).get('matched_count', 0),
                    'exact_matches': result.get('comparison', {}).get('exact_matches', 0),
                    'discrepancy_count': len(result.get('discrepancies', []))
                }
            json.dump(json_results, f, indent=2)
        print(f"📄 JSON results saved to: {json_filename}")
        
        return {
            'test_files': self.test_files,
            'results': all_results,
            'summary': summary,
            'summary_file': summary_filename,
            'json_file': json_filename
        }

    def generate_summary_report(self, all_results: List[Dict[str, Any]]) -> str:
        """
        Generate summary report for all tests
        
        Args:
            all_results: List of test results
            
        Returns:
            Summary report as string
        """
        report = []
        report.append("="*70)
        report.append("A/B TESTING SUMMARY REPORT")
        report.append("="*70)
        report.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total test files: {len(all_results)}")
        
        # Count successes
        successful_tests = sum(1 for r in all_results if r.get('success'))
        report.append(f"Successful tests: {successful_tests}/{len(all_results)}")
        
        if successful_tests == 0:
            report.append("\n❌ No successful tests to summarize")
            report.append("="*70)
            return "\n".join(report)
        
        # Aggregate statistics
        total_old_violations = 0
        total_new_violations = 0
        total_old_time = 0
        total_new_time = 0
        total_matched = 0
        total_exact_matches = 0
        total_discrepancies = 0
        
        for result in all_results:
            if result.get('success'):
                old_result = result.get('old_result', {})
                new_result = result.get('new_result', {})
                comparison = result.get('comparison', {})
                
                total_old_violations += old_result.get('total_violations', 0)
                total_new_violations += new_result.get('total_violations', 0)
                total_old_time += old_result.get('execution_time', 0)
                total_new_time += new_result.get('execution_time', 0)
                total_matched += comparison.get('matched_count', 0)
                total_exact_matches += comparison.get('exact_matches', 0)
                total_discrepancies += len(result.get('discrepancies', []))
        
        # Violation statistics
        report.append(f"\n{'-'*70}")
        report.append("VIOLATION STATISTICS")
        report.append(f"{'-'*70}")
        report.append(f"Total violations (old system): {total_old_violations}")
        report.append(f"Total violations (new system): {total_new_violations}")
        report.append(f"Difference: {total_new_violations - total_old_violations}")
        report.append(f"\nMatched violations: {total_matched}")
        report.append(f"  Exact matches: {total_exact_matches}")
        report.append(f"  Partial matches: {total_matched - total_exact_matches}")
        
        # Performance statistics
        report.append(f"\n{'-'*70}")
        report.append("PERFORMANCE STATISTICS")
        report.append(f"{'-'*70}")
        report.append(f"Total execution time (old system): {total_old_time:.2f}s")
        report.append(f"Total execution time (new system): {total_new_time:.2f}s")
        
        if total_old_time > 0:
            speedup = ((total_old_time - total_new_time) / total_old_time) * 100
            if speedup > 0:
                report.append(f"Overall speedup: {speedup:.1f}% faster")
            else:
                report.append(f"Overall slowdown: {abs(speedup):.1f}% slower")
        
        # Discrepancy statistics
        report.append(f"\n{'-'*70}")
        report.append("DISCREPANCY STATISTICS")
        report.append(f"{'-'*70}")
        report.append(f"Total discrepancies: {total_discrepancies}")
        
        if total_discrepancies > 0:
            # Categorize all discrepancies
            all_discrepancies = []
            for result in all_results:
                if result.get('success'):
                    all_discrepancies.extend(result.get('discrepancies', []))
            
            categories = {
                'missing_in_new': 0,
                'extra_in_new': 0,
                'field_difference': 0
            }
            
            for disc in all_discrepancies:
                disc_type = disc.get('type')
                if disc_type in categories:
                    categories[disc_type] += 1
            
            report.append(f"\nBy category:")
            report.append(f"  Missing in new system: {categories['missing_in_new']}")
            report.append(f"  Extra in new system: {categories['extra_in_new']}")
            report.append(f"  Field differences: {categories['field_difference']}")
        
        # Per-file results
        report.append(f"\n{'-'*70}")
        report.append("PER-FILE RESULTS")
        report.append(f"{'-'*70}")
        
        for result in all_results:
            test_file = result.get('test_file')
            report.append(f"\n{test_file}:")
            
            if not result.get('success'):
                report.append(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
                continue
            
            old_result = result.get('old_result', {})
            new_result = result.get('new_result', {})
            comparison = result.get('comparison', {})
            performance = result.get('performance', {})
            
            report.append(f"  Violations: {old_result.get('total_violations', 0)} → {new_result.get('total_violations', 0)}")
            report.append(f"  Execution time: {old_result.get('execution_time', 0):.2f}s → {new_result.get('execution_time', 0):.2f}s")
            report.append(f"  Matched: {comparison.get('matched_count', 0)} ({comparison.get('exact_matches', 0)} exact)")
            report.append(f"  Discrepancies: {len(result.get('discrepancies', []))}")
            
            if performance.get('faster'):
                report.append(f"  Performance: ✓ {performance.get('speedup_percentage', 0):.1f}% faster")
            else:
                report.append(f"  Performance: ⚠️  {abs(performance.get('speedup_percentage', 0)):.1f}% slower")
        
        # Overall assessment
        report.append(f"\n{'='*70}")
        report.append("OVERALL ASSESSMENT")
        report.append(f"{'='*70}")
        
        # Determine overall status
        all_match = all(
            r.get('comparison', {}).get('all_matched', False)
            for r in all_results if r.get('success')
        )
        
        no_discrepancies = total_discrepancies == 0
        
        if all_match and no_discrepancies:
            report.append("\n✅ ALL TESTS PASSED")
            report.append("\nBoth systems produce identical results:")
            report.append("  ✓ All violation counts match")
            report.append("  ✓ All violations match exactly")
            report.append("  ✓ No discrepancies detected")
        else:
            report.append("\n⚠️  SYSTEMS HAVE DIFFERENCES")
            report.append("\nSummary:")
            
            if all_match:
                report.append("  ✓ All violations match")
            else:
                report.append("  ⚠️  Some violations don't match")
            
            if no_discrepancies:
                report.append("  ✓ No discrepancies")
            else:
                report.append(f"  ⚠️  {total_discrepancies} total discrepancy(ies)")
        
        # Performance summary
        if total_old_time > 0:
            speedup = ((total_old_time - total_new_time) / total_old_time) * 100
            if speedup > 0:
                report.append(f"\n✓ New system is {speedup:.1f}% faster overall")
            else:
                report.append(f"\n⚠️  New system is {abs(speedup):.1f}% slower overall")
        
        report.append("="*70)
        
        return "\n".join(report)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="A/B Testing Framework for Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default file (exemple.json)
  python tests/ab_testing.py
  
  # Test with specific file
  python tests/ab_testing.py --file exemple.json
  
  # Test with multiple files
  python tests/ab_testing.py --file file1.json file2.json file3.json
        """
    )
    
    parser.add_argument(
        '--file',
        nargs='+',
        default=['exemple.json'],
        help='JSON file(s) to test (default: exemple.json)'
    )
    
    args = parser.parse_args()
    
    # Create framework
    framework = ABTestingFramework(test_files=args.file)
    
    # Run all tests
    results = framework.run_all_tests()
    
    # Determine exit code
    all_successful = all(
        r.get('success', False) for r in results['results']
    )
    
    if all_successful:
        print("\n✅ All A/B tests completed successfully")
        sys.exit(0)
    else:
        print("\n⚠️  Some A/B tests failed or had issues")
        sys.exit(1)


if __name__ == "__main__":
    main()
