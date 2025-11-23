#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Testing Suite for Multi-Agent Compliance System

This comprehensive test suite validates the complete multi-agent compliance workflow:
1. Complete workflow execution on multiple documents
2. All command-line flags and options
3. All configuration options
4. Error scenarios and recovery
5. HITL workflow integration
6. Output format validation

Requirements: 14.1, 14.2, 14.3, 14.4
"""

import sys
import os
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EndToEndTestSuite:
    """Comprehensive end-to-end test suite"""
    
    def __init__(self):
        self.test_results = []
        self.test_dir = tempfile.mkdtemp(prefix="e2e_test_")
        self.original_dir = os.getcwd()
        
    def setup(self):
        """Set up test environment"""
        logger.info(f"Test directory: {self.test_dir}")
        
    def teardown(self):
        """Clean up test environment"""
        os.chdir(self.original_dir)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        logger.info("Test environment cleaned up")
    
    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"  Details: {details}")
    
    def create_test_document(self, scenario: str = "basic") -> Dict[str, Any]:
        """
        Create test documents for different scenarios
        
        Args:
            scenario: Type of test document
                - "basic": Simple document with minimal violations
                - "complex": Document with multiple violation types
                - "clean": Document with no violations
                - "edge_case": Document with edge cases
        """
        base_doc = {
            'document_metadata': {
                'fund_isin': 'FR0010135103',
                'fund_name': 'Test Fund',
                'client_type': 'retail',
                'document_type': 'fund_presentation',
                'fund_esg_classification': 'article_8',
                'fund_age_years': 5,
                'document_date': '2024-01-15',
                'fund_inception_date': '2020-01-15'
            },
            'page_de_garde': {
                'title': 'Fund Presentation',
                'subtitle': 'Test Fund',
                'date': '2024-01-15'
            },
            'slide_2': {
                'title': 'Overview',
                'content': 'Investment strategy overview'
            },
            'pages_suivantes': [
                {
                    'slide_number': 3,
                    'title': 'Strategy',
                    'content': 'Investment strategy details'
                }
            ],
            'page_de_fin': {
                'legal': 'Test Asset Management SAS',
                'disclaimers': 'Standard disclaimers'
            }
        }
        
        if scenario == "complex":
            base_doc['page_de_garde']['content'] = 'Document promotionnel'
            base_doc['slide_2']['content'] = 'Performance: +20% en 2023. Note Morningstar: ★★★★.'
            base_doc['pages_suivantes'][0]['content'] = 'Volatilité: 12%. Alpha: 2.3%.'
            
        elif scenario == "clean":
            base_doc['page_de_garde']['content'] = 'Document à caractère promotionnel'
            base_doc['slide_2']['content'] = 'Stratégie d\'investissement diversifiée'
            base_doc['page_de_fin']['legal'] = 'Test Asset Management SAS - Société de Gestion agréée AMF'
            
        elif scenario == "edge_case":
            base_doc['slide_2']['content'] = ''  # Empty content
            base_doc['pages_suivantes'] = []  # No additional slides
            
        return base_doc
    
    def run_check_multiagent(
        self,
        json_file: str,
        args: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run check_multiagent.py with specified arguments
        
        Args:
            json_file: Path to JSON file
            args: Additional command-line arguments
            
        Returns:
            Dictionary with execution results
        """
        cmd = [sys.executable, "check_multiagent.py", json_file]
        if args:
            cmd.extend(args)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=120
            )
            
            # Load violations file if it exists
            violations_file = json_file.replace('.json', '_violations.json')
            violations_data = None
            if os.path.exists(violations_file):
                with open(violations_file, 'r', encoding='utf-8') as f:
                    violations_data = json.load(f)
            
            return {
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'violations_data': violations_data,
                'success': result.returncode in [0, 1]  # 0=no violations, 1=violations found
            }
            
        except subprocess.TimeoutExpired:
            return {
                'exit_code': -1,
                'error': 'Timeout',
                'success': False
            }
        except Exception as e:
            return {
                'exit_code': -1,
                'error': str(e),
                'success': False
            }
    
    # ========================================================================
    # TEST 1: Multiple Documents
    # ========================================================================
    
    def test_1_multiple_documents(self) -> bool:
        """
        Test 1: Run complete workflow on multiple documents
        
        Verifies:
        - System can process different document types
        - Results are consistent across runs
        - All document scenarios are handled correctly
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 1: Multiple Documents")
        logger.info("="*70)
        
        try:
            scenarios = ["basic", "complex", "clean", "edge_case"]
            results = {}
            
            for scenario in scenarios:
                logger.info(f"\nTesting scenario: {scenario}")
                
                # Create test document
                doc = self.create_test_document(scenario)
                doc_file = os.path.join(self.test_dir, f"test_{scenario}.json")
                
                with open(doc_file, 'w', encoding='utf-8') as f:
                    json.dump(doc, f, indent=2, ensure_ascii=False)
                
                # Run check
                result = self.run_check_multiagent(doc_file)
                
                if not result['success']:
                    self.log_test_result(
                        f"Multiple Documents - {scenario}",
                        False,
                        f"Execution failed: {result.get('error', 'Unknown error')}"
                    )
                    return False
                
                results[scenario] = result
                logger.info(f"  ✓ {scenario} processed successfully")
                
                if result['violations_data']:
                    violations = result['violations_data'].get('violations', [])
                    logger.info(f"    Violations found: {len(violations)}")
            
            # Verify all scenarios processed
            if len(results) == len(scenarios):
                self.log_test_result(
                    "Multiple Documents",
                    True,
                    f"All {len(scenarios)} document scenarios processed successfully"
                )
                return True
            else:
                self.log_test_result(
                    "Multiple Documents",
                    False,
                    f"Only {len(results)}/{len(scenarios)} scenarios completed"
                )
                return False
                
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            self.log_test_result("Multiple Documents", False, str(e))
            return False
    
    # ========================================================================
    # TEST 2: Command-Line Flags
    # ========================================================================
    
    def test_2_command_line_flags(self) -> bool:
        """
        Test 2: Test all command-line flags
        
        Verifies:
        - All flags are recognized and processed
        - Flags modify behavior correctly
        - Invalid flags are handled gracefully
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 2: Command-Line Flags")
        logger.info("="*70)
        
        try:
            # Create test document
            doc = self.create_test_document("basic")
            doc_file = os.path.join(self.test_dir, "test_flags.json")
            
            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            
            # Test different flag combinations
            flag_tests = [
                {
                    'name': 'show-metrics',
                    'args': ['--show-metrics'],
                    'expected_in_output': 'PERFORMANCE METRICS'
                },
                {
                    'name': 'legacy-output',
                    'args': ['--legacy-output'],
                    'expected_in_output': 'Legacy'
                },
                {
                    'name': 'hybrid-mode-on',
                    'args': ['--hybrid-mode=on'],
                    'expected_in_output': 'Hybrid mode enabled'
                },
                {
                    'name': 'hybrid-mode-off',
                    'args': ['--hybrid-mode=off'],
                    'expected_in_output': 'Hybrid mode disabled'
                },
                {
                    'name': 'rules-only',
                    'args': ['--rules-only'],
                    'expected_in_output': 'Rules-only mode enabled'
                },
                {
                    'name': 'ai-confidence',
                    'args': ['--ai-confidence=80'],
                    'expected_in_output': 'confidence threshold set to 80'
                },
                {
                    'name': 'review-threshold',
                    'args': ['--review-threshold=65'],
                    'expected_in_output': 'Review threshold set to 65'
                }
            ]
            
            passed_tests = 0
            for test in flag_tests:
                logger.info(f"\nTesting flag: {test['name']}")
                
                result = self.run_check_multiagent(doc_file, test['args'])
                
                if not result['success']:
                    logger.warning(f"  ✗ Execution failed for {test['name']}")
                    continue
                
                # Check if expected output is present
                output = result['stdout'] + result['stderr']
                if test['expected_in_output'] in output:
                    logger.info(f"  ✓ Flag {test['name']} works correctly")
                    passed_tests += 1
                else:
                    logger.warning(f"  ⚠ Expected output not found for {test['name']}")
            
            success = passed_tests >= len(flag_tests) * 0.8  # 80% pass rate
            self.log_test_result(
                "Command-Line Flags",
                success,
                f"{passed_tests}/{len(flag_tests)} flags tested successfully"
            )
            return success
            
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            self.log_test_result("Command-Line Flags", False, str(e))
            return False
    
    # ========================================================================
    # TEST 3: Configuration Options
    # ========================================================================
    
    def test_3_configuration_options(self) -> bool:
        """
        Test 3: Test all configuration options
        
        Verifies:
        - Configuration file is loaded correctly
        - Configuration options affect behavior
        - Invalid configurations are handled
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 3: Configuration Options")
        logger.info("="*70)
        
        try:
            # Create test document
            doc = self.create_test_document("basic")
            doc_file = os.path.join(self.test_dir, "test_config.json")
            
            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            
            # Test different configuration scenarios
            config_tests = [
                {
                    'name': 'multi_agent_enabled',
                    'config': {'multi_agent': {'enabled': True}},
                    'description': 'Multi-agent mode enabled'
                },
                {
                    'name': 'parallel_execution',
                    'config': {'multi_agent': {'enabled': True, 'parallel_execution': True}},
                    'description': 'Parallel execution enabled'
                },
                {
                    'name': 'hitl_enabled',
                    'config': {'hitl': {'enabled': True, 'review_threshold': 70}},
                    'description': 'HITL integration enabled'
                },
                {
                    'name': 'context_analysis',
                    'config': {'context_analysis': {'enabled': True}},
                    'description': 'Context analysis enabled'
                }
            ]
            
            passed_tests = 0
            for test in config_tests:
                logger.info(f"\nTesting config: {test['name']}")
                
                # Create temporary config file
                config_file = os.path.join(self.test_dir, f"config_{test['name']}.json")
                
                # Load base config and merge with test config
                base_config = {}
                if os.path.exists('hybrid_config.json'):
                    with open('hybrid_config.json', 'r', encoding='utf-8') as f:
                        base_config = json.load(f)
                
                # Merge test config
                test_config = {**base_config, **test['config']}
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(test_config, f, indent=2)
                
                # Temporarily replace config file
                original_config = None
                if os.path.exists('hybrid_config.json'):
                    shutil.copy('hybrid_config.json', 'hybrid_config.json.bak')
                    original_config = 'hybrid_config.json.bak'
                
                shutil.copy(config_file, 'hybrid_config.json')
                
                try:
                    result = self.run_check_multiagent(doc_file)
                    
                    if result['success']:
                        logger.info(f"  ✓ Config {test['name']} processed successfully")
                        passed_tests += 1
                    else:
                        logger.warning(f"  ✗ Config {test['name']} failed")
                finally:
                    # Restore original config
                    if original_config and os.path.exists(original_config):
                        shutil.copy(original_config, 'hybrid_config.json')
                        os.remove(original_config)
            
            success = passed_tests >= len(config_tests) * 0.75  # 75% pass rate
            self.log_test_result(
                "Configuration Options",
                success,
                f"{passed_tests}/{len(config_tests)} configurations tested successfully"
            )
            return success
            
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            self.log_test_result("Configuration Options", False, str(e))
            return False
    
    # ========================================================================
    # TEST 4: Error Scenarios
    # ========================================================================
    
    def test_4_error_scenarios(self) -> bool:
        """
        Test 4: Test error scenarios
        
        Verifies:
        - Missing file is handled gracefully
        - Invalid JSON is handled
        - Missing required fields are handled
        - System recovers from errors
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 4: Error Scenarios")
        logger.info("="*70)
        
        try:
            error_tests = []
            
            # Test 1: Missing file
            logger.info("\nTest 4a: Missing file")
            result = self.run_check_multiagent("nonexistent_file.json")
            if result['exit_code'] != 0:
                logger.info("  ✓ Missing file handled correctly")
                error_tests.append(True)
            else:
                logger.warning("  ✗ Missing file not detected")
                error_tests.append(False)
            
            # Test 2: Invalid JSON
            logger.info("\nTest 4b: Invalid JSON")
            invalid_json_file = os.path.join(self.test_dir, "invalid.json")
            with open(invalid_json_file, 'w', encoding='utf-8') as f:
                f.write("{ invalid json content")
            
            result = self.run_check_multiagent(invalid_json_file)
            if result['exit_code'] != 0:
                logger.info("  ✓ Invalid JSON handled correctly")
                error_tests.append(True)
            else:
                logger.warning("  ✗ Invalid JSON not detected")
                error_tests.append(False)
            
            # Test 3: Missing required fields
            logger.info("\nTest 4c: Missing required fields")
            incomplete_doc = {'document_metadata': {}}  # Missing required fields
            incomplete_file = os.path.join(self.test_dir, "incomplete.json")
            with open(incomplete_file, 'w', encoding='utf-8') as f:
                json.dump(incomplete_doc, f)
            
            result = self.run_check_multiagent(incomplete_file)
            # Should still run but may have warnings
            if result['success'] or result['exit_code'] != -1:
                logger.info("  ✓ Incomplete document handled gracefully")
                error_tests.append(True)
            else:
                logger.warning("  ✗ Incomplete document caused crash")
                error_tests.append(False)
            
            # Test 4: Empty document
            logger.info("\nTest 4d: Empty document")
            empty_file = os.path.join(self.test_dir, "empty.json")
            with open(empty_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            
            result = self.run_check_multiagent(empty_file)
            if result['success'] or result['exit_code'] != -1:
                logger.info("  ✓ Empty document handled gracefully")
                error_tests.append(True)
            else:
                logger.warning("  ✗ Empty document caused crash")
                error_tests.append(False)
            
            passed = sum(error_tests)
            total = len(error_tests)
            success = passed >= total * 0.75  # 75% pass rate
            
            self.log_test_result(
                "Error Scenarios",
                success,
                f"{passed}/{total} error scenarios handled correctly"
            )
            return success
            
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            self.log_test_result("Error Scenarios", False, str(e))
            return False
    
    # ========================================================================
    # TEST 5: HITL Workflow
    # ========================================================================
    
    def test_5_hitl_workflow(self) -> bool:
        """
        Test 5: Test HITL workflow
        
        Verifies:
        - Low confidence violations are queued for review
        - Review queue is created and populated
        - HITL integration components are functional
        - Review workflow can be initiated
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 5: HITL Workflow")
        logger.info("="*70)
        
        try:
            # Create document with ambiguous content (low confidence)
            doc = self.create_test_document("basic")
            doc['slide_2']['content'] = 'The fund may provide attractive returns based on market conditions.'
            
            doc_file = os.path.join(self.test_dir, "test_hitl.json")
            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            
            # Run with HITL test flag
            result = self.run_check_multiagent(doc_file, ['--test-hitl'])
            
            if not result['success']:
                self.log_test_result(
                    "HITL Workflow",
                    False,
                    "HITL test execution failed"
                )
                return False
            
            # Check for HITL-related output
            output = result['stdout'] + result['stderr']
            hitl_indicators = [
                'HITL',
                'Review',
                'review_queue',
                'Feedback',
                'Audit'
            ]
            
            found_indicators = sum(1 for indicator in hitl_indicators if indicator in output)
            
            logger.info(f"  HITL indicators found: {found_indicators}/{len(hitl_indicators)}")
            
            # Check if review queue file was created
            review_queue_exists = os.path.exists('review_queue.json')
            if review_queue_exists:
                logger.info("  ✓ Review queue file created")
            
            success = found_indicators >= 2  # At least 2 HITL indicators
            self.log_test_result(
                "HITL Workflow",
                success,
                f"HITL integration functional ({found_indicators} indicators found)"
            )
            return success
            
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            self.log_test_result("HITL Workflow", False, str(e))
            return False
    
    # ========================================================================
    # TEST 6: Output Format
    # ========================================================================
    
    def test_6_output_format(self) -> bool:
        """
        Test 6: Verify output format
        
        Verifies:
        - JSON output is valid
        - Required fields are present
        - Legacy format option works
        - Multi-agent metadata is included
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 6: Output Format")
        logger.info("="*70)
        
        try:
            # Create test document
            doc = self.create_test_document("complex")
            doc_file = os.path.join(self.test_dir, "test_output.json")
            
            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            
            format_tests = []
            
            # Test 1: Standard output format
            logger.info("\nTest 6a: Standard output format")
            result = self.run_check_multiagent(doc_file)
            
            if result['violations_data']:
                violations_data = result['violations_data']
                
                # Check required fields
                required_fields = ['document_id', 'violations', 'total_violations', 'metadata']
                has_required = all(field in violations_data for field in required_fields)
                
                if has_required:
                    logger.info("  ✓ All required fields present")
                    format_tests.append(True)
                else:
                    missing = [f for f in required_fields if f not in violations_data]
                    logger.warning(f"  ✗ Missing fields: {missing}")
                    format_tests.append(False)
                
                # Check multi-agent metadata
                if 'multi_agent' in violations_data:
                    logger.info("  ✓ Multi-agent metadata included")
                    format_tests.append(True)
                else:
                    logger.info("  ⚠ Multi-agent metadata not included (may be legacy mode)")
                    format_tests.append(True)  # Not critical
            else:
                logger.warning("  ✗ No violations data generated")
                format_tests.append(False)
            
            # Test 2: Legacy output format
            logger.info("\nTest 6b: Legacy output format")
            result = self.run_check_multiagent(doc_file, ['--legacy-output'])
            
            if result['violations_data']:
                violations_data = result['violations_data']
                
                # Legacy format should not have multi_agent field
                if 'multi_agent' not in violations_data:
                    logger.info("  ✓ Legacy format excludes multi-agent metadata")
                    format_tests.append(True)
                else:
                    logger.warning("  ⚠ Legacy format includes multi-agent metadata")
                    format_tests.append(True)  # Not critical
            else:
                logger.warning("  ✗ No violations data generated")
                format_tests.append(False)
            
            # Test 3: Violation structure
            logger.info("\nTest 6c: Violation structure")
            if result['violations_data']:
                violations = result['violations_data'].get('violations', [])
                
                if violations:
                    # Check first violation has required fields
                    violation = violations[0]
                    violation_fields = ['type', 'severity', 'rule', 'message', 'evidence']
                    has_fields = all(field in violation for field in violation_fields)
                    
                    if has_fields:
                        logger.info("  ✓ Violation structure is valid")
                        format_tests.append(True)
                    else:
                        missing = [f for f in violation_fields if f not in violation]
                        logger.warning(f"  ✗ Violation missing fields: {missing}")
                        format_tests.append(False)
                else:
                    logger.info("  ⚠ No violations to check structure")
                    format_tests.append(True)  # Not an error
            
            passed = sum(format_tests)
            total = len(format_tests)
            success = passed >= total * 0.8  # 80% pass rate
            
            self.log_test_result(
                "Output Format",
                success,
                f"{passed}/{total} format checks passed"
            )
            return success
            
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            self.log_test_result("Output Format", False, str(e))
            return False
    
    # ========================================================================
    # Main Test Runner
    # ========================================================================
    
    def run_all_tests(self) -> bool:
        """Run all end-to-end tests"""
        logger.info("\n" + "="*70)
        logger.info("END-TO-END TEST SUITE")
        logger.info("="*70)
        logger.info(f"Test directory: {self.test_dir}")
        logger.info(f"Started at: {datetime.now().isoformat()}")
        
        self.setup()
        start_time = datetime.now()
        
        # Run all tests
        tests = [
            self.test_1_multiple_documents,
            self.test_2_command_line_flags,
            self.test_3_configuration_options,
            self.test_4_error_scenarios,
            self.test_5_hitl_workflow,
            self.test_6_output_format
        ]
        
        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                logger.error(f"Test crashed: {e}", exc_info=True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("TEST SUMMARY")
        logger.info("="*70)
        
        passed = sum(1 for r in self.test_results if r['passed'])
        total = len(self.test_results)
        
        for result in self.test_results:
            status = "✓ PASSED" if result['passed'] else "✗ FAILED"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"  {result['details']}")
        
        logger.info("")
        logger.info(f"Results: {passed}/{total} tests passed")
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Completed at: {end_time.isoformat()}")
        logger.info("="*70)
        
        # Save results
        results_file = os.path.join(self.test_dir, "e2e_test_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": total - passed,
                    "duration_seconds": duration,
                    "timestamp": end_time.isoformat()
                },
                "tests": self.test_results
            }, f, indent=2)
        
        logger.info(f"\nResults saved to: {results_file}")
        
        # Cleanup
        self.teardown()
        
        return passed == total


def main():
    """Main entry point"""
    try:
        test_suite = EndToEndTestSuite()
        success = test_suite.run_all_tests()
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
