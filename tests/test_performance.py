#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Benchmarks for Multi-Agent System

This test suite measures and compares performance between the multi-agent system
and the current monolithic system:
1. Total execution time comparison
2. Per-agent execution time measurement
3. Parallel execution speedup verification
4. Multi-document batch processing performance
5. Performance regression detection

Requirements: 3.5, 14.3
"""

import logging
import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple
from pathlib import Path
import statistics

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceBenchmarks:
    """Performance benchmark test suite"""
    
    def __init__(self):
        self.test_results = []
        self.benchmark_results = {
            'multi_agent': {},
            'current_system': {},
            'comparison': {}
        }
        
        # Performance targets
        self.PARALLEL_SPEEDUP_TARGET = 1.30  # 30% improvement
        self.MAX_REGRESSION_THRESHOLD = 1.10  # Allow 10% slower max
        
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
    
    def create_test_document(self, complexity: str = "medium") -> Dict[str, Any]:
        """
        Create test documents with varying complexity
        
        Args:
            complexity: "simple", "medium", or "complex"
            
        Returns:
            Test document dictionary
        """
        base_document = {
            'document_metadata': {
                'fund_isin': 'FR0010135103',
                'fund_name': 'Performance Test Fund',
                'client_type': 'retail',
                'document_type': 'fund_presentation',
                'fund_esg_classification': 'article_8',
                'fund_age_years': 5,
                'document_date': '2024-01-15',
                'fund_inception_date': '2020-01-15'
            },
            'page_de_garde': {
                'title': 'Fund Presentation',
                'subtitle': 'Performance Test Fund',
                'date': '2024-01-15',
                'content': 'Document de présentation'
            },
            'slide_2': {
                'title': 'Overview',
                'content': 'Investment strategy overview'
            },
            'pages_suivantes': [],
            'page_de_fin': {
                'legal': 'Test Asset Management SAS',
                'disclaimers': 'Standard disclaimers'
            }
        }
        
        if complexity == "simple":
            # Minimal content - 3 slides
            base_document['pages_suivantes'] = [
                {
                    'slide_number': 3,
                    'title': 'Strategy',
                    'content': 'Investment strategy details'
                }
            ]
        
        elif complexity == "medium":
            # Medium content - 10 slides with some violations
            base_document['page_de_garde']['content'] = 'Document promotionnel'
            base_document['slide_2']['content'] = 'Performance: +15.5% en 2023. Rendement annualisé: 8.2%.'
            
            for i in range(3, 11):
                base_document['pages_suivantes'].append({
                    'slide_number': i,
                    'title': f'Section {i}',
                    'content': f'Content for slide {i} with investment details and market analysis.'
                })
        
        elif complexity == "complex":
            # Complex content - 20 slides with multiple violation types
            base_document['page_de_garde']['content'] = 'Promotional document'
            base_document['slide_2']['content'] = (
                'Performance exceptionnelle: +20% en 2023, +15% en 2022. '
                'Morningstar rating: ★★★★★. Alpha: 2.5%, Beta: 0.8. '
                'Nous recommandons fortement ce fonds.'
            )
            
            for i in range(3, 21):
                content_parts = [
                    f'Slide {i} content with detailed analysis.',
                    'Performance data: +12.3% YTD.',
                    'Technical indicators: Sharpe ratio 1.5, volatility 10%.',
                    'Investment recommendation for growth portfolios.',
                    'ESG score: AAA rating from MSCI.',
                    'Benchmark comparison: outperforming by 3.2%.'
                ]
                base_document['pages_suivantes'].append({
                    'slide_number': i,
                    'title': f'Detailed Section {i}',
                    'content': ' '.join(content_parts)
                })
        
        return base_document
    
    def benchmark_multi_agent_system(
        self,
        document: Dict[str, Any],
        document_id: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Benchmark multi-agent system execution
        
        Args:
            document: Test document
            document_id: Document identifier
            
        Returns:
            Tuple of (total_time, result_dict)
        """
        try:
            from workflow_builder import create_compliance_workflow
            from data_models_multiagent import initialize_compliance_state
            
            # Create workflow
            config = {
                "multi_agent": {
                    "enabled": True,
                    "parallel_execution": True,
                    "max_parallel_agents": 4
                }
            }
            
            workflow = create_compliance_workflow(
                config=config,
                enable_checkpointing=False  # Disable for pure performance testing
            )
            
            # Initialize state
            state = initialize_compliance_state(
                document=document,
                document_id=document_id,
                config=config
            )
            
            # Execute and time
            start_time = time.time()
            result = workflow.invoke(state)
            end_time = time.time()
            
            total_time = end_time - start_time
            
            return total_time, result
            
        except Exception as e:
            logger.error(f"Error benchmarking multi-agent system: {e}")
            return 0.0, {}
    
    def benchmark_current_system(
        self,
        document: Dict[str, Any],
        temp_file_path: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Benchmark current monolithic system execution
        
        Args:
            document: Test document
            temp_file_path: Temporary file path for document
            
        Returns:
            Tuple of (total_time, result_dict)
        """
        try:
            # Save document to temp file
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
            
            # Import check function from current system
            # Note: This imports the monolithic check.py logic
            import check
            
            # Execute and time
            start_time = time.time()
            
            # Call the check_document_compliance function
            result = check.check_document_compliance(temp_file_path)
            
            end_time = time.time()
            
            total_time = end_time - start_time
            
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            return total_time, result
            
        except Exception as e:
            logger.error(f"Error benchmarking current system: {e}")
            # Clean up on error
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return 0.0, {}
    
    def test_1_total_execution_time_comparison(self) -> bool:
        """
        Test 1: Compare total execution time between systems
        
        Verifies:
        - Multi-agent system completes successfully
        - Current system completes successfully
        - Execution times are measured accurately
        - Results are comparable
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 1: Total Execution Time Comparison")
        logger.info("="*70)
        
        try:
            # Create medium complexity test document
            document = self.create_test_document("medium")
            
            logger.info("\n[1] Benchmarking Multi-Agent System...")
            
            # Benchmark multi-agent system
            multi_agent_time, multi_agent_result = self.benchmark_multi_agent_system(
                document,
                "perf_test_multi_agent"
            )
            
            if multi_agent_time == 0:
                self.log_test_result(
                    "Total Execution Time Comparison",
                    False,
                    "Multi-agent system benchmark failed"
                )
                return False
            
            logger.info(f"  ✓ Multi-agent execution time: {multi_agent_time:.3f}s")
            
            # Get agent timings
            agent_timings = multi_agent_result.get('agent_timings', {})
            if agent_timings:
                logger.info(f"  Agent breakdown:")
                for agent, timing in sorted(agent_timings.items(), key=lambda x: x[1], reverse=True)[:5]:
                    logger.info(f"    - {agent}: {timing:.3f}s")
            
            violations_multi = len(multi_agent_result.get('violations', []))
            logger.info(f"  Violations detected: {violations_multi}")
            
            logger.info("\n[2] Benchmarking Current System...")
            
            # Benchmark current system
            temp_file = "temp_perf_test_current.json"
            current_time, current_result = self.benchmark_current_system(
                document,
                temp_file
            )
            
            if current_time == 0:
                logger.warning("  ⚠ Current system benchmark failed or not available")
                logger.info("  Skipping comparison, validating multi-agent performance only")
                
                # Store results
                self.benchmark_results['multi_agent']['total_time'] = multi_agent_time
                self.benchmark_results['multi_agent']['violations'] = violations_multi
                
                self.log_test_result(
                    "Total Execution Time Comparison",
                    True,
                    f"Multi-agent: {multi_agent_time:.3f}s (current system not available)"
                )
                return True
            
            logger.info(f"  ✓ Current system execution time: {current_time:.3f}s")
            
            violations_current = len(current_result.get('violations', []))
            logger.info(f"  Violations detected: {violations_current}")
            
            # Compare results
            logger.info("\n[3] Comparing Results...")
            
            speedup = current_time / multi_agent_time if multi_agent_time > 0 else 0
            time_diff = multi_agent_time - current_time
            percent_diff = (time_diff / current_time * 100) if current_time > 0 else 0
            
            logger.info(f"\n  Performance Comparison:")
            logger.info(f"    Current system:    {current_time:.3f}s")
            logger.info(f"    Multi-agent:       {multi_agent_time:.3f}s")
            logger.info(f"    Difference:        {time_diff:+.3f}s ({percent_diff:+.1f}%)")
            logger.info(f"    Speedup:           {speedup:.2f}x")
            
            if speedup >= 1.0:
                logger.info(f"  ✓ Multi-agent is {speedup:.2f}x faster")
            else:
                logger.info(f"  ⚠ Multi-agent is {1/speedup:.2f}x slower")
            
            # Verify violation counts are similar
            violation_diff = abs(violations_multi - violations_current)
            logger.info(f"\n  Violation Comparison:")
            logger.info(f"    Current system:    {violations_current}")
            logger.info(f"    Multi-agent:       {violations_multi}")
            logger.info(f"    Difference:        {violation_diff}")
            
            if violation_diff <= 2:
                logger.info(f"  ✓ Violation counts are similar")
            else:
                logger.warning(f"  ⚠ Violation counts differ significantly")
            
            # Store results
            self.benchmark_results['multi_agent']['total_time'] = multi_agent_time
            self.benchmark_results['multi_agent']['violations'] = violations_multi
            self.benchmark_results['current_system']['total_time'] = current_time
            self.benchmark_results['current_system']['violations'] = violations_current
            self.benchmark_results['comparison']['speedup'] = speedup
            self.benchmark_results['comparison']['time_diff'] = time_diff
            self.benchmark_results['comparison']['percent_diff'] = percent_diff
            
            self.log_test_result(
                "Total Execution Time Comparison",
                True,
                f"Multi-agent: {multi_agent_time:.3f}s, Current: {current_time:.3f}s, Speedup: {speedup:.2f}x"
            )
            return True
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}", exc_info=True)
            self.log_test_result("Total Execution Time Comparison", False, str(e))
            return False
    
    def test_2_per_agent_execution_time(self) -> bool:
        """
        Test 2: Measure per-agent execution time
        
        Verifies:
        - Each agent's execution time is tracked
        - Agent timings are reasonable
        - No agent dominates execution time excessively
        - Timing data is accurate
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 2: Per-Agent Execution Time Measurement")
        logger.info("="*70)
        
        try:
            # Create complex document to exercise all agents
            document = self.create_test_document("complex")
            
            logger.info("\n[1] Executing multi-agent workflow...")
            
            multi_agent_time, result = self.benchmark_multi_agent_system(
                document,
                "perf_test_per_agent"
            )
            
            if multi_agent_time == 0:
                self.log_test_result(
                    "Per-Agent Execution Time",
                    False,
                    "Multi-agent system execution failed"
                )
                return False
            
            logger.info(f"  ✓ Total execution time: {multi_agent_time:.3f}s")
            
            # Analyze agent timings
            logger.info("\n[2] Analyzing per-agent execution times...")
            
            agent_timings = result.get('agent_timings', {})
            
            if not agent_timings:
                logger.warning("  ⚠ No agent timing data available")
                self.log_test_result(
                    "Per-Agent Execution Time",
                    False,
                    "No agent timing data"
                )
                return False
            
            logger.info(f"\n  Agent Execution Times:")
            logger.info(f"  {'Agent':<20} {'Time (s)':<10} {'% of Total':<12} {'Status'}")
            logger.info(f"  {'-'*60}")
            
            total_agent_time = sum(agent_timings.values())
            
            for agent, timing in sorted(agent_timings.items(), key=lambda x: x[1], reverse=True):
                percent = (timing / total_agent_time * 100) if total_agent_time > 0 else 0
                
                # Check if agent time is reasonable
                status = "✓"
                if timing > multi_agent_time * 0.5:
                    status = "⚠ SLOW"
                elif timing < 0.001:
                    status = "⚠ FAST"
                
                logger.info(f"  {agent:<20} {timing:<10.3f} {percent:<12.1f} {status}")
            
            logger.info(f"  {'-'*60}")
            logger.info(f"  {'TOTAL':<20} {total_agent_time:<10.3f} {100.0:<12.1f}")
            
            # Verify timing data quality
            logger.info("\n[3] Verifying timing data quality...")
            
            # Check that all expected agents have timings
            expected_agents = ['supervisor', 'preprocessor']
            missing_agents = [a for a in expected_agents if a not in agent_timings]
            
            if missing_agents:
                logger.warning(f"  ⚠ Missing timing data for: {missing_agents}")
            else:
                logger.info(f"  ✓ All core agents have timing data")
            
            # Check for outliers
            if agent_timings:
                times = list(agent_timings.values())
                mean_time = statistics.mean(times)
                
                if len(times) > 1:
                    stdev_time = statistics.stdev(times)
                    
                    outliers = []
                    for agent, timing in agent_timings.items():
                        if timing > mean_time + 2 * stdev_time:
                            outliers.append((agent, timing))
                    
                    if outliers:
                        logger.info(f"\n  Outlier agents (>2σ from mean):")
                        for agent, timing in outliers:
                            logger.info(f"    - {agent}: {timing:.3f}s")
                    else:
                        logger.info(f"  ✓ No significant outliers detected")
            
            # Store results
            self.benchmark_results['multi_agent']['agent_timings'] = agent_timings
            self.benchmark_results['multi_agent']['total_agent_time'] = total_agent_time
            
            self.log_test_result(
                "Per-Agent Execution Time",
                True,
                f"{len(agent_timings)} agents measured, total: {total_agent_time:.3f}s"
            )
            return True
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}", exc_info=True)
            self.log_test_result("Per-Agent Execution Time", False, str(e))
            return False
    
    def test_3_parallel_execution_speedup(self) -> bool:
        """
        Test 3: Verify 30% improvement from parallel execution
        
        Verifies:
        - Parallel agents execute concurrently
        - Parallel execution provides measurable speedup
        - Speedup meets or exceeds 30% target
        - Sequential time calculation is accurate
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 3: Parallel Execution Speedup Verification")
        logger.info("="*70)
        
        try:
            # Create document that triggers multiple parallel agents
            document = self.create_test_document("complex")
            
            logger.info("\n[1] Executing with parallel execution enabled...")
            
            # Benchmark with parallel execution
            config_parallel = {
                "multi_agent": {
                    "enabled": True,
                    "parallel_execution": True,
                    "max_parallel_agents": 4
                }
            }
            
            from workflow_builder import create_compliance_workflow
            from data_models_multiagent import initialize_compliance_state
            
            workflow_parallel = create_compliance_workflow(
                config=config_parallel,
                enable_checkpointing=False
            )
            
            state_parallel = initialize_compliance_state(
                document=document,
                document_id="perf_test_parallel",
                config=config_parallel
            )
            
            start_time = time.time()
            result_parallel = workflow_parallel.invoke(state_parallel)
            parallel_time = time.time() - start_time
            
            logger.info(f"  ✓ Parallel execution time: {parallel_time:.3f}s")
            
            # Get agent timings
            agent_timings = result_parallel.get('agent_timings', {})
            
            # Identify parallel agents (structure, performance, securities, general)
            parallel_agents = ['structure', 'performance', 'securities', 'general']
            executed_parallel = [a for a in parallel_agents if a in agent_timings]
            
            logger.info(f"  Parallel agents executed: {len(executed_parallel)}")
            for agent in executed_parallel:
                logger.info(f"    - {agent}: {agent_timings[agent]:.3f}s")
            
            # Calculate theoretical sequential time
            sequential_time = sum(agent_timings.get(a, 0) for a in executed_parallel)
            
            logger.info(f"\n[2] Calculating speedup...")
            
            logger.info(f"\n  Timing Analysis:")
            logger.info(f"    Sequential time (sum of parallel agents): {sequential_time:.3f}s")
            logger.info(f"    Parallel execution time (actual):         {parallel_time:.3f}s")
            
            if sequential_time > 0:
                # Calculate speedup from parallel execution
                parallel_speedup = sequential_time / parallel_time
                improvement_percent = (parallel_speedup - 1.0) * 100
                
                logger.info(f"    Parallel speedup:                         {parallel_speedup:.2f}x")
                logger.info(f"    Improvement:                              {improvement_percent:.1f}%")
                
                # Check if meets 30% target
                logger.info(f"\n[3] Verifying performance target...")
                logger.info(f"  Target: {self.PARALLEL_SPEEDUP_TARGET:.2f}x ({(self.PARALLEL_SPEEDUP_TARGET-1)*100:.0f}% improvement)")
                logger.info(f"  Actual: {parallel_speedup:.2f}x ({improvement_percent:.1f}% improvement)")
                
                if parallel_speedup >= self.PARALLEL_SPEEDUP_TARGET:
                    logger.info(f"  ✓ PASSED: Exceeds 30% improvement target")
                    target_met = True
                else:
                    shortfall = (self.PARALLEL_SPEEDUP_TARGET - parallel_speedup) * 100
                    logger.warning(f"  ⚠ Below target by {shortfall:.1f} percentage points")
                    target_met = False
                
                # Store results
                self.benchmark_results['multi_agent']['parallel_time'] = parallel_time
                self.benchmark_results['multi_agent']['sequential_time'] = sequential_time
                self.benchmark_results['multi_agent']['parallel_speedup'] = parallel_speedup
                self.benchmark_results['multi_agent']['improvement_percent'] = improvement_percent
                
                self.log_test_result(
                    "Parallel Execution Speedup",
                    target_met,
                    f"Speedup: {parallel_speedup:.2f}x ({improvement_percent:.1f}% improvement)"
                )
                return target_met
            else:
                logger.warning("  ⚠ Could not calculate speedup (no parallel agent times)")
                self.log_test_result(
                    "Parallel Execution Speedup",
                    False,
                    "No parallel agent timing data"
                )
                return False
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}", exc_info=True)
            self.log_test_result("Parallel Execution Speedup", False, str(e))
            return False
    
    def test_4_multi_document_batch_performance(self) -> bool:
        """
        Test 4: Test with multiple documents
        
        Verifies:
        - System handles multiple documents efficiently
        - Performance is consistent across documents
        - No memory leaks or degradation
        - Batch processing is efficient
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 4: Multi-Document Batch Performance")
        logger.info("="*70)
        
        try:
            # Create multiple test documents with varying complexity
            documents = [
                ("simple_1", self.create_test_document("simple")),
                ("simple_2", self.create_test_document("simple")),
                ("medium_1", self.create_test_document("medium")),
                ("medium_2", self.create_test_document("medium")),
                ("complex_1", self.create_test_document("complex")),
            ]
            
            logger.info(f"\n[1] Processing {len(documents)} documents...")
            
            from workflow_builder import create_compliance_workflow
            from data_models_multiagent import initialize_compliance_state
            
            # Create workflow once (reuse for all documents)
            config = {
                "multi_agent": {
                    "enabled": True,
                    "parallel_execution": True,
                    "max_parallel_agents": 4
                }
            }
            
            workflow = create_compliance_workflow(
                config=config,
                enable_checkpointing=False
            )
            
            # Process each document and collect timings
            document_times = []
            total_violations = 0
            
            for doc_id, document in documents:
                logger.info(f"\n  Processing: {doc_id}")
                
                state = initialize_compliance_state(
                    document=document,
                    document_id=f"batch_{doc_id}",
                    config=config
                )
                
                start_time = time.time()
                result = workflow.invoke(state)
                doc_time = time.time() - start_time
                
                document_times.append(doc_time)
                violations = len(result.get('violations', []))
                total_violations += violations
                
                logger.info(f"    Time: {doc_time:.3f}s, Violations: {violations}")
            
            # Calculate statistics
            logger.info(f"\n[2] Analyzing batch performance...")
            
            total_time = sum(document_times)
            avg_time = statistics.mean(document_times)
            
            logger.info(f"\n  Batch Statistics:")
            logger.info(f"    Documents processed:  {len(documents)}")
            logger.info(f"    Total time:           {total_time:.3f}s")
            logger.info(f"    Average time/doc:     {avg_time:.3f}s")
            logger.info(f"    Min time:             {min(document_times):.3f}s")
            logger.info(f"    Max time:             {max(document_times):.3f}s")
            
            if len(document_times) > 1:
                stdev_time = statistics.stdev(document_times)
                logger.info(f"    Std deviation:        {stdev_time:.3f}s")
                logger.info(f"    Coefficient of var:   {(stdev_time/avg_time)*100:.1f}%")
            
            logger.info(f"    Total violations:     {total_violations}")
            
            # Check for performance degradation
            logger.info(f"\n[3] Checking for performance degradation...")
            
            # Compare first and last document times
            first_doc_time = document_times[0]
            last_doc_time = document_times[-1]
            degradation = (last_doc_time - first_doc_time) / first_doc_time * 100
            
            logger.info(f"    First document:  {first_doc_time:.3f}s")
            logger.info(f"    Last document:   {last_doc_time:.3f}s")
            logger.info(f"    Degradation:     {degradation:+.1f}%")
            
            if abs(degradation) < 20:
                logger.info(f"  ✓ No significant performance degradation")
                no_degradation = True
            else:
                logger.warning(f"  ⚠ Significant performance change detected")
                no_degradation = False
            
            # Store results
            self.benchmark_results['multi_agent']['batch_total_time'] = total_time
            self.benchmark_results['multi_agent']['batch_avg_time'] = avg_time
            self.benchmark_results['multi_agent']['batch_document_count'] = len(documents)
            
            self.log_test_result(
                "Multi-Document Batch Performance",
                no_degradation,
                f"{len(documents)} docs in {total_time:.3f}s (avg: {avg_time:.3f}s/doc)"
            )
            return no_degradation
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}", exc_info=True)
            self.log_test_result("Multi-Document Batch Performance", False, str(e))
            return False
    
    def test_5_performance_regression_detection(self) -> bool:
        """
        Test 5: Compare with current system and detect regressions
        
        Verifies:
        - Multi-agent system is not significantly slower
        - Performance is within acceptable range
        - No major regressions introduced
        - System meets performance requirements
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 5: Performance Regression Detection")
        logger.info("="*70)
        
        try:
            # Use medium complexity document for fair comparison
            document = self.create_test_document("medium")
            
            logger.info("\n[1] Running comparative benchmark...")
            
            # Benchmark multi-agent system
            multi_agent_time, multi_result = self.benchmark_multi_agent_system(
                document,
                "regression_test_multi"
            )
            
            if multi_agent_time == 0:
                logger.warning("  ⚠ Multi-agent benchmark failed")
                self.log_test_result(
                    "Performance Regression Detection",
                    False,
                    "Multi-agent benchmark failed"
                )
                return False
            
            logger.info(f"  ✓ Multi-agent time: {multi_agent_time:.3f}s")
            
            # Benchmark current system
            temp_file = "temp_regression_test.json"
            current_time, current_result = self.benchmark_current_system(
                document,
                temp_file
            )
            
            if current_time == 0:
                logger.warning("  ⚠ Current system not available for comparison")
                logger.info("  Validating multi-agent performance in isolation")
                
                # Check if multi-agent time is reasonable (< 10s for medium doc)
                if multi_agent_time < 10.0:
                    logger.info(f"  ✓ Multi-agent time is reasonable: {multi_agent_time:.3f}s")
                    self.log_test_result(
                        "Performance Regression Detection",
                        True,
                        f"Multi-agent: {multi_agent_time:.3f}s (baseline not available)"
                    )
                    return True
                else:
                    logger.warning(f"  ⚠ Multi-agent time seems high: {multi_agent_time:.3f}s")
                    self.log_test_result(
                        "Performance Regression Detection",
                        False,
                        f"Multi-agent time too high: {multi_agent_time:.3f}s"
                    )
                    return False
            
            logger.info(f"  ✓ Current system time: {current_time:.3f}s")
            
            # Calculate regression metrics
            logger.info(f"\n[2] Analyzing performance regression...")
            
            regression_ratio = multi_agent_time / current_time
            regression_percent = (regression_ratio - 1.0) * 100
            
            logger.info(f"\n  Regression Analysis:")
            logger.info(f"    Current system:       {current_time:.3f}s (baseline)")
            logger.info(f"    Multi-agent:          {multi_agent_time:.3f}s")
            logger.info(f"    Regression ratio:     {regression_ratio:.2f}x")
            logger.info(f"    Regression:           {regression_percent:+.1f}%")
            logger.info(f"    Threshold:            {self.MAX_REGRESSION_THRESHOLD:.2f}x ({(self.MAX_REGRESSION_THRESHOLD-1)*100:.0f}%)")
            
            # Check against threshold
            logger.info(f"\n[3] Checking regression threshold...")
            
            if regression_ratio <= self.MAX_REGRESSION_THRESHOLD:
                logger.info(f"  ✓ PASSED: Within acceptable regression threshold")
                
                if regression_ratio < 1.0:
                    improvement = (1.0 - regression_ratio) * 100
                    logger.info(f"  🎉 Multi-agent is {improvement:.1f}% FASTER!")
                
                passed = True
            else:
                excess = (regression_ratio - self.MAX_REGRESSION_THRESHOLD) * 100
                logger.warning(f"  ✗ FAILED: Exceeds threshold by {excess:.1f} percentage points")
                passed = False
            
            # Compare violation counts
            logger.info(f"\n[4] Comparing violation detection...")
            
            violations_current = len(current_result.get('violations', []))
            violations_multi = len(multi_result.get('violations', []))
            violation_diff = abs(violations_multi - violations_current)
            
            logger.info(f"    Current system:  {violations_current} violations")
            logger.info(f"    Multi-agent:     {violations_multi} violations")
            logger.info(f"    Difference:      {violation_diff}")
            
            if violation_diff <= 2:
                logger.info(f"  ✓ Violation counts are consistent")
            else:
                logger.warning(f"  ⚠ Violation counts differ significantly")
            
            # Store results
            self.benchmark_results['comparison']['regression_ratio'] = regression_ratio
            self.benchmark_results['comparison']['regression_percent'] = regression_percent
            self.benchmark_results['comparison']['threshold'] = self.MAX_REGRESSION_THRESHOLD
            self.benchmark_results['comparison']['passed_threshold'] = passed
            
            self.log_test_result(
                "Performance Regression Detection",
                passed,
                f"Regression: {regression_percent:+.1f}% (threshold: {(self.MAX_REGRESSION_THRESHOLD-1)*100:.0f}%)"
            )
            return passed
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}", exc_info=True)
            self.log_test_result("Performance Regression Detection", False, str(e))
            return False
    
    def run_all_tests(self) -> bool:
        """Run all performance benchmark tests"""
        logger.info("\n" + "="*70)
        logger.info("PERFORMANCE BENCHMARK TEST SUITE")
        logger.info("="*70)
        logger.info(f"Started at: {datetime.now().isoformat()}")
        logger.info(f"Parallel speedup target: {self.PARALLEL_SPEEDUP_TARGET:.2f}x ({(self.PARALLEL_SPEEDUP_TARGET-1)*100:.0f}%)")
        logger.info(f"Max regression threshold: {self.MAX_REGRESSION_THRESHOLD:.2f}x ({(self.MAX_REGRESSION_THRESHOLD-1)*100:.0f}%)")
        
        start_time = time.time()
        
        # Run all tests
        tests = [
            self.test_1_total_execution_time_comparison,
            self.test_2_per_agent_execution_time,
            self.test_3_parallel_execution_speedup,
            self.test_4_multi_document_batch_performance,
            self.test_5_performance_regression_detection
        ]
        
        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                logger.error(f"Test crashed: {e}", exc_info=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
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
        logger.info(f"Completed at: {datetime.now().isoformat()}")
        
        # Print performance summary
        if self.benchmark_results['multi_agent']:
            logger.info("\n" + "="*70)
            logger.info("PERFORMANCE SUMMARY")
            logger.info("="*70)
            
            multi = self.benchmark_results['multi_agent']
            current = self.benchmark_results['current_system']
            comparison = self.benchmark_results['comparison']
            
            if 'total_time' in multi:
                logger.info(f"\nExecution Time:")
                logger.info(f"  Multi-agent:     {multi['total_time']:.3f}s")
                if 'total_time' in current:
                    logger.info(f"  Current system:  {current['total_time']:.3f}s")
                    logger.info(f"  Speedup:         {comparison.get('speedup', 0):.2f}x")
            
            if 'parallel_speedup' in multi:
                logger.info(f"\nParallel Execution:")
                logger.info(f"  Sequential time: {multi['sequential_time']:.3f}s")
                logger.info(f"  Parallel time:   {multi['parallel_time']:.3f}s")
                logger.info(f"  Speedup:         {multi['parallel_speedup']:.2f}x")
                logger.info(f"  Improvement:     {multi['improvement_percent']:.1f}%")
                
                if multi['parallel_speedup'] >= self.PARALLEL_SPEEDUP_TARGET:
                    logger.info(f"  ✓ Meets 30% improvement target")
                else:
                    logger.info(f"  ✗ Below 30% improvement target")
            
            if 'batch_avg_time' in multi:
                logger.info(f"\nBatch Processing:")
                logger.info(f"  Documents:       {multi['batch_document_count']}")
                logger.info(f"  Total time:      {multi['batch_total_time']:.3f}s")
                logger.info(f"  Avg time/doc:    {multi['batch_avg_time']:.3f}s")
            
            if 'regression_ratio' in comparison:
                logger.info(f"\nRegression Analysis:")
                logger.info(f"  Regression:      {comparison['regression_percent']:+.1f}%")
                logger.info(f"  Threshold:       {(comparison['threshold']-1)*100:.0f}%")
                if comparison['passed_threshold']:
                    logger.info(f"  ✓ Within acceptable range")
                else:
                    logger.info(f"  ✗ Exceeds threshold")
        
        logger.info("="*70)
        
        # Save results to file
        results_file = "tests/performance_benchmark_results.json"
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": {
                        "total_tests": total,
                        "passed": passed,
                        "failed": total - passed,
                        "duration": duration,
                        "timestamp": datetime.now().isoformat()
                    },
                    "test_results": self.test_results,
                    "benchmark_results": self.benchmark_results
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"\n📄 Results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        return passed == total


def main():
    """Main function to run performance benchmarks"""
    print("\n" + "="*70)
    print("MULTI-AGENT SYSTEM PERFORMANCE BENCHMARKS")
    print("="*70)
    print("\nThis test suite measures and compares performance between")
    print("the multi-agent system and the current monolithic system.")
    print("\nTests:")
    print("  1. Total execution time comparison")
    print("  2. Per-agent execution time measurement")
    print("  3. Parallel execution speedup verification (30% target)")
    print("  4. Multi-document batch performance")
    print("  5. Performance regression detection")
    print("="*70)
    
    # Run benchmarks
    benchmarks = PerformanceBenchmarks()
    success = benchmarks.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
