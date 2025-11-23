#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalability Tests for Multi-Agent System

This test suite measures scalability characteristics of the multi-agent system:
1. Test with 10, 50, 100 documents
2. Measure memory usage during execution
3. Measure CPU usage during execution
4. Test concurrent workflow executions
5. Identify performance bottlenecks

Requirements: 3.5, 14.3
"""

import logging
import sys
import os
import json
import time
import threading
import psutil
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resources during test execution"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        self.samples = []
        self.interval = 0.5  # Sample every 0.5 seconds
        
    def start(self):
        """Start monitoring resources"""
        self.monitoring = True
        self.samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
        
    def stop(self):
        """Stop monitoring resources"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info(f"Resource monitoring stopped ({len(self.samples)} samples collected)")
        
    def _monitor_loop(self):
        """Monitoring loop that runs in background thread"""
        while self.monitoring:
            try:
                # Get memory info
                mem_info = self.process.memory_info()
                mem_percent = self.process.memory_percent()
                
                # Get CPU info
                cpu_percent = self.process.cpu_percent(interval=None)
                
                # Get thread count
                num_threads = self.process.num_threads()
                
                sample = {
                    'timestamp': time.time(),
                    'memory_rss_mb': mem_info.rss / (1024 * 1024),  # RSS in MB
                    'memory_vms_mb': mem_info.vms / (1024 * 1024),  # VMS in MB
                    'memory_percent': mem_percent,
                    'cpu_percent': cpu_percent,
                    'num_threads': num_threads
                }
                
                self.samples.append(sample)
                
            except Exception as e:
                logger.warning(f"Error collecting resource sample: {e}")
            
            time.sleep(self.interval)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics from collected samples"""
        if not self.samples:
            return {}
        
        memory_rss = [s['memory_rss_mb'] for s in self.samples]
        memory_vms = [s['memory_vms_mb'] for s in self.samples]
        memory_percent = [s['memory_percent'] for s in self.samples]
        cpu_percent = [s['cpu_percent'] for s in self.samples]
        num_threads = [s['num_threads'] for s in self.samples]
        
        return {
            'memory_rss_mb': {
                'min': min(memory_rss),
                'max': max(memory_rss),
                'avg': statistics.mean(memory_rss),
                'median': statistics.median(memory_rss),
                'stdev': statistics.stdev(memory_rss) if len(memory_rss) > 1 else 0
            },
            'memory_vms_mb': {
                'min': min(memory_vms),
                'max': max(memory_vms),
                'avg': statistics.mean(memory_vms),
                'median': statistics.median(memory_vms)
            },
            'memory_percent': {
                'min': min(memory_percent),
                'max': max(memory_percent),
                'avg': statistics.mean(memory_percent),
                'median': statistics.median(memory_percent)
            },
            'cpu_percent': {
                'min': min(cpu_percent),
                'max': max(cpu_percent),
                'avg': statistics.mean(cpu_percent),
                'median': statistics.median(cpu_percent)
            },
            'num_threads': {
                'min': min(num_threads),
                'max': max(num_threads),
                'avg': statistics.mean(num_threads),
                'median': statistics.median(num_threads)
            },
            'sample_count': len(self.samples),
            'duration': self.samples[-1]['timestamp'] - self.samples[0]['timestamp'] if len(self.samples) > 1 else 0
        }


class ScalabilityTests:
    """Scalability test suite"""
    
    def __init__(self):
        self.test_results = []
        self.scalability_results = {}
        
        # Scalability thresholds
        self.MAX_MEMORY_PER_DOC_MB = 50  # Max 50MB per document
        self.MAX_MEMORY_GROWTH_PERCENT = 20  # Max 20% memory growth
        self.MAX_CPU_PERCENT = 90  # Max 90% CPU usage
        
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
    
    def create_test_document(self, doc_id: int, complexity: str = "medium") -> Dict[str, Any]:
        """
        Create test documents with varying complexity
        
        Args:
            doc_id: Document identifier
            complexity: "simple", "medium", or "complex"
            
        Returns:
            Test document dictionary
        """
        base_document = {
            'document_metadata': {
                'fund_isin': f'FR00101351{doc_id:02d}',
                'fund_name': f'Scalability Test Fund {doc_id}',
                'client_type': 'retail',
                'document_type': 'fund_presentation',
                'fund_esg_classification': 'article_8',
                'fund_age_years': 5,
                'document_date': '2024-01-15',
                'fund_inception_date': '2020-01-15'
            },
            'page_de_garde': {
                'title': f'Fund Presentation {doc_id}',
                'subtitle': f'Scalability Test Fund {doc_id}',
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
            # Medium content - 10 slides
            base_document['page_de_garde']['content'] = 'Document promotionnel'
            base_document['slide_2']['content'] = 'Performance: +15.5% en 2023.'
            
            for i in range(3, 11):
                base_document['pages_suivantes'].append({
                    'slide_number': i,
                    'title': f'Section {i}',
                    'content': f'Content for slide {i} with investment details.'
                })
        
        elif complexity == "complex":
            # Complex content - 20 slides
            base_document['page_de_garde']['content'] = 'Promotional document'
            base_document['slide_2']['content'] = 'Performance: +20% en 2023. Morningstar rating: ★★★★★.'
            
            for i in range(3, 21):
                content_parts = [
                    f'Slide {i} content with detailed analysis.',
                    'Performance data: +12.3% YTD.',
                    'Technical indicators: Sharpe ratio 1.5.',
                    'Investment recommendation for portfolios.',
                    'ESG score: AAA rating from MSCI.'
                ]
                base_document['pages_suivantes'].append({
                    'slide_number': i,
                    'title': f'Detailed Section {i}',
                    'content': ' '.join(content_parts)
                })
        
        return base_document
    
    def process_single_document(
        self,
        workflow: Any,
        document: Dict[str, Any],
        document_id: str,
        config: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Process a single document and return timing
        
        Args:
            workflow: LangGraph workflow
            document: Document to process
            document_id: Document identifier
            config: Configuration dictionary
            
        Returns:
            Tuple of (execution_time, result)
        """
        try:
            from data_models_multiagent import initialize_compliance_state
            
            state = initialize_compliance_state(
                document=document,
                document_id=document_id,
                config=config
            )
            
            start_time = time.time()
            result = workflow.invoke(state)
            execution_time = time.time() - start_time
            
            return execution_time, result
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            return 0.0, {}

    def test_1_scale_to_10_documents(self) -> bool:
        """
        Test 1: Test with 10 documents
        
        Verifies:
        - System can process 10 documents successfully
        - Memory usage is reasonable
        - CPU usage is reasonable
        - Performance is consistent
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 1: Scale to 10 Documents")
        logger.info("="*70)
        
        try:
            from workflow_builder import create_compliance_workflow
            
            num_documents = 10
            
            logger.info(f"\n[1] Creating {num_documents} test documents...")
            
            documents = []
            for i in range(num_documents):
                doc = self.create_test_document(i, "medium")
                documents.append((f"scale_10_doc_{i}", doc))
            
            logger.info(f"  ✓ Created {len(documents)} documents")
            
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
                enable_checkpointing=False
            )
            
            logger.info(f"  ✓ Workflow created")
            
            # Start resource monitoring
            logger.info(f"\n[2] Processing {num_documents} documents with resource monitoring...")
            
            monitor = ResourceMonitor()
            monitor.start()
            
            start_time = time.time()
            
            execution_times = []
            total_violations = 0
            
            for doc_id, document in documents:
                exec_time, result = self.process_single_document(
                    workflow, document, doc_id, config
                )
                execution_times.append(exec_time)
                total_violations += len(result.get('violations', []))
                
                logger.info(f"  Processed {doc_id}: {exec_time:.3f}s, {len(result.get('violations', []))} violations")
            
            total_time = time.time() - start_time
            
            monitor.stop()
            
            # Analyze results
            logger.info(f"\n[3] Analyzing results...")
            
            avg_time = statistics.mean(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            stdev_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            
            logger.info(f"\n  Execution Time Statistics:")
            logger.info(f"    Total time:       {total_time:.3f}s")
            logger.info(f"    Average per doc:  {avg_time:.3f}s")
            logger.info(f"    Min time:         {min_time:.3f}s")
            logger.info(f"    Max time:         {max_time:.3f}s")
            logger.info(f"    Std deviation:    {stdev_time:.3f}s")
            logger.info(f"    Total violations: {total_violations}")
            
            # Analyze resource usage
            resource_stats = monitor.get_statistics()
            
            if resource_stats:
                logger.info(f"\n  Resource Usage Statistics:")
                logger.info(f"    Memory RSS (MB):")
                logger.info(f"      Min:     {resource_stats['memory_rss_mb']['min']:.1f}")
                logger.info(f"      Max:     {resource_stats['memory_rss_mb']['max']:.1f}")
                logger.info(f"      Average: {resource_stats['memory_rss_mb']['avg']:.1f}")
                logger.info(f"      Median:  {resource_stats['memory_rss_mb']['median']:.1f}")
                
                logger.info(f"    Memory Percent:")
                logger.info(f"      Min:     {resource_stats['memory_percent']['min']:.1f}%")
                logger.info(f"      Max:     {resource_stats['memory_percent']['max']:.1f}%")
                logger.info(f"      Average: {resource_stats['memory_percent']['avg']:.1f}%")
                
                logger.info(f"    CPU Percent:")
                logger.info(f"      Min:     {resource_stats['cpu_percent']['min']:.1f}%")
                logger.info(f"      Max:     {resource_stats['cpu_percent']['max']:.1f}%")
                logger.info(f"      Average: {resource_stats['cpu_percent']['avg']:.1f}%")
                
                logger.info(f"    Threads:")
                logger.info(f"      Min:     {resource_stats['num_threads']['min']:.0f}")
                logger.info(f"      Max:     {resource_stats['num_threads']['max']:.0f}")
                logger.info(f"      Average: {resource_stats['num_threads']['avg']:.1f}")
            
            # Store results
            self.scalability_results['10_documents'] = {
                'num_documents': num_documents,
                'total_time': total_time,
                'avg_time': avg_time,
                'execution_times': execution_times,
                'total_violations': total_violations,
                'resource_stats': resource_stats
            }
            
            # Check if performance is acceptable
            passed = True
            if avg_time > 10.0:
                logger.warning(f"  ⚠ Average time per document is high: {avg_time:.3f}s")
                passed = False
            
            if resource_stats and resource_stats['memory_rss_mb']['max'] > 1000:
                logger.warning(f"  ⚠ Peak memory usage is high: {resource_stats['memory_rss_mb']['max']:.1f}MB")
                passed = False
            
            self.log_test_result(
                "Scale to 10 Documents",
                passed,
                f"{num_documents} docs in {total_time:.3f}s (avg: {avg_time:.3f}s/doc)"
            )
            return passed
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}", exc_info=True)
            self.log_test_result("Scale to 10 Documents", False, str(e))
            return False
    
    def test_2_scale_to_50_documents(self) -> bool:
        """
        Test 2: Test with 50 documents
        
        Verifies:
        - System can handle larger batch of 50 documents
        - Memory usage scales linearly
        - No memory leaks
        - Performance remains consistent
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 2: Scale to 50 Documents")
        logger.info("="*70)
        
        try:
            from workflow_builder import create_compliance_workflow
            
            num_documents = 50
            
            logger.info(f"\n[1] Creating {num_documents} test documents...")
            
            documents = []
            for i in range(num_documents):
                # Mix of complexities
                complexity = "simple" if i % 3 == 0 else "medium"
                doc = self.create_test_document(i, complexity)
                documents.append((f"scale_50_doc_{i}", doc))
            
            logger.info(f"  ✓ Created {len(documents)} documents")
            
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
                enable_checkpointing=False
            )
            
            logger.info(f"  ✓ Workflow created")
            
            # Start resource monitoring
            logger.info(f"\n[2] Processing {num_documents} documents...")
            
            monitor = ResourceMonitor()
            monitor.start()
            
            start_time = time.time()
            
            execution_times = []
            total_violations = 0
            
            # Process in batches to show progress
            batch_size = 10
            for batch_start in range(0, num_documents, batch_size):
                batch_end = min(batch_start + batch_size, num_documents)
                logger.info(f"  Processing batch {batch_start//batch_size + 1} (docs {batch_start+1}-{batch_end})...")
                
                for i in range(batch_start, batch_end):
                    doc_id, document = documents[i]
                    exec_time, result = self.process_single_document(
                        workflow, document, doc_id, config
                    )
                    execution_times.append(exec_time)
                    total_violations += len(result.get('violations', []))
            
            total_time = time.time() - start_time
            
            monitor.stop()
            
            # Analyze results
            logger.info(f"\n[3] Analyzing results...")
            
            avg_time = statistics.mean(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            stdev_time = statistics.stdev(execution_times)
            
            logger.info(f"\n  Execution Time Statistics:")
            logger.info(f"    Total time:       {total_time:.3f}s ({total_time/60:.1f} minutes)")
            logger.info(f"    Average per doc:  {avg_time:.3f}s")
            logger.info(f"    Min time:         {min_time:.3f}s")
            logger.info(f"    Max time:         {max_time:.3f}s")
            logger.info(f"    Std deviation:    {stdev_time:.3f}s")
            logger.info(f"    Throughput:       {num_documents/total_time:.2f} docs/sec")
            logger.info(f"    Total violations: {total_violations}")
            
            # Analyze resource usage
            resource_stats = monitor.get_statistics()
            
            if resource_stats:
                logger.info(f"\n  Resource Usage Statistics:")
                logger.info(f"    Memory RSS (MB):")
                logger.info(f"      Min:     {resource_stats['memory_rss_mb']['min']:.1f}")
                logger.info(f"      Max:     {resource_stats['memory_rss_mb']['max']:.1f}")
                logger.info(f"      Average: {resource_stats['memory_rss_mb']['avg']:.1f}")
                logger.info(f"      Growth:  {resource_stats['memory_rss_mb']['max'] - resource_stats['memory_rss_mb']['min']:.1f}MB")
                
                logger.info(f"    CPU Percent:")
                logger.info(f"      Average: {resource_stats['cpu_percent']['avg']:.1f}%")
                logger.info(f"      Max:     {resource_stats['cpu_percent']['max']:.1f}%")
            
            # Check for memory leaks
            logger.info(f"\n[4] Checking for memory leaks...")
            
            # Compare first 10 and last 10 documents
            first_10_avg = statistics.mean(execution_times[:10])
            last_10_avg = statistics.mean(execution_times[-10:])
            degradation = (last_10_avg - first_10_avg) / first_10_avg * 100
            
            logger.info(f"    First 10 docs avg: {first_10_avg:.3f}s")
            logger.info(f"    Last 10 docs avg:  {last_10_avg:.3f}s")
            logger.info(f"    Degradation:       {degradation:+.1f}%")
            
            if abs(degradation) < 15:
                logger.info(f"  ✓ No significant performance degradation")
                no_leak = True
            else:
                logger.warning(f"  ⚠ Possible memory leak or performance degradation")
                no_leak = False
            
            # Store results
            self.scalability_results['50_documents'] = {
                'num_documents': num_documents,
                'total_time': total_time,
                'avg_time': avg_time,
                'execution_times': execution_times,
                'total_violations': total_violations,
                'resource_stats': resource_stats,
                'degradation_percent': degradation
            }
            
            # Check if performance is acceptable
            passed = no_leak
            if resource_stats and resource_stats['memory_rss_mb']['max'] > 2000:
                logger.warning(f"  ⚠ Peak memory usage is high: {resource_stats['memory_rss_mb']['max']:.1f}MB")
                passed = False
            
            self.log_test_result(
                "Scale to 50 Documents",
                passed,
                f"{num_documents} docs in {total_time:.1f}s (avg: {avg_time:.3f}s/doc, degradation: {degradation:+.1f}%)"
            )
            return passed
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}", exc_info=True)
            self.log_test_result("Scale to 50 Documents", False, str(e))
            return False
    
    def test_3_scale_to_100_documents(self) -> bool:
        """
        Test 3: Test with 100 documents
        
        Verifies:
        - System can handle large batch of 100 documents
        - Memory usage remains bounded
        - Performance is acceptable
        - System is stable under load
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 3: Scale to 100 Documents")
        logger.info("="*70)
        
        try:
            from workflow_builder import create_compliance_workflow
            
            num_documents = 100
            
            logger.info(f"\n[1] Creating {num_documents} test documents...")
            
            documents = []
            for i in range(num_documents):
                # Mix of complexities: 40% simple, 50% medium, 10% complex
                if i % 10 < 4:
                    complexity = "simple"
                elif i % 10 < 9:
                    complexity = "medium"
                else:
                    complexity = "complex"
                
                doc = self.create_test_document(i, complexity)
                documents.append((f"scale_100_doc_{i}", doc))
            
            logger.info(f"  ✓ Created {len(documents)} documents")
            
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
                enable_checkpointing=False
            )
            
            logger.info(f"  ✓ Workflow created")
            
            # Start resource monitoring
            logger.info(f"\n[2] Processing {num_documents} documents...")
            logger.info(f"  This may take several minutes...")
            
            monitor = ResourceMonitor()
            monitor.start()
            
            start_time = time.time()
            
            execution_times = []
            total_violations = 0
            
            # Process in batches to show progress
            batch_size = 20
            for batch_start in range(0, num_documents, batch_size):
                batch_end = min(batch_start + batch_size, num_documents)
                batch_start_time = time.time()
                
                logger.info(f"  Processing batch {batch_start//batch_size + 1}/5 (docs {batch_start+1}-{batch_end})...")
                
                for i in range(batch_start, batch_end):
                    doc_id, document = documents[i]
                    exec_time, result = self.process_single_document(
                        workflow, document, doc_id, config
                    )
                    execution_times.append(exec_time)
                    total_violations += len(result.get('violations', []))
                
                batch_time = time.time() - batch_start_time
                logger.info(f"    Batch completed in {batch_time:.1f}s")
            
            total_time = time.time() - start_time
            
            monitor.stop()
            
            # Analyze results
            logger.info(f"\n[3] Analyzing results...")
            
            avg_time = statistics.mean(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            stdev_time = statistics.stdev(execution_times)
            median_time = statistics.median(execution_times)
            
            logger.info(f"\n  Execution Time Statistics:")
            logger.info(f"    Total time:       {total_time:.3f}s ({total_time/60:.1f} minutes)")
            logger.info(f"    Average per doc:  {avg_time:.3f}s")
            logger.info(f"    Median per doc:   {median_time:.3f}s")
            logger.info(f"    Min time:         {min_time:.3f}s")
            logger.info(f"    Max time:         {max_time:.3f}s")
            logger.info(f"    Std deviation:    {stdev_time:.3f}s")
            logger.info(f"    Throughput:       {num_documents/total_time:.2f} docs/sec")
            logger.info(f"    Total violations: {total_violations}")
            
            # Analyze resource usage
            resource_stats = monitor.get_statistics()
            
            if resource_stats:
                logger.info(f"\n  Resource Usage Statistics:")
                logger.info(f"    Memory RSS (MB):")
                logger.info(f"      Min:     {resource_stats['memory_rss_mb']['min']:.1f}")
                logger.info(f"      Max:     {resource_stats['memory_rss_mb']['max']:.1f}")
                logger.info(f"      Average: {resource_stats['memory_rss_mb']['avg']:.1f}")
                logger.info(f"      Growth:  {resource_stats['memory_rss_mb']['max'] - resource_stats['memory_rss_mb']['min']:.1f}MB")
                logger.info(f"      Per doc: {(resource_stats['memory_rss_mb']['max'] - resource_stats['memory_rss_mb']['min'])/num_documents:.2f}MB/doc")
                
                logger.info(f"    CPU Percent:")
                logger.info(f"      Average: {resource_stats['cpu_percent']['avg']:.1f}%")
                logger.info(f"      Max:     {resource_stats['cpu_percent']['max']:.1f}%")
            
            # Check for memory leaks and degradation
            logger.info(f"\n[4] Checking for memory leaks and performance degradation...")
            
            # Compare first 20 and last 20 documents
            first_20_avg = statistics.mean(execution_times[:20])
            last_20_avg = statistics.mean(execution_times[-20:])
            degradation = (last_20_avg - first_20_avg) / first_20_avg * 100
            
            logger.info(f"    First 20 docs avg: {first_20_avg:.3f}s")
            logger.info(f"    Last 20 docs avg:  {last_20_avg:.3f}s")
            logger.info(f"    Degradation:       {degradation:+.1f}%")
            
            if abs(degradation) < 15:
                logger.info(f"  ✓ No significant performance degradation")
                no_leak = True
            else:
                logger.warning(f"  ⚠ Possible memory leak or performance degradation")
                no_leak = False
            
            # Store results
            self.scalability_results['100_documents'] = {
                'num_documents': num_documents,
                'total_time': total_time,
                'avg_time': avg_time,
                'median_time': median_time,
                'execution_times': execution_times,
                'total_violations': total_violations,
                'resource_stats': resource_stats,
                'degradation_percent': degradation
            }
            
            # Check if performance is acceptable
            passed = no_leak
            if resource_stats:
                memory_per_doc = (resource_stats['memory_rss_mb']['max'] - resource_stats['memory_rss_mb']['min']) / num_documents
                if memory_per_doc > self.MAX_MEMORY_PER_DOC_MB:
                    logger.warning(f"  ⚠ Memory per document is high: {memory_per_doc:.2f}MB/doc")
                    passed = False
                
                if resource_stats['memory_rss_mb']['max'] > 3000:
                    logger.warning(f"  ⚠ Peak memory usage is high: {resource_stats['memory_rss_mb']['max']:.1f}MB")
                    passed = False
            
            self.log_test_result(
                "Scale to 100 Documents",
                passed,
                f"{num_documents} docs in {total_time/60:.1f}min (avg: {avg_time:.3f}s/doc, degradation: {degradation:+.1f}%)"
            )
            return passed
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}", exc_info=True)
            self.log_test_result("Scale to 100 Documents", False, str(e))
            return False

    def test_4_concurrent_workflow_executions(self) -> bool:
        """
        Test 4: Test concurrent workflow executions
        
        Verifies:
        - Multiple workflows can run concurrently
        - Thread safety is maintained
        - Resource usage is reasonable under concurrent load
        - No race conditions or deadlocks
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 4: Concurrent Workflow Executions")
        logger.info("="*70)
        
        try:
            from workflow_builder import create_compliance_workflow
            
            num_concurrent = 5
            docs_per_workflow = 3
            total_documents = num_concurrent * docs_per_workflow
            
            logger.info(f"\n[1] Setting up {num_concurrent} concurrent workflows...")
            logger.info(f"  Each workflow will process {docs_per_workflow} documents")
            logger.info(f"  Total documents: {total_documents}")
            
            # Create documents for each workflow
            workflow_documents = []
            for workflow_id in range(num_concurrent):
                docs = []
                for doc_id in range(docs_per_workflow):
                    doc = self.create_test_document(
                        workflow_id * docs_per_workflow + doc_id,
                        "medium"
                    )
                    docs.append((f"concurrent_w{workflow_id}_d{doc_id}", doc))
                workflow_documents.append(docs)
            
            logger.info(f"  ✓ Created documents for {num_concurrent} workflows")
            
            # Create config
            config = {
                "multi_agent": {
                    "enabled": True,
                    "parallel_execution": True,
                    "max_parallel_agents": 4
                }
            }
            
            def process_workflow(workflow_id: int, documents: List[Tuple[str, Dict]]) -> Dict[str, Any]:
                """Process a workflow with its documents"""
                try:
                    # Each workflow gets its own instance
                    workflow = create_compliance_workflow(
                        config=config,
                        enable_checkpointing=False
                    )
                    
                    results = []
                    for doc_id, document in documents:
                        exec_time, result = self.process_single_document(
                            workflow, document, doc_id, config
                        )
                        results.append({
                            'doc_id': doc_id,
                            'exec_time': exec_time,
                            'violations': len(result.get('violations', []))
                        })
                    
                    return {
                        'workflow_id': workflow_id,
                        'success': True,
                        'results': results,
                        'total_time': sum(r['exec_time'] for r in results)
                    }
                    
                except Exception as e:
                    logger.error(f"Workflow {workflow_id} failed: {e}")
                    return {
                        'workflow_id': workflow_id,
                        'success': False,
                        'error': str(e)
                    }
            
            # Start resource monitoring
            logger.info(f"\n[2] Executing {num_concurrent} workflows concurrently...")
            
            monitor = ResourceMonitor()
            monitor.start()
            
            start_time = time.time()
            
            # Execute workflows concurrently using ThreadPoolExecutor
            workflow_results = []
            with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                futures = []
                for workflow_id, documents in enumerate(workflow_documents):
                    future = executor.submit(process_workflow, workflow_id, documents)
                    futures.append(future)
                
                # Collect results as they complete
                for future in as_completed(futures):
                    result = future.result()
                    workflow_results.append(result)
                    if result['success']:
                        logger.info(f"  Workflow {result['workflow_id']} completed: {result['total_time']:.3f}s")
                    else:
                        logger.error(f"  Workflow {result['workflow_id']} failed: {result.get('error')}")
            
            total_time = time.time() - start_time
            
            monitor.stop()
            
            # Analyze results
            logger.info(f"\n[3] Analyzing concurrent execution results...")
            
            successful_workflows = [r for r in workflow_results if r['success']]
            failed_workflows = [r for r in workflow_results if not r['success']]
            
            logger.info(f"\n  Workflow Execution:")
            logger.info(f"    Total workflows:     {num_concurrent}")
            logger.info(f"    Successful:          {len(successful_workflows)}")
            logger.info(f"    Failed:              {len(failed_workflows)}")
            logger.info(f"    Total time:          {total_time:.3f}s")
            
            if successful_workflows:
                workflow_times = [r['total_time'] for r in successful_workflows]
                avg_workflow_time = statistics.mean(workflow_times)
                
                logger.info(f"    Avg workflow time:   {avg_workflow_time:.3f}s")
                logger.info(f"    Min workflow time:   {min(workflow_times):.3f}s")
                logger.info(f"    Max workflow time:   {max(workflow_times):.3f}s")
                
                # Calculate total documents processed
                total_docs_processed = sum(len(r['results']) for r in successful_workflows)
                total_violations = sum(sum(d['violations'] for d in r['results']) for r in successful_workflows)
                
                logger.info(f"    Documents processed: {total_docs_processed}")
                logger.info(f"    Total violations:    {total_violations}")
                logger.info(f"    Throughput:          {total_docs_processed/total_time:.2f} docs/sec")
            
            # Analyze resource usage
            resource_stats = monitor.get_statistics()
            
            if resource_stats:
                logger.info(f"\n  Resource Usage (Concurrent):")
                logger.info(f"    Memory RSS (MB):")
                logger.info(f"      Max:     {resource_stats['memory_rss_mb']['max']:.1f}")
                logger.info(f"      Average: {resource_stats['memory_rss_mb']['avg']:.1f}")
                
                logger.info(f"    CPU Percent:")
                logger.info(f"      Max:     {resource_stats['cpu_percent']['max']:.1f}%")
                logger.info(f"      Average: {resource_stats['cpu_percent']['avg']:.1f}%")
                
                logger.info(f"    Threads:")
                logger.info(f"      Max:     {resource_stats['num_threads']['max']:.0f}")
                logger.info(f"      Average: {resource_stats['num_threads']['avg']:.1f}")
            
            # Store results
            self.scalability_results['concurrent_execution'] = {
                'num_concurrent': num_concurrent,
                'docs_per_workflow': docs_per_workflow,
                'total_time': total_time,
                'successful_workflows': len(successful_workflows),
                'failed_workflows': len(failed_workflows),
                'workflow_results': workflow_results,
                'resource_stats': resource_stats
            }
            
            # Check if all workflows succeeded
            passed = len(failed_workflows) == 0
            
            if not passed:
                logger.warning(f"  ⚠ {len(failed_workflows)} workflow(s) failed")
            
            # Check resource usage
            if resource_stats:
                if resource_stats['cpu_percent']['max'] > self.MAX_CPU_PERCENT:
                    logger.warning(f"  ⚠ Peak CPU usage is high: {resource_stats['cpu_percent']['max']:.1f}%")
                
                if resource_stats['memory_rss_mb']['max'] > 2000:
                    logger.warning(f"  ⚠ Peak memory usage is high: {resource_stats['memory_rss_mb']['max']:.1f}MB")
            
            self.log_test_result(
                "Concurrent Workflow Executions",
                passed,
                f"{num_concurrent} workflows, {len(successful_workflows)} succeeded, {total_time:.3f}s"
            )
            return passed
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}", exc_info=True)
            self.log_test_result("Concurrent Workflow Executions", False, str(e))
            return False
    
    def test_5_identify_bottlenecks(self) -> bool:
        """
        Test 5: Identify performance bottlenecks
        
        Verifies:
        - Bottlenecks are identified and documented
        - Resource usage patterns are analyzed
        - Performance characteristics are understood
        - Optimization opportunities are identified
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 5: Identify Performance Bottlenecks")
        logger.info("="*70)
        
        try:
            logger.info(f"\n[1] Analyzing scalability test results...")
            
            if not self.scalability_results:
                logger.warning("  ⚠ No scalability results available")
                self.log_test_result(
                    "Identify Performance Bottlenecks",
                    False,
                    "No test results to analyze"
                )
                return False
            
            bottlenecks = []
            recommendations = []
            
            # Analyze memory usage scaling
            logger.info(f"\n[2] Analyzing memory usage scaling...")
            
            if '10_documents' in self.scalability_results and '50_documents' in self.scalability_results:
                mem_10 = self.scalability_results['10_documents']['resource_stats']['memory_rss_mb']['max']
                mem_50 = self.scalability_results['50_documents']['resource_stats']['memory_rss_mb']['max']
                
                mem_growth = mem_50 - mem_10
                mem_growth_per_doc = mem_growth / 40  # 40 additional documents
                
                logger.info(f"    Memory at 10 docs:  {mem_10:.1f}MB")
                logger.info(f"    Memory at 50 docs:  {mem_50:.1f}MB")
                logger.info(f"    Growth:             {mem_growth:.1f}MB")
                logger.info(f"    Growth per doc:     {mem_growth_per_doc:.2f}MB/doc")
                
                if mem_growth_per_doc > self.MAX_MEMORY_PER_DOC_MB:
                    bottlenecks.append({
                        'type': 'memory_scaling',
                        'severity': 'HIGH',
                        'description': f'Memory grows by {mem_growth_per_doc:.2f}MB per document (threshold: {self.MAX_MEMORY_PER_DOC_MB}MB)',
                        'impact': 'System may run out of memory with large document batches'
                    })
                    recommendations.append('Implement document streaming or batch processing with memory cleanup')
                    recommendations.append('Review state management for memory leaks')
                else:
                    logger.info(f"  ✓ Memory scaling is acceptable")
            
            # Analyze performance degradation
            logger.info(f"\n[3] Analyzing performance degradation...")
            
            for test_name in ['50_documents', '100_documents']:
                if test_name in self.scalability_results:
                    degradation = self.scalability_results[test_name].get('degradation_percent', 0)
                    
                    logger.info(f"    {test_name}: {degradation:+.1f}% degradation")
                    
                    if abs(degradation) > self.MAX_MEMORY_GROWTH_PERCENT:
                        bottlenecks.append({
                            'type': 'performance_degradation',
                            'severity': 'MEDIUM',
                            'description': f'{test_name} shows {degradation:+.1f}% performance degradation',
                            'impact': 'Processing slows down over time'
                        })
                        recommendations.append('Investigate memory leaks or resource accumulation')
                        recommendations.append('Consider periodic garbage collection')
            
            # Analyze CPU usage
            logger.info(f"\n[4] Analyzing CPU usage patterns...")
            
            if 'concurrent_execution' in self.scalability_results:
                cpu_stats = self.scalability_results['concurrent_execution']['resource_stats']['cpu_percent']
                
                logger.info(f"    Average CPU: {cpu_stats['avg']:.1f}%")
                logger.info(f"    Max CPU:     {cpu_stats['max']:.1f}%")
                
                if cpu_stats['avg'] < 50:
                    logger.info(f"  ℹ CPU utilization is low - system may be I/O bound")
                    bottlenecks.append({
                        'type': 'cpu_underutilization',
                        'severity': 'LOW',
                        'description': f'Average CPU usage is only {cpu_stats["avg"]:.1f}%',
                        'impact': 'System may be waiting on I/O or external services'
                    })
                    recommendations.append('Profile I/O operations and API calls')
                    recommendations.append('Consider increasing parallelism if I/O bound')
                elif cpu_stats['max'] > self.MAX_CPU_PERCENT:
                    logger.info(f"  ⚠ Peak CPU usage is high")
                    bottlenecks.append({
                        'type': 'cpu_saturation',
                        'severity': 'MEDIUM',
                        'description': f'Peak CPU usage reaches {cpu_stats["max"]:.1f}%',
                        'impact': 'System may become unresponsive under load'
                    })
                    recommendations.append('Optimize CPU-intensive operations')
                    recommendations.append('Consider rate limiting or load balancing')
                else:
                    logger.info(f"  ✓ CPU usage is balanced")
            
            # Analyze throughput scaling
            logger.info(f"\n[5] Analyzing throughput scaling...")
            
            throughputs = {}
            for test_name in ['10_documents', '50_documents', '100_documents']:
                if test_name in self.scalability_results:
                    num_docs = self.scalability_results[test_name]['num_documents']
                    total_time = self.scalability_results[test_name]['total_time']
                    throughput = num_docs / total_time
                    throughputs[test_name] = throughput
                    
                    logger.info(f"    {test_name}: {throughput:.2f} docs/sec")
            
            if len(throughputs) >= 2:
                throughput_values = list(throughputs.values())
                throughput_variance = max(throughput_values) / min(throughput_values)
                
                logger.info(f"    Throughput variance: {throughput_variance:.2f}x")
                
                if throughput_variance > 1.5:
                    logger.info(f"  ℹ Throughput varies significantly with batch size")
                    recommendations.append('Investigate batch size optimization')
                else:
                    logger.info(f"  ✓ Throughput is consistent across batch sizes")
            
            # Print bottleneck summary
            logger.info(f"\n[6] Bottleneck Summary...")
            
            if bottlenecks:
                logger.info(f"\n  Identified {len(bottlenecks)} bottleneck(s):")
                for i, bottleneck in enumerate(bottlenecks, 1):
                    logger.info(f"\n  {i}. [{bottleneck['severity']}] {bottleneck['type']}")
                    logger.info(f"     Description: {bottleneck['description']}")
                    logger.info(f"     Impact: {bottleneck['impact']}")
            else:
                logger.info(f"  ✓ No significant bottlenecks identified")
            
            # Print recommendations
            if recommendations:
                logger.info(f"\n  Optimization Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    logger.info(f"    {i}. {rec}")
            
            # Store results
            self.scalability_results['bottleneck_analysis'] = {
                'bottlenecks': bottlenecks,
                'recommendations': recommendations,
                'analyzed_at': datetime.now().isoformat()
            }
            
            # Test passes if no HIGH severity bottlenecks
            high_severity = [b for b in bottlenecks if b['severity'] == 'HIGH']
            passed = len(high_severity) == 0
            
            if not passed:
                logger.warning(f"  ⚠ Found {len(high_severity)} HIGH severity bottleneck(s)")
            
            self.log_test_result(
                "Identify Performance Bottlenecks",
                passed,
                f"{len(bottlenecks)} bottlenecks identified, {len(high_severity)} high severity"
            )
            return passed
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}", exc_info=True)
            self.log_test_result("Identify Performance Bottlenecks", False, str(e))
            return False
    
    def run_all_tests(self) -> bool:
        """Run all scalability tests"""
        logger.info("\n" + "="*70)
        logger.info("SCALABILITY TEST SUITE")
        logger.info("="*70)
        logger.info(f"Started at: {datetime.now().isoformat()}")
        logger.info(f"\nThresholds:")
        logger.info(f"  Max memory per doc:     {self.MAX_MEMORY_PER_DOC_MB}MB")
        logger.info(f"  Max memory growth:      {self.MAX_MEMORY_GROWTH_PERCENT}%")
        logger.info(f"  Max CPU usage:          {self.MAX_CPU_PERCENT}%")
        
        start_time = time.time()
        
        # Run all tests
        tests = [
            self.test_1_scale_to_10_documents,
            self.test_2_scale_to_50_documents,
            self.test_3_scale_to_100_documents,
            self.test_4_concurrent_workflow_executions,
            self.test_5_identify_bottlenecks
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
        logger.info(f"Duration: {duration:.2f}s ({duration/60:.1f} minutes)")
        logger.info(f"Completed at: {datetime.now().isoformat()}")
        
        # Print scalability summary
        if self.scalability_results:
            logger.info("\n" + "="*70)
            logger.info("SCALABILITY SUMMARY")
            logger.info("="*70)
            
            for test_name in ['10_documents', '50_documents', '100_documents']:
                if test_name in self.scalability_results:
                    result = self.scalability_results[test_name]
                    logger.info(f"\n{test_name.replace('_', ' ').title()}:")
                    logger.info(f"  Documents:       {result['num_documents']}")
                    logger.info(f"  Total time:      {result['total_time']:.1f}s")
                    logger.info(f"  Avg time/doc:    {result['avg_time']:.3f}s")
                    logger.info(f"  Throughput:      {result['num_documents']/result['total_time']:.2f} docs/sec")
                    
                    if 'resource_stats' in result and result['resource_stats']:
                        stats = result['resource_stats']
                        logger.info(f"  Peak memory:     {stats['memory_rss_mb']['max']:.1f}MB")
                        logger.info(f"  Avg CPU:         {stats['cpu_percent']['avg']:.1f}%")
                    
                    if 'degradation_percent' in result:
                        logger.info(f"  Degradation:     {result['degradation_percent']:+.1f}%")
            
            if 'concurrent_execution' in self.scalability_results:
                result = self.scalability_results['concurrent_execution']
                logger.info(f"\nConcurrent Execution:")
                logger.info(f"  Workflows:       {result['num_concurrent']}")
                logger.info(f"  Successful:      {result['successful_workflows']}")
                logger.info(f"  Total time:      {result['total_time']:.1f}s")
                
                if 'resource_stats' in result and result['resource_stats']:
                    stats = result['resource_stats']
                    logger.info(f"  Peak memory:     {stats['memory_rss_mb']['max']:.1f}MB")
                    logger.info(f"  Peak CPU:        {stats['cpu_percent']['max']:.1f}%")
            
            if 'bottleneck_analysis' in self.scalability_results:
                analysis = self.scalability_results['bottleneck_analysis']
                logger.info(f"\nBottleneck Analysis:")
                logger.info(f"  Bottlenecks:     {len(analysis['bottlenecks'])}")
                logger.info(f"  Recommendations: {len(analysis['recommendations'])}")
        
        logger.info("="*70)
        
        # Save results to file
        results_file = "tests/scalability_test_results.json"
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
                    "scalability_results": self.scalability_results
                }, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"\n📄 Results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        return passed == total


def main():
    """Main function to run scalability tests"""
    print("\n" + "="*70)
    print("MULTI-AGENT SYSTEM SCALABILITY TESTS")
    print("="*70)
    print("\nThis test suite measures scalability characteristics:")
    print("  1. Test with 10 documents")
    print("  2. Test with 50 documents")
    print("  3. Test with 100 documents")
    print("  4. Test concurrent workflow executions")
    print("  5. Identify performance bottlenecks")
    print("\nNote: These tests may take several minutes to complete.")
    print("="*70)
    
    # Check if psutil is available
    try:
        import psutil
    except ImportError:
        print("\n❌ Error: psutil library is required for scalability tests")
        print("   Install it with: pip install psutil")
        sys.exit(1)
    
    # Run tests
    tests = ScalabilityTests()
    success = tests.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
