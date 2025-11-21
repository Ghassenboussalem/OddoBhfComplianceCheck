#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async Processor - Asynchronous Processing Architecture
Provides async/await support for AI calls and concurrent processing
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Coroutine
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncTaskStatus(Enum):
    """Status of async tasks"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"


@dataclass
class AsyncResult:
    """Result of an async operation"""
    task_id: str
    status: AsyncTaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def get_duration(self) -> Optional[float]:
        """Get task duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class AsyncAIEngine:
    """
    Async wrapper for AI Engine
    Converts synchronous AI calls to async operations
    """
    
    def __init__(self, ai_engine, executor: Optional[ThreadPoolExecutor] = None):
        """
        Initialize async AI engine
        
        Args:
            ai_engine: Synchronous AIEngine instance
            executor: Optional thread pool executor
        """
        self.ai_engine = ai_engine
        self.executor = executor or ThreadPoolExecutor(max_workers=10)
        logger.info("Async AI Engine initialized")
    
    async def analyze_async(self, document: Dict, check_type: str, 
                           rule_hints: Dict, **kwargs) -> Optional[Dict]:
        """
        Async version of AI analysis
        
        Args:
            document: Document to analyze
            check_type: Type of compliance check
            rule_hints: Hints from rule-based analysis
            **kwargs: Additional parameters
            
        Returns:
            Analysis result dict
        """
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.ai_engine.analyze(document, check_type, rule_hints, **kwargs)
            )
            return result
        except Exception as e:
            logger.error(f"Async AI analysis failed: {e}")
            return None
    
    async def call_with_cache_async(self, prompt: str, system_message: str = "", 
                                   **kwargs) -> Optional[Any]:
        """
        Async version of cached AI call
        
        Args:
            prompt: User prompt
            system_message: System message
            **kwargs: Additional parameters
            
        Returns:
            AI response
        """
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.ai_engine.call_with_cache(prompt, system_message, **kwargs)
            )
            return result
        except Exception as e:
            logger.error(f"Async AI call failed: {e}")
            return None
    
    async def batch_analyze_async(self, tasks: List[Dict]) -> List[Optional[Dict]]:
        """
        Process multiple AI analysis tasks concurrently
        
        Args:
            tasks: List of task dicts with 'document', 'check_type', 'rule_hints'
            
        Returns:
            List of results in same order as tasks
        """
        coroutines = [
            self.analyze_async(
                task['document'],
                task['check_type'],
                task['rule_hints'],
                **task.get('kwargs', {})
            )
            for task in tasks
        ]
        
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Convert exceptions to None
        return [r if not isinstance(r, Exception) else None for r in results]


class AsyncComplianceChecker:
    """
    Async wrapper for HybridComplianceChecker
    Enables concurrent processing of different check types
    """
    
    def __init__(self, compliance_checker, async_ai_engine: Optional[AsyncAIEngine] = None,
                 default_timeout: float = 30.0):
        """
        Initialize async compliance checker
        
        Args:
            compliance_checker: HybridComplianceChecker instance
            async_ai_engine: Optional AsyncAIEngine instance
            default_timeout: Default timeout for async operations in seconds
        """
        self.compliance_checker = compliance_checker
        self.async_ai_engine = async_ai_engine
        self.default_timeout = default_timeout
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Replace AI engine with async version if provided
        if async_ai_engine:
            self.original_ai_engine = compliance_checker.ai_engine
            compliance_checker.ai_engine = async_ai_engine
        
        logger.info(f"Async Compliance Checker initialized (timeout={default_timeout}s)")
    
    async def check_compliance_async(self, document: Dict, check_type: str, 
                                    timeout: Optional[float] = None, 
                                    **kwargs) -> Optional[Any]:
        """
        Async version of compliance check with timeout handling
        
        Args:
            document: Document to check
            check_type: Type of compliance check
            timeout: Timeout in seconds (None = use default)
            **kwargs: Additional parameters
            
        Returns:
            ComplianceResult or None
        """
        timeout = timeout or self.default_timeout
        
        try:
            # Create async task with timeout
            task = asyncio.create_task(
                self._run_check_async(document, check_type, **kwargs)
            )
            
            # Wait with timeout
            result = await asyncio.wait_for(task, timeout=timeout)
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Check {check_type} timed out after {timeout}s")
            return self._create_timeout_result(check_type, timeout)
        except Exception as e:
            logger.error(f"Async check {check_type} failed: {e}")
            return None
    
    async def _run_check_async(self, document: Dict, check_type: str, **kwargs) -> Optional[Any]:
        """
        Run compliance check in async context
        
        Args:
            document: Document to check
            check_type: Type of check
            **kwargs: Additional parameters
            
        Returns:
            ComplianceResult or None
        """
        loop = asyncio.get_event_loop()
        
        # Run synchronous check in executor
        result = await loop.run_in_executor(
            self.executor,
            lambda: self.compliance_checker.check_compliance(document, check_type, **kwargs)
        )
        
        return result
    
    async def check_all_compliance_async(self, document: Dict, 
                                        timeout_per_check: Optional[float] = None,
                                        **kwargs) -> List[Any]:
        """
        Run all compliance checks concurrently with timeout handling
        
        Args:
            document: Document to check
            timeout_per_check: Timeout per individual check
            **kwargs: Additional parameters
            
        Returns:
            List of ComplianceResult objects
        """
        from hybrid_compliance_checker import CheckType
        
        # Create tasks for all check types
        tasks = [
            self.check_compliance_async(document, check_type.value, timeout_per_check, **kwargs)
            for check_type in CheckType
        ]
        
        # Run all checks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        violations = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Check {list(CheckType)[i].value} raised exception: {result}")
            elif result is not None:
                violations.append(result)
        
        logger.info(f"Async compliance check complete: {len(violations)} violations found")
        return violations
    
    async def check_multiple_documents_async(self, documents: List[Dict],
                                            check_types: Optional[List[str]] = None,
                                            timeout_per_document: Optional[float] = None,
                                            **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple documents concurrently
        
        Args:
            documents: List of documents to check
            check_types: Optional list of specific check types (None = all)
            timeout_per_document: Timeout per document
            **kwargs: Additional parameters
            
        Returns:
            List of results with document_id and violations
        """
        if check_types:
            # Run specific check types
            tasks = [
                self._check_document_with_types_async(doc, check_types, timeout_per_document, **kwargs)
                for doc in documents
            ]
        else:
            # Run all checks
            tasks = [
                self._check_document_all_async(doc, timeout_per_document, **kwargs)
                for doc in documents
            ]
        
        # Process all documents concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results):
            doc_id = documents[i].get('id', f'doc_{i}')
            
            if isinstance(result, Exception):
                formatted_results.append({
                    'document_id': doc_id,
                    'status': 'error',
                    'error': str(result),
                    'violations': []
                })
            else:
                formatted_results.append({
                    'document_id': doc_id,
                    'status': 'completed',
                    'violations': result
                })
        
        return formatted_results
    
    async def _check_document_with_types_async(self, document: Dict, check_types: List[str],
                                              timeout: Optional[float], **kwargs) -> List[Any]:
        """
        Check document with specific check types concurrently
        
        Args:
            document: Document to check
            check_types: List of check types
            timeout: Timeout per check
            **kwargs: Additional parameters
            
        Returns:
            List of violations
        """
        tasks = [
            self.check_compliance_async(document, check_type, timeout, **kwargs)
            for check_type in check_types
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter valid results
        violations = [r for r in results if r is not None and not isinstance(r, Exception)]
        return violations
    
    async def _check_document_all_async(self, document: Dict, timeout: Optional[float],
                                       **kwargs) -> List[Any]:
        """
        Check document with all check types concurrently
        
        Args:
            document: Document to check
            timeout: Timeout per check
            **kwargs: Additional parameters
            
        Returns:
            List of violations
        """
        return await self.check_all_compliance_async(document, timeout, **kwargs)
    
    def _create_timeout_result(self, check_type: str, timeout: float) -> AsyncResult:
        """
        Create timeout result
        
        Args:
            check_type: Type of check that timed out
            timeout: Timeout value
            
        Returns:
            AsyncResult with timeout status
        """
        return AsyncResult(
            task_id=f"{check_type}_timeout",
            status=AsyncTaskStatus.TIMEOUT,
            error=f"Check timed out after {timeout}s"
        )
    
    async def aggregate_results_async(self, results: List[Any]) -> Dict[str, Any]:
        """
        Aggregate multiple compliance results asynchronously
        
        Args:
            results: List of ComplianceResult objects
            
        Returns:
            Aggregated statistics and summary
        """
        loop = asyncio.get_event_loop()
        
        # Run aggregation in executor to avoid blocking
        aggregated = await loop.run_in_executor(
            self.executor,
            lambda: self._aggregate_results_sync(results)
        )
        
        return aggregated
    
    def _aggregate_results_sync(self, results: List[Any]) -> Dict[str, Any]:
        """
        Synchronous result aggregation logic
        
        Args:
            results: List of results to aggregate
            
        Returns:
            Aggregated statistics
        """
        if not results:
            return {
                'total_violations': 0,
                'by_type': {},
                'by_severity': {},
                'by_status': {},
                'average_confidence': 0.0
            }
        
        by_type = {}
        by_severity = {}
        by_status = {}
        total_confidence = 0
        count = 0
        
        for result in results:
            if result is None:
                continue
            
            # Count by type
            check_type = getattr(result, 'check_type', 'unknown')
            if hasattr(check_type, 'value'):
                check_type = check_type.value
            by_type[check_type] = by_type.get(check_type, 0) + 1
            
            # Count by severity
            severity = getattr(result, 'severity', 'unknown')
            by_severity[severity] = by_severity.get(severity, 0) + 1
            
            # Count by status
            status = getattr(result, 'status', 'unknown')
            if hasattr(status, 'value'):
                status = status.value
            by_status[status] = by_status.get(status, 0) + 1
            
            # Sum confidence
            confidence = getattr(result, 'confidence', 0)
            total_confidence += confidence
            count += 1
        
        avg_confidence = total_confidence / count if count > 0 else 0.0
        
        return {
            'total_violations': len([r for r in results if r is not None]),
            'by_type': by_type,
            'by_severity': by_severity,
            'by_status': by_status,
            'average_confidence': round(avg_confidence, 2)
        }
    
    async def batch_process_with_concurrency(self, documents: List[Dict],
                                            max_concurrent: int = 5,
                                            timeout_per_document: Optional[float] = None,
                                            progress_callback: Optional[Callable] = None,
                                            **kwargs) -> List[Dict[str, Any]]:
        """
        Process documents in batches with controlled concurrency
        
        Args:
            documents: List of documents to process
            max_concurrent: Maximum concurrent document processing
            timeout_per_document: Timeout per document
            progress_callback: Optional callback for progress updates
            **kwargs: Additional parameters
            
        Returns:
            List of results
        """
        results = []
        total = len(documents)
        
        # Process in batches
        for i in range(0, total, max_concurrent):
            batch = documents[i:i + max_concurrent]
            batch_results = await self.check_multiple_documents_async(
                batch, timeout_per_document=timeout_per_document, **kwargs
            )
            results.extend(batch_results)
            
            # Call progress callback
            if progress_callback:
                progress = {
                    'completed': len(results),
                    'total': total,
                    'percentage': (len(results) / total) * 100
                }
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(progress)
                else:
                    progress_callback(progress)
        
        return results
    
    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        logger.info("Async Compliance Checker closed")


class AsyncResultAggregator:
    """
    Aggregates results from multiple async operations
    Provides real-time statistics and summaries
    """
    
    def __init__(self):
        """Initialize result aggregator"""
        self.results: List[Any] = []
        self.lock = asyncio.Lock()
        self.stats = {
            'total': 0,
            'completed': 0,
            'failed': 0,
            'timeout': 0
        }
    
    async def add_result(self, result: Any):
        """
        Add a result to the aggregator
        
        Args:
            result: Result to add
        """
        async with self.lock:
            self.results.append(result)
            self.stats['total'] += 1
            
            if isinstance(result, AsyncResult):
                if result.status == AsyncTaskStatus.COMPLETED:
                    self.stats['completed'] += 1
                elif result.status == AsyncTaskStatus.FAILED:
                    self.stats['failed'] += 1
                elif result.status == AsyncTaskStatus.TIMEOUT:
                    self.stats['timeout'] += 1
    
    async def get_results(self) -> List[Any]:
        """Get all results"""
        async with self.lock:
            return self.results.copy()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        async with self.lock:
            return self.stats.copy()
    
    async def clear(self):
        """Clear all results"""
        async with self.lock:
            self.results.clear()
            self.stats = {
                'total': 0,
                'completed': 0,
                'failed': 0,
                'timeout': 0
            }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def run_with_timeout(coro: Coroutine, timeout: float, 
                          fallback_value: Any = None) -> Any:
    """
    Run a coroutine with timeout
    
    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        fallback_value: Value to return on timeout
        
    Returns:
        Result or fallback_value on timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        return fallback_value


async def gather_with_timeout(tasks: List[Coroutine], timeout: float,
                              return_exceptions: bool = True) -> List[Any]:
    """
    Gather multiple tasks with overall timeout
    
    Args:
        tasks: List of coroutines
        timeout: Overall timeout in seconds
        return_exceptions: Whether to return exceptions as results
        
    Returns:
        List of results
    """
    try:
        return await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=return_exceptions),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"Gather operation timed out after {timeout}s")
        return [None] * len(tasks)


async def process_with_semaphore(tasks: List[Coroutine], max_concurrent: int) -> List[Any]:
    """
    Process tasks with concurrency limit using semaphore
    
    Args:
        tasks: List of coroutines
        max_concurrent: Maximum concurrent tasks
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_task(task):
        async with semaphore:
            return await task
    
    bounded_tasks = [bounded_task(task) for task in tasks]
    return await asyncio.gather(*bounded_tasks, return_exceptions=True)


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

async def example_async_processing():
    """Example of async processing"""
    print("="*70)
    print("Async Processor - Example Usage")
    print("="*70)
    
    # Simulate AI engine
    class MockAIEngine:
        async def analyze_async(self, document, check_type, rule_hints, **kwargs):
            await asyncio.sleep(0.5)  # Simulate API call
            return {
                'violation': True,
                'confidence': 85,
                'reasoning': f'Mock analysis for {check_type}'
            }
    
    # Simulate compliance checker
    class MockComplianceChecker:
        def __init__(self):
            self.ai_engine = None
        
        def check_compliance(self, document, check_type, **kwargs):
            time.sleep(0.5)  # Simulate processing
            return {
                'type': check_type,
                'violation': True,
                'confidence': 80
            }
    
    # Create async checker
    mock_checker = MockComplianceChecker()
    async_ai = AsyncAIEngine(MockAIEngine())
    async_checker = AsyncComplianceChecker(mock_checker, async_ai, default_timeout=5.0)
    
    print("\n🧪 Testing single async check...")
    doc = {'id': 'test_doc', 'content': 'test content'}
    result = await async_checker.check_compliance_async(doc, 'structure', timeout=2.0)
    print(f"  ✓ Result: {result}")
    
    print("\n🧪 Testing concurrent checks...")
    start = time.time()
    results = await async_checker.check_all_compliance_async(doc, timeout_per_check=2.0)
    duration = time.time() - start
    print(f"  ✓ Completed {len(results)} checks in {duration:.2f}s")
    
    print("\n🧪 Testing multiple documents...")
    docs = [{'id': f'doc_{i}', 'content': f'content {i}'} for i in range(5)]
    start = time.time()
    results = await async_checker.check_multiple_documents_async(docs, timeout_per_document=3.0)
    duration = time.time() - start
    print(f"  ✓ Processed {len(results)} documents in {duration:.2f}s")
    
    print("\n🧪 Testing result aggregation...")
    aggregated = await async_checker.aggregate_results_async(results)
    print(f"  ✓ Aggregated stats: {aggregated}")
    
    print("\n🧪 Testing timeout handling...")
    try:
        result = await async_checker.check_compliance_async(doc, 'structure', timeout=0.1)
        print(f"  ✓ Timeout handled: {result}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    async_checker.close()
    
    print("\n" + "="*70)
    print("✓ Async processing examples complete")
    print("="*70)


if __name__ == "__main__":
    # Run example
    asyncio.run(example_async_processing())