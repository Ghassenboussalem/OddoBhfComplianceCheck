#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Processor - Document Queue Management and Parallel Processing
Handles batch processing of multiple documents with rate limiting and progress tracking
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    """Status of batch processing"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class DocumentStatus(Enum):
    """Status of individual document processing"""
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class DocumentTask:
    """Represents a document to be processed"""
    document_id: str
    document: Dict
    check_types: List[str]
    metadata: Dict = field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.QUEUED
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def get_processing_time(self) -> Optional[float]:
        """Get processing time in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class BatchJob:
    """Represents a batch processing job"""
    batch_id: str
    tasks: List[DocumentTask]
    status: BatchStatus = BatchStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    def __post_init__(self):
        self.total_tasks = len(self.tasks)
    
    def get_progress(self) -> float:
        """Get progress percentage"""
        if self.total_tasks == 0:
            return 100.0
        return (self.completed_tasks + self.failed_tasks) / self.total_tasks * 100
    
    def get_duration(self) -> Optional[float]:
        """Get total duration in seconds"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_calls_per_second: float = 10.0, max_calls_per_minute: float = 100.0):
        """
        Initialize rate limiter
        
        Args:
            max_calls_per_second: Maximum calls per second
            max_calls_per_minute: Maximum calls per minute
        """
        self.max_calls_per_second = max_calls_per_second
        self.max_calls_per_minute = max_calls_per_minute
        
        self.call_times_second: List[float] = []
        self.call_times_minute: List[float] = []
        self.lock = threading.Lock()
        
        logger.info(f"Rate limiter initialized: {max_calls_per_second}/s, {max_calls_per_minute}/min")
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a call
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if permission granted, False if timeout
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                current_time = time.time()
                
                # Clean up old timestamps
                self.call_times_second = [t for t in self.call_times_second if current_time - t < 1.0]
                self.call_times_minute = [t for t in self.call_times_minute if current_time - t < 60.0]
                
                # Check if we can make a call
                if (len(self.call_times_second) < self.max_calls_per_second and
                    len(self.call_times_minute) < self.max_calls_per_minute):
                    self.call_times_second.append(current_time)
                    self.call_times_minute.append(current_time)
                    return True
            
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            # Wait before retrying
            time.sleep(0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        with self.lock:
            current_time = time.time()
            recent_second = [t for t in self.call_times_second if current_time - t < 1.0]
            recent_minute = [t for t in self.call_times_minute if current_time - t < 60.0]
            
            return {
                "calls_last_second": len(recent_second),
                "calls_last_minute": len(recent_minute),
                "max_per_second": self.max_calls_per_second,
                "max_per_minute": self.max_calls_per_minute
            }


class BatchProcessor:
    """
    Batch processor for multiple documents
    Manages document queue, parallel processing, and progress tracking
    """
    
    def __init__(self, compliance_checker, max_workers: int = 5,
                 rate_limiter: Optional[RateLimiter] = None):
        """
        Initialize batch processor
        
        Args:
            compliance_checker: HybridComplianceChecker instance
            max_workers: Maximum number of parallel workers
            rate_limiter: Optional rate limiter for API calls
        """
        self.compliance_checker = compliance_checker
        self.max_workers = max_workers
        self.rate_limiter = rate_limiter or RateLimiter()
        
        self.jobs: Dict[str, BatchJob] = {}
        self.job_lock = threading.Lock()
        
        logger.info(f"Batch processor initialized (max_workers={max_workers})")
    
    def create_batch(self, documents: List[Dict], check_types: Optional[List[str]] = None,
                    batch_id: Optional[str] = None) -> str:
        """
        Create a new batch job
        
        Args:
            documents: List of documents to process
            check_types: List of check types to run (None = all)
            batch_id: Optional custom batch ID
            
        Returns:
            Batch ID
        """
        if not batch_id:
            batch_id = f"batch_{int(time.time() * 1000)}"
        
        # Create tasks for each document
        tasks = []
        for i, doc in enumerate(documents):
            doc_id = doc.get('id', f"doc_{i}")
            task = DocumentTask(
                document_id=doc_id,
                document=doc,
                check_types=check_types or []
            )
            tasks.append(task)
        
        # Create batch job
        job = BatchJob(batch_id=batch_id, tasks=tasks)
        
        with self.job_lock:
            self.jobs[batch_id] = job
        
        logger.info(f"Created batch {batch_id} with {len(tasks)} documents")
        return batch_id

    
    def process_batch(self, batch_id: str, progress_callback: Optional[Callable] = None) -> BatchJob:
        """
        Process a batch job
        
        Args:
            batch_id: Batch ID to process
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Completed BatchJob
        """
        with self.job_lock:
            if batch_id not in self.jobs:
                raise ValueError(f"Batch {batch_id} not found")
            job = self.jobs[batch_id]
        
        if job.status != BatchStatus.PENDING:
            raise ValueError(f"Batch {batch_id} is not in PENDING status")
        
        logger.info(f"Starting batch processing: {batch_id}")
        job.status = BatchStatus.PROCESSING
        job.started_at = time.time()
        
        try:
            # Process tasks in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self._process_task, task): task
                    for task in job.tasks
                }
                
                # Process completed tasks
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    
                    try:
                        result = future.result()
                        task.result = result
                        task.status = DocumentStatus.COMPLETED
                        job.completed_tasks += 1
                        
                    except Exception as e:
                        logger.error(f"Task {task.document_id} failed: {e}")
                        task.error = str(e)
                        task.status = DocumentStatus.FAILED
                        job.failed_tasks += 1
                    
                    # Call progress callback
                    if progress_callback:
                        progress_callback(job)
            
            job.status = BatchStatus.COMPLETED
            job.completed_at = time.time()
            
            logger.info(f"Batch {batch_id} completed: {job.completed_tasks} succeeded, "
                       f"{job.failed_tasks} failed")
            
        except Exception as e:
            logger.error(f"Batch {batch_id} failed: {e}")
            job.status = BatchStatus.FAILED
            job.completed_at = time.time()
            raise
        
        return job
    
    def _process_task(self, task: DocumentTask) -> List[Any]:
        """
        Process a single document task
        
        Args:
            task: DocumentTask to process
            
        Returns:
            List of compliance results
        """
        task.status = DocumentStatus.PROCESSING
        task.start_time = time.time()
        
        try:
            # Acquire rate limit permission
            if not self.rate_limiter.acquire(timeout=30):
                raise TimeoutError("Rate limit timeout")
            
            # Process document
            violations = []
            
            if task.check_types:
                # Run specific check types
                for check_type in task.check_types:
                    result = self.compliance_checker.check_compliance(
                        task.document, check_type
                    )
                    if result:
                        violations.append(result)
            else:
                # Run all checks
                violations = self.compliance_checker.check_all_compliance(task.document)
            
            task.end_time = time.time()
            return violations
            
        except Exception as e:
            task.end_time = time.time()
            logger.error(f"Error processing task {task.document_id}: {e}")
            raise
    
    def get_batch_status(self, batch_id: str) -> Optional[BatchJob]:
        """Get status of a batch job"""
        with self.job_lock:
            return self.jobs.get(batch_id)
    
    def get_batch_progress(self, batch_id: str) -> Dict[str, Any]:
        """
        Get detailed progress information for a batch
        
        Returns:
            Dict with progress details
        """
        with self.job_lock:
            job = self.jobs.get(batch_id)
        
        if not job:
            return {"error": "Batch not found"}
        
        return {
            "batch_id": batch_id,
            "status": job.status.value,
            "progress": job.get_progress(),
            "total_tasks": job.total_tasks,
            "completed_tasks": job.completed_tasks,
            "failed_tasks": job.failed_tasks,
            "pending_tasks": job.total_tasks - job.completed_tasks - job.failed_tasks,
            "duration": job.get_duration(),
            "started_at": job.started_at,
            "completed_at": job.completed_at
        }
    
    def cancel_batch(self, batch_id: str):
        """Cancel a batch job"""
        with self.job_lock:
            if batch_id in self.jobs:
                job = self.jobs[batch_id]
                if job.status == BatchStatus.PROCESSING:
                    job.status = BatchStatus.CANCELLED
                    logger.info(f"Batch {batch_id} cancelled")
    
    def list_batches(self) -> List[str]:
        """List all batch IDs"""
        with self.job_lock:
            return list(self.jobs.keys())
    
    def get_rate_limiter_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return self.rate_limiter.get_stats()


class ProgressTracker:
    """Track and report batch processing progress"""
    
    def __init__(self):
        """Initialize progress tracker"""
        self.progress_history: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
    
    def update(self, batch_job: BatchJob):
        """
        Update progress for a batch job
        
        Args:
            batch_job: BatchJob to track
        """
        with self.lock:
            progress_entry = {
                "timestamp": time.time(),
                "batch_id": batch_job.batch_id,
                "status": batch_job.status.value,
                "progress": batch_job.get_progress(),
                "completed": batch_job.completed_tasks,
                "failed": batch_job.failed_tasks,
                "total": batch_job.total_tasks
            }
            self.progress_history.append(progress_entry)
    
    def get_history(self, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get progress history
        
        Args:
            batch_id: Optional batch ID to filter by
            
        Returns:
            List of progress entries
        """
        with self.lock:
            if batch_id:
                return [e for e in self.progress_history if e["batch_id"] == batch_id]
            return self.progress_history.copy()
    
    def clear_history(self):
        """Clear progress history"""
        with self.lock:
            self.progress_history.clear()


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("Batch Processor - Document Queue Management")
    print("="*70)
    
    # Test rate limiter
    print("\n🧪 Testing Rate Limiter...")
    limiter = RateLimiter(max_calls_per_second=5, max_calls_per_minute=20)
    
    for i in range(10):
        if limiter.acquire(timeout=2):
            print(f"  ✓ Call {i+1} approved")
        else:
            print(f"  ✗ Call {i+1} rate limited")
        time.sleep(0.1)
    
    print(f"\n📊 Rate limiter stats: {limiter.get_stats()}")
    
    # Test progress tracker
    print("\n🧪 Testing Progress Tracker...")
    tracker = ProgressTracker()
    
    # Simulate batch progress
    test_job = BatchJob(
        batch_id="test_batch",
        tasks=[DocumentTask(f"doc_{i}", {}, []) for i in range(10)]
    )
    
    test_job.status = BatchStatus.PROCESSING
    test_job.completed_tasks = 5
    tracker.update(test_job)
    
    test_job.completed_tasks = 10
    test_job.status = BatchStatus.COMPLETED
    tracker.update(test_job)
    
    history = tracker.get_history("test_batch")
    print(f"  ✓ Progress history entries: {len(history)}")
    for entry in history:
        print(f"    - {entry['status']}: {entry['progress']:.1f}% complete")
    
    print("\n" + "="*70)
    print("✓ Batch Processor ready")
    print("="*70)
