#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics Tracker

This module provides functionality for the multi-agent compliance system.
"""

"""
Performance Metrics Tracker - Multi-Agent System Monitoring

This module provides comprehensive performance metrics tracking for the
LangGraph-based multi-agent compliance system. It tracks agent execution times,
success/failure rates, cache hit rates, API call counts, and provides metrics
aggregation and reporting capabilities.

Integration with PerformanceMonitor:
- Bridges multi-agent metrics with existing monitoring infrastructure
- Provides unified metrics dashboard
- Supports alerting and historical tracking
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import statistics


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import PerformanceMonitor for integration
try:
    from performance_monitor import PerformanceMonitor, ProcessingLayer
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    logger.warning("PerformanceMonitor not available - integration disabled")
    PERFORMANCE_MONITOR_AVAILABLE = False
    PerformanceMonitor = None
    ProcessingLayer = None

# Import PerformanceAlerting for alerting support
try:
    from performance_alerting import PerformanceAlerting, AlertThresholds, Alert
    ALERTING_AVAILABLE = True
except ImportError:
    logger.warning("PerformanceAlerting not available - alerting disabled")
    ALERTING_AVAILABLE = False
    PerformanceAlerting = None
    AlertThresholds = None
    Alert = None


class MetricType(str, Enum):
    """Types of metrics tracked"""
    EXECUTION_TIME = "execution_time"
    SUCCESS_RATE = "success_rate"
    FAILURE_RATE = "failure_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    API_CALLS = "api_calls"
    THROUGHPUT = "throughput"
    LATENCY = "latency"


@dataclass
class AgentMetrics:
    """Metrics for a single agent"""
    agent_name: str

    # Execution metrics
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    skipped_executions: int = 0

    # Timing metrics (in seconds)
    total_duration: float = 0.0
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    avg_duration: float = 0.0

    # Performance metrics
    total_api_calls: int = 0
    total_cache_hits: int = 0
    cache_hit_rate: float = 0.0

    # Violation metrics
    total_violations_found: int = 0
    avg_violations_per_execution: float = 0.0

    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_timestamp: Optional[str] = None

    # Timestamps
    first_execution: Optional[str] = None
    last_execution: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentMetrics':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class WorkflowMetrics:
    """Metrics for complete workflow executions"""
    workflow_id: str
    document_id: str

    # Execution details
    started_at: str
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    status: str = "running"

    # Agent metrics
    agents_executed: List[str] = field(default_factory=list)
    agent_durations: Dict[str, float] = field(default_factory=dict)

    # Performance metrics
    total_api_calls: int = 0
    total_cache_hits: int = 0
    cache_hit_rate: float = 0.0

    # Results
    total_violations: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'WorkflowMetrics':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class SystemMetrics:
    """System-wide aggregated metrics"""
    # Overall statistics
    total_workflows: int = 0
    successful_workflows: int = 0
    failed_workflows: int = 0

    # Timing
    total_processing_time: float = 0.0
    avg_workflow_duration: float = 0.0
    min_workflow_duration: Optional[float] = None
    max_workflow_duration: Optional[float] = None

    # Performance
    total_api_calls: int = 0
    total_cache_hits: int = 0
    overall_cache_hit_rate: float = 0.0

    # Throughput
    workflows_per_hour: float = 0.0
    documents_processed: int = 0

    # Violations
    total_violations_found: int = 0
    avg_violations_per_document: float = 0.0

    # Timestamps
    tracking_started: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class MetricsTracker:
    """
    Comprehensive performance metrics tracker for multi-agent system

    Features:
    - Track agent execution times and success rates
    - Monitor API calls and cache hit rates
    - Aggregate metrics across workflows
    - Real-time metrics updates
    - Historical metrics storage
    - Thread-safe operations
    - Metrics export and reporting
    """

    def __init__(self,
                 metrics_dir: str = "./monitoring/metrics/",
                 metrics_file: str = "metrics.json",
                 history_window: int = 1000,
                 auto_save: bool = True,
                 save_interval: int = 60,
                 performance_monitor: Optional[Any] = None,
                 enable_performance_monitor_integration: bool = True):
        """
        Initialize metrics tracker

        Args:
            metrics_dir: Directory for metrics storage
            metrics_file: Name of metrics file
            history_window: Number of recent executions to keep in memory
            auto_save: Whether to auto-save metrics periodically
            save_interval: Auto-save interval in seconds
            performance_monitor: Optional PerformanceMonitor instance for integration
            enable_performance_monitor_integration: Whether to enable PerformanceMonitor integration
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.metrics_dir / metrics_file
        self.history_window = history_window
        self.auto_save = auto_save
        self.save_interval = save_interval

        # Agent metrics storage
        self.agent_metrics: Dict[str, AgentMetrics] = {}

        # Workflow metrics storage (recent history)
        self.workflow_metrics: deque = deque(maxlen=history_window)

        # System-wide metrics
        self.system_metrics = SystemMetrics()

        # Real-time tracking
        self.active_workflows: Dict[str, WorkflowMetrics] = {}
        self.active_agents: Dict[str, Dict[str, Any]] = {}

        # Thread safety
        self.lock = threading.Lock()

        # PerformanceMonitor integration
        self.performance_monitor = performance_monitor
        self.enable_performance_monitor_integration = (
            enable_performance_monitor_integration and PERFORMANCE_MONITOR_AVAILABLE
        )

        # Initialize PerformanceMonitor if not provided and integration is enabled
        if self.enable_performance_monitor_integration and self.performance_monitor is None:
            try:
                self.performance_monitor = PerformanceMonitor()
                logger.info("✓ PerformanceMonitor integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize PerformanceMonitor: {e}")
                self.enable_performance_monitor_integration = False

        # Alerting integration
        self.alerting: Optional[Any] = None
        if ALERTING_AVAILABLE:
            try:
                self.alerting = PerformanceAlerting()
                logger.info("✓ Performance alerting enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize alerting: {e}")

        # Auto-save thread
        self._stop_auto_save = threading.Event()
        if self.auto_save:
            self._auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
            self._auto_save_thread.start()

        # Load existing metrics
        self._load_metrics()

        logger.info(f"MetricsTracker initialized with {len(self.agent_metrics)} agents tracked")

    def start_workflow(self, workflow_id: str, document_id: str) -> None:
        """
        Start tracking a new workflow

        Args:
            workflow_id: Unique workflow identifier
            document_id: Document being processed
        """
        with self.lock:
            workflow_metrics = WorkflowMetrics(
                workflow_id=workflow_id,
                document_id=document_id,
                started_at=datetime.now().isoformat(),
                status="running"
            )
            self.active_workflows[workflow_id] = workflow_metrics
            logger.debug(f"Started tracking workflow: {workflow_id}")

    def complete_workflow(self,
                         workflow_id: str,
                         status: str = "completed",
                         total_violations: int = 0,
                         errors: Optional[List[str]] = None) -> None:
        """
        Complete workflow tracking

        Args:
            workflow_id: Workflow identifier
            status: Final status (completed, failed, interrupted)
            total_violations: Total violations found
            errors: List of errors encountered
        """
        with self.lock:
            if workflow_id not in self.active_workflows:
                logger.warning(f"Workflow {workflow_id} not found in active workflows")
                return

            workflow = self.active_workflows[workflow_id]
            workflow.completed_at = datetime.now().isoformat()
            workflow.status = status
            workflow.total_violations = total_violations
            workflow.errors = errors or []

            # Calculate duration
            started = datetime.fromisoformat(workflow.started_at)
            completed = datetime.fromisoformat(workflow.completed_at)
            workflow.duration_seconds = (completed - started).total_seconds()

            # Calculate cache hit rate
            total_requests = workflow.total_api_calls + workflow.total_cache_hits
            if total_requests > 0:
                workflow.cache_hit_rate = workflow.total_cache_hits / total_requests

            # Move to history
            self.workflow_metrics.append(workflow)
            del self.active_workflows[workflow_id]

            # Update system metrics
            self._update_system_metrics(workflow)

            logger.info(f"Completed workflow {workflow_id}: {status}, "
                       f"{workflow.duration_seconds:.2f}s, {total_violations} violations")

    def start_agent_execution(self,
                             workflow_id: str,
                             agent_name: str,
                             execution_id: str) -> None:
        """
        Start tracking an agent execution

        Args:
            workflow_id: Parent workflow ID
            agent_name: Name of the agent
            execution_id: Unique execution ID
        """
        with self.lock:
            self.active_agents[execution_id] = {
                "workflow_id": workflow_id,
                "agent_name": agent_name,
                "started_at": datetime.now().isoformat(),
                "api_calls": 0,
                "cache_hits": 0
            }
            logger.debug(f"Started tracking agent: {agent_name} ({execution_id})")

    def complete_agent_execution(self,
                                execution_id: str,
                                status: str = "completed",
                                violations_found: int = 0,
                                api_calls: int = 0,
                                cache_hits: int = 0,
                                error: Optional[str] = None) -> None:
        """
        Complete agent execution tracking

        Args:
            execution_id: Execution identifier
            status: Final status (completed, failed, skipped)
            violations_found: Number of violations found
            api_calls: Number of API calls made
            cache_hits: Number of cache hits
            error: Error message if failed
        """
        with self.lock:
            if execution_id not in self.active_agents:
                logger.warning(f"Agent execution {execution_id} not found")
                return

            agent_exec = self.active_agents[execution_id]
            agent_name = agent_exec["agent_name"]
            workflow_id = agent_exec["workflow_id"]

            # Calculate duration
            started = datetime.fromisoformat(agent_exec["started_at"])
            completed = datetime.now()
            duration = (completed - started).total_seconds()

            # Update agent metrics
            if agent_name not in self.agent_metrics:
                self.agent_metrics[agent_name] = AgentMetrics(agent_name=agent_name)

            metrics = self.agent_metrics[agent_name]
            metrics.total_executions += 1

            if status == "completed":
                metrics.successful_executions += 1
            elif status == "failed":
                metrics.failed_executions += 1
                metrics.error_count += 1
                metrics.last_error = error
                metrics.last_error_timestamp = completed.isoformat()
            elif status == "skipped":
                metrics.skipped_executions += 1

            # Update timing metrics
            metrics.total_duration += duration
            metrics.avg_duration = metrics.total_duration / metrics.total_executions

            if metrics.min_duration is None or duration < metrics.min_duration:
                metrics.min_duration = duration
            if metrics.max_duration is None or duration > metrics.max_duration:
                metrics.max_duration = duration

            # Update performance metrics
            metrics.total_api_calls += api_calls
            metrics.total_cache_hits += cache_hits
            total_requests = metrics.total_api_calls + metrics.total_cache_hits
            if total_requests > 0:
                metrics.cache_hit_rate = metrics.total_cache_hits / total_requests

            # Update violation metrics
            metrics.total_violations_found += violations_found
            metrics.avg_violations_per_execution = (
                metrics.total_violations_found / metrics.total_executions
            )

            # Update timestamps
            if metrics.first_execution is None:
                metrics.first_execution = agent_exec["started_at"]
            metrics.last_execution = completed.isoformat()

            # Update workflow metrics
            if workflow_id in self.active_workflows:
                workflow = self.active_workflows[workflow_id]
                workflow.agents_executed.append(agent_name)
                workflow.agent_durations[agent_name] = duration
                workflow.total_api_calls += api_calls
                workflow.total_cache_hits += cache_hits
                if error:
                    workflow.errors.append(f"{agent_name}: {error}")

            # Integrate with PerformanceMonitor
            if self.enable_performance_monitor_integration and self.performance_monitor:
                self._sync_to_performance_monitor(
                    agent_name=agent_name,
                    duration=duration,
                    api_calls=api_calls,
                    cache_hits=cache_hits,
                    status=status,
                    violations_found=violations_found
                )

            # Clean up
            del self.active_agents[execution_id]

            logger.debug(f"Completed agent {agent_name}: {status}, {duration:.2f}s")

    def record_api_call(self, execution_id: str) -> None:
        """
        Record an API call for an active agent execution

        Args:
            execution_id: Agent execution ID
        """
        with self.lock:
            if execution_id in self.active_agents:
                self.active_agents[execution_id]["api_calls"] += 1

    def record_cache_hit(self, execution_id: str) -> None:
        """
        Record a cache hit for an active agent execution

        Args:
            execution_id: Agent execution ID
        """
        with self.lock:
            if execution_id in self.active_agents:
                self.active_agents[execution_id]["cache_hits"] += 1

    def get_agent_metrics(self, agent_name: Optional[str] = None) -> Dict[str, AgentMetrics]:
        """
        Get metrics for one or all agents

        Args:
            agent_name: Optional agent name to filter by

        Returns:
            Dictionary of agent name to AgentMetrics
        """
        with self.lock:
            if agent_name:
                return {agent_name: self.agent_metrics.get(agent_name)}
            return self.agent_metrics.copy()

    def get_workflow_metrics(self,
                           workflow_id: Optional[str] = None,
                           limit: Optional[int] = None) -> List[WorkflowMetrics]:
        """
        Get workflow metrics

        Args:
            workflow_id: Optional workflow ID to filter by
            limit: Maximum number of workflows to return (most recent)

        Returns:
            List of WorkflowMetrics
        """
        with self.lock:
            if workflow_id:
                # Search in history
                workflows = [w for w in self.workflow_metrics if w.workflow_id == workflow_id]
                # Check active workflows
                if workflow_id in self.active_workflows:
                    workflows.append(self.active_workflows[workflow_id])
                return workflows

            # Return recent workflows
            workflows = list(self.workflow_metrics)
            if limit:
                workflows = workflows[-limit:]
            return workflows

    def get_system_metrics(self) -> SystemMetrics:
        """
        Get system-wide aggregated metrics

        Returns:
            SystemMetrics object
        """
        with self.lock:
            return self.system_metrics

    def get_success_rate(self, agent_name: Optional[str] = None) -> float:
        """
        Calculate success rate for agent(s)

        Args:
            agent_name: Optional agent name to filter by

        Returns:
            Success rate as percentage (0-100)
        """
        with self.lock:
            if agent_name:
                if agent_name not in self.agent_metrics:
                    return 0.0
                metrics = self.agent_metrics[agent_name]
                if metrics.total_executions == 0:
                    return 0.0
                return (metrics.successful_executions / metrics.total_executions) * 100

            # Overall success rate
            total = sum(m.total_executions for m in self.agent_metrics.values())
            successful = sum(m.successful_executions for m in self.agent_metrics.values())
            if total == 0:
                return 0.0
            return (successful / total) * 100

    def get_failure_rate(self, agent_name: Optional[str] = None) -> float:
        """
        Calculate failure rate for agent(s)

        Args:
            agent_name: Optional agent name to filter by

        Returns:
            Failure rate as percentage (0-100)
        """
        return 100.0 - self.get_success_rate(agent_name)

    def get_cache_hit_rate(self, agent_name: Optional[str] = None) -> float:
        """
        Get cache hit rate for agent(s)

        Args:
            agent_name: Optional agent name to filter by

        Returns:
            Cache hit rate as percentage (0-100)
        """
        with self.lock:
            if agent_name:
                if agent_name not in self.agent_metrics:
                    return 0.0
                return self.agent_metrics[agent_name].cache_hit_rate * 100

            # Overall cache hit rate
            total_api = sum(m.total_api_calls for m in self.agent_metrics.values())
            total_cache = sum(m.total_cache_hits for m in self.agent_metrics.values())
            total_requests = total_api + total_cache
            if total_requests == 0:
                return 0.0
            return (total_cache / total_requests) * 100

    def get_average_execution_time(self, agent_name: Optional[str] = None) -> float:
        """
        Get average execution time for agent(s)

        Args:
            agent_name: Optional agent name to filter by

        Returns:
            Average execution time in seconds
        """
        with self.lock:
            if agent_name:
                if agent_name not in self.agent_metrics:
                    return 0.0
                return self.agent_metrics[agent_name].avg_duration

            # Overall average
            durations = [m.avg_duration for m in self.agent_metrics.values() if m.total_executions > 0]
            if not durations:
                return 0.0
            return statistics.mean(durations)

    def get_throughput(self, time_window_hours: float = 1.0) -> float:
        """
        Calculate workflow throughput (workflows per hour)

        Args:
            time_window_hours: Time window to calculate throughput over

        Returns:
            Workflows per hour
        """
        with self.lock:
            if not self.workflow_metrics:
                return 0.0

            # Get workflows in time window
            cutoff = datetime.now() - timedelta(hours=time_window_hours)
            recent_workflows = [
                w for w in self.workflow_metrics
                if datetime.fromisoformat(w.completed_at) > cutoff
            ]

            if not recent_workflows:
                return 0.0

            return len(recent_workflows) / time_window_hours

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary

        Returns:
            Dictionary with performance metrics
        """
        with self.lock:
            return {
                "system": {
                    "total_workflows": self.system_metrics.total_workflows,
                    "successful_workflows": self.system_metrics.successful_workflows,
                    "failed_workflows": self.system_metrics.failed_workflows,
                    "success_rate": self.get_success_rate(),
                    "avg_workflow_duration": self.system_metrics.avg_workflow_duration,
                    "throughput_per_hour": self.get_throughput(),
                    "cache_hit_rate": self.get_cache_hit_rate(),
                    "total_violations": self.system_metrics.total_violations_found
                },
                "agents": {
                    name: {
                        "total_executions": metrics.total_executions,
                        "success_rate": self.get_success_rate(name),
                        "avg_duration": metrics.avg_duration,
                        "cache_hit_rate": metrics.cache_hit_rate * 100,
                        "violations_found": metrics.total_violations_found
                    }
                    for name, metrics in self.agent_metrics.items()
                },
                "recent_workflows": [
                    {
                        "workflow_id": w.workflow_id,
                        "duration": w.duration_seconds,
                        "violations": w.total_violations,
                        "status": w.status
                    }
                    for w in list(self.workflow_metrics)[-10:]
                ]
            }

    def _update_system_metrics(self, workflow: WorkflowMetrics) -> None:
        """
        Update system-wide metrics from completed workflow

        Args:
            workflow: Completed workflow metrics
        """
        self.system_metrics.total_workflows += 1

        if workflow.status == "completed":
            self.system_metrics.successful_workflows += 1
        elif workflow.status == "failed":
            self.system_metrics.failed_workflows += 1

        # Update timing
        if workflow.duration_seconds:
            self.system_metrics.total_processing_time += workflow.duration_seconds
            self.system_metrics.avg_workflow_duration = (
                self.system_metrics.total_processing_time / self.system_metrics.total_workflows
            )

            if (self.system_metrics.min_workflow_duration is None or
                workflow.duration_seconds < self.system_metrics.min_workflow_duration):
                self.system_metrics.min_workflow_duration = workflow.duration_seconds

            if (self.system_metrics.max_workflow_duration is None or
                workflow.duration_seconds > self.system_metrics.max_workflow_duration):
                self.system_metrics.max_workflow_duration = workflow.duration_seconds

        # Update performance
        self.system_metrics.total_api_calls += workflow.total_api_calls
        self.system_metrics.total_cache_hits += workflow.total_cache_hits
        total_requests = self.system_metrics.total_api_calls + self.system_metrics.total_cache_hits
        if total_requests > 0:
            self.system_metrics.overall_cache_hit_rate = (
                self.system_metrics.total_cache_hits / total_requests
            )

        # Update violations
        self.system_metrics.total_violations_found += workflow.total_violations
        self.system_metrics.documents_processed += 1
        self.system_metrics.avg_violations_per_document = (
            self.system_metrics.total_violations_found / self.system_metrics.documents_processed
        )

        # Update throughput
        tracking_duration = (
            datetime.now() - datetime.fromisoformat(self.system_metrics.tracking_started)
        ).total_seconds() / 3600  # Convert to hours
        if tracking_duration > 0:
            self.system_metrics.workflows_per_hour = (
                self.system_metrics.total_workflows / tracking_duration
            )

        self.system_metrics.last_updated = datetime.now().isoformat()

    def _sync_to_performance_monitor(self,
                                     agent_name: str,
                                     duration: float,
                                     api_calls: int,
                                     cache_hits: int,
                                     status: str,
                                     violations_found: int) -> None:
        """
        Sync metrics to PerformanceMonitor (non-blocking)

        Args:
            agent_name: Name of the agent
            duration: Execution duration in seconds
            api_calls: Number of API calls
            cache_hits: Number of cache hits
            status: Execution status
            violations_found: Number of violations found
        """
        try:
            if not self.performance_monitor:
                return

            duration_ms = duration * 1000

            # Record API calls (simplified to avoid deadlocks)
            if api_calls > 0:
                self.performance_monitor.record_api_call(
                    provider='multi_agent',
                    tokens=api_calls * 500,  # Estimated average per call
                    latency_ms=duration_ms / api_calls,
                    cached=False,
                    success=(status == "completed")
                )

            # Record cache hits
            if cache_hits > 0:
                self.performance_monitor.record_api_call(
                    provider='multi_agent',
                    tokens=0,
                    latency_ms=0,
                    cached=True,
                    success=True
                )

            # Record check results for accuracy tracking
            if violations_found > 0:
                self.performance_monitor.record_check_result(
                    check_type=agent_name,
                    predicted_violation=True,
                    actual_violation=None,  # Unknown until human review
                    human_review=False
                )

            logger.debug(f"Synced {agent_name} metrics to PerformanceMonitor")

        except Exception as e:
            logger.warning(f"Failed to sync to PerformanceMonitor: {e}")

    def get_performance_monitor(self) -> Optional[Any]:
        """
        Get the integrated PerformanceMonitor instance

        Returns:
            PerformanceMonitor instance or None
        """
        return self.performance_monitor

    def get_unified_metrics(self) -> Dict[str, Any]:
        """
        Get unified metrics combining MetricsTracker and PerformanceMonitor

        Returns:
            Dictionary with unified metrics
        """
        unified = {
            'multi_agent': None,
            'performance_monitor': None
        }

        # Get multi-agent metrics (already thread-safe with lock)
        try:
            unified['multi_agent'] = self.get_performance_summary()
        except Exception as e:
            logger.warning(f"Failed to get multi-agent summary: {e}")

        # Get PerformanceMonitor metrics (simplified to avoid hanging)
        if self.enable_performance_monitor_integration and self.performance_monitor:
            try:
                # Get basic metrics without full summary to avoid potential issues
                pm_metrics = {
                    'total_cost_usd': self.performance_monitor.get_total_cost(),
                    'cache_efficiency': self.performance_monitor.get_cache_efficiency(),
                    'integration_enabled': True
                }
                unified['performance_monitor'] = pm_metrics
            except Exception as e:
                logger.warning(f"Failed to get PerformanceMonitor metrics: {e}")
                unified['performance_monitor'] = {'integration_enabled': True, 'error': str(e)}

        return unified

    def print_unified_dashboard(self) -> None:
        """
        Print unified dashboard combining both monitoring systems
        """
        print("\n" + "="*80)
        print("📊 UNIFIED PERFORMANCE DASHBOARD")
        print("="*80)

        # Print multi-agent metrics
        print("\n🤖 MULTI-AGENT SYSTEM METRICS")
        print("-" * 80)
        self.print_summary()

        # Print PerformanceMonitor integration status
        if self.enable_performance_monitor_integration and self.performance_monitor:
            print("\n⚡ PERFORMANCE MONITOR INTEGRATION")
            print("-" * 80)
            try:
                total_cost = self.performance_monitor.get_total_cost()
                cache_eff = self.performance_monitor.get_cache_efficiency()
                print(f"Status: ✓ Enabled")
                print(f"Total Cost: ${total_cost:.4f}")
                print(f"Cache Hit Rate: {cache_eff.get('cache_hit_rate', 0):.1f}%")
                print(f"Cached Calls: {cache_eff.get('cached_calls', 0)}/{cache_eff.get('total_calls', 0)}")
            except Exception as e:
                print(f"Status: ⚠️  Error - {e}")

        # Check for alerts
        try:
            alerts = self.check_alerts()
            if alerts:
                print("\n🚨 ACTIVE ALERTS")
                print("-" * 80)
                for alert in alerts:
                    severity_icon = "🔴" if alert.severity.value == "CRITICAL" else "🟡" if alert.severity.value == "WARNING" else "🔵"
                    print(f"{severity_icon} [{alert.severity.value}] {alert.message}")
        except Exception as e:
            logger.warning(f"Failed to check alerts: {e}")

        print("="*80 + "\n")

    def check_alerts(self) -> List[Any]:
        """
        Check metrics against alert thresholds

        Returns:
            List of triggered alerts
        """
        if not self.alerting:
            return []

        try:
            alerts = []

            # Check multi-agent specific alerts (simplified)
            with self.lock:
                # Check workflow success rate
                if self.system_metrics.total_workflows > 0:
                    success_rate = (self.system_metrics.successful_workflows /
                                  self.system_metrics.total_workflows * 100)

                    if success_rate < 90 and ALERTING_AVAILABLE and Alert:
                        from performance_alerting import AlertType, AlertSeverity
                        alerts.append(Alert(
                            alert_type=AlertType.HIGH_ERROR_RATE,
                            severity=AlertSeverity.CRITICAL if success_rate < 80 else AlertSeverity.WARNING,
                            message=f"Multi-agent workflow success rate ({success_rate:.1f}%) is below threshold",
                            timestamp=datetime.now().isoformat(),
                            metric_value=success_rate,
                            threshold=90.0,
                            details={'metric': 'workflow_success_rate', 'system': 'multi_agent'}
                        ))

                # Check cache hit rate
                if self.system_metrics.overall_cache_hit_rate < 0.3:
                    cache_rate = self.system_metrics.overall_cache_hit_rate * 100
                    if ALERTING_AVAILABLE and Alert:
                        from performance_alerting import AlertType, AlertSeverity
                        alerts.append(Alert(
                            alert_type=AlertType.LOW_CACHE_HIT_RATE,
                            severity=AlertSeverity.WARNING,
                            message=f"Multi-agent cache hit rate ({cache_rate:.1f}%) is below threshold (30%)",
                            timestamp=datetime.now().isoformat(),
                            metric_value=cache_rate,
                            threshold=30.0,
                            details={'metric': 'cache_hit_rate', 'system': 'multi_agent'}
                        ))

            return alerts

        except Exception as e:
            logger.warning(f"Failed to check alerts: {e}")
            return []

    def _check_multi_agent_alerts(self, summary: Dict[str, Any]) -> List[Any]:
        """
        Check multi-agent specific metrics for alerts

        Args:
            summary: Multi-agent performance summary

        Returns:
            List of alerts
        """
        if not ALERTING_AVAILABLE or not Alert:
            return []

        alerts = []
        system = summary.get('system', {})

        # Check workflow success rate
        success_rate = system.get('success_rate', 100)
        if success_rate < 90:
            from performance_alerting import AlertType, AlertSeverity
            alerts.append(Alert(
                alert_type=AlertType.HIGH_ERROR_RATE,
                severity=AlertSeverity.CRITICAL if success_rate < 80 else AlertSeverity.WARNING,
                message=f"Multi-agent workflow success rate ({success_rate:.1f}%) is below threshold",
                timestamp=datetime.now().isoformat(),
                metric_value=success_rate,
                threshold=90.0,
                details={'metric': 'workflow_success_rate', 'system': 'multi_agent'}
            ))

        # Check average workflow duration
        avg_duration = system.get('avg_workflow_duration', 0)
        if avg_duration > 30:  # 30 seconds threshold
            from performance_alerting import AlertType, AlertSeverity
            alerts.append(Alert(
                alert_type=AlertType.HIGH_LATENCY,
                severity=AlertSeverity.WARNING,
                message=f"Average workflow duration ({avg_duration:.2f}s) exceeds threshold (30s)",
                timestamp=datetime.now().isoformat(),
                metric_value=avg_duration,
                threshold=30.0,
                details={'metric': 'avg_workflow_duration', 'system': 'multi_agent'}
            ))

        # Check cache hit rate
        cache_rate = system.get('cache_hit_rate', 0)
        if cache_rate < 30:
            from performance_alerting import AlertType, AlertSeverity
            alerts.append(Alert(
                alert_type=AlertType.LOW_CACHE_HIT_RATE,
                severity=AlertSeverity.WARNING,
                message=f"Multi-agent cache hit rate ({cache_rate:.1f}%) is below threshold (30%)",
                timestamp=datetime.now().isoformat(),
                metric_value=cache_rate,
                threshold=30.0,
                details={'metric': 'cache_hit_rate', 'system': 'multi_agent'}
            ))

        return alerts

    def export_metrics(self, filepath: str, format: str = "json") -> None:
        """
        Export metrics to file

        Args:
            filepath: Output file path
            format: Export format (json, csv)
        """
        with self.lock:
            if format == "json":
                data = {
                    "exported_at": datetime.now().isoformat(),
                    "system_metrics": self.system_metrics.to_dict(),
                    "agent_metrics": {
                        name: metrics.to_dict()
                        for name, metrics in self.agent_metrics.items()
                    },
                    "recent_workflows": [
                        w.to_dict() for w in list(self.workflow_metrics)[-100:]
                    ]
                }

                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)

                logger.info(f"Exported metrics to {filepath}")

            elif format == "csv":
                import csv

                # Calculate success rates outside of file writing to avoid nested lock issues
                agent_data = []
                for name, metrics in self.agent_metrics.items():
                    if metrics.total_executions == 0:
                        success_rate = 0.0
                    else:
                        success_rate = (metrics.successful_executions / metrics.total_executions) * 100

                    agent_data.append({
                        "name": name,
                        "executions": metrics.total_executions,
                        "success_rate": success_rate,
                        "avg_duration": metrics.avg_duration,
                        "cache_hit_rate": metrics.cache_hit_rate * 100,
                        "violations": metrics.total_violations_found
                    })

                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)

                    # Write agent metrics
                    writer.writerow(["Agent Metrics"])
                    writer.writerow([
                        "Agent", "Total Executions", "Success Rate", "Avg Duration",
                        "Cache Hit Rate", "Violations Found"
                    ])

                    for data in agent_data:
                        writer.writerow([
                            data["name"],
                            data["executions"],
                            f"{data['success_rate']:.1f}%",
                            f"{data['avg_duration']:.2f}s",
                            f"{data['cache_hit_rate']:.1f}%",
                            data["violations"]
                        ])

                logger.info(f"Exported metrics to {filepath}")

            else:
                raise ValueError(f"Unsupported format: {format}")

    def print_summary(self, agent_name: Optional[str] = None) -> None:
        """
        Print metrics summary to console

        Args:
            agent_name: Optional agent name to filter by
        """
        print("\n" + "="*80)
        print("PERFORMANCE METRICS SUMMARY")
        print("="*80)

        if agent_name:
            # Print specific agent metrics
            if agent_name not in self.agent_metrics:
                print(f"\nNo metrics found for agent: {agent_name}")
                return

            metrics = self.agent_metrics[agent_name]
            print(f"\nAgent: {agent_name}")
            print("-" * 80)
            print(f"Total Executions:     {metrics.total_executions}")
            print(f"Successful:           {metrics.successful_executions}")
            print(f"Failed:               {metrics.failed_executions}")
            print(f"Skipped:              {metrics.skipped_executions}")
            print(f"Success Rate:         {self.get_success_rate(agent_name):.1f}%")
            print(f"\nTiming:")
            print(f"  Average Duration:   {metrics.avg_duration:.2f}s")
            print(f"  Min Duration:       {metrics.min_duration:.2f}s" if metrics.min_duration else "  Min Duration:       N/A")
            print(f"  Max Duration:       {metrics.max_duration:.2f}s" if metrics.max_duration else "  Max Duration:       N/A")
            print(f"\nPerformance:")
            print(f"  API Calls:          {metrics.total_api_calls}")
            print(f"  Cache Hits:         {metrics.total_cache_hits}")
            print(f"  Cache Hit Rate:     {metrics.cache_hit_rate * 100:.1f}%")
            print(f"\nViolations:")
            print(f"  Total Found:        {metrics.total_violations_found}")
            print(f"  Avg per Execution:  {metrics.avg_violations_per_execution:.1f}")

        else:
            # Print system-wide metrics
            print(f"\nSystem Metrics")
            print("-" * 80)
            print(f"Total Workflows:      {self.system_metrics.total_workflows}")
            print(f"Successful:           {self.system_metrics.successful_workflows}")
            print(f"Failed:               {self.system_metrics.failed_workflows}")
            print(f"Success Rate:         {self.get_success_rate():.1f}%")
            print(f"\nTiming:")
            print(f"  Avg Workflow:       {self.system_metrics.avg_workflow_duration:.2f}s")
            print(f"  Min Workflow:       {self.system_metrics.min_workflow_duration:.2f}s" if self.system_metrics.min_workflow_duration else "  Min Workflow:       N/A")
            print(f"  Max Workflow:       {self.system_metrics.max_workflow_duration:.2f}s" if self.system_metrics.max_workflow_duration else "  Max Workflow:       N/A")
            print(f"\nPerformance:")
            print(f"  Total API Calls:    {self.system_metrics.total_api_calls}")
            print(f"  Total Cache Hits:   {self.system_metrics.total_cache_hits}")
            print(f"  Cache Hit Rate:     {self.system_metrics.overall_cache_hit_rate * 100:.1f}%")
            print(f"  Throughput:         {self.system_metrics.workflows_per_hour:.1f} workflows/hour")
            print(f"\nViolations:")
            print(f"  Total Found:        {self.system_metrics.total_violations_found}")
            print(f"  Avg per Document:   {self.system_metrics.avg_violations_per_document:.1f}")

            print(f"\nAgent Performance:")
            print("-" * 80)
            print(f"{'Agent':<20} {'Executions':<12} {'Success Rate':<15} {'Avg Time':<12} {'Cache Hit':<12}")
            print("-" * 80)

            for name, metrics in sorted(self.agent_metrics.items()):
                success_rate = self.get_success_rate(name)
                print(f"{name:<20} {metrics.total_executions:<12} "
                      f"{success_rate:>6.1f}%{'':<8} {metrics.avg_duration:>6.2f}s{'':<5} "
                      f"{metrics.cache_hit_rate * 100:>6.1f}%")

        print("="*80 + "\n")

    def reset_metrics(self, agent_name: Optional[str] = None) -> None:
        """
        Reset metrics

        Args:
            agent_name: Optional agent name to reset (resets all if None)
        """
        with self.lock:
            if agent_name:
                if agent_name in self.agent_metrics:
                    del self.agent_metrics[agent_name]
                    logger.info(f"Reset metrics for agent: {agent_name}")
            else:
                self.agent_metrics.clear()
                self.workflow_metrics.clear()
                self.system_metrics = SystemMetrics()
                logger.info("Reset all metrics")

    def _load_metrics(self) -> None:
        """Load metrics from file"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)

                # Load system metrics
                if "system_metrics" in data:
                    self.system_metrics = SystemMetrics(**data["system_metrics"])

                # Load agent metrics
                if "agent_metrics" in data:
                    for name, metrics_data in data["agent_metrics"].items():
                        self.agent_metrics[name] = AgentMetrics.from_dict(metrics_data)

                # Load recent workflows
                if "recent_workflows" in data:
                    for workflow_data in data["recent_workflows"]:
                        self.workflow_metrics.append(WorkflowMetrics.from_dict(workflow_data))

                logger.info(f"Loaded metrics from {self.metrics_file}")

        except Exception as e:
            logger.error(f"Error loading metrics: {e}")

    def _save_metrics(self) -> None:
        """Save metrics to file"""
        try:
            data = {
                "schema_version": 1,
                "last_saved": datetime.now().isoformat(),
                "system_metrics": self.system_metrics.to_dict(),
                "agent_metrics": {
                    name: metrics.to_dict()
                    for name, metrics in self.agent_metrics.items()
                },
                "recent_workflows": [
                    w.to_dict() for w in list(self.workflow_metrics)
                ]
            }

            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved metrics to {self.metrics_file}")

        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def _auto_save_loop(self) -> None:
        """Auto-save loop running in background thread"""
        while not self._stop_auto_save.is_set():
            time.sleep(self.save_interval)
            if not self._stop_auto_save.is_set():
                self._save_metrics()

    def shutdown(self) -> None:
        """Shutdown metrics tracker and save final state"""
        logger.info("Shutting down MetricsTracker")

        # Stop auto-save thread
        if self.auto_save:
            self._stop_auto_save.set()
            self._auto_save_thread.join(timeout=5)

        # Final save
        self._save_metrics()

        logger.info("MetricsTracker shutdown complete")


# Global metrics tracker instance
_global_tracker: Optional[MetricsTracker] = None


def get_metrics_tracker() -> MetricsTracker:
    """
    Get global metrics tracker instance (singleton)

    Returns:
        MetricsTracker instance
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = MetricsTracker()
    return _global_tracker


def initialize_metrics_tracker(metrics_dir: str = "./monitoring/metrics/",
                               history_window: int = 1000,
                               auto_save: bool = True,
                               save_interval: int = 60) -> MetricsTracker:
    """
    Initialize global metrics tracker with custom settings

    Args:
        metrics_dir: Directory for metrics storage
        history_window: Number of recent executions to keep
        auto_save: Whether to auto-save metrics
        save_interval: Auto-save interval in seconds

    Returns:
        MetricsTracker instance
    """
    global _global_tracker
    _global_tracker = MetricsTracker(
        metrics_dir=metrics_dir,
        history_window=history_window,
        auto_save=auto_save,
        save_interval=save_interval
    )
    return _global_tracker


# Export all public symbols
__all__ = [
    "MetricsTracker",
    "AgentMetrics",
    "WorkflowMetrics",
    "SystemMetrics",
    "MetricType",
    "get_metrics_tracker",
    "initialize_metrics_tracker"
]



if __name__ == "__main__":
    # Example usage and testing
    logger.info("="*80)
    logger.info("MetricsTracker - Performance Metrics for Multi-Agent System")
    logger.info("="*80)

    # Initialize tracker
    tracker = MetricsTracker(
        metrics_dir="./test_metrics/",
        auto_save=False
    )

    logger.info(f"\n✓ MetricsTracker initialized")
    logger.info(f"  Metrics directory: ./test_metrics/")
    logger.info(f"  Agents tracked: {len(tracker.agent_metrics)}")

    # Simulate workflow execution
    logger.info("\n📊 Simulating workflow execution...")

    workflow_id = "workflow_test_001"
    document_id = "doc_001"

    tracker.start_workflow(workflow_id, document_id)

    # Simulate agent executions
    agents = ["structure", "performance", "securities", "general"]

    for i, agent_name in enumerate(agents):
        execution_id = f"exec_{i}"

        tracker.start_agent_execution(workflow_id, agent_name, execution_id)

        # Simulate work
        time.sleep(0.1)

        # Record some API calls and cache hits
        for _ in range(3):
            tracker.record_api_call(execution_id)
        for _ in range(2):
            tracker.record_cache_hit(execution_id)

        tracker.complete_agent_execution(
            execution_id=execution_id,
            status="completed",
            violations_found=i + 1,
            api_calls=3,
            cache_hits=2
        )

    # Complete workflow
    tracker.complete_workflow(
        workflow_id=workflow_id,
        status="completed",
        total_violations=10
    )

    logger.info(f"  ✓ Completed workflow with {len(agents)} agents")

    # Print summary
    tracker.print_summary()

    # Export metrics
    logger.info("📁 Exporting metrics...")
    tracker.export_metrics("./test_metrics/metrics_export.json", format="json")
    tracker.export_metrics("./test_metrics/metrics_export.csv", format="csv")
    logger.info("  ✓ Metrics exported")

    # Shutdown
    tracker.shutdown()
    logger.info("\n✓ MetricsTracker test complete")
