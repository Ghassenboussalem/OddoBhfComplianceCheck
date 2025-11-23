#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard

This module provides functionality for the multi-agent compliance system.
"""

"""
Monitoring Dashboard - Real-Time Multi-Agent System Monitoring

This module provides a comprehensive monitoring dashboard for the
LangGraph-based multi-agent compliance system. It displays real-time
agent status, performance metrics, workflow execution paths, and
provides alerting for failures.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

# Import monitoring components
try:
    from monitoring.agent_logger import AgentLogger, get_agent_logger
    from monitoring.metrics_tracker import MetricsTracker, get_metrics_tracker
    from monitoring.workflow_visualizer import (
        WorkflowVisualizer,
        get_workflow_visualizer,
        NodeStatus
    )
except ModuleNotFoundError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from monitoring.agent_logger import AgentLogger, get_agent_logger
    from monitoring.metrics_tracker import MetricsTracker, get_metrics_tracker
    from monitoring.workflow_visualizer import (
        WorkflowVisualizer,
        get_workflow_visualizer,
        NodeStatus
    )


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert notification"""
    alert_id: str
    timestamp: str
    level: AlertLevel
    title: str
    message: str
    source: str  # agent name or "system"
    workflow_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class MonitoringDashboard:
    """
    Comprehensive monitoring dashboard for multi-agent system

    Features:
    - Real-time agent status display
    - Performance metrics visualization
    - Workflow execution path tracking
    - Alert management and notifications
    - Historical data analysis
    - System health monitoring
    """

    def __init__(self,
                 agent_logger: Optional[AgentLogger] = None,
                 metrics_tracker: Optional[MetricsTracker] = None,
                 workflow_visualizer: Optional[WorkflowVisualizer] = None,
                 alert_threshold_error_rate: float = 0.2,
                 alert_threshold_duration: float = 30.0,
                 alert_threshold_cache_hit_rate: float = 0.3,
                 refresh_interval: int = 5):
        """
        Initialize monitoring dashboard

        Args:
            agent_logger: AgentLogger instance (uses global if None)
            metrics_tracker: MetricsTracker instance (uses global if None)
            workflow_visualizer: WorkflowVisualizer instance (uses global if None)
            alert_threshold_error_rate: Error rate threshold for alerts (0-1)
            alert_threshold_duration: Duration threshold for alerts (seconds)
            alert_threshold_cache_hit_rate: Cache hit rate threshold for alerts (0-1)
            refresh_interval: Dashboard refresh interval in seconds
        """
        self.agent_logger = agent_logger or get_agent_logger()
        self.metrics_tracker = metrics_tracker or get_metrics_tracker()
        self.workflow_visualizer = workflow_visualizer or get_workflow_visualizer()

        # Alert thresholds
        self.alert_threshold_error_rate = alert_threshold_error_rate
        self.alert_threshold_duration = alert_threshold_duration
        self.alert_threshold_cache_hit_rate = alert_threshold_cache_hit_rate
        self.refresh_interval = refresh_interval

        # Alert storage
        self.alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.max_alerts = 100
        self.max_alert_history = 1000

        # Thread safety
        self.lock = threading.Lock()

        # Auto-refresh thread
        self._stop_refresh = threading.Event()
        self._refresh_thread = None

        logger.info("MonitoringDashboard initialized")

    def start_monitoring(self) -> None:
        """Start real-time monitoring with auto-refresh"""
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            logger.warning("Monitoring already started")
            return

        self._stop_refresh.clear()
        self._refresh_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._refresh_thread.start()

        logger.info("Started real-time monitoring")

    def stop_monitoring(self) -> None:
        """Stop real-time monitoring"""
        if self._refresh_thread is None or not self._refresh_thread.is_alive():
            logger.warning("Monitoring not running")
            return

        self._stop_refresh.set()
        self._refresh_thread.join(timeout=10)

        logger.info("Stopped real-time monitoring")

    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while not self._stop_refresh.is_set():
            try:
                # Check for issues and generate alerts
                self._check_agent_health()
                self._check_workflow_health()
                self._check_system_health()

                # Sleep until next refresh
                time.sleep(self.refresh_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.refresh_interval)

    def _check_agent_health(self) -> None:
        """Check health of all agents and generate alerts"""
        agent_metrics = self.metrics_tracker.get_agent_metrics()

        for agent_name, metrics in agent_metrics.items():
            # Check error rate
            if metrics.total_executions > 0:
                error_rate = metrics.failed_executions / metrics.total_executions

                if error_rate > self.alert_threshold_error_rate:
                    self._create_alert(
                        level=AlertLevel.ERROR,
                        title=f"High Error Rate: {agent_name}",
                        message=f"Agent {agent_name} has error rate of {error_rate:.1%} "
                               f"(threshold: {self.alert_threshold_error_rate:.1%})",
                        source=agent_name
                    )

            # Check execution duration
            if metrics.avg_duration > self.alert_threshold_duration:
                self._create_alert(
                    level=AlertLevel.WARNING,
                    title=f"Slow Execution: {agent_name}",
                    message=f"Agent {agent_name} avg duration is {metrics.avg_duration:.1f}s "
                           f"(threshold: {self.alert_threshold_duration:.1f}s)",
                    source=agent_name
                )

            # Check cache hit rate
            if metrics.total_api_calls + metrics.total_cache_hits > 10:
                if metrics.cache_hit_rate < self.alert_threshold_cache_hit_rate:
                    self._create_alert(
                        level=AlertLevel.WARNING,
                        title=f"Low Cache Hit Rate: {agent_name}",
                        message=f"Agent {agent_name} cache hit rate is {metrics.cache_hit_rate:.1%} "
                               f"(threshold: {self.alert_threshold_cache_hit_rate:.1%})",
                        source=agent_name
                    )

    def _check_workflow_health(self) -> None:
        """Check health of recent workflows"""
        recent_workflows = self.metrics_tracker.get_workflow_metrics(limit=10)

        for workflow in recent_workflows:
            # Check for failed workflows
            if workflow.status == "failed":
                self._create_alert(
                    level=AlertLevel.ERROR,
                    title=f"Workflow Failed",
                    message=f"Workflow {workflow.workflow_id} failed for document {workflow.document_id}. "
                           f"Errors: {', '.join(workflow.errors[:3])}",
                    source="system",
                    workflow_id=workflow.workflow_id
                )

            # Check for slow workflows
            if workflow.duration_seconds and workflow.duration_seconds > 60.0:
                self._create_alert(
                    level=AlertLevel.WARNING,
                    title=f"Slow Workflow",
                    message=f"Workflow {workflow.workflow_id} took {workflow.duration_seconds:.1f}s "
                           f"(threshold: 60s)",
                    source="system",
                    workflow_id=workflow.workflow_id
                )

    def _check_system_health(self) -> None:
        """Check overall system health"""
        system_metrics = self.metrics_tracker.get_system_metrics()

        # Check overall success rate
        if system_metrics.total_workflows > 10:
            success_rate = system_metrics.successful_workflows / system_metrics.total_workflows

            if success_rate < 0.8:
                self._create_alert(
                    level=AlertLevel.CRITICAL,
                    title="Low System Success Rate",
                    message=f"System success rate is {success_rate:.1%} (threshold: 80%)",
                    source="system"
                )

    def _create_alert(self,
                     level: AlertLevel,
                     title: str,
                     message: str,
                     source: str,
                     workflow_id: Optional[str] = None) -> None:
        """
        Create and store an alert

        Args:
            level: Alert severity level
            title: Alert title
            message: Alert message
            source: Source of alert (agent name or "system")
            workflow_id: Optional workflow ID
        """
        with self.lock:
            # Check if similar alert already exists (avoid duplicates)
            for existing_alert in self.alerts:
                if (existing_alert.title == title and
                    existing_alert.source == source and
                    existing_alert.workflow_id == workflow_id):
                    # Update timestamp of existing alert
                    existing_alert.timestamp = datetime.now().isoformat()
                    return

            # Create new alert
            alert = Alert(
                alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                timestamp=datetime.now().isoformat(),
                level=level,
                title=title,
                message=message,
                source=source,
                workflow_id=workflow_id
            )

            self.alerts.append(alert)

            # Maintain max alerts limit
            if len(self.alerts) > self.max_alerts:
                # Move oldest to history
                old_alert = self.alerts.pop(0)
                self.alert_history.append(old_alert)

                # Maintain history limit
                if len(self.alert_history) > self.max_alert_history:
                    self.alert_history.pop(0)

            logger.info(f"Created alert: [{level}] {title}")

    def get_alerts(self,
                   level: Optional[AlertLevel] = None,
                   source: Optional[str] = None,
                   limit: Optional[int] = None) -> List[Alert]:
        """
        Get alerts with optional filtering

        Args:
            level: Filter by alert level
            source: Filter by source
            limit: Maximum number of alerts to return

        Returns:
            List of Alert objects
        """
        with self.lock:
            alerts = list(self.alerts)

            # Apply filters
            if level:
                alerts = [a for a in alerts if a.level == level]

            if source:
                alerts = [a for a in alerts if a.source == source]

            # Apply limit
            if limit:
                alerts = alerts[-limit:]

            return alerts

    def clear_alerts(self, level: Optional[AlertLevel] = None) -> int:
        """
        Clear alerts

        Args:
            level: Optional level to clear (clears all if None)

        Returns:
            Number of alerts cleared
        """
        with self.lock:
            if level:
                cleared = [a for a in self.alerts if a.level == level]
                self.alerts = [a for a in self.alerts if a.level != level]
                count = len(cleared)
            else:
                count = len(self.alerts)
                self.alerts = []

            logger.info(f"Cleared {count} alerts")
            return count

    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get current status of all agents

        Returns:
            Dictionary with agent status information
        """
        agent_metrics = self.metrics_tracker.get_agent_metrics()

        status = {}
        for agent_name, metrics in agent_metrics.items():
            # Determine health status
            health = "healthy"
            if metrics.total_executions > 0:
                error_rate = metrics.failed_executions / metrics.total_executions
                if error_rate > self.alert_threshold_error_rate:
                    health = "unhealthy"
                elif error_rate > self.alert_threshold_error_rate / 2:
                    health = "degraded"

            status[agent_name] = {
                "health": health,
                "total_executions": metrics.total_executions,
                "success_rate": (metrics.successful_executions / metrics.total_executions * 100)
                               if metrics.total_executions > 0 else 0.0,
                "avg_duration": metrics.avg_duration,
                "last_execution": metrics.last_execution,
                "violations_found": metrics.total_violations_found,
                "cache_hit_rate": metrics.cache_hit_rate * 100
            }

        return status

    def get_workflow_status(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get workflow execution status

        Args:
            workflow_id: Optional workflow ID (returns recent workflows if None)

        Returns:
            Dictionary with workflow status
        """
        if workflow_id:
            workflows = self.metrics_tracker.get_workflow_metrics(workflow_id=workflow_id)
            if not workflows:
                return {}
            workflow = workflows[0]

            # Get execution path from visualizer
            execution_summary = self.workflow_visualizer.generate_execution_summary(workflow_id)

            return {
                "workflow_id": workflow.workflow_id,
                "document_id": workflow.document_id,
                "status": workflow.status,
                "duration": workflow.duration_seconds,
                "agents_executed": workflow.agents_executed,
                "violations": workflow.total_violations,
                "errors": workflow.errors,
                "execution_path": execution_summary.get("execution_path", [])
            }
        else:
            # Return recent workflows
            recent = self.metrics_tracker.get_workflow_metrics(limit=10)
            return {
                "recent_workflows": [
                    {
                        "workflow_id": w.workflow_id,
                        "document_id": w.document_id,
                        "status": w.status,
                        "duration": w.duration_seconds,
                        "violations": w.total_violations
                    }
                    for w in recent
                ]
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics

        Returns:
            Dictionary with performance metrics
        """
        return self.metrics_tracker.get_performance_summary()

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status

        Returns:
            Dictionary with system health information
        """
        system_metrics = self.metrics_tracker.get_system_metrics()
        agent_metrics = self.metrics_tracker.get_agent_metrics()

        # Calculate health indicators
        total_agents = len(agent_metrics)
        healthy_agents = 0
        degraded_agents = 0
        unhealthy_agents = 0

        for metrics in agent_metrics.values():
            if metrics.total_executions > 0:
                error_rate = metrics.failed_executions / metrics.total_executions
                if error_rate > self.alert_threshold_error_rate:
                    unhealthy_agents += 1
                elif error_rate > self.alert_threshold_error_rate / 2:
                    degraded_agents += 1
                else:
                    healthy_agents += 1

        # Overall health status
        if unhealthy_agents > 0:
            overall_health = "unhealthy"
        elif degraded_agents > total_agents / 2:
            overall_health = "degraded"
        else:
            overall_health = "healthy"

        # Count alerts by level
        alert_counts = defaultdict(int)
        for alert in self.alerts:
            alert_counts[alert.level.value] += 1

        return {
            "overall_health": overall_health,
            "agents": {
                "total": total_agents,
                "healthy": healthy_agents,
                "degraded": degraded_agents,
                "unhealthy": unhealthy_agents
            },
            "workflows": {
                "total": system_metrics.total_workflows,
                "successful": system_metrics.successful_workflows,
                "failed": system_metrics.failed_workflows,
                "success_rate": (system_metrics.successful_workflows / system_metrics.total_workflows * 100)
                               if system_metrics.total_workflows > 0 else 0.0
            },
            "performance": {
                "avg_workflow_duration": system_metrics.avg_workflow_duration,
                "throughput_per_hour": system_metrics.workflows_per_hour,
                "cache_hit_rate": system_metrics.overall_cache_hit_rate * 100
            },
            "alerts": {
                "total": len(self.alerts),
                "by_level": dict(alert_counts)
            }
        }

    def display_dashboard(self, clear_screen: bool = True) -> None:
        """
        Display dashboard in console

        Args:
            clear_screen: Whether to clear screen before displaying
        """
        if clear_screen:
            # Clear screen (works on Windows and Unix)
            import os
            os.system('cls' if os.name == 'nt' else 'clear')

        # Get data
        system_health = self.get_system_health()
        agent_status = self.get_agent_status()
        workflow_status = self.get_workflow_status()
        alerts = self.get_alerts(limit=10)

        # Display header
        print("\n" + "="*100)
        print(" "*35 + "MULTI-AGENT SYSTEM MONITORING DASHBOARD")
        print("="*100)
        print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)

        # Display system health
        print("\n+-- SYSTEM HEALTH " + "-"*80 + "+")
        health_emoji = {"healthy": "[OK]", "degraded": "[!]", "unhealthy": "[X]"}
        health_color = {"healthy": "green", "degraded": "yellow", "unhealthy": "red"}

        print(f"| Overall Status: {health_emoji.get(system_health['overall_health'], '?')} "
              f"{system_health['overall_health'].upper():<20} "
              f"Workflows: {system_health['workflows']['total']:<5} "
              f"Success Rate: {system_health['workflows']['success_rate']:.1f}%{' '*20}|")
        print(f"| Agents: {system_health['agents']['healthy']} healthy, "
              f"{system_health['agents']['degraded']} degraded, "
              f"{system_health['agents']['unhealthy']} unhealthy{' '*30}|")
        print(f"| Throughput: {system_health['performance']['throughput_per_hour']:.1f} workflows/hour  "
              f"Avg Duration: {system_health['performance']['avg_workflow_duration']:.1f}s  "
              f"Cache Hit: {system_health['performance']['cache_hit_rate']:.1f}%{' '*10}|")
        print("+--" + "-"*96 + "+")

        # Display agent status
        print("\n+-- AGENT STATUS " + "-"*81 + "+")
        print(f"| {'Agent':<20} {'Status':<12} {'Executions':<12} {'Success':<10} "
              f"{'Avg Time':<10} {'Cache Hit':<10}|")
        print("+--" + "-"*96 + "+")

        for agent_name, status in sorted(agent_status.items())[:10]:
            health_symbol = health_emoji.get(status['health'], '?')
            print(f"| {agent_name:<20} {health_symbol} {status['health']:<10} "
                  f"{status['total_executions']:<12} {status['success_rate']:>6.1f}%{' '*3} "
                  f"{status['avg_duration']:>6.2f}s{' '*3} {status['cache_hit_rate']:>6.1f}%{' '*3}|")

        print("+--" + "-"*96 + "+")

        # Display recent workflows
        print("\n+-- RECENT WORKFLOWS " + "-"*77 + "+")
        print(f"| {'Workflow ID':<25} {'Document ID':<20} {'Status':<12} "
              f"{'Duration':<10} {'Violations':<10}|")
        print("+--" + "-"*96 + "+")

        for workflow in workflow_status.get('recent_workflows', [])[:5]:
            status_symbol = "[OK]" if workflow['status'] == "completed" else "[X]"
            duration_str = f"{workflow['duration']:.1f}s" if workflow['duration'] else "N/A"
            print(f"| {workflow['workflow_id']:<25} {workflow['document_id']:<20} "
                  f"{status_symbol} {workflow['status']:<10} {duration_str:<10} "
                  f"{workflow['violations']:<10}|")

        print("+--" + "-"*96 + "+")

        # Display alerts
        print("\n+-- ACTIVE ALERTS " + "-"*80 + "+")

        if not alerts:
            print("| No active alerts" + " "*81 + "|")
        else:
            for alert in alerts[-5:]:  # Show last 5 alerts
                level_emoji = {
                    AlertLevel.INFO: "[i]",
                    AlertLevel.WARNING: "[!]",
                    AlertLevel.ERROR: "[X]",
                    AlertLevel.CRITICAL: "[!!]"
                }
                symbol = level_emoji.get(alert.level, "[?]")
                timestamp = datetime.fromisoformat(alert.timestamp).strftime('%H:%M:%S')

                # Truncate message if too long
                message = alert.message
                if len(message) > 70:
                    message = message[:67] + "..."

                print(f"| {symbol} [{timestamp}] {alert.title:<30} {message:<40}|")

        print("+--" + "-"*96 + "+")

        print("\n" + "="*100)
        print("Press Ctrl+C to stop monitoring")
        print("="*100 + "\n")

    def export_dashboard_data(self, output_path: str) -> None:
        """
        Export dashboard data to JSON file

        Args:
            output_path: Output file path
        """
        data = {
            "exported_at": datetime.now().isoformat(),
            "system_health": self.get_system_health(),
            "agent_status": self.get_agent_status(),
            "workflow_status": self.get_workflow_status(),
            "performance_metrics": self.get_performance_metrics(),
            "alerts": [alert.to_dict() for alert in self.alerts]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported dashboard data to {output_path}")

    def generate_report(self, output_path: str, format: str = "text") -> None:
        """
        Generate monitoring report

        Args:
            output_path: Output file path
            format: Report format (text, json, html)
        """
        if format == "json":
            self.export_dashboard_data(output_path)
            return

        elif format == "text":
            with open(output_path, 'w') as f:
                # Write header
                f.write("="*100 + "\n")
                f.write(" "*30 + "MULTI-AGENT SYSTEM MONITORING REPORT\n")
                f.write("="*100 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*100 + "\n\n")

                # System health
                system_health = self.get_system_health()
                f.write("SYSTEM HEALTH\n")
                f.write("-"*100 + "\n")
                f.write(f"Overall Status: {system_health['overall_health'].upper()}\n")
                f.write(f"Total Workflows: {system_health['workflows']['total']}\n")
                f.write(f"Success Rate: {system_health['workflows']['success_rate']:.1f}%\n")
                f.write(f"Throughput: {system_health['performance']['throughput_per_hour']:.1f} workflows/hour\n")
                f.write(f"Avg Duration: {system_health['performance']['avg_workflow_duration']:.1f}s\n")
                f.write(f"Cache Hit Rate: {system_health['performance']['cache_hit_rate']:.1f}%\n\n")

                # Agent status
                agent_status = self.get_agent_status()
                f.write("AGENT STATUS\n")
                f.write("-"*100 + "\n")
                f.write(f"{'Agent':<20} {'Health':<12} {'Executions':<12} {'Success Rate':<15} "
                       f"{'Avg Duration':<15} {'Cache Hit':<12}\n")
                f.write("-"*100 + "\n")

                for agent_name, status in sorted(agent_status.items()):
                    f.write(f"{agent_name:<20} {status['health']:<12} {status['total_executions']:<12} "
                           f"{status['success_rate']:>6.1f}%{' '*8} {status['avg_duration']:>6.2f}s{' '*8} "
                           f"{status['cache_hit_rate']:>6.1f}%\n")

                f.write("\n")

                # Alerts
                f.write("ACTIVE ALERTS\n")
                f.write("-"*100 + "\n")

                alerts = self.get_alerts()
                if not alerts:
                    f.write("No active alerts\n")
                else:
                    for alert in alerts:
                        f.write(f"[{alert.level.value.upper()}] {alert.title}\n")
                        f.write(f"  Time: {alert.timestamp}\n")
                        f.write(f"  Source: {alert.source}\n")
                        f.write(f"  Message: {alert.message}\n\n")

            logger.info(f"Generated text report: {output_path}")

        else:
            raise ValueError(f"Unsupported format: {format}")

    def run_interactive_dashboard(self, refresh_interval: Optional[int] = None) -> None:
        """
        Run interactive dashboard with auto-refresh

        Args:
            refresh_interval: Refresh interval in seconds (uses default if None)
        """
        interval = refresh_interval or self.refresh_interval

        print("\n" + "="*100)
        print(" "*30 + "STARTING INTERACTIVE DASHBOARD")
        print("="*100)
        print(f"\nRefresh Interval: {interval} seconds")
        print("Press Ctrl+C to exit\n")

        try:
            while True:
                self.display_dashboard(clear_screen=True)
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n" + "="*100)
            print(" "*35 + "DASHBOARD STOPPED")
            print("="*100 + "\n")


# Global dashboard instance
_global_dashboard: Optional[MonitoringDashboard] = None


def get_monitoring_dashboard() -> MonitoringDashboard:
    """
    Get global monitoring dashboard instance (singleton)

    Returns:
        MonitoringDashboard instance
    """
    global _global_dashboard
    if _global_dashboard is None:
        _global_dashboard = MonitoringDashboard()
    return _global_dashboard


def initialize_monitoring_dashboard(
    agent_logger: Optional[AgentLogger] = None,
    metrics_tracker: Optional[MetricsTracker] = None,
    workflow_visualizer: Optional[WorkflowVisualizer] = None,
    alert_threshold_error_rate: float = 0.2,
    alert_threshold_duration: float = 30.0,
    alert_threshold_cache_hit_rate: float = 0.3
) -> MonitoringDashboard:
    """
    Initialize global monitoring dashboard with custom settings

    Args:
        agent_logger: AgentLogger instance
        metrics_tracker: MetricsTracker instance
        workflow_visualizer: WorkflowVisualizer instance
        alert_threshold_error_rate: Error rate threshold for alerts
        alert_threshold_duration: Duration threshold for alerts
        alert_threshold_cache_hit_rate: Cache hit rate threshold for alerts

    Returns:
        MonitoringDashboard instance
    """
    global _global_dashboard
    _global_dashboard = MonitoringDashboard(
        agent_logger=agent_logger,
        metrics_tracker=metrics_tracker,
        workflow_visualizer=workflow_visualizer,
        alert_threshold_error_rate=alert_threshold_error_rate,
        alert_threshold_duration=alert_threshold_duration,
        alert_threshold_cache_hit_rate=alert_threshold_cache_hit_rate
    )
    return _global_dashboard


# Export all public symbols
__all__ = [
    "MonitoringDashboard",
    "Alert",
    "AlertLevel",
    "get_monitoring_dashboard",
    "initialize_monitoring_dashboard"
]



if __name__ == "__main__":
    # Example usage and testing
    import sys

    logger.info("="*100)
    logger.info(" "*30 + "MONITORING DASHBOARD - DEMO MODE")
    logger.info("="*100)

    # Initialize components
    logger.info("\n[*] Initializing monitoring components...")

    try:
        from monitoring.agent_logger import initialize_agent_logger
        from monitoring.metrics_tracker import initialize_metrics_tracker
        from monitoring.workflow_visualizer import initialize_workflow_visualizer
    except ModuleNotFoundError:
        # Handle direct execution
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from monitoring.agent_logger import initialize_agent_logger
        from monitoring.metrics_tracker import initialize_metrics_tracker
        from monitoring.workflow_visualizer import initialize_workflow_visualizer

    agent_logger = initialize_agent_logger(log_dir="./test_monitoring/logs/")
    metrics_tracker = initialize_metrics_tracker(
        metrics_dir="./test_monitoring/metrics/",
        auto_save=False  # Disable auto-save to prevent blocking
    )
    workflow_visualizer = initialize_workflow_visualizer(
        output_dir="./test_monitoring/visualizations/"
    )

    logger.info("  [OK] AgentLogger initialized")
    logger.info("  [OK] MetricsTracker initialized")
    logger.info("  [OK] WorkflowVisualizer initialized")

    # Initialize dashboard
    dashboard = MonitoringDashboard(
        agent_logger=agent_logger,
        metrics_tracker=metrics_tracker,
        workflow_visualizer=workflow_visualizer
    )

    logger.info("  [OK] MonitoringDashboard initialized")

    # Simulate some data
    logger.info("\n[*] Simulating workflow executions...")

    # Simulate workflow 1
    workflow_id_1 = "workflow_demo_001"
    metrics_tracker.start_workflow(workflow_id_1, "doc_001")
    workflow_visualizer.start_execution_tracking(workflow_id_1, "doc_001")

    agents = ["supervisor", "preprocessor", "structure", "performance", "securities", "general"]

    for i, agent_name in enumerate(agents):
        exec_id = f"exec_{i}"
        started = datetime.now().isoformat()

        metrics_tracker.start_agent_execution(workflow_id_1, agent_name, exec_id)
        workflow_visualizer.record_node_execution(
            workflow_id_1, agent_name, NodeStatus.RUNNING, started_at=started
        )

        time.sleep(0.05)

        completed = datetime.now().isoformat()
        metrics_tracker.complete_agent_execution(
            exec_id, status="completed", violations_found=i,
            api_calls=3, cache_hits=2
        )
        workflow_visualizer.record_node_execution(
            workflow_id_1, agent_name, NodeStatus.COMPLETED,
            started_at=started, completed_at=completed, duration_seconds=0.05
        )

        agent_logger.log_agent_execution(
            agent_name=agent_name,
            execution_id=exec_id,
            workflow_id=workflow_id_1,
            started_at=started,
            completed_at=completed,
            duration_seconds=0.05,
            status="completed",
            input_state={"document_id": "doc_001"},
            output_state={"document_id": "doc_001", "violations": []},
            violations_added=i,
            api_calls=3,
            cache_hits=2
        )

    metrics_tracker.complete_workflow(workflow_id_1, status="completed", total_violations=15)
    workflow_visualizer.complete_execution_tracking(workflow_id_1, status="completed")

    logger.info(f"  [OK] Simulated workflow: {workflow_id_1}")

    # Simulate workflow 2 with failure
    workflow_id_2 = "workflow_demo_002"
    metrics_tracker.start_workflow(workflow_id_2, "doc_002")

    exec_id = "exec_fail"
    metrics_tracker.start_agent_execution(workflow_id_2, "structure", exec_id)
    time.sleep(0.05)
    metrics_tracker.complete_agent_execution(
        exec_id, status="failed", violations_found=0,
        api_calls=5, cache_hits=0, error="Connection timeout"
    )

    metrics_tracker.complete_workflow(
        workflow_id_2, status="failed", total_violations=0,
        errors=["structure: Connection timeout"]
    )

    logger.info(f"  [OK] Simulated failed workflow: {workflow_id_2}")

    # Display dashboard
    logger.info("\n" + "="*100)
    logger.info("Displaying dashboard...")
    logger.info("="*100 + "\n")

    time.sleep(1)
    dashboard.display_dashboard(clear_screen=False)

    # Export data
    logger.info("\n[*] Exporting dashboard data...")
    try:
        dashboard.export_dashboard_data("./test_monitoring/dashboard_data.json")
        logger.info("  [OK] Exported dashboard data")
    except Exception as e:
        logger.info(f"  [ERROR] Failed to export dashboard data: {e}")

    try:
        dashboard.generate_report("./test_monitoring/monitoring_report.txt", format="text")
        logger.info("  [OK] Generated monitoring report")
    except Exception as e:
        logger.info(f"  [ERROR] Failed to generate report: {e}")

    # Cleanup (skip shutdown in demo mode to avoid hanging)
    logger.info("\n[*] Demo complete (skipping cleanup to avoid blocking)")

    # Test interactive mode
    logger.info("\n" + "="*100)
    logger.info("To run interactive dashboard, use:")
    logger.info("  python monitoring/dashboard.py --interactive")
    logger.info("="*100)

    # Check for interactive flag
    if "--interactive" in sys.argv:
        logger.info("\n[!] Starting interactive mode (Press Ctrl+C to exit)...")
        time.sleep(2)
        dashboard.run_interactive_dashboard(refresh_interval=5)

    logger.info("\n[OK] Dashboard demo complete")
