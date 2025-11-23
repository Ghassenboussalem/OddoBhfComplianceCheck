#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent Logger

This module provides functionality for the multi-agent compliance system.
"""

"""
Agent Execution Logger - Structured Logging for Multi-Agent System

This module provides comprehensive logging for agent executions in the
LangGraph-based multi-agent compliance system. It tracks all agent invocations,
inputs, outputs, durations, and errors with structured JSON logging and
automatic log rotation.
"""

import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import hashlib


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Log levels for agent execution"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class AgentExecutionLog:
    """Single agent execution log entry"""
    log_id: str
    timestamp: str  # ISO format
    agent_name: str
    execution_id: str  # Unique ID for this execution
    workflow_id: Optional[str]  # ID of the workflow this execution belongs to

    # Execution details
    started_at: str
    completed_at: Optional[str]
    duration_seconds: Optional[float]
    status: str  # running, completed, failed, skipped

    # Input/Output
    input_state_summary: Dict[str, Any]  # Summary of input state
    output_state_summary: Dict[str, Any]  # Summary of output state
    violations_added: int

    # Error information
    error: Optional[str]
    error_type: Optional[str]
    stack_trace: Optional[str]

    # Performance metrics
    api_calls: int
    cache_hits: int
    tools_invoked: List[str]

    # Additional metadata
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentExecutionLog':
        """Create AgentExecutionLog from dictionary"""
        return cls(**data)


@dataclass
class WorkflowExecutionLog:
    """Log entry for complete workflow execution"""
    workflow_id: str
    timestamp: str
    document_id: str

    # Workflow details
    started_at: str
    completed_at: Optional[str]
    duration_seconds: Optional[float]
    status: str  # running, completed, failed, interrupted

    # Agent executions
    agents_executed: List[str]
    total_violations: int

    # Performance summary
    total_api_calls: int
    total_cache_hits: int
    total_duration_by_agent: Dict[str, float]

    # Error information
    errors: List[Dict[str, Any]]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'WorkflowExecutionLog':
        """Create WorkflowExecutionLog from dictionary"""
        return cls(**data)


class AgentLogger:
    """
    Structured logger for agent executions

    Features:
    - Structured JSON logging
    - Agent execution tracking with inputs/outputs
    - Performance metrics collection
    - Automatic log rotation
    - Thread-safe operations
    - Workflow-level aggregation
    """

    def __init__(self,
                 log_dir: str = "./monitoring/logs/",
                 agent_log_file: str = "agent_executions.json",
                 workflow_log_file: str = "workflow_executions.json",
                 max_entries_per_file: int = 10000,
                 rotation_size_mb: int = 50):
        """
        Initialize agent logger

        Args:
            log_dir: Directory for log storage
            agent_log_file: Name of agent execution log file
            workflow_log_file: Name of workflow execution log file
            max_entries_per_file: Maximum entries before rotating
            rotation_size_mb: Maximum file size in MB before rotating
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.agent_log_file = self.log_dir / agent_log_file
        self.workflow_log_file = self.log_dir / workflow_log_file
        self.max_entries_per_file = max_entries_per_file
        self.rotation_size_mb = rotation_size_mb

        # In-memory log entries
        self.agent_logs: List[AgentExecutionLog] = []
        self.workflow_logs: List[WorkflowExecutionLog] = []

        # Thread safety
        self.lock = threading.Lock()

        # Load existing logs
        self._load_logs()

        logger.info(f"AgentLogger initialized with {len(self.agent_logs)} agent logs, "
                   f"{len(self.workflow_logs)} workflow logs")

    def log_agent_execution(self,
                           agent_name: str,
                           execution_id: str,
                           workflow_id: Optional[str],
                           started_at: str,
                           completed_at: Optional[str],
                           duration_seconds: Optional[float],
                           status: str,
                           input_state: Dict[str, Any],
                           output_state: Optional[Dict[str, Any]],
                           violations_added: int = 0,
                           error: Optional[str] = None,
                           error_type: Optional[str] = None,
                           stack_trace: Optional[str] = None,
                           api_calls: int = 0,
                           cache_hits: int = 0,
                           tools_invoked: Optional[List[str]] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """
        Log an agent execution

        Args:
            agent_name: Name of the agent
            execution_id: Unique execution ID
            workflow_id: ID of parent workflow
            started_at: Start timestamp (ISO format)
            completed_at: Completion timestamp (ISO format)
            duration_seconds: Execution duration
            status: Execution status (running, completed, failed, skipped)
            input_state: Input state (will be summarized)
            output_state: Output state (will be summarized)
            violations_added: Number of violations added
            error: Error message if failed
            error_type: Type of error
            stack_trace: Full stack trace
            api_calls: Number of API calls made
            cache_hits: Number of cache hits
            tools_invoked: List of tool names invoked
            metadata: Additional metadata
        """
        with self.lock:
            # Generate log ID
            log_id = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            # Summarize states (avoid storing full document content)
            input_summary = self._summarize_state(input_state)
            output_summary = self._summarize_state(output_state) if output_state else {}

            # Create log entry
            log_entry = AgentExecutionLog(
                log_id=log_id,
                timestamp=datetime.now().isoformat(),
                agent_name=agent_name,
                execution_id=execution_id,
                workflow_id=workflow_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration_seconds,
                status=status,
                input_state_summary=input_summary,
                output_state_summary=output_summary,
                violations_added=violations_added,
                error=error,
                error_type=error_type,
                stack_trace=stack_trace,
                api_calls=api_calls,
                cache_hits=cache_hits,
                tools_invoked=tools_invoked or [],
                metadata=metadata or {}
            )

            # Add to logs
            self.agent_logs.append(log_entry)

            # Save to disk
            self._save_agent_logs()

            # Check if rotation needed
            self._check_rotation()

            logger.debug(f"Logged agent execution: {agent_name} ({execution_id})")

    def log_workflow_execution(self,
                              workflow_id: str,
                              document_id: str,
                              started_at: str,
                              completed_at: Optional[str],
                              duration_seconds: Optional[float],
                              status: str,
                              agents_executed: List[str],
                              total_violations: int,
                              total_api_calls: int,
                              total_cache_hits: int,
                              total_duration_by_agent: Dict[str, float],
                              errors: Optional[List[Dict[str, Any]]] = None):
        """
        Log a complete workflow execution

        Args:
            workflow_id: Unique workflow ID
            document_id: Document being processed
            started_at: Start timestamp (ISO format)
            completed_at: Completion timestamp (ISO format)
            duration_seconds: Total workflow duration
            status: Workflow status (running, completed, failed, interrupted)
            agents_executed: List of agent names executed
            total_violations: Total violations found
            total_api_calls: Total API calls across all agents
            total_cache_hits: Total cache hits across all agents
            total_duration_by_agent: Duration breakdown by agent
            errors: List of errors encountered
        """
        with self.lock:
            # Create log entry
            log_entry = WorkflowExecutionLog(
                workflow_id=workflow_id,
                timestamp=datetime.now().isoformat(),
                document_id=document_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration_seconds,
                status=status,
                agents_executed=agents_executed,
                total_violations=total_violations,
                total_api_calls=total_api_calls,
                total_cache_hits=total_cache_hits,
                total_duration_by_agent=total_duration_by_agent,
                errors=errors or []
            )

            # Add to logs
            self.workflow_logs.append(log_entry)

            # Save to disk
            self._save_workflow_logs()

            # Check if rotation needed
            self._check_rotation()

            logger.info(f"Logged workflow execution: {workflow_id} ({document_id})")

    def _summarize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of state without full document content

        Args:
            state: Full state dictionary

        Returns:
            Summarized state
        """
        if not state:
            return {}

        summary = {
            "document_id": state.get("document_id"),
            "document_type": state.get("document_type"),
            "client_type": state.get("client_type"),
            "current_agent": state.get("current_agent"),
            "workflow_status": state.get("workflow_status"),
            "violations_count": len(state.get("violations", [])),
            "whitelist_size": len(state.get("whitelist", set())),
            "review_queue_size": len(state.get("review_queue", [])),
            "error_count": len(state.get("error_log", [])),
            "has_metadata": bool(state.get("metadata")),
            "has_context_analysis": bool(state.get("context_analysis")),
            "has_evidence": bool(state.get("evidence_extractions"))
        }

        return summary

    def get_agent_logs(self,
                       agent_name: Optional[str] = None,
                       workflow_id: Optional[str] = None,
                       status: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> List[AgentExecutionLog]:
        """
        Get agent execution logs with optional filtering

        Args:
            agent_name: Filter by agent name
            workflow_id: Filter by workflow ID
            status: Filter by status
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)

        Returns:
            List of AgentExecutionLog objects
        """
        with self.lock:
            logs = list(self.agent_logs)

            # Apply filters
            if agent_name:
                logs = [log for log in logs if log.agent_name == agent_name]

            if workflow_id:
                logs = [log for log in logs if log.workflow_id == workflow_id]

            if status:
                logs = [log for log in logs if log.status == status]

            if start_date:
                start = datetime.fromisoformat(start_date)
                logs = [log for log in logs
                       if datetime.fromisoformat(log.timestamp) >= start]

            if end_date:
                end = datetime.fromisoformat(end_date)
                logs = [log for log in logs
                       if datetime.fromisoformat(log.timestamp) <= end]

            return logs

    def get_workflow_logs(self,
                         document_id: Optional[str] = None,
                         status: Optional[str] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> List[WorkflowExecutionLog]:
        """
        Get workflow execution logs with optional filtering

        Args:
            document_id: Filter by document ID
            status: Filter by status
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)

        Returns:
            List of WorkflowExecutionLog objects
        """
        with self.lock:
            logs = list(self.workflow_logs)

            # Apply filters
            if document_id:
                logs = [log for log in logs if log.document_id == document_id]

            if status:
                logs = [log for log in logs if log.status == status]

            if start_date:
                start = datetime.fromisoformat(start_date)
                logs = [log for log in logs
                       if datetime.fromisoformat(log.timestamp) >= start]

            if end_date:
                end = datetime.fromisoformat(end_date)
                logs = [log for log in logs
                       if datetime.fromisoformat(log.timestamp) <= end]

            return logs

    def get_agent_statistics(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get execution statistics for agents

        Args:
            agent_name: Optional agent name to filter by

        Returns:
            Dictionary with statistics
        """
        logs = self.get_agent_logs(agent_name=agent_name)

        if not logs:
            return {
                "total_executions": 0,
                "by_status": {},
                "avg_duration_seconds": 0.0,
                "total_violations": 0,
                "total_api_calls": 0,
                "total_cache_hits": 0,
                "error_rate": 0.0
            }

        # Calculate statistics
        by_status = {}
        total_duration = 0.0
        duration_count = 0
        total_violations = 0
        total_api_calls = 0
        total_cache_hits = 0
        error_count = 0

        for log in logs:
            # Count by status
            by_status[log.status] = by_status.get(log.status, 0) + 1

            # Duration
            if log.duration_seconds is not None:
                total_duration += log.duration_seconds
                duration_count += 1

            # Violations
            total_violations += log.violations_added

            # API calls and cache
            total_api_calls += log.api_calls
            total_cache_hits += log.cache_hits

            # Errors
            if log.status == "failed":
                error_count += 1

        avg_duration = total_duration / duration_count if duration_count > 0 else 0.0
        error_rate = error_count / len(logs) if logs else 0.0

        return {
            "total_executions": len(logs),
            "by_status": by_status,
            "avg_duration_seconds": avg_duration,
            "total_violations": total_violations,
            "total_api_calls": total_api_calls,
            "total_cache_hits": total_cache_hits,
            "cache_hit_rate": total_cache_hits / (total_api_calls + total_cache_hits) if (total_api_calls + total_cache_hits) > 0 else 0.0,
            "error_rate": error_rate
        }

    def export_logs(self,
                   filepath: str,
                   log_type: str = "agent",
                   filters: Optional[Dict] = None):
        """
        Export logs to JSON file

        Args:
            filepath: Output file path
            log_type: Type of logs to export ("agent" or "workflow")
            filters: Optional filters to apply
        """
        if log_type == "agent":
            logs = self.get_agent_logs(**(filters or {}))
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "log_type": "agent_executions",
                "total_entries": len(logs),
                "entries": [log.to_dict() for log in logs]
            }
        elif log_type == "workflow":
            logs = self.get_workflow_logs(**(filters or {}))
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "log_type": "workflow_executions",
                "total_entries": len(logs),
                "entries": [log.to_dict() for log in logs]
            }
        else:
            raise ValueError(f"Invalid log_type: {log_type}")

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(logs)} {log_type} logs to {filepath}")

    def _load_logs(self):
        """Load logs from files"""
        try:
            # Load agent logs
            if self.agent_log_file.exists():
                with open(self.agent_log_file, 'r') as f:
                    data = json.load(f)

                for entry_data in data.get('entries', []):
                    log = AgentExecutionLog.from_dict(entry_data)
                    self.agent_logs.append(log)

                logger.info(f"Loaded {len(self.agent_logs)} agent logs")

            # Load workflow logs
            if self.workflow_log_file.exists():
                with open(self.workflow_log_file, 'r') as f:
                    data = json.load(f)

                for entry_data in data.get('entries', []):
                    log = WorkflowExecutionLog.from_dict(entry_data)
                    self.workflow_logs.append(log)

                logger.info(f"Loaded {len(self.workflow_logs)} workflow logs")

        except Exception as e:
            logger.error(f"Error loading logs: {e}")

    def _save_agent_logs(self):
        """Save agent logs to file"""
        try:
            data = {
                "schema_version": 1,
                "last_updated": datetime.now().isoformat(),
                "total_entries": len(self.agent_logs),
                "entries": [log.to_dict() for log in self.agent_logs]
            }

            with open(self.agent_log_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self.agent_logs)} agent logs")

        except Exception as e:
            logger.error(f"Error saving agent logs: {e}")

    def _save_workflow_logs(self):
        """Save workflow logs to file"""
        try:
            data = {
                "schema_version": 1,
                "last_updated": datetime.now().isoformat(),
                "total_entries": len(self.workflow_logs),
                "entries": [log.to_dict() for log in self.workflow_logs]
            }

            with open(self.workflow_log_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self.workflow_logs)} workflow logs")

        except Exception as e:
            logger.error(f"Error saving workflow logs: {e}")

    def _check_rotation(self):
        """Check if log rotation is needed"""
        # Check agent logs
        if len(self.agent_logs) >= self.max_entries_per_file:
            self._rotate_agent_logs()

        # Check workflow logs
        if len(self.workflow_logs) >= self.max_entries_per_file:
            self._rotate_workflow_logs()

        # Check file sizes
        if self.agent_log_file.exists():
            size_mb = self.agent_log_file.stat().st_size / (1024 * 1024)
            if size_mb >= self.rotation_size_mb:
                self._rotate_agent_logs()

        if self.workflow_log_file.exists():
            size_mb = self.workflow_log_file.stat().st_size / (1024 * 1024)
            if size_mb >= self.rotation_size_mb:
                self._rotate_workflow_logs()

    def _rotate_agent_logs(self):
        """Rotate agent logs"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_name = f"agent_executions_{timestamp}.json"
            archive_path = self.log_dir / archive_name

            # Save current logs to archive
            archive_data = {
                "schema_version": 1,
                "archived_at": datetime.now().isoformat(),
                "total_entries": len(self.agent_logs),
                "entries": [log.to_dict() for log in self.agent_logs]
            }

            with open(archive_path, 'w') as f:
                json.dump(archive_data, f, indent=2)

            logger.info(f"Rotated agent logs: archived {len(self.agent_logs)} entries to {archive_name}")

            # Clear in-memory logs
            self.agent_logs = []

            # Save empty log file
            self._save_agent_logs()

        except Exception as e:
            logger.error(f"Error rotating agent logs: {e}")

    def _rotate_workflow_logs(self):
        """Rotate workflow logs"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_name = f"workflow_executions_{timestamp}.json"
            archive_path = self.log_dir / archive_name

            # Save current logs to archive
            archive_data = {
                "schema_version": 1,
                "archived_at": datetime.now().isoformat(),
                "total_entries": len(self.workflow_logs),
                "entries": [log.to_dict() for log in self.workflow_logs]
            }

            with open(archive_path, 'w') as f:
                json.dump(archive_data, f, indent=2)

            logger.info(f"Rotated workflow logs: archived {len(self.workflow_logs)} entries to {archive_name}")

            # Clear in-memory logs
            self.workflow_logs = []

            # Save empty log file
            self._save_workflow_logs()

        except Exception as e:
            logger.error(f"Error rotating workflow logs: {e}")

    def clear_logs(self, log_type: Optional[str] = None):
        """
        Clear logs from memory and disk

        Args:
            log_type: Type of logs to clear ("agent", "workflow", or None for both)
        """
        with self.lock:
            if log_type is None or log_type == "agent":
                self.agent_logs = []
                self._save_agent_logs()
                logger.info("Cleared agent logs")

            if log_type is None or log_type == "workflow":
                self.workflow_logs = []
                self._save_workflow_logs()
                logger.info("Cleared workflow logs")

    def print_statistics(self, agent_name: Optional[str] = None):
        """
        Print execution statistics to console

        Args:
            agent_name: Optional agent name to filter by
        """
        stats = self.get_agent_statistics(agent_name)

        print("\n" + "="*70)
        print(f"Agent Execution Statistics{' - ' + agent_name if agent_name else ''}")
        print("="*70)
        print(f"\nTotal Executions: {stats['total_executions']}")
        print(f"\nBy Status:")
        for status, count in stats['by_status'].items():
            print(f"  {status}: {count}")
        print(f"\nPerformance:")
        print(f"  Avg Duration: {stats['avg_duration_seconds']:.2f}s")
        print(f"  Total Violations: {stats['total_violations']}")
        print(f"  Total API Calls: {stats['total_api_calls']}")
        print(f"  Total Cache Hits: {stats['total_cache_hits']}")
        print(f"  Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
        print(f"  Error Rate: {stats['error_rate']:.1%}")
        print("="*70 + "\n")


# Global logger instance
_global_logger: Optional[AgentLogger] = None


def get_agent_logger() -> AgentLogger:
    """
    Get global agent logger instance (singleton)

    Returns:
        AgentLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = AgentLogger()
    return _global_logger


def initialize_agent_logger(log_dir: str = "./monitoring/logs/",
                           max_entries_per_file: int = 10000,
                           rotation_size_mb: int = 50) -> AgentLogger:
    """
    Initialize global agent logger with custom settings

    Args:
        log_dir: Directory for log storage
        max_entries_per_file: Maximum entries before rotating
        rotation_size_mb: Maximum file size in MB before rotating

    Returns:
        AgentLogger instance
    """
    global _global_logger
    _global_logger = AgentLogger(
        log_dir=log_dir,
        max_entries_per_file=max_entries_per_file,
        rotation_size_mb=rotation_size_mb
    )
    return _global_logger


# Export all public symbols
__all__ = [
    "AgentLogger",
    "AgentExecutionLog",
    "WorkflowExecutionLog",
    "LogLevel",
    "get_agent_logger",
    "initialize_agent_logger"
]


if __name__ == "__main__":
    # Example usage and testing
    logger.info("="*70)
    logger.info("Agent Logger - Structured Logging for Multi-Agent System")
    logger.info("="*70)

    # Initialize logger
    agent_logger = AgentLogger(log_dir="./test_monitoring_logs/")

    logger.info(f"\n✓ Agent logger initialized")
    logger.info(f"  Log directory: ./test_monitoring_logs/")
    logger.info(f"  Existing agent logs: {len(agent_logger.agent_logs)}")
    logger.info(f"  Existing workflow logs: {len(agent_logger.workflow_logs)}")

    # Simulate agent executions
    logger.info("\n📝 Logging test agent executions...")

    workflow_id = "workflow_test_001"

    for i in range(3):
        agent_name = ["structure", "performance", "securities"][i]
        execution_id = f"exec_{i}"
        started_at = datetime.now().isoformat()

        # Simulate execution
        import time
        time.sleep(0.1)

        completed_at = datetime.now().isoformat()
        duration = 0.1

        # Mock state
        input_state = {
            "document_id": "doc_001",
            "document_type": "fund_presentation",
            "client_type": "retail",
            "violations": [],
            "whitelist": {"fund", "performance"}
        }

        output_state = input_state.copy()
        output_state["violations"] = [{"rule": f"rule_{i}", "type": agent_name}]

        agent_logger.log_agent_execution(
            agent_name=agent_name,
            execution_id=execution_id,
            workflow_id=workflow_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            status="completed",
            input_state=input_state,
            output_state=output_state,
            violations_added=1,
            api_calls=2,
            cache_hits=1,
            tools_invoked=[f"check_{agent_name}"],
            metadata={"test": True}
        )

    logger.info(f"  ✓ Logged {3} agent executions")

    # Log workflow execution
    logger.info("\n📊 Logging workflow execution...")

    agent_logger.log_workflow_execution(
        workflow_id=workflow_id,
        document_id="doc_001",
        started_at=datetime.now().isoformat(),
        completed_at=datetime.now().isoformat(),
        duration_seconds=0.3,
        status="completed",
        agents_executed=["structure", "performance", "securities"],
        total_violations=3,
        total_api_calls=6,
        total_cache_hits=3,
        total_duration_by_agent={
            "structure": 0.1,
            "performance": 0.1,
            "securities": 0.1
        }
    )

    logger.info(f"  ✓ Logged workflow execution")

    # Print statistics
    agent_logger.print_statistics()

    # Export logs
    logger.info("\n💾 Exporting logs...")
    agent_logger.export_logs("test_agent_logs.json", log_type="agent")
    agent_logger.export_logs("test_workflow_logs.json", log_type="workflow")
    logger.info(f"  ✓ Exported logs")

    logger.info("\n" + "="*70)
    logger.info("✓ Agent Logger test complete")
    logger.info("="*70)
