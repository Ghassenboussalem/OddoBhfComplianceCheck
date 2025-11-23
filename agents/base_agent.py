#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Agent

This module provides functionality for the multi-agent compliance system.
"""

"""
Base Agent Framework for Multi-Agent Compliance System

This module provides the foundational classes and utilities for building
specialized compliance checking agents in the LangGraph-based system.
"""

import logging
import time
import functools
import traceback
from typing import Callable, Optional, Dict, Any, List, Type
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Import state models
from data_models_multiagent import (
    ComplianceState,
    WorkflowStatus,
    ViolationRecord,
    AgentExecutionRecord,
    update_state_timestamp
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Status of agent execution"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AgentConfig:
    """Configuration for an agent"""
    name: str
    enabled: bool = True
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    log_level: str = "INFO"
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get custom setting value"""
        return self.custom_settings.get(key, default)


class AgentRegistry:
    """
    Registry for dynamically loading and managing agents

    Allows agents to be registered by name and instantiated on demand.
    """

    _instance = None
    _agents: Dict[str, Type['BaseAgent']] = {}

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize registry"""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            logger.info("AgentRegistry initialized")

    @classmethod
    def register(cls, name: str, agent_class: Type['BaseAgent']):
        """
        Register an agent class

        Args:
            name: Unique name for the agent
            agent_class: Agent class (must inherit from BaseAgent)
        """
        if not issubclass(agent_class, BaseAgent):
            raise TypeError(f"Agent class must inherit from BaseAgent")

        cls._agents[name] = agent_class
        logger.info(f"Registered agent: {name} ({agent_class.__name__})")

    @classmethod
    def unregister(cls, name: str):
        """
        Unregister an agent

        Args:
            name: Agent name to unregister
        """
        if name in cls._agents:
            del cls._agents[name]
            logger.info(f"Unregistered agent: {name}")

    @classmethod
    def get_agent_class(cls, name: str) -> Optional[Type['BaseAgent']]:
        """
        Get agent class by name

        Args:
            name: Agent name

        Returns:
            Agent class or None if not found
        """
        return cls._agents.get(name)

    @classmethod
    def create_agent(cls, name: str, config: Optional[AgentConfig] = None, **kwargs) -> Optional['BaseAgent']:
        """
        Create an agent instance

        Args:
            name: Agent name
            config: Agent configuration
            **kwargs: Additional arguments for agent constructor

        Returns:
            Agent instance or None if not found
        """
        agent_class = cls.get_agent_class(name)
        if not agent_class:
            logger.error(f"Agent not found: {name}")
            return None

        if config is None:
            config = AgentConfig(name=name)

        try:
            agent = agent_class(config=config, **kwargs)
            logger.info(f"Created agent instance: {name}")
            return agent
        except Exception as e:
            logger.error(f"Failed to create agent {name}: {e}")
            return None

    @classmethod
    def list_agents(cls) -> List[str]:
        """
        Get list of registered agent names

        Returns:
            List of agent names
        """
        return list(cls._agents.keys())

    @classmethod
    def clear(cls):
        """Clear all registered agents"""
        cls._agents.clear()
        logger.info("Agent registry cleared")


def agent_timing(func: Callable) -> Callable:
    """
    Decorator to measure and log agent execution time with detailed profiling

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(self, state: ComplianceState, *args, **kwargs) -> ComplianceState:
        agent_name = getattr(self, 'name', func.__name__)
        start_time = time.time()
        start_cpu = time.process_time()

        logger.info(f"[{agent_name}] Starting execution")

        try:
            result = func(self, state, *args, **kwargs)

            duration = time.time() - start_time
            cpu_time = time.process_time() - start_cpu

            # Record detailed timing in state
            if "agent_timings" not in result:
                result["agent_timings"] = {}
            result["agent_timings"][agent_name] = duration

            # Record detailed profiling data
            if "agent_profiling" not in result:
                result["agent_profiling"] = {}
            result["agent_profiling"][agent_name] = {
                "wall_time": duration,
                "cpu_time": cpu_time,
                "io_wait": duration - cpu_time,
                "timestamp": datetime.now().isoformat()
            }

            # Log performance warning if slow
            if duration > 5.0:
                logger.warning(f"[{agent_name}] Slow execution: {duration:.2f}s (CPU: {cpu_time:.2f}s, I/O: {duration - cpu_time:.2f}s)")
            else:
                logger.info(f"[{agent_name}] Completed in {duration:.2f}s (CPU: {cpu_time:.2f}s)")

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{agent_name}] Failed after {duration:.2f}s: {e}")
            raise

    return wrapper



def agent_error_handler(func: Callable) -> Callable:
    """
    Decorator to handle agent errors gracefully with retry and fallback

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(self, state: ComplianceState, *args, **kwargs) -> ComplianceState:
        agent_name = getattr(self, 'name', func.__name__)
        max_retries = getattr(self, 'retry_attempts', 3)

        last_exception = None

        # Try with retries
        for attempt in range(max_retries):
            try:
                return func(self, state, *args, **kwargs)

            except Exception as e:
                last_exception = e
                error_type = type(e).__name__

                if attempt < max_retries - 1:
                    # Calculate exponential backoff delay
                    delay = min(1.0 * (2 ** attempt), 10.0)
                    logger.warning(
                        f"[{agent_name}] Attempt {attempt + 1}/{max_retries} failed with {error_type}: {str(e)[:100]}"
                    )
                    logger.info(f"[{agent_name}] Retrying in {delay:.1f}s... (exponential backoff)")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"[{agent_name}] ❌ All {max_retries} attempts exhausted. "
                        f"Final error: {error_type}: {str(e)[:200]}"
                    )
                    logger.error(f"[{agent_name}] Context: Processing document_id={state.get('document_id', 'unknown')}")
                    logger.debug(f"[{agent_name}] Full traceback:", exc_info=True)

        # All retries exhausted, handle error
        # Record detailed error in state
        if "error_log" not in state:
            state["error_log"] = []

        error_record = {
            "agent": agent_name,
            "error": str(last_exception),
            "error_type": type(last_exception).__name__,
            "error_message": f"{type(last_exception).__name__}: {str(last_exception)[:500]}",
            "timestamp": datetime.now().isoformat(),
            "function": func.__name__,
            "attempts": max_retries,
            "document_id": state.get("document_id", "unknown"),
            "workflow_status": state.get("workflow_status", "unknown"),
            "traceback": traceback.format_exc(),
            "resolution_hint": _get_error_resolution_hint(last_exception, agent_name)
        }

        state["error_log"].append(error_record)

        # Update state with error status
        state["current_agent"] = agent_name
        state = update_state_timestamp(state)

        # Check if we should fail fast
        if getattr(self, 'fail_fast', False):
            logger.error(
                f"[{agent_name}] Fail-fast mode enabled. Raising exception to halt workflow."
            )
            raise last_exception

        # Try fallback if available
        fallback_method = getattr(self, 'fallback_process', None)
        if fallback_method and callable(fallback_method):
            try:
                logger.info(f"[{agent_name}] 🔄 Attempting fallback processing strategy...")
                return fallback_method(state, last_exception)
            except Exception as fallback_error:
                logger.error(
                    f"[{agent_name}] ❌ Fallback strategy also failed: "
                    f"{type(fallback_error).__name__}: {str(fallback_error)[:200]}"
                )
                state["error_log"].append({
                    "agent": agent_name,
                    "error": str(fallback_error),
                    "error_type": "fallback_failure",
                    "error_message": f"Fallback failed: {type(fallback_error).__name__}: {str(fallback_error)[:500]}",
                    "timestamp": datetime.now().isoformat(),
                    "resolution_hint": "Both primary and fallback processing failed. Check agent configuration and input data."
                })

        # Return state with error logged (graceful degradation)
        logger.warning(
            f"[{agent_name}] ⚠️  Continuing workflow with graceful degradation. "
            f"Agent {agent_name} results will be incomplete."
        )
        return state

    return wrapper


def _get_error_resolution_hint(exception: Exception, agent_name: str) -> str:
    """
    Generate helpful resolution hints based on error type

    Args:
        exception: The exception that occurred
        agent_name: Name of the agent that failed

    Returns:
        Human-readable hint for resolving the error
    """
    error_type = type(exception).__name__
    error_msg = str(exception).lower()

    # API/Network errors
    if "api" in error_msg or "connection" in error_msg or "timeout" in error_msg:
        return (
            f"Network or API error in {agent_name}. "
            "Check: 1) Internet connectivity, 2) API keys in .env file, "
            "3) API service status, 4) Rate limits not exceeded."
        )

    # Authentication errors
    if "auth" in error_msg or "key" in error_msg or "token" in error_msg:
        return (
            f"Authentication error in {agent_name}. "
            "Check: 1) OPENAI_API_KEY in .env file is valid, "
            "2) API key has not expired, 3) API key has sufficient permissions."
        )

    # Data/parsing errors
    if "json" in error_msg or "parse" in error_msg or "decode" in error_msg:
        return (
            f"Data parsing error in {agent_name}. "
            "Check: 1) Input document format is valid JSON, "
            "2) Document structure matches expected schema, 3) No corrupted data."
        )

    # Missing data errors
    if "keyerror" in error_type.lower() or "not found" in error_msg:
        return (
            f"Missing required data in {agent_name}. "
            "Check: 1) Document contains all required fields, "
            "2) Preprocessor agent ran successfully, 3) Document metadata is complete."
        )

    # Type errors
    if "type" in error_type.lower():
        return (
            f"Data type mismatch in {agent_name}. "
            "Check: 1) Input data types match expected types, "
            "2) Document structure is correct, 3) No None values where data expected."
        )

    # Import errors
    if "import" in error_type.lower() or "module" in error_msg:
        return (
            f"Missing dependency in {agent_name}. "
            "Check: 1) All packages in requirements.txt are installed, "
            "2) Run 'pip install -r requirements.txt', 3) Virtual environment is activated."
        )

    # File errors
    if "file" in error_msg or "path" in error_msg:
        return (
            f"File access error in {agent_name}. "
            "Check: 1) Required files exist in expected locations, "
            "2) File permissions are correct, 3) File paths in config are valid."
        )

    # Memory errors
    if "memory" in error_msg:
        return (
            f"Memory error in {agent_name}. "
            "Check: 1) Document size is reasonable, "
            "2) System has sufficient RAM, 3) Consider processing smaller batches."
        )

    # Generic fallback
    return (
        f"Error in {agent_name}: {error_type}. "
        "Check: 1) Agent configuration is correct, 2) Input data is valid, "
        "3) Review error log for details, 4) Enable debug logging for more information."
    )


def agent_logging(log_level: str = "INFO"):
    """
    Decorator to add detailed logging to agent execution

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, state: ComplianceState, *args, **kwargs) -> ComplianceState:
            agent_name = getattr(self, 'name', func.__name__)

            # Log entry
            log_func = getattr(logger, log_level.lower(), logger.info)
            log_func(f"[{agent_name}] Entering {func.__name__}")
            log_func(f"[{agent_name}] Current workflow status: {state.get('workflow_status', 'unknown')}")
            log_func(f"[{agent_name}] Violations so far: {len(state.get('violations', []))}")

            # Execute
            result = func(self, state, *args, **kwargs)

            # Log exit
            log_func(f"[{agent_name}] Exiting {func.__name__}")
            log_func(f"[{agent_name}] New violations: {len(result.get('violations', [])) - len(state.get('violations', []))}")

            return result

        return wrapper
    return decorator


class BaseAgent(ABC):
    """
    Abstract base class for all compliance checking agents

    All specialized agents (Structure, Performance, Securities, etc.) must
    inherit from this class and implement the abstract methods.

    Provides:
    - Standard interface for state processing
    - Logging and error handling
    - Timing and performance tracking
    - Configuration management
    """

    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """
        Initialize base agent

        Args:
            config: Agent configuration
            **kwargs: Additional configuration options
        """
        self.config = config or AgentConfig(name=self.__class__.__name__)
        self.name = self.config.name
        self.enabled = self.config.enabled
        self.timeout_seconds = self.config.timeout_seconds
        self.retry_attempts = self.config.retry_attempts
        self.fail_fast = kwargs.get('fail_fast', False)
        self.status = AgentStatus.IDLE

        # Set up agent-specific logger
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))

        # Execution tracking
        self.execution_count = 0
        self.total_duration = 0.0
        self.error_count = 0

        self.logger.info(f"Agent initialized: {self.name}")

    @agent_timing
    @agent_error_handler
    def __call__(self, state: ComplianceState) -> ComplianceState:
        """
        Main entry point for agent execution

        This method is called by LangGraph when the agent node is executed.
        It wraps the process() method with timing and error handling.

        Args:
            state: Current compliance state

        Returns:
            Updated compliance state
        """
        # Check if agent is enabled
        if not self.enabled:
            self.logger.info(f"Agent {self.name} is disabled, skipping")
            self.status = AgentStatus.SKIPPED
            return state

        # Update execution tracking
        self.execution_count += 1
        self.status = AgentStatus.RUNNING

        # Update state
        state["current_agent"] = self.name
        state = update_state_timestamp(state)

        # Create execution record
        execution_record = AgentExecutionRecord(
            agent_name=self.name,
            started_at=datetime.now().isoformat()
        )

        try:
            # Call the agent's process method
            result = self.process(state)

            # Update execution record
            execution_record.completed_at = datetime.now().isoformat()
            execution_record.status = "completed"
            execution_record.violations_found = len(result.get("violations", [])) - len(state.get("violations", []))

            self.status = AgentStatus.COMPLETED

            return result

        except Exception as e:
            # Update execution record with error
            execution_record.completed_at = datetime.now().isoformat()
            execution_record.status = "failed"
            execution_record.error = str(e)

            self.status = AgentStatus.FAILED
            self.error_count += 1

            raise

    @abstractmethod
    def process(self, state: ComplianceState) -> ComplianceState:
        """
        Process the compliance state

        This is the main method that each agent must implement.
        It should perform the agent's specific compliance checks and
        return an updated state.

        Args:
            state: Current compliance state

        Returns:
            Updated compliance state with new violations, analysis, etc.
        """
        pass

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def add_violation(self, state: ComplianceState, violation: Dict[str, Any]) -> ComplianceState:
        """
        Add a violation to the state

        Args:
            state: Current state
            violation: Violation dictionary

        Returns:
            Updated state
        """
        # Ensure violation has agent name
        if "agent" not in violation:
            violation["agent"] = self.name

        # Ensure violation has timestamp
        if "timestamp" not in violation:
            violation["timestamp"] = datetime.now().isoformat()

        # Add to violations list
        if "violations" not in state:
            state["violations"] = []

        state["violations"] = list(state["violations"]) + [violation]

        self.logger.debug(f"Added violation: {violation.get('rule', 'unknown')}")

        return state

    def add_violations(self, state: ComplianceState, violations: List[Dict[str, Any]]) -> ComplianceState:
        """
        Add multiple violations to the state

        Args:
            state: Current state
            violations: List of violation dictionaries

        Returns:
            Updated state
        """
        for violation in violations:
            state = self.add_violation(state, violation)

        return state

    def log_execution_stats(self):
        """Log execution statistics for this agent"""
        avg_duration = self.total_duration / self.execution_count if self.execution_count > 0 else 0

        self.logger.info(f"Agent Statistics for {self.name}:")
        self.logger.info(f"  Executions: {self.execution_count}")
        self.logger.info(f"  Total Duration: {self.total_duration:.2f}s")
        self.logger.info(f"  Average Duration: {avg_duration:.2f}s")
        self.logger.info(f"  Errors: {self.error_count}")
        self.logger.info(f"  Status: {self.status.value}")

    def reset_stats(self):
        """Reset execution statistics"""
        self.execution_count = 0
        self.total_duration = 0.0
        self.error_count = 0
        self.status = AgentStatus.IDLE
        self.logger.info(f"Statistics reset for {self.name}")

    def fallback_process(self, state: ComplianceState, error: Exception) -> ComplianceState:
        """
        Fallback processing when main process fails

        This method can be overridden by subclasses to provide
        agent-specific fallback behavior (e.g., rule-based checking
        when AI is unavailable).

        Default implementation: log error and return state unchanged

        Args:
            state: Current compliance state
            error: Exception that caused the failure

        Returns:
            Updated compliance state
        """
        error_type = type(error).__name__
        error_msg = str(error)[:200]

        self.logger.warning(
            f"[{self.name}] ⚠️  No custom fallback implementation available. Using default graceful degradation."
        )
        self.logger.info(
            f"[{self.name}] Agent will be skipped. Error: {error_type}: {error_msg}"
        )
        self.logger.info(
            f"[{self.name}] Impact: Compliance checks for this agent will not be performed. "
            f"Workflow will continue with remaining agents."
        )

        # Mark agent as failed but allow workflow to continue
        if "skipped_agents" not in state:
            state["skipped_agents"] = []

        state["skipped_agents"].append({
            "agent": self.name,
            "reason": "error_with_no_fallback",
            "error": str(error),
            "error_type": error_type,
            "timestamp": datetime.now().isoformat(),
            "impact": f"Compliance checks for {self.name} were not performed",
            "recommendation": (
                f"Review error logs for {self.name}, fix the underlying issue, "
                f"and re-run the compliance check to get complete results."
            )
        })

        return state

    def __repr__(self) -> str:
        """String representation of agent"""
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.enabled}, status={self.status.value})"


class AgentConfigManager:
    """
    Manages agent configurations from config files and runtime updates
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager

        Args:
            config_dict: Dictionary with agent configurations
        """
        self.config_dict = config_dict or {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self._load_configs()

        logger.info("AgentConfigManager initialized")

    def _load_configs(self):
        """Load agent configurations from config dictionary"""
        agents_config = self.config_dict.get("agents", {})

        for agent_name, agent_settings in agents_config.items():
            if isinstance(agent_settings, dict):
                config = AgentConfig(
                    name=agent_name,
                    enabled=agent_settings.get("enabled", True),
                    timeout_seconds=agent_settings.get("timeout_seconds", 30.0),
                    retry_attempts=agent_settings.get("retry_attempts", 3),
                    log_level=agent_settings.get("log_level", "INFO"),
                    custom_settings=agent_settings.get("custom_settings", {})
                )
                self.agent_configs[agent_name] = config
                logger.debug(f"Loaded config for agent: {agent_name}")

    def get_config(self, agent_name: str) -> AgentConfig:
        """
        Get configuration for an agent

        Args:
            agent_name: Name of the agent

        Returns:
            AgentConfig instance
        """
        if agent_name in self.agent_configs:
            return self.agent_configs[agent_name]

        # Return default config if not found
        logger.warning(f"No config found for {agent_name}, using defaults")
        return AgentConfig(name=agent_name)

    def update_config(self, agent_name: str, **kwargs):
        """
        Update configuration for an agent at runtime

        Args:
            agent_name: Name of the agent
            **kwargs: Configuration parameters to update
        """
        if agent_name not in self.agent_configs:
            self.agent_configs[agent_name] = AgentConfig(name=agent_name)

        config = self.agent_configs[agent_name]

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"Updated {agent_name}.{key} = {value}")
            else:
                config.custom_settings[key] = value
                logger.info(f"Added custom setting {agent_name}.{key} = {value}")

    def set_agent_enabled(self, agent_name: str, enabled: bool):
        """
        Enable or disable an agent

        Args:
            agent_name: Name of the agent
            enabled: True to enable, False to disable
        """
        self.update_config(agent_name, enabled=enabled)
        logger.info(f"Agent {agent_name} {'enabled' if enabled else 'disabled'}")

    def get_all_configs(self) -> Dict[str, AgentConfig]:
        """
        Get all agent configurations

        Returns:
            Dictionary of agent name to AgentConfig
        """
        return self.agent_configs.copy()

    def list_enabled_agents(self) -> List[str]:
        """
        Get list of enabled agent names

        Returns:
            List of enabled agent names
        """
        return [name for name, config in self.agent_configs.items() if config.enabled]

    def reload_from_dict(self, config_dict: Dict[str, Any]):
        """
        Reload configurations from a new config dictionary

        Args:
            config_dict: New configuration dictionary
        """
        self.config_dict = config_dict
        self.agent_configs.clear()
        self._load_configs()
        logger.info("Agent configurations reloaded")


# Export all public symbols
__all__ = [
    "BaseAgent",
    "AgentConfig",
    "AgentRegistry",
    "AgentConfigManager",
    "AgentStatus",
    "agent_timing",
    "agent_error_handler",
    "agent_logging"
]
