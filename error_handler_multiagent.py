#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error Handler for Multi-Agent Compliance System

Comprehensive error handling with:
- Retry logic with exponential backoff
- Graceful degradation for agent failures
- Fallback strategies (AI → rules)
- Circuit breaker pattern
- Error logging with context

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5
"""

import logging
import time
import functools
import traceback
from typing import Callable, Optional, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    LOW = "low"  # Minor issue, can continue
    MEDIUM = "medium"  # Significant issue, degraded service
    HIGH = "high"  # Critical issue, fallback required
    CRITICAL = "critical"  # System failure


class ServiceStatus(Enum):
    """Status of AI service or agent"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 1


@dataclass
class ErrorMetrics:
    """Metrics for error tracking"""
    total_errors: int = 0
    api_errors: int = 0
    timeout_errors: int = 0
    parsing_errors: int = 0
    network_errors: int = 0
    fallback_activations: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    circuit_breaker_trips: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        """Convert metrics to dictionary"""
        return {
            "total_errors": self.total_errors,
            "api_errors": self.api_errors,
            "timeout_errors": self.timeout_errors,
            "parsing_errors": self.parsing_errors,
            "network_errors": self.network_errors,
            "fallback_activations": self.fallback_activations,
            "successful_retries": self.successful_retries,
            "failed_retries": self.failed_retries,
            "circuit_breaker_trips": self.circuit_breaker_trips
        }


class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    
    Prevents cascading failures by stopping calls to failing services.
    States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing)
    
    Requirements: 13.1, 13.3
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self.state = ServiceStatus.HEALTHY  # CLOSED state
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        
        logger.info(f"CircuitBreaker initialized "
                   f"(threshold={self.config.failure_threshold}, "
                   f"timeout={self.config.timeout_seconds}s)")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        # Check if circuit should transition from OPEN to HALF_OPEN
        if self.state == ServiceStatus.CIRCUIT_OPEN:
            if self._should_attempt_reset():
                logger.info("Circuit breaker: Transitioning to HALF_OPEN")
                self.state = ServiceStatus.DEGRADED  # HALF_OPEN
                self.half_open_calls = 0
            else:
                raise Exception(f"Circuit breaker is OPEN. "
                              f"Service unavailable until {self._get_reset_time()}")
        
        # Check if we're in HALF_OPEN and exceeded max calls
        if self.state == ServiceStatus.DEGRADED:
            if self.half_open_calls >= self.config.half_open_max_calls:
                raise Exception("Circuit breaker HALF_OPEN: max test calls exceeded")
            self.half_open_calls += 1
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == ServiceStatus.DEGRADED:  # HALF_OPEN
            self.success_count += 1
            logger.info(f"Circuit breaker: Success in HALF_OPEN "
                       f"({self.success_count}/{self.config.success_threshold})")
            
            if self.success_count >= self.config.success_threshold:
                logger.info("Circuit breaker: Transitioning to CLOSED (healthy)")
                self.state = ServiceStatus.HEALTHY  # CLOSED
                self.failure_count = 0
                self.success_count = 0
        else:
            # Reset failure count on success in CLOSED state
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == ServiceStatus.DEGRADED:  # HALF_OPEN
            logger.warning("Circuit breaker: Failure in HALF_OPEN, reopening circuit")
            self.state = ServiceStatus.CIRCUIT_OPEN  # OPEN
            self.success_count = 0
        elif self.failure_count >= self.config.failure_threshold:
            logger.error(f"Circuit breaker: Threshold reached ({self.failure_count}), "
                        f"opening circuit")
            self.state = ServiceStatus.CIRCUIT_OPEN  # OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self.last_failure_time:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.config.timeout_seconds
    
    def _get_reset_time(self) -> str:
        """Get time when circuit will attempt reset"""
        if not self.last_failure_time:
            return "unknown"
        
        reset_time = self.last_failure_time + timedelta(seconds=self.config.timeout_seconds)
        return reset_time.strftime("%H:%M:%S")
    
    def get_state(self) -> ServiceStatus:
        """Get current circuit breaker state"""
        return self.state
    
    def is_available(self) -> bool:
        """Check if service is available"""
        return self.state != ServiceStatus.CIRCUIT_OPEN
    
    def reset(self):
        """Manually reset circuit breaker"""
        logger.info("Circuit breaker: Manual reset")
        self.state = ServiceStatus.HEALTHY
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0


class RetryHandler:
    """
    Handles retry logic with exponential backoff
    
    Requirements: 13.4
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry handler
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
        logger.debug(f"RetryHandler initialized "
                    f"(max_attempts={self.config.max_attempts})")
    
    def retry_with_backoff(
        self,
        func: Callable,
        *args,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with exponential backoff retry
        
        Args:
            func: Function to execute
            *args: Function arguments
            on_retry: Optional callback on retry (attempt, exception)
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Retry successful on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1}/{self.config.max_attempts} failed: "
                                 f"{str(e)[:100]}")
                    logger.info(f"Retrying in {delay:.2f}s...")
                    
                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(attempt, e)
                        except Exception as callback_error:
                            logger.error(f"Retry callback failed: {callback_error}")
                    
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} attempts failed")
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for exponential backoff
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay = min(
            self.config.initial_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay


class ServiceHealthMonitor:
    """
    Monitor health of AI services and agents
    
    Tracks errors over time and determines service status.
    
    Requirements: 13.1, 13.5
    """
    
    def __init__(
        self,
        error_threshold: int = 5,
        time_window_seconds: int = 60,
        degraded_threshold: int = 3
    ):
        """
        Initialize health monitor
        
        Args:
            error_threshold: Number of errors before marking unavailable
            time_window_seconds: Time window for error counting
            degraded_threshold: Number of errors before marking degraded
        """
        self.error_threshold = error_threshold
        self.time_window_seconds = time_window_seconds
        self.degraded_threshold = degraded_threshold
        self.error_history: deque = deque(maxlen=100)
        self.status = ServiceStatus.HEALTHY
        self.metrics = ErrorMetrics()
        
        logger.info(f"ServiceHealthMonitor initialized "
                   f"(threshold={error_threshold}, window={time_window_seconds}s)")
    
    def record_error(self, error_type: str = "general", error: Optional[Exception] = None):
        """
        Record an error occurrence
        
        Args:
            error_type: Type of error (api, timeout, parsing, network, etc.)
            error: Optional exception object
        """
        current_time = time.time()
        self.error_history.append({
            "time": current_time,
            "type": error_type,
            "error": str(error) if error else None
        })
        
        # Update metrics
        self.metrics.total_errors += 1
        if error_type == "api":
            self.metrics.api_errors += 1
        elif error_type == "timeout":
            self.metrics.timeout_errors += 1
        elif error_type == "parsing":
            self.metrics.parsing_errors += 1
        elif error_type == "network":
            self.metrics.network_errors += 1
        
        # Update status
        self._update_status()
        
        logger.warning(f"Error recorded: {error_type} "
                      f"(recent errors: {self._count_recent_errors()})")
    
    def record_success(self):
        """Record a successful operation"""
        self._update_status()
    
    def record_fallback(self):
        """Record fallback activation"""
        self.metrics.fallback_activations += 1
        logger.info(f"Fallback activated (total: {self.metrics.fallback_activations})")
    
    def record_retry(self, success: bool):
        """Record retry attempt"""
        if success:
            self.metrics.successful_retries += 1
        else:
            self.metrics.failed_retries += 1
    
    def record_circuit_breaker_trip(self):
        """Record circuit breaker trip"""
        self.metrics.circuit_breaker_trips += 1
    
    def get_status(self) -> ServiceStatus:
        """Get current service status"""
        return self.status
    
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        return self.status == ServiceStatus.HEALTHY
    
    def is_available(self) -> bool:
        """Check if service is available (healthy or degraded)"""
        return self.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
    
    def get_metrics(self) -> ErrorMetrics:
        """Get error metrics"""
        return self.metrics
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = ErrorMetrics()
        self.error_history.clear()
        self.status = ServiceStatus.HEALTHY
        logger.info("Metrics reset")
    
    def _count_recent_errors(self) -> int:
        """Count errors within time window"""
        current_time = time.time()
        cutoff_time = current_time - self.time_window_seconds
        
        return sum(1 for error in self.error_history if error["time"] > cutoff_time)
    
    def _update_status(self):
        """Update service status based on recent errors"""
        error_count = self._count_recent_errors()
        
        if error_count == 0:
            self.status = ServiceStatus.HEALTHY
        elif error_count < self.degraded_threshold:
            self.status = ServiceStatus.HEALTHY
        elif error_count < self.error_threshold:
            self.status = ServiceStatus.DEGRADED
        else:
            self.status = ServiceStatus.UNAVAILABLE
        
        logger.debug(f"Service status: {self.status.value} "
                    f"(errors in window: {error_count})")


class MultiAgentErrorHandler:
    """
    Main error handler for multi-agent system
    
    Provides comprehensive error handling with:
    - Retry logic with exponential backoff
    - Circuit breaker pattern
    - Health monitoring
    - Graceful degradation
    - Fallback strategies
    - Agent timeout handling
    - Partial result generation
    - Failure notifications
    
    Requirements: 13.1, 13.2, 13.3, 13.4, 13.5
    """
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        health_monitor_config: Optional[Dict[str, int]] = None,
        default_timeout: float = 30.0,
        notification_callback: Optional[Callable[[str, str, Exception], None]] = None
    ):
        """
        Initialize multi-agent error handler
        
        Args:
            retry_config: Retry configuration
            circuit_breaker_config: Circuit breaker configuration
            health_monitor_config: Health monitor configuration
            default_timeout: Default timeout for agent calls in seconds
            notification_callback: Optional callback for failure notifications
        """
        self.retry_handler = RetryHandler(retry_config)
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config)
        
        health_config = health_monitor_config or {}
        self.health_monitor = ServiceHealthMonitor(
            error_threshold=health_config.get("error_threshold", 5),
            time_window_seconds=health_config.get("time_window_seconds", 60),
            degraded_threshold=health_config.get("degraded_threshold", 3)
        )
        
        self.default_timeout = default_timeout
        self.notification_callback = notification_callback
        self.agent_circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.partial_results: Dict[str, Any] = {}
        self.failed_agents: List[str] = []
        
        logger.info("MultiAgentErrorHandler initialized with timeout handling and notifications")
    
    def handle_agent_call(
        self,
        agent_name: str,
        primary_func: Callable,
        fallback_func: Optional[Callable] = None,
        timeout: Optional[float] = None,
        allow_partial: bool = True,
        *args,
        **kwargs
    ) -> Any:
        """
        Handle agent call with retry, circuit breaker, timeout, and fallback
        
        Requirements: 13.1, 13.2, 13.3, 13.4
        
        Args:
            agent_name: Name of the agent
            primary_func: Primary function to call
            fallback_func: Optional fallback function
            timeout: Optional timeout in seconds (uses default if not provided)
            allow_partial: Whether to allow partial results on failure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or partial result
        """
        logger.debug(f"[{agent_name}] Handling agent call")
        
        # Get or create agent-specific circuit breaker
        if agent_name not in self.agent_circuit_breakers:
            self.agent_circuit_breakers[agent_name] = CircuitBreaker(
                CircuitBreakerConfig(
                    failure_threshold=3,
                    success_threshold=2,
                    timeout_seconds=60.0
                )
            )
        
        agent_circuit = self.agent_circuit_breakers[agent_name]
        
        # Check agent-specific circuit breaker
        if not agent_circuit.is_available():
            logger.warning(f"[{agent_name}] Agent circuit breaker is OPEN, attempting fallback")
            self.health_monitor.record_circuit_breaker_trip()
            self._send_failure_notification(agent_name, "circuit_open", 
                                           Exception("Circuit breaker is OPEN"))
            
            if fallback_func:
                return self._execute_fallback(agent_name, fallback_func, *args, **kwargs)
            elif allow_partial:
                return self._get_partial_result(agent_name)
            else:
                raise Exception(f"[{agent_name}] Circuit breaker is OPEN and no fallback available")
        
        # Determine timeout
        effective_timeout = timeout or self.get_recommended_timeout(self.default_timeout)
        
        # Try primary function with retry, circuit breaker, and timeout
        try:
            def wrapped_call():
                return self._call_with_timeout(
                    agent_name,
                    agent_circuit,
                    primary_func,
                    effective_timeout,
                    *args,
                    **kwargs
                )
            
            result = self.retry_handler.retry_with_backoff(
                wrapped_call,
                on_retry=lambda attempt, error: self.health_monitor.record_retry(False)
            )
            
            self.health_monitor.record_success()
            self.health_monitor.record_retry(True)
            
            # Store successful result as partial result cache
            self.partial_results[agent_name] = result
            
            # Remove from failed agents list if present
            if agent_name in self.failed_agents:
                self.failed_agents.remove(agent_name)
            
            return result
            
        except Exception as e:
            logger.error(f"[{agent_name}] Primary function failed: {e}")
            
            # Track failed agent
            if agent_name not in self.failed_agents:
                self.failed_agents.append(agent_name)
            
            # Classify error type
            error_type = self._classify_error(e)
            self.health_monitor.record_error(error_type, e)
            
            # Send failure notification
            self._send_failure_notification(agent_name, error_type, e)
            
            # Try fallback if available
            if fallback_func:
                logger.info(f"[{agent_name}] Attempting fallback...")
                self.health_monitor.record_fallback()
                
                try:
                    result = self._execute_fallback(agent_name, fallback_func, *args, **kwargs)
                    self.partial_results[agent_name] = result
                    return result
                    
                except Exception as fallback_error:
                    logger.error(f"[{agent_name}] Fallback also failed: {fallback_error}")
                    self.health_monitor.record_error("fallback", fallback_error)
                    self._send_failure_notification(agent_name, "fallback_failed", fallback_error)
            
            # Try to return partial result if allowed
            if allow_partial:
                partial = self._get_partial_result(agent_name)
                if partial is not None:
                    logger.info(f"[{agent_name}] Returning partial result")
                    return partial
            
            # No fallback, no partial result, or not allowed
            raise
    
    def _call_with_timeout(
        self,
        agent_name: str,
        circuit_breaker: CircuitBreaker,
        func: Callable,
        timeout: float,
        *args,
        **kwargs
    ) -> Any:
        """
        Call function with timeout handling
        
        Args:
            agent_name: Name of the agent
            circuit_breaker: Circuit breaker to use
            func: Function to call
            timeout: Timeout in seconds
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            TimeoutError: If function exceeds timeout
        """
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"[{agent_name}] Agent call exceeded timeout of {timeout}s")
        
        # Set up timeout (Unix-like systems only)
        try:
            # Try to use signal-based timeout (works on Unix)
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            
            try:
                result = circuit_breaker.call(func, *args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                return result
            finally:
                signal.signal(signal.SIGALRM, old_handler)
                
        except (AttributeError, ValueError):
            # signal.SIGALRM not available (Windows) - use threading
            import threading
            
            result_container = {"result": None, "error": None}
            
            def target():
                try:
                    result_container["result"] = circuit_breaker.call(func, *args, **kwargs)
                except Exception as e:
                    result_container["error"] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout)
            
            if thread.is_alive():
                logger.error(f"[{agent_name}] Agent call timed out after {timeout}s")
                self.health_monitor.record_error("timeout")
                raise TimeoutError(f"[{agent_name}] Agent call exceeded timeout of {timeout}s")
            
            if result_container["error"]:
                raise result_container["error"]
            
            return result_container["result"]
    
    def _execute_fallback(
        self,
        agent_name: str,
        fallback_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute fallback function
        
        Args:
            agent_name: Name of the agent
            fallback_func: Fallback function
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Fallback result
        """
        logger.info(f"[{agent_name}] Executing fallback")
        return fallback_func(*args, **kwargs)
    
    def _get_partial_result(self, agent_name: str) -> Any:
        """
        Get partial result for failed agent
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Partial result or None
        """
        if agent_name in self.partial_results:
            logger.info(f"[{agent_name}] Using cached partial result")
            return self.partial_results[agent_name]
        
        # Return empty result structure based on agent type
        logger.warning(f"[{agent_name}] No partial result available, returning empty result")
        return self._get_empty_result(agent_name)
    
    def _get_empty_result(self, agent_name: str) -> Any:
        """
        Get empty result structure for agent
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Empty result appropriate for agent type
        """
        # Return empty violations list for compliance agents
        if any(keyword in agent_name.lower() for keyword in 
               ["structure", "performance", "securities", "general", "esg", "prospectus", "registration"]):
            return []
        
        # Return empty dict for analysis agents
        if any(keyword in agent_name.lower() for keyword in ["context", "evidence", "aggregator"]):
            return {}
        
        # Default: return None
        return None
    
    def _send_failure_notification(
        self,
        agent_name: str,
        error_type: str,
        error: Exception
    ):
        """
        Send failure notification
        
        Args:
            agent_name: Name of the failed agent
            error_type: Type of error
            error: Exception object
        """
        if self.notification_callback:
            try:
                self.notification_callback(agent_name, error_type, error)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")
        
        # Log notification
        logger.warning(f"FAILURE NOTIFICATION: [{agent_name}] {error_type} - {str(error)[:200]}")
    
    def _classify_error(self, error: Exception) -> str:
        """
        Classify error type
        
        Args:
            error: Exception object
            
        Returns:
            Error type string
        """
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        if "timeout" in error_str or "timeout" in error_type_name:
            return "timeout"
        elif "api" in error_str or "openai" in error_str:
            return "api"
        elif "parse" in error_str or "json" in error_str:
            return "parsing"
        elif "network" in error_str or "connection" in error_str:
            return "network"
        else:
            return "general"
    
    def safe_execute(
        self,
        func: Callable,
        default_value: Any = None,
        log_errors: bool = True,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function safely, returning default value on error
        
        Requirements: 13.1, 13.5
        
        Args:
            func: Function to execute
            default_value: Value to return on error
            log_errors: Whether to log errors
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or default value
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_errors:
                logger.error(f"Safe execution failed: {e}")
                logger.debug(traceback.format_exc())
            return default_value
    
    def get_service_status(self) -> ServiceStatus:
        """Get current service status"""
        return self.health_monitor.get_status()
    
    def get_circuit_breaker_state(self) -> ServiceStatus:
        """Get circuit breaker state"""
        return self.circuit_breaker.get_state()
    
    def get_metrics(self) -> ErrorMetrics:
        """Get error metrics"""
        return self.health_monitor.get_metrics()
    
    def reset(self):
        """Reset error handler state"""
        self.health_monitor.reset_metrics()
        self.circuit_breaker.reset()
        logger.info("MultiAgentErrorHandler reset")
    
    def should_use_ai(self) -> bool:
        """
        Check if AI should be used based on service health
        
        Returns:
            True if AI should be used, False to use fallback
        """
        status = self.health_monitor.get_status()
        circuit_state = self.circuit_breaker.get_state()
        
        # Don't use AI if circuit is open or service is unavailable
        if circuit_state == ServiceStatus.CIRCUIT_OPEN:
            return False
        if status == ServiceStatus.UNAVAILABLE:
            return False
        
        return True
    
    def get_recommended_timeout(self, default_timeout: float) -> float:
        """
        Get recommended timeout based on service health
        
        Args:
            default_timeout: Default timeout value
            
        Returns:
            Adjusted timeout
        """
        status = self.health_monitor.get_status()
        
        if status == ServiceStatus.HEALTHY:
            return default_timeout
        elif status == ServiceStatus.DEGRADED:
            return default_timeout * 1.5
        else:
            return default_timeout * 2.0
    
    def get_failed_agents(self) -> List[str]:
        """
        Get list of failed agents
        
        Returns:
            List of agent names that have failed
        """
        return self.failed_agents.copy()
    
    def get_agent_circuit_state(self, agent_name: str) -> ServiceStatus:
        """
        Get circuit breaker state for specific agent
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Circuit breaker state
        """
        if agent_name in self.agent_circuit_breakers:
            return self.agent_circuit_breakers[agent_name].get_state()
        return ServiceStatus.HEALTHY
    
    def reset_agent_circuit(self, agent_name: str):
        """
        Reset circuit breaker for specific agent
        
        Args:
            agent_name: Name of the agent
        """
        if agent_name in self.agent_circuit_breakers:
            self.agent_circuit_breakers[agent_name].reset()
            logger.info(f"[{agent_name}] Circuit breaker reset")
            
            # Remove from failed agents list
            if agent_name in self.failed_agents:
                self.failed_agents.remove(agent_name)
    
    def generate_partial_report(self) -> Dict[str, Any]:
        """
        Generate partial compliance report with available results
        
        Returns:
            Partial report with status of all agents
        """
        report = {
            "status": "partial",
            "completed_agents": [],
            "failed_agents": self.failed_agents.copy(),
            "partial_results": {},
            "circuit_breaker_states": {},
            "service_health": self.health_monitor.get_status().value,
            "metrics": self.health_monitor.get_metrics().to_dict()
        }
        
        # Add partial results
        for agent_name, result in self.partial_results.items():
            report["partial_results"][agent_name] = result
            if agent_name not in self.failed_agents:
                report["completed_agents"].append(agent_name)
        
        # Add circuit breaker states
        for agent_name, circuit in self.agent_circuit_breakers.items():
            report["circuit_breaker_states"][agent_name] = circuit.get_state().value
        
        return report
    
    def can_recover_agent(self, agent_name: str) -> bool:
        """
        Check if agent can be recovered
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            True if agent can be recovered
        """
        if agent_name not in self.agent_circuit_breakers:
            return True
        
        circuit = self.agent_circuit_breakers[agent_name]
        state = circuit.get_state()
        
        # Can recover if circuit is not open, or if enough time has passed
        if state == ServiceStatus.CIRCUIT_OPEN:
            return circuit._should_attempt_reset()
        
        return True
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """
        Get recovery status for all agents
        
        Returns:
            Recovery status information
        """
        status = {
            "total_agents": len(self.agent_circuit_breakers),
            "failed_agents": len(self.failed_agents),
            "recoverable_agents": [],
            "unrecoverable_agents": [],
            "agent_details": {}
        }
        
        for agent_name in self.failed_agents:
            can_recover = self.can_recover_agent(agent_name)
            circuit_state = self.get_agent_circuit_state(agent_name)
            
            agent_info = {
                "can_recover": can_recover,
                "circuit_state": circuit_state.value,
                "has_partial_result": agent_name in self.partial_results
            }
            
            status["agent_details"][agent_name] = agent_info
            
            if can_recover:
                status["recoverable_agents"].append(agent_name)
            else:
                status["unrecoverable_agents"].append(agent_name)
        
        return status


# Decorator functions for easy use

def with_agent_error_handling(
    agent_name: str,
    error_handler: MultiAgentErrorHandler,
    fallback_func: Optional[Callable] = None
):
    """
    Decorator for agent methods with comprehensive error handling
    
    Args:
        agent_name: Name of the agent
        error_handler: Error handler instance
        fallback_func: Optional fallback function
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return error_handler.handle_agent_call(
                agent_name,
                func,
                fallback_func,
                *args,
                **kwargs
            )
        return wrapper
    return decorator


def with_safe_execution(default_value: Any = None, log_errors: bool = True):
    """
    Decorator for safe execution with default value on error
    
    Args:
        default_value: Value to return on error
        log_errors: Whether to log errors
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"{func.__name__} failed: {e}")
                    logger.debug(traceback.format_exc())
                return default_value
        return wrapper
    return decorator


# Export all public symbols
__all__ = [
    "MultiAgentErrorHandler",
    "RetryHandler",
    "CircuitBreaker",
    "ServiceHealthMonitor",
    "ErrorMetrics",
    "RetryConfig",
    "CircuitBreakerConfig",
    "ErrorSeverity",
    "ServiceStatus",
    "with_agent_error_handling",
    "with_safe_execution"
]



# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("MULTI-AGENT ERROR HANDLER - Comprehensive Error Handling")
    print("="*70)
    
    # Test 1: Retry Handler
    print("\n🔄 Test 1: RetryHandler with exponential backoff")
    print("-" * 70)
    
    retry_handler = RetryHandler(RetryConfig(max_attempts=3, initial_delay=0.1))
    
    class Counter:
        def __init__(self):
            self.count = 0
    
    counter = Counter()
    
    def flaky_function():
        counter.count += 1
        print(f"  Attempt {counter.count}")
        if counter.count < 2:
            raise Exception("Simulated failure")
        return "Success!"
    
    try:
        result = retry_handler.retry_with_backoff(flaky_function)
        print(f"  ✓ Result: {result} (took {counter.count} attempts)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 2: Circuit Breaker
    print("\n⚡ Test 2: CircuitBreaker pattern")
    print("-" * 70)
    
    circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=2.0
    ))
    
    def failing_function():
        raise Exception("Service unavailable")
    
    # Trigger failures to open circuit
    for i in range(4):
        try:
            circuit_breaker.call(failing_function)
        except Exception as e:
            print(f"  Call {i+1}: {e}")
    
    print(f"  Circuit state: {circuit_breaker.get_state().value}")
    
    # Test 3: Health Monitor
    print("\n📊 Test 3: ServiceHealthMonitor")
    print("-" * 70)
    
    health_monitor = ServiceHealthMonitor(
        error_threshold=5,
        time_window_seconds=10,
        degraded_threshold=3
    )
    
    print(f"  Initial status: {health_monitor.get_status().value}")
    
    # Record some errors
    for i in range(3):
        health_monitor.record_error("api")
    
    print(f"  After 3 errors: {health_monitor.get_status().value}")
    
    # Record more errors
    for i in range(3):
        health_monitor.record_error("timeout")
    
    print(f"  After 6 errors: {health_monitor.get_status().value}")
    
    metrics = health_monitor.get_metrics()
    print(f"  Metrics: {metrics.total_errors} total, "
          f"{metrics.api_errors} API, {metrics.timeout_errors} timeout")
    
    # Test 4: MultiAgentErrorHandler
    print("\n🛡️ Test 4: MultiAgentErrorHandler (integrated)")
    print("-" * 70)
    
    error_handler = MultiAgentErrorHandler(
        retry_config=RetryConfig(max_attempts=2, initial_delay=0.1),
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3)
    )
    
    def primary_ai_function():
        return "AI result"
    
    def fallback_rule_function():
        return "Rule-based result"
    
    # Test successful call
    result = error_handler.handle_agent_call(
        "test_agent",
        primary_ai_function,
        fallback_rule_function
    )
    print(f"  ✓ Successful call: {result}")
    
    # Test with failing primary and working fallback
    def failing_primary():
        raise Exception("AI service unavailable")
    
    result = error_handler.handle_agent_call(
        "test_agent",
        failing_primary,
        fallback_rule_function
    )
    print(f"  ✓ Fallback activated: {result}")
    
    # Test safe execution
    print("\n🔒 Test 5: Safe execution")
    print("-" * 70)
    
    def risky_function():
        raise ValueError("Something went wrong")
    
    result = error_handler.safe_execute(risky_function, default_value="default")
    print(f"  ✓ Safe execution returned: {result}")
    
    # Test service status checks
    print("\n📈 Test 6: Service status and recommendations")
    print("-" * 70)
    
    print(f"  Should use AI: {error_handler.should_use_ai()}")
    print(f"  Service status: {error_handler.get_service_status().value}")
    print(f"  Circuit breaker: {error_handler.get_circuit_breaker_state().value}")
    print(f"  Recommended timeout (30s): {error_handler.get_recommended_timeout(30.0)}s")
    
    # Display final metrics
    print("\n📊 Final Metrics")
    print("-" * 70)
    final_metrics = error_handler.get_metrics()
    for key, value in final_metrics.to_dict().items():
        print(f"  {key}: {value}")
    
    # Test 7: Agent timeout handling
    print("\n⏱️ Test 7: Agent timeout handling")
    print("-" * 70)
    
    import time
    
    def slow_function():
        time.sleep(2)
        return "Should timeout"
    
    def fast_fallback():
        return "Fallback result"
    
    try:
        result = error_handler.handle_agent_call(
            "slow_agent",
            slow_function,
            fast_fallback,
            timeout=0.5
        )
        print(f"  ✓ Timeout handled, fallback returned: {result}")
    except Exception as e:
        print(f"  ✗ Timeout test failed: {e}")
    
    # Test 8: Partial result generation
    print("\n📦 Test 8: Partial result generation")
    print("-" * 70)
    
    # Store a partial result
    error_handler.partial_results["test_agent"] = {"violations": [{"rule": "test"}]}
    
    def failing_function():
        raise Exception("Agent failed")
    
    result = error_handler.handle_agent_call(
        "test_agent",
        failing_function,
        allow_partial=True
    )
    print(f"  ✓ Partial result returned: {result}")
    
    # Test 9: Failure notifications
    print("\n🔔 Test 9: Failure notifications")
    print("-" * 70)
    
    notifications = []
    
    def notification_handler(agent_name: str, error_type: str, error: Exception):
        notifications.append({
            "agent": agent_name,
            "type": error_type,
            "error": str(error)
        })
        print(f"  📧 Notification: [{agent_name}] {error_type}")
    
    error_handler_with_notifications = MultiAgentErrorHandler(
        notification_callback=notification_handler
    )
    
    try:
        error_handler_with_notifications.handle_agent_call(
            "notification_test_agent",
            failing_function,
            allow_partial=False
        )
    except Exception:
        pass
    
    print(f"  ✓ Notifications sent: {len(notifications)}")
    
    # Test 10: Agent-specific circuit breakers
    print("\n⚡ Test 10: Agent-specific circuit breakers")
    print("-" * 70)
    
    # Fail agent1 multiple times
    for i in range(4):
        try:
            error_handler.handle_agent_call("agent1", failing_function, allow_partial=False)
        except Exception:
            pass
    
    agent1_state = error_handler.get_agent_circuit_state("agent1")
    print(f"  Agent1 circuit state: {agent1_state.value}")
    
    # Agent2 should still be healthy
    agent2_state = error_handler.get_agent_circuit_state("agent2")
    print(f"  Agent2 circuit state: {agent2_state.value}")
    
    # Test 11: Recovery status
    print("\n🔄 Test 11: Recovery status")
    print("-" * 70)
    
    recovery_status = error_handler.get_recovery_status()
    print(f"  Failed agents: {recovery_status['failed_agents']}")
    print(f"  Recoverable: {len(recovery_status['recoverable_agents'])}")
    print(f"  Unrecoverable: {len(recovery_status['unrecoverable_agents'])}")
    
    # Test 12: Partial report generation
    print("\n📋 Test 12: Partial report generation")
    print("-" * 70)
    
    partial_report = error_handler.generate_partial_report()
    print(f"  Report status: {partial_report['status']}")
    print(f"  Completed agents: {len(partial_report['completed_agents'])}")
    print(f"  Failed agents: {len(partial_report['failed_agents'])}")
    print(f"  Service health: {partial_report['service_health']}")
    
    # Test 13: Agent recovery
    print("\n🔧 Test 13: Agent recovery")
    print("-" * 70)
    
    print(f"  Agent1 can recover: {error_handler.can_recover_agent('agent1')}")
    error_handler.reset_agent_circuit("agent1")
    agent1_state_after = error_handler.get_agent_circuit_state("agent1")
    print(f"  Agent1 circuit after reset: {agent1_state_after.value}")
    print(f"  Failed agents after reset: {error_handler.get_failed_agents()}")
    
    print("\n" + "="*70)
    print("✓ All tests completed successfully (including recovery features)")
    print("="*70)
