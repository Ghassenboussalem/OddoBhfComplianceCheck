#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error Handler - Comprehensive Error Handling and Fallback Logic
Implements fallback mechanisms, retry logic, and graceful degradation for AI services
"""

import logging
import time
import functools
from typing import Callable, Optional, Any, Dict, List
from dataclasses import dataclass
from enum import Enum
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Status of AI service"""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    LOW = "LOW"  # Minor issue, can continue
    MEDIUM = "MEDIUM"  # Significant issue, degraded service
    HIGH = "HIGH"  # Critical issue, fallback required
    CRITICAL = "CRITICAL"  # System failure


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class ErrorMetrics:
    """Metrics for error tracking"""
    total_errors: int = 0
    api_errors: int = 0
    timeout_errors: int = 0
    parsing_errors: int = 0
    fallback_activations: int = 0
    successful_retries: int = 0
    failed_retries: int = 0


class ServiceHealthMonitor:
    """
    Monitor health of AI services
    Tracks errors and determines service status
    """
    
    def __init__(self, error_threshold: int = 5, time_window: int = 60):
        """
        Initialize health monitor
        
        Args:
            error_threshold: Number of errors before marking degraded
            time_window: Time window in seconds for error counting
        """
        self.error_threshold = error_threshold
        self.time_window = time_window
        self.error_history: List[float] = []
        self.status = ServiceStatus.UNKNOWN
        self.metrics = ErrorMetrics()
        
        logger.info(f"ServiceHealthMonitor initialized "
                   f"(threshold={error_threshold}, window={time_window}s)")
    
    def record_error(self, error_type: str = "general"):
        """
        Record an error occurrence
        
        Args:
            error_type: Type of error (api, timeout, parsing, etc.)
        """
        current_time = time.time()
        self.error_history.append(current_time)
        
        # Update metrics
        self.metrics.total_errors += 1
        if error_type == "api":
            self.metrics.api_errors += 1
        elif error_type == "timeout":
            self.metrics.timeout_errors += 1
        elif error_type == "parsing":
            self.metrics.parsing_errors += 1
        
        # Clean old errors outside time window
        self._clean_old_errors(current_time)
        
        # Update status
        self._update_status()
        
        logger.warning(f"Error recorded: {error_type} "
                      f"(recent errors: {len(self.error_history)})")
    
    def record_success(self):
        """Record a successful operation"""
        current_time = time.time()
        self._clean_old_errors(current_time)
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
        self.status = ServiceStatus.UNKNOWN
        logger.info("Metrics reset")
    
    def _clean_old_errors(self, current_time: float):
        """Remove errors outside time window"""
        cutoff_time = current_time - self.time_window
        self.error_history = [t for t in self.error_history if t > cutoff_time]
    
    def _update_status(self):
        """Update service status based on recent errors"""
        error_count = len(self.error_history)
        
        if error_count == 0:
            self.status = ServiceStatus.HEALTHY
        elif error_count < self.error_threshold:
            self.status = ServiceStatus.DEGRADED
        else:
            self.status = ServiceStatus.UNAVAILABLE
        
        logger.debug(f"Service status: {self.status.value} "
                    f"(errors in window: {error_count})")


class RetryHandler:
    """
    Handles retry logic with exponential backoff
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry handler
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
        logger.info(f"RetryHandler initialized "
                   f"(max_attempts={self.config.max_attempts})")
    
    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with exponential backoff retry
        
        Args:
            func: Function to execute
            *args: Function arguments
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
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)[:100]}")
                    logger.info(f"Retrying in {delay:.2f}s...")
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


def with_retry(max_attempts: int = 3, initial_delay: float = 1.0):
    """
    Decorator for automatic retry with exponential backoff
    
    Args:
        max_attempts: Maximum retry attempts
        initial_delay: Initial delay in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                initial_delay=initial_delay
            )
            handler = RetryHandler(config)
            return handler.retry_with_backoff(func, *args, **kwargs)
        return wrapper
    return decorator


def with_fallback(fallback_func: Callable):
    """
    Decorator for automatic fallback on failure
    
    Args:
        fallback_func: Fallback function to call on error
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"{func.__name__} failed: {e}")
                logger.info(f"Activating fallback: {fallback_func.__name__}")
                return fallback_func(*args, **kwargs)
        return wrapper
    return decorator


def with_timeout(timeout_seconds: float):
    """
    Decorator for function timeout
    
    Args:
        timeout_seconds: Timeout in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"{func.__name__} timed out after {timeout_seconds}s")
            
            # Set timeout (Unix-like systems only)
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout_seconds))
                
                result = func(*args, **kwargs)
                
                signal.alarm(0)  # Cancel alarm
                return result
                
            except AttributeError:
                # Windows doesn't support SIGALRM, just run normally
                logger.warning("Timeout not supported on this platform")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class ErrorHandler:
    """
    Main error handler with fallback and monitoring
    """
    
    def __init__(self, health_monitor: Optional[ServiceHealthMonitor] = None,
                 retry_handler: Optional[RetryHandler] = None):
        """
        Initialize error handler
        
        Args:
            health_monitor: Service health monitor
            retry_handler: Retry handler
        """
        self.health_monitor = health_monitor or ServiceHealthMonitor()
        self.retry_handler = retry_handler or RetryHandler()
        
        logger.info("ErrorHandler initialized")
    
    def handle_ai_call(self, primary_func: Callable, fallback_func: Optional[Callable] = None,
                      *args, **kwargs) -> Any:
        """
        Handle AI call with retry and fallback
        
        Args:
            primary_func: Primary function to call
            fallback_func: Optional fallback function
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        try:
            # Try primary function with retry
            result = self.retry_handler.retry_with_backoff(
                primary_func, *args, **kwargs
            )
            
            self.health_monitor.record_success()
            return result
            
        except Exception as e:
            logger.error(f"Primary function failed: {e}")
            self.health_monitor.record_error("api")
            
            # Try fallback if available
            if fallback_func:
                try:
                    logger.info("Attempting fallback...")
                    self.health_monitor.record_fallback()
                    result = fallback_func(*args, **kwargs)
                    return result
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    self.health_monitor.record_error("fallback")
            
            # No fallback or fallback failed
            raise
    
    def safe_execute(self, func: Callable, default_value: Any = None,
                    *args, **kwargs) -> Any:
        """
        Execute function safely, returning default value on error
        
        Args:
            func: Function to execute
            default_value: Value to return on error
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or default value
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Safe execution failed: {e}")
            logger.debug(traceback.format_exc())
            return default_value
    
    def get_service_status(self) -> ServiceStatus:
        """Get current service status"""
        return self.health_monitor.get_status()
    
    def get_metrics(self) -> ErrorMetrics:
        """Get error metrics"""
        return self.health_monitor.get_metrics()
    
    def reset(self):
        """Reset error handler state"""
        self.health_monitor.reset_metrics()
        logger.info("ErrorHandler reset")


class GracefulDegradation:
    """
    Manages graceful degradation of service
    """
    
    def __init__(self, error_handler: ErrorHandler):
        """
        Initialize graceful degradation manager
        
        Args:
            error_handler: Error handler instance
        """
        self.error_handler = error_handler
        self.degradation_level = 0  # 0 = full service, 3 = minimal service
        
        logger.info("GracefulDegradation initialized")
    
    def check_and_adjust(self) -> int:
        """
        Check service health and adjust degradation level
        
        Returns:
            Current degradation level (0-3)
        """
        status = self.error_handler.get_service_status()
        
        if status == ServiceStatus.HEALTHY:
            self.degradation_level = 0
        elif status == ServiceStatus.DEGRADED:
            self.degradation_level = 1
        elif status == ServiceStatus.UNAVAILABLE:
            self.degradation_level = 2
        else:
            self.degradation_level = 3
        
        if self.degradation_level > 0:
            logger.warning(f"Service degradation level: {self.degradation_level}")
        
        return self.degradation_level
    
    def should_use_ai(self) -> bool:
        """Check if AI should be used based on degradation level"""
        return self.degradation_level < 2
    
    def should_use_cache_aggressively(self) -> bool:
        """Check if cache should be used more aggressively"""
        return self.degradation_level >= 1
    
    def should_skip_optional_checks(self) -> bool:
        """Check if optional checks should be skipped"""
        return self.degradation_level >= 2
    
    def get_recommended_timeout(self, default_timeout: float) -> float:
        """
        Get recommended timeout based on degradation level
        
        Args:
            default_timeout: Default timeout value
            
        Returns:
            Adjusted timeout
        """
        if self.degradation_level == 0:
            return default_timeout
        elif self.degradation_level == 1:
            return default_timeout * 1.5
        else:
            return default_timeout * 2.0


if __name__ == "__main__":
    # Example usage and testing
    print("="*70)
    print("Error Handler - Comprehensive Error Handling")
    print("="*70)
    
    # Test health monitor
    print("\n📊 Testing ServiceHealthMonitor...")
    monitor = ServiceHealthMonitor(error_threshold=3, time_window=10)
    
    print(f"  Initial status: {monitor.get_status().value}")
    
    monitor.record_error("api")
    print(f"  After 1 error: {monitor.get_status().value}")
    
    monitor.record_error("api")
    monitor.record_error("timeout")
    print(f"  After 3 errors: {monitor.get_status().value}")
    
    monitor.record_success()
    print(f"  After success: {monitor.get_status().value}")
    
    metrics = monitor.get_metrics()
    print(f"  Metrics: {metrics.total_errors} total, {metrics.api_errors} API errors")
    
    # Test retry handler
    print("\n🔄 Testing RetryHandler...")
    retry_handler = RetryHandler(RetryConfig(max_attempts=3, initial_delay=0.1))
    
    class Counter:
        def __init__(self):
            self.count = 0
    
    counter = Counter()
    
    def flaky_function():
        counter.count += 1
        if counter.count < 2:
            raise Exception("Simulated failure")
        return "Success!"
    
    try:
        result = retry_handler.retry_with_backoff(flaky_function)
        print(f"  ✓ Retry succeeded: {result} (attempts: {counter.count})")
    except Exception as e:
        print(f"  ✗ Retry failed: {e}")
    
    # Test decorators
    print("\n🎨 Testing decorators...")
    
    @with_retry(max_attempts=2, initial_delay=0.1)
    def test_retry_decorator():
        return "Decorator works!"
    
    result = test_retry_decorator()
    print(f"  ✓ @with_retry: {result}")
    
    def fallback_function():
        return "Fallback result"
    
    @with_fallback(fallback_function)
    def test_fallback_decorator():
        raise Exception("Primary failed")
    
    result = test_fallback_decorator()
    print(f"  ✓ @with_fallback: {result}")
    
    # Test error handler
    print("\n🛡️ Testing ErrorHandler...")
    error_handler = ErrorHandler()
    
    def primary_ai_call():
        return "AI result"
    
    def fallback_rule_call():
        return "Rule-based result"
    
    result = error_handler.handle_ai_call(primary_ai_call, fallback_rule_call)
    print(f"  ✓ AI call handled: {result}")
    
    # Test graceful degradation
    print("\n⚠️ Testing GracefulDegradation...")
    degradation = GracefulDegradation(error_handler)
    
    level = degradation.check_and_adjust()
    print(f"  Degradation level: {level}")
    print(f"  Should use AI: {degradation.should_use_ai()}")
    print(f"  Aggressive caching: {degradation.should_use_cache_aggressively()}")
    print(f"  Recommended timeout: {degradation.get_recommended_timeout(30.0)}s")
    
    print("\n" + "="*70)
    print("✓ Error Handler tests complete")
    print("="*70)
