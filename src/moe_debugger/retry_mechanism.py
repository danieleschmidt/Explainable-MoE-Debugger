"""Intelligent retry mechanisms with exponential backoff and jitter."""

import asyncio
import random
import time
import logging
from typing import Callable, Any, Optional, Type, Union, List
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = "exponential"  # exponential, linear, fixed
    retryable_exceptions: tuple = (Exception,)


class RetryExhaustedException(Exception):
    """Exception raised when all retry attempts are exhausted."""
    
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Retry exhausted after {attempts} attempts. Last error: {last_exception}")


class RetryMechanism:
    """Intelligent retry mechanism with multiple backoff strategies."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.total_attempts = 0
        self.successful_attempts = 0
        self.failed_attempts = 0
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with retry logic."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                self.total_attempts += 1
                logger.debug(f"Attempt {attempt}/{self.config.max_attempts} for {func.__name__}")
                
                result = func(*args, **kwargs)
                self.successful_attempts += 1
                
                if attempt > 1:
                    logger.info(f"Function {func.__name__} succeeded on attempt {attempt}")
                
                return result
                
            except self.config.retryable_exceptions as e:
                last_exception = e
                self.failed_attempts += 1
                
                if attempt == self.config.max_attempts:
                    logger.error(f"Function {func.__name__} failed after {attempt} attempts: {e}")
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Function {func.__name__} failed on attempt {attempt}: {e}. "
                             f"Retrying in {delay:.2f}s")
                
                time.sleep(delay)
        
        raise RetryExhaustedException(self.config.max_attempts, last_exception)
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                self.total_attempts += 1
                logger.debug(f"Async attempt {attempt}/{self.config.max_attempts} for {func.__name__}")
                
                result = await func(*args, **kwargs)
                self.successful_attempts += 1
                
                if attempt > 1:
                    logger.info(f"Async function {func.__name__} succeeded on attempt {attempt}")
                
                return result
                
            except self.config.retryable_exceptions as e:
                last_exception = e
                self.failed_attempts += 1
                
                if attempt == self.config.max_attempts:
                    logger.error(f"Async function {func.__name__} failed after {attempt} attempts: {e}")
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Async function {func.__name__} failed on attempt {attempt}: {e}. "
                             f"Retrying in {delay:.2f}s")
                
                await asyncio.sleep(delay)
        
        raise RetryExhaustedException(self.config.max_attempts, last_exception)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry attempt."""
        if self.config.backoff_strategy == "exponential":
            delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        elif self.config.backoff_strategy == "linear":
            delay = self.config.base_delay * attempt
        else:  # fixed
            delay = self.config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_amount = delay * 0.1
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def get_stats(self) -> dict:
        """Get retry statistics."""
        success_rate = (self.successful_attempts / self.total_attempts * 100
                       if self.total_attempts > 0 else 0)
        
        return {
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
            "success_rate": success_rate
        }


class AdaptiveRetryMechanism(RetryMechanism):
    """Adaptive retry mechanism that learns from failure patterns."""
    
    def __init__(self, config: RetryConfig):
        super().__init__(config)
        self.failure_patterns = {}  # Track failure patterns by exception type
        self.success_patterns = {}  # Track success patterns by attempt number
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with adaptive retry logic."""
        # Analyze historical patterns to adjust retry strategy
        self._adapt_config(func)
        
        try:
            result = super().execute(func, *args, **kwargs)
            self._record_success_pattern(func, self.config.max_attempts)
            return result
            
        except RetryExhaustedException as e:
            self._record_failure_pattern(func, e.last_exception)
            raise
    
    def _adapt_config(self, func: Callable):
        """Adapt retry configuration based on historical data."""
        func_name = func.__name__
        
        # If this function frequently fails, increase max attempts
        if func_name in self.failure_patterns:
            failure_rate = self.failure_patterns[func_name].get("rate", 0)
            if failure_rate > 0.5:  # More than 50% failure rate
                self.config.max_attempts = min(self.config.max_attempts + 1, 10)
                logger.debug(f"Increased max_attempts to {self.config.max_attempts} for {func_name}")
        
        # If this function typically succeeds on later attempts, adjust base delay
        if func_name in self.success_patterns:
            avg_success_attempt = self.success_patterns[func_name].get("avg_attempt", 1)
            if avg_success_attempt > 2:
                self.config.base_delay = min(self.config.base_delay * 1.5, 5.0)
                logger.debug(f"Increased base_delay to {self.config.base_delay} for {func_name}")
    
    def _record_success_pattern(self, func: Callable, attempts_made: int):
        """Record successful execution pattern."""
        func_name = func.__name__
        
        if func_name not in self.success_patterns:
            self.success_patterns[func_name] = {"attempts": [], "avg_attempt": 1}
        
        self.success_patterns[func_name]["attempts"].append(attempts_made)
        attempts_list = self.success_patterns[func_name]["attempts"]
        self.success_patterns[func_name]["avg_attempt"] = sum(attempts_list) / len(attempts_list)
    
    def _record_failure_pattern(self, func: Callable, exception: Exception):
        """Record failure pattern."""
        func_name = func.__name__
        exception_type = type(exception).__name__
        
        if func_name not in self.failure_patterns:
            self.failure_patterns[func_name] = {"exceptions": {}, "rate": 0, "total": 0}
        
        pattern = self.failure_patterns[func_name]
        pattern["total"] += 1
        
        if exception_type not in pattern["exceptions"]:
            pattern["exceptions"][exception_type] = 0
        pattern["exceptions"][exception_type] += 1
        
        # Update failure rate
        pattern["rate"] = pattern["total"] / (pattern["total"] + self.successful_attempts)


def retry(config: Optional[RetryConfig] = None):
    """Decorator for applying retry mechanism."""
    retry_config = config or RetryConfig()
    mechanism = RetryMechanism(retry_config)
    
    def decorator(func: Callable) -> Callable:
        return mechanism(func)
    return decorator


def adaptive_retry(config: Optional[RetryConfig] = None):
    """Decorator for applying adaptive retry mechanism."""
    retry_config = config or RetryConfig()
    mechanism = AdaptiveRetryMechanism(retry_config)
    
    def decorator(func: Callable) -> Callable:
        return mechanism(func)
    return decorator


# Predefined retry configurations
class RetryConfigs:
    """Predefined retry configurations for common scenarios."""
    
    NETWORK = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=30.0,
        retryable_exceptions=(ConnectionError, TimeoutError)
    )
    
    DATABASE = RetryConfig(
        max_attempts=5,
        base_delay=0.5,
        max_delay=10.0,
        exponential_base=1.5,
        retryable_exceptions=(ConnectionError, TimeoutError)
    )
    
    API_CALL = RetryConfig(
        max_attempts=3,
        base_delay=2.0,
        max_delay=60.0,
        jitter=True,
        retryable_exceptions=(ConnectionError, TimeoutError, Exception)
    )
    
    FILE_IO = RetryConfig(
        max_attempts=2,
        base_delay=0.1,
        max_delay=1.0,
        backoff_strategy="linear",
        retryable_exceptions=(IOError, OSError)
    )
    
    QUICK_OPERATION = RetryConfig(
        max_attempts=2,
        base_delay=0.1,
        max_delay=0.5,
        backoff_strategy="fixed",
        jitter=False
    )