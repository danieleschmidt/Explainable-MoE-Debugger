"""Circuit breaker pattern for robust error handling and fault tolerance."""

import asyncio
import time
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker tripped, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    success_threshold: int = 2  # For half-open state


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._lock = threading.RLock()
        
        # Metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.circuit_open_events = 0
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            self.total_calls += 1
            
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker moving to HALF_OPEN state")
                else:
                    logger.warning("Circuit breaker is OPEN, rejecting call")
                    raise CircuitBreakerOpenError("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.config.expected_exception as e:
                self._on_failure()
                raise e
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        with self._lock:
            self.total_calls += 1
            
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker moving to HALF_OPEN state")
                else:
                    logger.warning("Circuit breaker is OPEN, rejecting async call")
                    raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self.successful_calls += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker reset to CLOSED state")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failed_calls += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if (self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN] and
                self.failure_count >= self.config.failure_threshold):
                self.state = CircuitState.OPEN
                self.circuit_open_events += 1
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        if self.last_failure_time is None:
            return False
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            success_rate = (self.successful_calls / self.total_calls * 100
                          if self.total_calls > 0 else 0)
            
            return {
                "state": self.state.value,
                "total_calls": self.total_calls,
                "successful_calls": self.successful_calls,
                "failed_calls": self.failed_calls,
                "success_rate": success_rate,
                "failure_count": self.failure_count,
                "circuit_open_events": self.circuit_open_events,
                "last_failure_time": self.last_failure_time
            }
    
    def reset(self):
        """Manually reset circuit breaker."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            logger.info("Circuit breaker manually reset")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class AdvancedCircuitBreaker(CircuitBreaker):
    """Advanced circuit breaker with sliding window and gradual recovery."""
    
    def __init__(self, config: CircuitBreakerConfig, window_size: int = 100):
        super().__init__(config)
        self.window_size = window_size
        self.call_history = []  # Sliding window of success/failure
        self.recovery_factor = 0.1  # Start with 10% of normal load
    
    def _update_call_history(self, success: bool):
        """Update sliding window of call results."""
        self.call_history.append(success)
        if len(self.call_history) > self.window_size:
            self.call_history.pop(0)
    
    def _calculate_failure_rate(self) -> float:
        """Calculate failure rate from sliding window."""
        if not self.call_history:
            return 0.0
        failures = sum(1 for success in self.call_history if not success)
        return failures / len(self.call_history)
    
    def _should_reject_call(self) -> bool:
        """Determine if call should be rejected based on recovery factor."""
        if self.state != CircuitState.HALF_OPEN:
            return False
        
        # Gradually increase load during recovery
        import random
        return random.random() > self.recovery_factor
    
    def _on_success(self):
        """Enhanced success handling with sliding window."""
        self._update_call_history(True)
        
        if self.state == CircuitState.HALF_OPEN:
            self.recovery_factor = min(1.0, self.recovery_factor + 0.1)
            if self.recovery_factor >= 1.0:
                self.state = CircuitState.CLOSED
                logger.info("Advanced circuit breaker fully recovered")
        
        super()._on_success()
    
    def _on_failure(self):
        """Enhanced failure handling with sliding window."""
        self._update_call_history(False)
        failure_rate = self._calculate_failure_rate()
        
        if (self.state == CircuitState.CLOSED and 
            failure_rate >= (self.config.failure_threshold / self.window_size)):
            self.state = CircuitState.OPEN
            self.recovery_factor = 0.1
            logger.error(f"Advanced circuit breaker opened, failure rate: {failure_rate:.2%}")
        
        super()._on_failure()


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a named circuit breaker."""
    if name not in _circuit_breakers:
        config = config or CircuitBreakerConfig()
        _circuit_breakers[name] = CircuitBreaker(config)
    return _circuit_breakers[name]


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for applying circuit breaker pattern."""
    def decorator(func: Callable) -> Callable:
        cb = get_circuit_breaker(name, config)
        return cb(func)
    return decorator