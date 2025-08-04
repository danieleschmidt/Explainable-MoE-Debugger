"""Comprehensive logging configuration for MoE debugger."""

import logging
import logging.handlers
import os
import sys
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
import threading
from datetime import datetime

from .validation import safe_json_dumps


class PerformanceFilter(logging.Filter):
    """Filter to track performance metrics in logs."""
    
    def filter(self, record):
        # Add timing information
        if not hasattr(record, 'duration'):
            record.duration = 0.0
        
        # Add memory usage if available
        try:
            import psutil
            record.memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        except:
            record.memory_mb = 0.0
        
        return True


class SecurityFilter(logging.Filter):
    """Filter to detect and flag security-related log entries."""
    
    SECURITY_KEYWORDS = [
        'password', 'token', 'key', 'secret', 'credential',
        'authorization', 'authentication', 'login', 'session'
    ]
    
    def filter(self, record):
        # Flag potential security issues
        message_lower = record.getMessage().lower()
        record.is_security_sensitive = any(
            keyword in message_lower for keyword in self.SECURITY_KEYWORDS
        )
        
        # Sanitize sensitive information
        if record.is_security_sensitive:
            record.msg = self._sanitize_message(record.msg)
        
        return True
    
    def _sanitize_message(self, message: str) -> str:
        """Sanitize sensitive information from log messages."""
        # Replace common patterns with placeholders
        import re
        
        patterns = [
            (r'password[=:]\s*\S+', 'password=***'),
            (r'token[=:]\s*\S+', 'token=***'),
            (r'key[=:]\s*\S+', 'key=***'),
            (r'secret[=:]\s*\S+', 'secret=***'),
        ]
        
        sanitized = message
        for pattern, replacement in patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        return sanitized


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add custom attributes
        for attr in ['duration', 'memory_mb', 'is_security_sensitive']:
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        try:
            return safe_json_dumps(log_entry)
        except Exception:
            # Fallback to simple format if JSON serialization fails
            return f"{record.levelname}: {record.getMessage()}"


class DebuggerLogManager:
    """Centralized logging manager for MoE debugger."""
    
    def __init__(self, log_level: str = "INFO", log_dir: Optional[str] = None):
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.loggers: Dict[str, logging.Logger] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup comprehensive logging configuration."""
        # Create logs directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(SecurityFilter())
        root_logger.addHandler(console_handler)
        
        # Setup file handlers
        self._setup_file_handlers(root_logger)
        
        # Setup component-specific loggers
        self._setup_component_loggers()
    
    def _setup_file_handlers(self, logger: logging.Logger):
        """Setup file-based logging handlers."""
        
        # Main application log
        main_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "moe_debugger.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        main_handler.setLevel(logging.DEBUG)
        main_handler.setFormatter(JsonFormatter())
        main_handler.addFilter(PerformanceFilter())
        main_handler.addFilter(SecurityFilter())
        logger.addHandler(main_handler)
        
        # Error-only log
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JsonFormatter())
        logger.addHandler(error_handler)
        
        # Performance log
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "performance.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(JsonFormatter())
        perf_handler.addFilter(PerformanceFilter())
        # Only log performance-related messages
        perf_handler.addFilter(lambda record: 'performance' in record.getMessage().lower() or 
                                            hasattr(record, 'duration'))
        logger.addHandler(perf_handler)
        
        # Security log
        security_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "security.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=10  # Keep more security logs
        )
        security_handler.setLevel(logging.WARNING)
        security_handler.setFormatter(JsonFormatter())
        security_handler.addFilter(lambda record: getattr(record, 'is_security_sensitive', False))
        logger.addHandler(security_handler)
    
    def _setup_component_loggers(self):
        """Setup loggers for specific components."""
        components = [
            'moe_debugger.analyzer',
            'moe_debugger.profiler',
            'moe_debugger.debugger',
            'moe_debugger.server',
            'moe_debugger.hooks',
            'moe_debugger.validation',
        ]
        
        for component in components:
            logger = logging.getLogger(component)
            logger.setLevel(self.log_level)
            self.loggers[component] = logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger for a specific component."""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self.log_level)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def log_performance(self, logger_name: str, operation: str, 
                       duration: float, extra_data: Optional[Dict[str, Any]] = None):
        """Log performance metrics."""
        logger = self.get_logger(logger_name)
        
        extra_fields = {
            'operation': operation,
            'duration': duration,
            'performance_metric': True
        }
        
        if extra_data:
            extra_fields.update(extra_data)
        
        # Create a custom LogRecord with extra fields
        record = logger.makeRecord(
            logger.name, logging.INFO, __file__, 0,
            f"Performance: {operation} completed in {duration:.3f}s",
            (), None
        )
        record.extra_fields = extra_fields
        
        logger.handle(record)
    
    def log_security_event(self, logger_name: str, event_type: str, 
                          details: str, severity: str = "WARNING"):
        """Log security-related events."""
        logger = self.get_logger(logger_name)
        
        level = getattr(logging, severity.upper(), logging.WARNING)
        
        extra_fields = {
            'security_event': True,
            'event_type': event_type,
            'is_security_sensitive': True
        }
        
        record = logger.makeRecord(
            logger.name, level, __file__, 0,
            f"Security event [{event_type}]: {details}",
            (), None
        )
        record.extra_fields = extra_fields
        record.is_security_sensitive = True
        
        logger.handle(record)
    
    def log_error_with_context(self, logger_name: str, error: Exception, 
                              context: Optional[Dict[str, Any]] = None):
        """Log errors with additional context."""
        logger = self.get_logger(logger_name)
        
        extra_fields = {
            'error_type': type(error).__name__,
            'error_message': str(error),
        }
        
        if context:
            extra_fields['context'] = context
        
        record = logger.makeRecord(
            logger.name, logging.ERROR, __file__, 0,
            f"Error occurred: {error}",
            (), (type(error), error, error.__traceback__)
        )
        record.extra_fields = extra_fields
        
        logger.handle(record)
    
    def set_log_level(self, level: str):
        """Dynamically change log level."""
        new_level = getattr(logging, level.upper(), logging.INFO)
        
        # Update all loggers
        for logger in self.loggers.values():
            logger.setLevel(new_level)
        
        # Update handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(new_level)
        
        self.log_level = new_level
        logging.getLogger(__name__).info(f"Log level changed to {level}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = {
            'log_level': logging.getLevelName(self.log_level),
            'log_directory': str(self.log_dir),
            'active_loggers': len(self.loggers),
            'log_files': {}
        }
        
        # Get log file sizes
        for log_file in self.log_dir.glob("*.log"):
            try:
                stats['log_files'][log_file.name] = {
                    'size_mb': log_file.stat().st_size / 1024 / 1024,
                    'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                }
            except Exception:
                stats['log_files'][log_file.name] = {'error': 'Cannot read file stats'}
        
        return stats


# Global log manager instance
_log_manager: Optional[DebuggerLogManager] = None
_lock = threading.Lock()


def get_log_manager(log_level: str = "INFO", log_dir: Optional[str] = None) -> DebuggerLogManager:
    """Get or create the global log manager."""
    global _log_manager
    
    with _lock:
        if _log_manager is None:
            _log_manager = DebuggerLogManager(log_level, log_dir)
        return _log_manager


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific component."""
    return get_log_manager().get_logger(name)


def performance_timer(logger_name: str, operation: str):
    """Decorator for timing operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                get_log_manager().log_performance(logger_name, operation, duration)
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                get_log_manager().log_performance(
                    logger_name, f"{operation}_failed", duration, {'error': str(e)}
                )
                raise
        return wrapper
    return decorator


# Configure logging on module import
try:
    # Initialize with environment variables or defaults
    log_level = os.getenv('MOE_DEBUG_LOG_LEVEL', 'INFO')
    log_dir = os.getenv('MOE_DEBUG_LOG_DIR', None)
    get_log_manager(log_level, log_dir)
except Exception as e:
    # Fallback to basic logging if setup fails
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).warning(f"Failed to setup advanced logging: {e}")