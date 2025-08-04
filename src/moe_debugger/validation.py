"""Input validation and data sanitization for MoE debugger."""

import re
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
import logging

from .models import RoutingEvent, HookConfiguration, ModelArchitecture

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class ConfigurationError(ValidationError):
    """Configuration validation error."""
    pass


class DataValidationError(ValidationError):
    """Data validation error."""
    pass


class SecurityError(ValidationError):
    """Security validation error."""
    pass


class InputValidator:
    """Comprehensive input validation for MoE debugger."""
    
    # Security patterns to detect potentially dangerous input
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS attempts
        r'javascript:',                # JavaScript URLs
        r'data:.*base64',              # Base64 data URLs
        r'eval\s*\(',                  # eval() calls
        r'exec\s*\(',                  # exec() calls
        r'import\s+os',                # OS imports
        r'__import__',                 # Dynamic imports
        r'\.\./',                      # Path traversal
        r'\.\./\.\.',                  # Multiple path traversal
        r'[;|&`$]',                    # Shell injection characters
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                for pattern in self.DANGEROUS_PATTERNS]
    
    def validate_string_input(self, value: str, field_name: str, 
                            max_length: int = 1000, allow_empty: bool = True) -> str:
        """Validate string input for security and length."""
        if not isinstance(value, str):
            raise DataValidationError(f"{field_name} must be a string, got {type(value)}")
        
        if not allow_empty and not value.strip():
            raise DataValidationError(f"{field_name} cannot be empty")
        
        if len(value) > max_length:
            raise DataValidationError(f"{field_name} exceeds maximum length of {max_length}")
        
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(value):
                logger.warning(f"Potentially dangerous input detected in {field_name}: {value[:100]}...")
                raise SecurityError(f"Invalid characters detected in {field_name}")
        
        return value.strip()
    
    def validate_numeric_input(self, value: Union[int, float], field_name: str,
                             min_value: Optional[float] = None, 
                             max_value: Optional[float] = None) -> Union[int, float]:
        """Validate numeric input."""
        if not isinstance(value, (int, float)):
            raise DataValidationError(f"{field_name} must be numeric, got {type(value)}")
        
        if min_value is not None and value < min_value:
            raise DataValidationError(f"{field_name} must be >= {min_value}, got {value}")
        
        if max_value is not None and value > max_value:
            raise DataValidationError(f"{field_name} must be <= {max_value}, got {value}")
        
        return value
    
    def validate_list_input(self, value: List[Any], field_name: str,
                          max_length: int = 10000, item_type: Optional[type] = None) -> List[Any]:
        """Validate list input."""
        if not isinstance(value, list):
            raise DataValidationError(f"{field_name} must be a list, got {type(value)}")
        
        if len(value) > max_length:
            raise DataValidationError(f"{field_name} exceeds maximum length of {max_length}")
        
        if item_type is not None:
            for i, item in enumerate(value):
                if not isinstance(item, item_type):
                    raise DataValidationError(
                        f"{field_name}[{i}] must be {item_type.__name__}, got {type(item)}"
                    )
        
        return value
    
    def validate_dict_input(self, value: Dict[str, Any], field_name: str,
                          max_keys: int = 1000) -> Dict[str, Any]:
        """Validate dictionary input."""
        if not isinstance(value, dict):
            raise DataValidationError(f"{field_name} must be a dict, got {type(value)}")
        
        if len(value) > max_keys:
            raise DataValidationError(f"{field_name} exceeds maximum keys of {max_keys}")
        
        # Validate all keys are strings and safe
        for key in value.keys():
            if not isinstance(key, str):
                raise DataValidationError(f"{field_name} keys must be strings, got {type(key)}")
            self.validate_string_input(key, f"{field_name} key", max_length=100, allow_empty=False)
        
        return value
    
    def validate_file_path(self, path: str, field_name: str) -> Path:
        """Validate file path for security."""
        try:
            path_obj = Path(path).resolve()
            
            # Check for path traversal attempts
            if '..' in path or path.startswith('/'):
                logger.warning(f"Potentially dangerous path in {field_name}: {path}")
                raise SecurityError(f"Invalid path format in {field_name}")
            
            return path_obj
            
        except Exception as e:
            raise DataValidationError(f"Invalid path in {field_name}: {e}")
    
    def validate_routing_event(self, event: Dict[str, Any]) -> RoutingEvent:
        """Validate and create a RoutingEvent from dictionary."""
        try:
            # Validate required fields
            timestamp = self.validate_numeric_input(
                event.get('timestamp', 0), 'timestamp', min_value=0
            )
            
            layer_idx = self.validate_numeric_input(
                event.get('layer_idx', 0), 'layer_idx', min_value=0, max_value=1000
            )
            
            token_position = self.validate_numeric_input(
                event.get('token_position', 0), 'token_position', min_value=0, max_value=100000
            )
            
            token = self.validate_string_input(
                event.get('token', ''), 'token', max_length=1000, allow_empty=False
            )
            
            expert_weights = self.validate_list_input(
                event.get('expert_weights', []), 'expert_weights', max_length=1000, item_type=float
            )
            
            selected_experts = self.validate_list_input(
                event.get('selected_experts', []), 'selected_experts', max_length=100, item_type=int
            )
            
            routing_confidence = self.validate_numeric_input(
                event.get('routing_confidence', 0.0), 'routing_confidence', min_value=0.0, max_value=1.0
            )
            
            sequence_id = self.validate_string_input(
                event.get('sequence_id', ''), 'sequence_id', max_length=100, allow_empty=False
            )
            
            # Additional validation
            if not expert_weights:
                raise DataValidationError("expert_weights cannot be empty")
            
            if not selected_experts:
                raise DataValidationError("selected_experts cannot be empty")
            
            if max(selected_experts) >= len(expert_weights):
                raise DataValidationError("selected_experts indices exceed expert_weights length")
            
            return RoutingEvent(
                timestamp=timestamp,
                layer_idx=int(layer_idx),
                token_position=int(token_position),
                token=token,
                expert_weights=expert_weights,
                selected_experts=selected_experts,
                routing_confidence=routing_confidence,
                sequence_id=sequence_id
            )
            
        except Exception as e:
            logger.error(f"Failed to validate routing event: {e}")
            raise DataValidationError(f"Invalid routing event: {e}")
    
    def validate_hook_configuration(self, config: Dict[str, Any]) -> HookConfiguration:
        """Validate and create HookConfiguration from dictionary."""
        try:
            enabled_hooks = self.validate_dict_input(
                config.get('enabled_hooks', {}), 'enabled_hooks', max_keys=50
            )
            
            sampling_rate = self.validate_numeric_input(
                config.get('sampling_rate', 0.1), 'sampling_rate', min_value=0.0, max_value=1.0
            )
            
            buffer_size = self.validate_numeric_input(
                config.get('buffer_size', 1000), 'buffer_size', min_value=1, max_value=1000000
            )
            
            save_gradients = config.get('save_gradients', False)
            if not isinstance(save_gradients, bool):
                raise DataValidationError("save_gradients must be boolean")
            
            save_activations = config.get('save_activations', True)
            if not isinstance(save_activations, bool):
                raise DataValidationError("save_activations must be boolean")
            
            track_parameters = self.validate_list_input(
                config.get('track_parameters', []), 'track_parameters', max_length=100, item_type=str
            )
            
            memory_limit_mb = self.validate_numeric_input(
                config.get('memory_limit_mb', 2048), 'memory_limit_mb', min_value=100, max_value=100000
            )
            
            return HookConfiguration(
                enabled_hooks=enabled_hooks,
                sampling_rate=sampling_rate,
                buffer_size=int(buffer_size),
                save_gradients=save_gradients,
                save_activations=save_activations,
                track_parameters=track_parameters,
                memory_limit_mb=int(memory_limit_mb)
            )
            
        except Exception as e:
            logger.error(f"Failed to validate hook configuration: {e}")
            raise ConfigurationError(f"Invalid hook configuration: {e}")
    
    def validate_model_architecture(self, arch: Dict[str, Any]) -> ModelArchitecture:
        """Validate and create ModelArchitecture from dictionary."""
        try:
            num_layers = self.validate_numeric_input(
                arch.get('num_layers', 1), 'num_layers', min_value=1, max_value=1000
            )
            
            num_experts_per_layer = self.validate_numeric_input(
                arch.get('num_experts_per_layer', 1), 'num_experts_per_layer', 
                min_value=1, max_value=1000
            )
            
            hidden_size = self.validate_numeric_input(
                arch.get('hidden_size', 768), 'hidden_size', min_value=1, max_value=100000
            )
            
            intermediate_size = self.validate_numeric_input(
                arch.get('intermediate_size', 3072), 'intermediate_size', 
                min_value=1, max_value=1000000
            )
            
            vocab_size = self.validate_numeric_input(
                arch.get('vocab_size', 32000), 'vocab_size', min_value=1, max_value=10000000
            )
            
            max_sequence_length = self.validate_numeric_input(
                arch.get('max_sequence_length', 2048), 'max_sequence_length', 
                min_value=1, max_value=1000000
            )
            
            expert_capacity = self.validate_numeric_input(
                arch.get('expert_capacity', 2.0), 'expert_capacity', min_value=0.1, max_value=100.0
            )
            
            router_type = self.validate_string_input(
                arch.get('router_type', 'top_k'), 'router_type', max_length=50, allow_empty=False
            )
            
            expert_types = self.validate_dict_input(
                arch.get('expert_types', {}), 'expert_types', max_keys=1000
            )
            
            return ModelArchitecture(
                num_layers=int(num_layers),
                num_experts_per_layer=int(num_experts_per_layer),
                hidden_size=int(hidden_size),
                intermediate_size=int(intermediate_size),
                vocab_size=int(vocab_size),
                max_sequence_length=int(max_sequence_length),
                expert_capacity=expert_capacity,
                router_type=router_type,
                expert_types=expert_types
            )
            
        except Exception as e:
            logger.error(f"Failed to validate model architecture: {e}")
            raise DataValidationError(f"Invalid model architecture: {e}")
    
    def sanitize_json_output(self, data: Any) -> Any:
        """Sanitize data before JSON serialization."""
        if isinstance(data, dict):
            return {
                self.validate_string_input(str(k), "json_key", max_length=200): 
                self.sanitize_json_output(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            if len(data) > 10000:  # Prevent huge arrays
                logger.warning("Large array truncated in JSON output")
                data = data[:10000]
            return [self.sanitize_json_output(item) for item in data]
        elif isinstance(data, str):
            return self.validate_string_input(data, "json_string", max_length=10000)
        elif isinstance(data, (int, float, bool, type(None))):
            return data
        else:
            # Convert unknown types to string
            return str(data)[:1000]  # Limit length


# Global validator instance
validator = InputValidator()


def validate_input(func):
    """Decorator for automatic input validation."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError:
            raise  # Re-raise validation errors
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise DataValidationError(f"Input validation failed: {e}")
    return wrapper


def safe_json_loads(json_str: str, max_size: int = 1024 * 1024) -> Dict[str, Any]:
    """Safely load JSON with size limits."""
    if len(json_str) > max_size:
        raise DataValidationError(f"JSON input too large: {len(json_str)} bytes")
    
    try:
        data = json.loads(json_str)
        return validator.sanitize_json_output(data)
    except json.JSONDecodeError as e:
        raise DataValidationError(f"Invalid JSON: {e}")


def safe_json_dumps(data: Any, max_size: int = 10 * 1024 * 1024) -> str:
    """Safely dump JSON with size limits."""
    try:
        sanitized_data = validator.sanitize_json_output(data)
        json_str = json.dumps(sanitized_data, ensure_ascii=True, separators=(',', ':'))
        
        if len(json_str) > max_size:
            logger.warning("JSON output truncated due to size limit")
            raise DataValidationError(f"JSON output too large: {len(json_str)} bytes")
        
        return json_str
    except Exception as e:
        raise DataValidationError(f"JSON serialization failed: {e}")