"""Model loading and management for MoE Debugger with Hugging Face integration."""

import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import json

# Try to import torch and transformers, fall back to mock if not available
try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        AutoConfig,
        PreTrainedModel,
        PreTrainedTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    from .mock_torch import torch, nn
    TRANSFORMERS_AVAILABLE = False
    # Mock transformers classes
    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return None
    
    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return None
    
    class AutoConfig:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return None

from .models import SessionInfo

logger = logging.getLogger(__name__)

# Supported MoE model architectures
SUPPORTED_MOE_MODELS = {
    'mixtral': [
        'mistralai/Mixtral-8x7B-v0.1',
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'mistralai/Mixtral-8x22B-v0.1',
        'mistralai/Mixtral-8x22B-Instruct-v0.1'
    ],
    'switch_transformer': [
        'google/switch-base-8',
        'google/switch-base-16',
        'google/switch-base-32'
    ],
    'glam': [
        'google/glam-64b'
    ]
}

class ModelLoader:
    """Handles loading and management of MoE models from Hugging Face."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'moe_debugger'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_model: Optional[Any] = None
        self.current_tokenizer: Optional[Any] = None
        self.current_config: Optional[Dict[str, Any]] = None
        self.model_info: Optional[Dict[str, Any]] = None
        
        logger.info(f"ModelLoader initialized with cache_dir: {self.cache_dir}")
        logger.info(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    
    def list_supported_models(self) -> Dict[str, List[str]]:
        """Get list of supported MoE model architectures."""
        return SUPPORTED_MOE_MODELS
    
    def load_model(
        self, 
        model_name: str, 
        device: Optional[str] = None,
        torch_dtype: Optional[str] = 'auto',
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False
    ) -> Tuple[bool, str]:
        """
        Load a MoE model from Hugging Face.
        
        Args:
            model_name: Model identifier (e.g., 'mistralai/Mixtral-8x7B-v0.1')
            device: Target device ('cuda', 'cpu', etc.)
            torch_dtype: Data type for model weights
            load_in_8bit: Whether to load in 8-bit precision
            load_in_4bit: Whether to load in 4-bit precision  
            trust_remote_code: Whether to trust remote code execution
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not TRANSFORMERS_AVAILABLE:
            return False, "Transformers library not available. Please install with: pip install transformers"
        
        try:
            logger.info(f"Loading model: {model_name}")
            
            # Validate model architecture
            if not self._is_moe_model(model_name):
                return False, f"Model {model_name} is not a supported MoE architecture"
            
            # Determine device
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Load configuration first
            logger.info("Loading model configuration...")
            config = AutoConfig.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                trust_remote_code=trust_remote_code
            )
            
            # Extract MoE-specific configuration
            moe_config = self._extract_moe_config(config)
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                trust_remote_code=trust_remote_code
            )
            
            # Prepare model loading arguments
            model_kwargs = {
                'cache_dir': str(self.cache_dir),
                'torch_dtype': torch_dtype if torch_dtype != 'auto' else torch.float16,
                'trust_remote_code': trust_remote_code,
                'device_map': 'auto' if device == 'cuda' else device
            }
            
            if load_in_8bit:
                model_kwargs['load_in_8bit'] = True
            elif load_in_4bit:
                model_kwargs['load_in_4bit'] = True
            
            # Load model
            logger.info(f"Loading model weights to {device}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Store loaded components
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_config = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
            
            # Extract detailed model info
            self.model_info = {
                'name': model_name,
                'architecture': config.model_type if hasattr(config, 'model_type') else 'unknown',
                'num_parameters': self._count_parameters(model),
                'device': str(device),
                'dtype': str(model_kwargs.get('torch_dtype', 'unknown')),
                'precision': '8bit' if load_in_8bit else '4bit' if load_in_4bit else 'fp16',
                **moe_config
            }
            
            logger.info(f"Model loaded successfully: {self.model_info}")
            return True, f"Successfully loaded {model_name}"
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False, f"Failed to load model: {str(e)}"
    
    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        if self.current_model is not None:
            logger.info("Unloading current model...")
            del self.current_model
            del self.current_tokenizer
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.current_model = None
            self.current_tokenizer = None
            self.current_config = None
            self.model_info = None
            
            logger.info("Model unloaded successfully")
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently loaded model."""
        return self.model_info
    
    def get_model(self) -> Optional[Any]:
        """Get the currently loaded model."""
        return self.current_model
    
    def get_tokenizer(self) -> Optional[Any]:
        """Get the current tokenizer."""
        return self.current_tokenizer
    
    def get_session_info(self) -> Optional[SessionInfo]:
        """Create a SessionInfo object from the current model."""
        if not self.model_info:
            return None
        
        return SessionInfo(
            session_id=f"session_{int(torch.randint(0, 1000000, (1,)).item())}",
            model_name=self.model_info['name'],
            model_architecture=self.model_info['architecture'],
            num_experts=self.model_info.get('num_experts', 8),
            num_layers=self.model_info.get('num_layers', 32),
            created_at=torch.tensor([]).new_empty(0).item(),  # Current timestamp
            last_active=torch.tensor([]).new_empty(0).item(),
            total_tokens_processed=0,
            status='active'
        )
    
    def _is_moe_model(self, model_name: str) -> bool:
        """Check if a model is a supported MoE architecture."""
        for arch_models in SUPPORTED_MOE_MODELS.values():
            if any(model_name.startswith(model.split('/')[0]) or model_name == model 
                   for model in arch_models):
                return True
        return False
    
    def _extract_moe_config(self, config) -> Dict[str, Any]:
        """Extract MoE-specific configuration from model config."""
        moe_config = {}
        
        # Common MoE parameters
        for param in [
            'num_local_experts', 'num_experts_per_tok', 'num_experts',
            'expert_capacity', 'router_aux_loss_coef', 'router_z_loss_coef',
            'num_hidden_layers', 'hidden_size', 'intermediate_size'
        ]:
            if hasattr(config, param):
                value = getattr(config, param)
                if param == 'num_local_experts':
                    moe_config['num_experts'] = value
                elif param == 'num_experts_per_tok':
                    moe_config['experts_per_token'] = value
                elif param == 'num_hidden_layers':
                    moe_config['num_layers'] = value
                else:
                    moe_config[param] = value
        
        # Set defaults for missing values
        moe_config.setdefault('num_experts', 8)
        moe_config.setdefault('num_layers', 32)
        moe_config.setdefault('experts_per_token', 2)
        
        return moe_config
    
    def _count_parameters(self, model) -> int:
        """Count the number of parameters in the model."""
        if model is None:
            return 0
        
        try:
            return sum(p.numel() for p in model.parameters())
        except Exception:
            return 0
    
    def save_model_info(self, filepath: Path) -> None:
        """Save current model information to a JSON file."""
        if self.model_info:
            with open(filepath, 'w') as f:
                json.dump(self.model_info, f, indent=2, default=str)
    
    def load_model_info(self, filepath: Path) -> Dict[str, Any]:
        """Load model information from a JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)