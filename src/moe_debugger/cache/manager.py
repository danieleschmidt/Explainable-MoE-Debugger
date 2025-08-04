"""Cache manager with multiple backend support."""

import os
import json
import hashlib
from typing import Any, Optional, Dict, Union
from datetime import datetime, timedelta
import logging

from .memory_cache import MemoryCache
from .redis_cache import RedisCache

logger = logging.getLogger(__name__)


class CacheManager:
    """Unified cache manager supporting multiple backends."""
    
    def __init__(self, cache_type: str = "auto", **kwargs):
        self.cache_type = cache_type
        self.cache = self._create_cache(cache_type, **kwargs)
        
    def _create_cache(self, cache_type: str, **kwargs):
        """Create appropriate cache backend."""
        if cache_type == "redis":
            return RedisCache(**kwargs)
        elif cache_type == "memory":
            return MemoryCache(**kwargs)
        elif cache_type == "auto":
            # Try Redis first, fallback to memory
            redis_url = kwargs.get("redis_url") or os.getenv("REDIS_URL")
            if redis_url:
                try:
                    cache = RedisCache(redis_url=redis_url)
                    if cache.is_available():
                        logger.info("Using Redis cache backend")
                        return cache
                except Exception as e:
                    logger.warning(f"Redis not available, falling back to memory cache: {e}")
            
            logger.info("Using memory cache backend")
            return MemoryCache(**kwargs)
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
    
    def _generate_key(self, key: Union[str, Dict[str, Any]]) -> str:
        """Generate cache key from string or dict."""
        if isinstance(key, str):
            return key
        elif isinstance(key, dict):
            # Create deterministic hash from dict
            key_str = json.dumps(key, sort_keys=True)
            return hashlib.md5(key_str.encode()).hexdigest()
        else:
            return str(key)
    
    def get(self, key: Union[str, Dict[str, Any]], default: Any = None) -> Any:
        """Get value from cache."""
        cache_key = self._generate_key(key)
        try:
            value = self.cache.get(cache_key)
            return value if value is not None else default
        except Exception as e:
            logger.error(f"Cache get error for key {cache_key}: {e}")
            return default
    
    def set(self, key: Union[str, Dict[str, Any]], value: Any, 
            ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        cache_key = self._generate_key(key)
        try:
            return self.cache.set(cache_key, value, ttl)
        except Exception as e:
            logger.error(f"Cache set error for key {cache_key}: {e}")
            return False
    
    def delete(self, key: Union[str, Dict[str, Any]]) -> bool:
        """Delete value from cache."""
        cache_key = self._generate_key(key)
        try:
            return self.cache.delete(cache_key)
        except Exception as e:
            logger.error(f"Cache delete error for key {cache_key}: {e}")
            return False
    
    def exists(self, key: Union[str, Dict[str, Any]]) -> bool:
        """Check if key exists in cache."""
        cache_key = self._generate_key(key)
        try:
            return self.cache.exists(cache_key)
        except Exception as e:
            logger.error(f"Cache exists error for key {cache_key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            return self.cache.clear()
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            stats = self.cache.get_stats()
            stats["cache_type"] = self.cache_type
            return stats
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"cache_type": self.cache_type, "error": str(e)}
    
    def cache_routing_stats(self, session_id: str, stats: Dict[str, Any], 
                           ttl: int = 300) -> bool:
        """Cache routing statistics."""
        key = f"routing_stats:{session_id}"
        return self.set(key, stats, ttl)
    
    def get_routing_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached routing statistics."""
        key = f"routing_stats:{session_id}"
        return self.get(key)
    
    def cache_expert_metrics(self, session_id: str, metrics: Dict[str, Any], 
                            ttl: int = 300) -> bool:
        """Cache expert metrics."""
        key = f"expert_metrics:{session_id}"
        return self.set(key, metrics, ttl)
    
    def get_expert_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached expert metrics."""
        key = f"expert_metrics:{session_id}"
        return self.get(key)
    
    def cache_analysis_result(self, analysis_type: str, session_id: str, 
                             result: Any, ttl: int = 600) -> bool:
        """Cache analysis result."""
        key = f"analysis:{analysis_type}:{session_id}"
        return self.set(key, result, ttl)
    
    def get_analysis_result(self, analysis_type: str, session_id: str) -> Any:
        """Get cached analysis result."""
        key = f"analysis:{analysis_type}:{session_id}"
        return self.get(key)
    
    def cache_visualization_data(self, session_id: str, viz_type: str, 
                                data: Any, ttl: int = 180) -> bool:
        """Cache visualization data."""
        key = f"viz:{viz_type}:{session_id}"
        return self.set(key, data, ttl)
    
    def get_visualization_data(self, session_id: str, viz_type: str) -> Any:
        """Get cached visualization data."""
        key = f"viz:{viz_type}:{session_id}"
        return self.get(key)
    
    def invalidate_session_cache(self, session_id: str) -> bool:
        """Invalidate all cache entries for a session."""
        patterns = [
            f"routing_stats:{session_id}",
            f"expert_metrics:{session_id}",
            f"analysis:*:{session_id}",
            f"viz:*:{session_id}"
        ]
        
        success = True
        for pattern in patterns:
            if "*" in pattern:
                # Handle wildcard patterns
                keys = self.cache.keys(pattern) if hasattr(self.cache, 'keys') else []
                for key in keys:
                    success &= self.delete(key)
            else:
                success &= self.delete(pattern)
        
        return success
    
    def is_available(self) -> bool:
        """Check if cache is available."""
        return self.cache.is_available()


# Global cache manager instance
cache_manager = CacheManager()