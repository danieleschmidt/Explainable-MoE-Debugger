"""Redis cache implementation."""

import json
import pickle
import os
from typing import Any, Optional, Dict, List
import logging

try:
    import redis
    from redis.exceptions import ConnectionError, TimeoutError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, redis_url: Optional[str] = None, 
                 prefix: str = "moe_debugger:", 
                 serializer: str = "json",
                 **redis_kwargs):
        self.prefix = prefix
        self.serializer = serializer
        self.redis_client = None
        self.available = False
        
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - install with: pip install redis")
            return
        
        # Get Redis URL
        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        
        try:
            # Create Redis client
            if redis_url.startswith("redis://") or redis_url.startswith("rediss://"):
                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    **redis_kwargs
                )
            else:
                # Parse host:port format
                if ":" in redis_url:
                    host, port = redis_url.split(":", 1)
                    port = int(port)
                else:
                    host, port = redis_url, 6379
                
                self.redis_client = redis.Redis(
                    host=host,
                    port=port,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    **redis_kwargs
                )
            
            # Test connection
            self.redis_client.ping()
            self.available = True
            logger.info(f"Redis cache connected: {redis_url}")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.available = False
    
    def _serialize(self, value: Any) -> str:
        """Serialize value for storage."""
        if self.serializer == "json":
            try:
                return json.dumps(value)
            except (TypeError, ValueError):
                # Fallback to pickle for non-JSON serializable objects
                return pickle.dumps(value).hex()
        elif self.serializer == "pickle":
            return pickle.dumps(value).hex()
        else:
            return str(value)
    
    def _deserialize(self, value: str) -> Any:
        """Deserialize value from storage."""
        if not value:
            return None
        
        if self.serializer == "json":
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # Try pickle fallback
                try:
                    return pickle.loads(bytes.fromhex(value))
                except Exception:
                    return value
        elif self.serializer == "pickle":
            try:
                return pickle.loads(bytes.fromhex(value))
            except Exception:
                return value
        else:
            return value
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        if not self.available:
            return None
        
        try:
            redis_key = self._make_key(key)
            value = self.redis_client.get(redis_key)
            
            if value is None:
                return None
            
            return self._deserialize(value)
            
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error during get: {e}")
            return None
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self.available:
            return False
        
        try:
            redis_key = self._make_key(key)
            serialized_value = self._serialize(value)
            
            if ttl is not None:
                return self.redis_client.setex(redis_key, ttl, serialized_value)
            else:
                return self.redis_client.set(redis_key, serialized_value)
                
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error during set: {e}")
            return False
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self.available:
            return False
        
        try:
            redis_key = self._make_key(key)
            result = self.redis_client.delete(redis_key)
            return result > 0
            
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error during delete: {e}")
            return False
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.available:
            return False
        
        try:
            redis_key = self._make_key(key)
            return self.redis_client.exists(redis_key) > 0
            
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error during exists: {e}")
            return False
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries with prefix."""
        if not self.available:
            return False
        
        try:
            pattern = f"{self.prefix}*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                return self.redis_client.delete(*keys) > 0
            
            return True
            
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error during clear: {e}")
            return False
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    def keys(self, pattern: str = "*") -> List[str]:
        """Get cache keys matching pattern."""
        if not self.available:
            return []
        
        try:
            redis_pattern = f"{self.prefix}{pattern}"
            keys = self.redis_client.keys(redis_pattern)
            
            # Remove prefix from keys
            prefix_len = len(self.prefix)
            return [key[prefix_len:] for key in keys]
            
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error during keys: {e}")
            return []
        except Exception as e:
            logger.error(f"Redis keys error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "backend": "redis",
            "available": self.available
        }
        
        if not self.available:
            return stats
        
        try:
            info = self.redis_client.info()
            
            stats.update({
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "expired_keys": info.get("expired_keys", 0),
                "evicted_keys": info.get("evicted_keys", 0)
            })
            
            # Calculate hit rate
            hits = stats.get("keyspace_hits", 0)
            misses = stats.get("keyspace_misses", 0)
            if hits + misses > 0:
                stats["hit_rate"] = hits / (hits + misses)
            else:
                stats["hit_rate"] = 0
            
            # Count keys with our prefix
            prefix_keys = self.redis_client.keys(f"{self.prefix}*")
            stats["prefix_key_count"] = len(prefix_keys)
            
        except Exception as e:
            stats["error"] = str(e)
            logger.error(f"Redis stats error: {e}")
        
        return stats
    
    def is_available(self) -> bool:
        """Check if Redis is available."""
        if not self.available:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False
    
    def flush_db(self) -> bool:
        """Flush entire Redis database (use with caution)."""
        if not self.available:
            return False
        
        try:
            self.redis_client.flushdb()
            logger.warning("Redis database flushed")
            return True
        except Exception as e:
            logger.error(f"Redis flush error: {e}")
            return False