"""Advanced caching system for MoE debugging with intelligent eviction and optimization."""

import time
import threading
import hashlib
import pickle
import json
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import statistics

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .logging_config import get_logger
from .validation import safe_json_dumps, safe_json_loads

logger = get_logger(__name__)


class CacheKey:
    """Smart cache key with automatic generation and validation."""
    
    def __init__(self, namespace: str, operation: str, params: Dict[str, Any]):
        self.namespace = namespace
        self.operation = operation 
        self.params = params
        self._key = None
        self._hash = None
    
    @property
    def key(self) -> str:
        """Generate cache key string."""
        if self._key is None:
            # Sort parameters for consistent key generation
            param_str = safe_json_dumps(sorted(self.params.items()))
            key_data = f"{self.namespace}:{self.operation}:{param_str}"
            self._key = hashlib.sha256(key_data.encode()).hexdigest()[:32]
        return self._key
    
    @property
    def hash(self) -> str:
        """Get hash of the key."""
        if self._hash is None:
            self._hash = hashlib.md5(self.key.encode()).hexdigest()
        return self._hash
    
    def __str__(self) -> str:
        return self.key
    
    def __eq__(self, other) -> bool:
        return isinstance(other, CacheKey) and self.key == other.key
    
    def __hash__(self) -> int:
        return hash(self.key)


class CacheEntry:
    """Cache entry with metadata and TTL support."""
    
    def __init__(self, value: Any, ttl: Optional[float] = None, 
                 size: Optional[int] = None, access_count: int = 0):
        self.value = value
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = access_count
        self.hit_count = 0
        self.ttl = ttl
        self.size = size or self._calculate_size(value)
        self.tags: set = set()
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate size of cached value."""
        try:
            return len(pickle.dumps(value))
        except:
            try:
                return len(safe_json_dumps(value))
            except:
                return len(str(value))
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1
        self.hit_count += 1
    
    def age(self) -> float:
        """Get age in seconds."""
        return time.time() - self.created_at
    
    def add_tag(self, tag: str):
        """Add tag to entry."""
        self.tags.add(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if entry has tag."""
        return tag in self.tags


class CacheEvictionPolicy(ABC):
    """Abstract base for cache eviction policies."""
    
    @abstractmethod
    def should_evict(self, entries: Dict[str, CacheEntry], max_size: int) -> List[str]:
        """Determine which keys should be evicted."""
        pass


class LRUEvictionPolicy(CacheEvictionPolicy):
    """Least Recently Used eviction policy."""
    
    def should_evict(self, entries: Dict[str, CacheEntry], max_size: int) -> List[str]:
        current_size = sum(entry.size for entry in entries.values())
        if current_size <= max_size:
            return []
        
        # Sort by last access time (oldest first)
        sorted_entries = sorted(
            entries.items(), 
            key=lambda x: x[1].last_accessed
        )
        
        keys_to_evict = []
        size_to_free = current_size - max_size
        freed_size = 0
        
        for key, entry in sorted_entries:
            keys_to_evict.append(key)
            freed_size += entry.size
            if freed_size >= size_to_free:
                break
        
        return keys_to_evict


class LFUEvictionPolicy(CacheEvictionPolicy):
    """Least Frequently Used eviction policy."""
    
    def should_evict(self, entries: Dict[str, CacheEntry], max_size: int) -> List[str]:
        current_size = sum(entry.size for entry in entries.values())
        if current_size <= max_size:
            return []
        
        # Sort by access frequency (least used first)
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: (x[1].access_count, x[1].last_accessed)
        )
        
        keys_to_evict = []
        size_to_free = current_size - max_size
        freed_size = 0
        
        for key, entry in sorted_entries:
            keys_to_evict.append(key)
            freed_size += entry.size
            if freed_size >= size_to_free:
                break
        
        return keys_to_evict


class AdaptiveEvictionPolicy(CacheEvictionPolicy):
    """Adaptive eviction policy that combines multiple strategies."""
    
    def __init__(self):
        self.lru = LRUEvictionPolicy()
        self.lfu = LFUEvictionPolicy()
        self.hit_rate_threshold = 0.1
    
    def should_evict(self, entries: Dict[str, CacheEntry], max_size: int) -> List[str]:
        current_size = sum(entry.size for entry in entries.values())
        if current_size <= max_size:
            return []
        
        # Calculate hit rates
        total_accesses = sum(entry.access_count for entry in entries.values())
        
        # Use LFU for low hit rate entries, LRU for others
        low_hit_entries = {}
        normal_entries = {}
        
        for key, entry in entries.items():
            hit_rate = entry.hit_count / max(entry.access_count, 1)
            if hit_rate < self.hit_rate_threshold:
                low_hit_entries[key] = entry
            else:
                normal_entries[key] = entry
        
        # Prioritize evicting low hit rate entries first
        keys_to_evict = []
        size_to_free = current_size - max_size
        freed_size = 0
        
        # Evict low hit rate entries with LFU
        if low_hit_entries:
            lfu_evictions = self.lfu.should_evict(low_hit_entries, 0)  # Evict all if needed
            for key in lfu_evictions:
                keys_to_evict.append(key)
                freed_size += entries[key].size
                if freed_size >= size_to_free:
                    return keys_to_evict
        
        # If still need space, evict normal entries with LRU
        if normal_entries and freed_size < size_to_free:
            remaining_size = max_size - freed_size
            lru_evictions = self.lru.should_evict(normal_entries, remaining_size)
            keys_to_evict.extend(lru_evictions)
        
        return keys_to_evict


class InMemoryCache:
    """High-performance in-memory cache with intelligent eviction."""
    
    def __init__(self, max_size: int = 100 * 1024 * 1024,  # 100MB
                 default_ttl: Optional[float] = 3600,  # 1 hour
                 eviction_policy: Optional[CacheEvictionPolicy] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy or AdaptiveEvictionPolicy()
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'writes': 0,
            'deletes': 0,
            'size_bytes': 0,
            'entry_count': 0
        }
        
        # Performance tracking
        self.operation_times = defaultdict(list)
        self._cleanup_thread = None
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_loop():
            while True:
                try:
                    self._cleanup_expired()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    time.sleep(60)
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _ensure_capacity(self):
        """Ensure cache doesn't exceed capacity."""
        keys_to_evict = self.eviction_policy.should_evict(self._cache, self.max_size)
        
        for key in keys_to_evict:
            self._remove_entry(key)
            self.stats['evictions'] += 1
        
        if keys_to_evict:
            logger.debug(f"Evicted {len(keys_to_evict)} cache entries")
    
    def _remove_entry(self, key: str):
        """Remove entry and update stats."""
        if key in self._cache:
            entry = self._cache[key]
            del self._cache[key]
            self.stats['size_bytes'] -= entry.size
            self.stats['entry_count'] -= 1
    
    def get(self, key: Union[str, CacheKey], default: Any = None) -> Any:
        """Get value from cache."""
        start_time = time.perf_counter()
        
        cache_key = key.key if isinstance(key, CacheKey) else str(key)
        
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                if entry.is_expired():
                    self._remove_entry(cache_key)
                    self.stats['misses'] += 1
                    result = default
                else:
                    entry.touch()
                    self.stats['hits'] += 1
                    result = entry.value
            else:
                self.stats['misses'] += 1
                result = default
        
        duration = time.perf_counter() - start_time
        self.operation_times['get'].append(duration)
        
        return result
    
    def set(self, key: Union[str, CacheKey], value: Any, 
            ttl: Optional[float] = None, tags: Optional[List[str]] = None) -> bool:
        """Set value in cache."""
        start_time = time.perf_counter()
        
        cache_key = key.key if isinstance(key, CacheKey) else str(key)
        ttl = ttl or self.default_ttl
        
        try:
            entry = CacheEntry(value, ttl)
            
            if tags:
                for tag in tags:
                    entry.add_tag(tag)
            
            with self._lock:
                # Remove existing entry if present
                if cache_key in self._cache:
                    self._remove_entry(cache_key)
                
                # Add new entry
                self._cache[cache_key] = entry
                self.stats['size_bytes'] += entry.size
                self.stats['entry_count'] += 1
                self.stats['writes'] += 1
                
                # Ensure we don't exceed capacity
                self._ensure_capacity()
            
            duration = time.perf_counter() - start_time
            self.operation_times['set'].append(duration)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {cache_key}: {e}")
            return False
    
    def delete(self, key: Union[str, CacheKey]) -> bool:
        """Delete value from cache."""
        cache_key = key.key if isinstance(key, CacheKey) else str(key)
        
        with self._lock:
            if cache_key in self._cache:
                self._remove_entry(cache_key)
                self.stats['deletes'] += 1
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self.stats['size_bytes'] = 0
            self.stats['entry_count'] = 0
    
    def clear_by_tag(self, tag: str) -> int:
        """Clear entries with specific tag."""
        with self._lock:
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if entry.has_tag(tag)
            ]
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_ops = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_ops if total_ops > 0 else 0
            
            stats = self.stats.copy()
            stats.update({
                'hit_rate': hit_rate,
                'miss_rate': 1 - hit_rate,
                'utilization': self.stats['size_bytes'] / self.max_size,
                'avg_entry_size': self.stats['size_bytes'] / max(self.stats['entry_count'], 1)
            })
            
            # Add operation timing stats
            for op, times in self.operation_times.items():
                if times:
                    stats[f'{op}_avg_time_ms'] = statistics.mean(times) * 1000
                    stats[f'{op}_p95_time_ms'] = statistics.quantiles(times, n=20)[18] * 1000 if len(times) > 20 else statistics.mean(times) * 1000
            
            return stats


class DistributedCache:
    """Redis-based distributed cache for scaling across instances."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0",
                 default_ttl: int = 3600, key_prefix: str = "moe_debug:"):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")
        
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'deletes': 0,
            'errors': 0
        }
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.key_prefix}{key}"
    
    def get(self, key: Union[str, CacheKey], default: Any = None) -> Any:
        """Get value from distributed cache."""
        cache_key = key.key if isinstance(key, CacheKey) else str(key)
        redis_key = self._make_key(cache_key)
        
        try:
            data = self.redis_client.get(redis_key)
            if data is not None:
                self.stats['hits'] += 1
                return pickle.loads(data)
            else:
                self.stats['misses'] += 1
                return default
        except Exception as e:
            logger.error(f"Redis get error for key {cache_key}: {e}")
            self.stats['errors'] += 1
            return default
    
    def set(self, key: Union[str, CacheKey], value: Any, 
            ttl: Optional[int] = None) -> bool:
        """Set value in distributed cache."""
        cache_key = key.key if isinstance(key, CacheKey) else str(key)
        redis_key = self._make_key(cache_key)
        ttl = ttl or self.default_ttl
        
        try:
            data = pickle.dumps(value)
            result = self.redis_client.setex(redis_key, ttl, data)
            self.stats['writes'] += 1
            return bool(result)
        except Exception as e:
            logger.error(f"Redis set error for key {cache_key}: {e}")
            self.stats['errors'] += 1
            return False
    
    def delete(self, key: Union[str, CacheKey]) -> bool:
        """Delete value from distributed cache."""
        cache_key = key.key if isinstance(key, CacheKey) else str(key)
        redis_key = self._make_key(cache_key)
        
        try:
            result = self.redis_client.delete(redis_key)
            self.stats['deletes'] += 1
            return bool(result)
        except Exception as e:
            logger.error(f"Redis delete error for key {cache_key}: {e}")
            self.stats['errors'] += 1
            return False
    
    def clear(self):
        """Clear all cache entries with prefix."""
        try:
            keys = self.redis_client.keys(f"{self.key_prefix}*")
            if keys:
                self.redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} cache entries")
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            self.stats['errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_ops = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_ops if total_ops > 0 else 0
        
        stats = self.stats.copy()
        stats.update({
            'hit_rate': hit_rate,
            'miss_rate': 1 - hit_rate
        })
        
        try:
            info = self.redis_client.info()
            stats.update({
                'redis_memory_used': info.get('used_memory', 0),
                'redis_connected_clients': info.get('connected_clients', 0),
                'redis_uptime_seconds': info.get('uptime_in_seconds', 0)
            })
        except Exception as e:
            logger.warning(f"Could not get Redis info: {e}")
        
        return stats


class TieredCache:
    """Multi-tiered cache with L1 (memory) and L2 (distributed) caches."""
    
    def __init__(self, l1_cache: InMemoryCache, l2_cache: Optional[DistributedCache] = None):
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache
        
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'total_misses': 0,
            'l1_to_l2_promotions': 0
        }
    
    def get(self, key: Union[str, CacheKey], default: Any = None) -> Any:
        """Get value from tiered cache."""
        # Try L1 first
        result = self.l1_cache.get(key, None)
        if result is not None:
            self.stats['l1_hits'] += 1
            return result
        
        # Try L2 if available
        if self.l2_cache:
            result = self.l2_cache.get(key, None)
            if result is not None:
                self.stats['l2_hits'] += 1
                # Promote to L1
                try:
                    self.l1_cache.set(key, result)
                    self.stats['l1_to_l2_promotions'] += 1
                except Exception as e:
                    logger.warning(f"Failed to promote cache entry to L1: {e}")
                return result
        
        self.stats['total_misses'] += 1
        return default
    
    def set(self, key: Union[str, CacheKey], value: Any, 
            ttl: Optional[float] = None) -> bool:
        """Set value in tiered cache."""
        # Set in L1
        l1_success = self.l1_cache.set(key, value, ttl)
        
        # Set in L2 if available
        l2_success = True
        if self.l2_cache:
            l2_ttl = int(ttl) if ttl else None
            l2_success = self.l2_cache.set(key, value, l2_ttl)
        
        return l1_success and l2_success
    
    def delete(self, key: Union[str, CacheKey]) -> bool:
        """Delete value from tiered cache."""
        l1_result = self.l1_cache.delete(key)
        l2_result = True
        
        if self.l2_cache:
            l2_result = self.l2_cache.delete(key)
        
        return l1_result or l2_result
    
    def clear(self):
        """Clear all caches."""
        self.l1_cache.clear()
        if self.l2_cache:
            self.l2_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self.stats.copy()
        
        total_hits = stats['l1_hits'] + stats['l2_hits']
        total_ops = total_hits + stats['total_misses']
        
        stats.update({
            'total_hit_rate': total_hits / total_ops if total_ops > 0 else 0,
            'l1_hit_rate': stats['l1_hits'] / total_ops if total_ops > 0 else 0,
            'l2_hit_rate': stats['l2_hits'] / total_ops if total_ops > 0 else 0,
            'l1_stats': self.l1_cache.get_stats(),
        })
        
        if self.l2_cache:
            stats['l2_stats'] = self.l2_cache.get_stats()
        
        return stats


def cached(cache: Union[InMemoryCache, TieredCache], ttl: Optional[float] = None,
           key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': sorted(kwargs.items()) if kwargs else {}
                }
                cache_key = CacheKey("function", func.__name__, key_data)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# Global cache instances
_global_cache: Optional[TieredCache] = None
_cache_lock = threading.Lock()


def get_global_cache() -> TieredCache:
    """Get or create global cache instance."""
    global _global_cache
    
    with _cache_lock:
        if _global_cache is None:
            # Create L1 cache
            l1_cache = InMemoryCache(
                max_size=50 * 1024 * 1024,  # 50MB
                default_ttl=1800  # 30 minutes
            )
            
            # Try to create L2 cache
            l2_cache = None
            try:
                import os
                redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
                l2_cache = DistributedCache(redis_url)
                logger.info("Initialized tiered cache with Redis L2")
            except Exception as e:
                logger.info(f"Using memory-only cache (Redis unavailable: {e})")
            
            _global_cache = TieredCache(l1_cache, l2_cache)
        
        return _global_cache