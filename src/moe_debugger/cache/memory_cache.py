"""In-memory cache implementation."""

import time
import threading
from typing import Any, Optional, Dict, List
from dataclasses import dataclass
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = 0


class MemoryCache:
    """Thread-safe in-memory cache with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 10000, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._deletes = 0
        self._evictions = 0
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self._cleanup_thread.start()
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            # Check expiration
            current_time = time.time()
            if entry.expires_at and current_time > entry.expires_at:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = current_time
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        with self._lock:
            current_time = time.time()
            
            # Calculate expiration
            expires_at = None
            if ttl is not None:
                expires_at = current_time + ttl
            elif self.default_ttl is not None:
                expires_at = current_time + self.default_ttl
            
            # Create entry
            entry = CacheEntry(
                value=value,
                created_at=current_time,
                expires_at=expires_at,
                last_accessed=current_time
            )
            
            # Add to cache
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            # Evict if over capacity
            self._evict_if_needed()
            
            self._sets += 1
            return True
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._deletes += 1
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            
            # Check expiration
            if entry.expires_at and time.time() > entry.expires_at:
                del self._cache[key]
                return False
            
            return True
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            return True
    
    def keys(self, pattern: str = "*") -> List[str]:
        """Get cache keys matching pattern."""
        with self._lock:
            if pattern == "*":
                return list(self._cache.keys())
            
            # Simple pattern matching
            import fnmatch
            return [key for key in self._cache.keys() if fnmatch.fnmatch(key, pattern)]
    
    def size(self) -> int:
        """Get number of entries in cache."""
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
            
            return {
                "backend": "memory",
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "sets": self._sets,
                "deletes": self._deletes,
                "evictions": self._evictions,
                "default_ttl": self.default_ttl
            }
    
    def _evict_if_needed(self):
        """Evict least recently used entries if over capacity."""
        while len(self._cache) > self.max_size:
            # Remove oldest entry (least recently used)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._evictions += 1
    
    def _cleanup_expired(self):
        """Background thread to clean up expired entries."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                
                current_time = time.time()
                expired_keys = []
                
                with self._lock:
                    for key, entry in self._cache.items():
                        if entry.expires_at and current_time > entry.expires_at:
                            expired_keys.append(key)
                
                # Delete expired entries
                with self._lock:
                    for key in expired_keys:
                        if key in self._cache:  # Double-check in case it was already deleted
                            del self._cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
            except Exception as e:
                logger.error(f"Error in cache cleanup thread: {e}")
    
    def is_available(self) -> bool:
        """Check if cache is available."""
        return True  # Memory cache is always available