"""Intelligent multi-tier caching with predictive prefetching and smart eviction."""

import asyncio
import hashlib
import json
import time
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry in the intelligent cache."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    size_bytes: int = 0
    priority: int = 5  # 1-10, higher = more important
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def get_age(self) -> float:
        """Get age in seconds."""
        return time.time() - self.timestamp
    
    def get_access_frequency(self) -> float:
        """Calculate access frequency (accesses per second)."""
        age = self.get_age()
        return self.access_count / max(1, age)


class CacheEvictionStrategy(ABC):
    """Abstract base class for cache eviction strategies."""
    
    @abstractmethod
    def select_victims(self, entries: Dict[str, CacheEntry], 
                      required_space: int) -> List[str]:
        """Select entries to evict."""
        pass


class IntelligentLRUStrategy(CacheEvictionStrategy):
    """LRU with intelligence based on access patterns and priority."""
    
    def select_victims(self, entries: Dict[str, CacheEntry], 
                      required_space: int) -> List[str]:
        """Select victims using intelligent LRU algorithm."""
        # Sort by composite score (recency, frequency, priority)
        scored_entries = []
        
        current_time = time.time()
        for key, entry in entries.items():
            # Calculate composite score
            recency_score = 1.0 / max(1, current_time - entry.last_access)
            frequency_score = entry.get_access_frequency()
            priority_score = entry.priority / 10.0
            
            # Composite score with weights
            score = (recency_score * 0.4 + 
                    frequency_score * 0.4 + 
                    priority_score * 0.2)
            
            scored_entries.append((key, entry, score))
        
        # Sort by score (lowest first for eviction)
        scored_entries.sort(key=lambda x: x[2])
        
        # Select victims
        victims = []
        freed_space = 0
        
        for key, entry, score in scored_entries:
            if freed_space >= required_space:
                break
            victims.append(key)
            freed_space += entry.size_bytes
        
        return victims


class AdaptiveEvictionStrategy(CacheEvictionStrategy):
    """Adaptive strategy that learns from access patterns."""
    
    def __init__(self):
        self.access_pattern_weights = {
            'recency': 0.4,
            'frequency': 0.4,
            'priority': 0.2
        }
        self.pattern_history = []
        self.max_history = 1000
    
    def select_victims(self, entries: Dict[str, CacheEntry], 
                      required_space: int) -> List[str]:
        """Select victims using adaptive algorithm."""
        # Adapt weights based on historical performance
        self._adapt_weights()
        
        scored_entries = []
        current_time = time.time()
        
        for key, entry in entries.items():
            recency_score = 1.0 / max(1, current_time - entry.last_access)
            frequency_score = entry.get_access_frequency()
            priority_score = entry.priority / 10.0
            
            # Use adaptive weights
            score = (recency_score * self.access_pattern_weights['recency'] + 
                    frequency_score * self.access_pattern_weights['frequency'] + 
                    priority_score * self.access_pattern_weights['priority'])
            
            scored_entries.append((key, entry, score))
        
        scored_entries.sort(key=lambda x: x[2])
        
        victims = []
        freed_space = 0
        
        for key, entry, score in scored_entries:
            if freed_space >= required_space:
                break
            victims.append(key)
            freed_space += entry.size_bytes
        
        return victims
    
    def _adapt_weights(self):
        """Adapt weights based on access pattern history."""
        if len(self.pattern_history) < 100:
            return
        
        # Analyze recent patterns and adjust weights
        # This is a simplified implementation
        recent_patterns = self.pattern_history[-100:]
        
        # Calculate hit rates for different patterns
        recency_hits = sum(1 for p in recent_patterns if p.get('hit_type') == 'recency')
        frequency_hits = sum(1 for p in recent_patterns if p.get('hit_type') == 'frequency')
        priority_hits = sum(1 for p in recent_patterns if p.get('hit_type') == 'priority')
        
        total_hits = recency_hits + frequency_hits + priority_hits
        
        if total_hits > 0:
            self.access_pattern_weights['recency'] = recency_hits / total_hits
            self.access_pattern_weights['frequency'] = frequency_hits / total_hits
            self.access_pattern_weights['priority'] = priority_hits / total_hits


class PredictivePrefetcher:
    """Predictive prefetching based on access patterns."""
    
    def __init__(self, cache_instance: 'IntelligentCache'):
        self.cache = cache_instance
        self.access_sequences = defaultdict(list)
        self.pattern_graph = defaultdict(lambda: defaultdict(int))
        self.max_sequence_length = 10
        self.min_pattern_confidence = 0.3
        
    def record_access(self, key: str):
        """Record an access for pattern learning."""
        # Add to access sequences for each active session/context
        for session_id in self.access_sequences:
            sequence = self.access_sequences[session_id]
            sequence.append((key, time.time()))
            
            # Keep sequence length manageable
            if len(sequence) > self.max_sequence_length:
                sequence.pop(0)
            
            # Update pattern graph
            if len(sequence) >= 2:
                prev_key = sequence[-2][0]
                self.pattern_graph[prev_key][key] += 1
    
    def predict_next_keys(self, current_key: str, limit: int = 3) -> List[str]:
        """Predict next keys likely to be accessed."""
        if current_key not in self.pattern_graph:
            return []
        
        # Get patterns for current key
        patterns = self.pattern_graph[current_key]
        total_accesses = sum(patterns.values())
        
        if total_accesses == 0:
            return []
        
        # Calculate confidence scores
        predictions = []
        for next_key, count in patterns.items():
            confidence = count / total_accesses
            if confidence >= self.min_pattern_confidence:
                predictions.append((next_key, confidence))
        
        # Sort by confidence and return top predictions
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [key for key, confidence in predictions[:limit]]
    
    def prefetch(self, current_key: str, loader: Callable[[str], Any]):
        """Prefetch predicted keys."""
        predicted_keys = self.predict_next_keys(current_key)
        
        for key in predicted_keys:
            if not self.cache.contains(key):
                try:
                    # Load in background
                    threading.Thread(
                        target=self._background_load,
                        args=(key, loader),
                        daemon=True
                    ).start()
                except Exception as e:
                    logger.debug(f"Prefetch failed for {key}: {e}")
    
    def _background_load(self, key: str, loader: Callable[[str], Any]):
        """Load key in background."""
        try:
            value = loader(key)
            self.cache.put(key, value, priority=3)  # Lower priority for prefetched items
        except Exception as e:
            logger.debug(f"Background load failed for {key}: {e}")


class IntelligentCache:
    """High-performance intelligent cache with multi-tier storage."""
    
    def __init__(self,
                 max_memory_entries: int = 10000,
                 max_memory_size_mb: int = 256,
                 eviction_strategy: Optional[CacheEvictionStrategy] = None,
                 enable_prefetching: bool = True,
                 enable_compression: bool = True):
        
        self.max_memory_entries = max_memory_entries
        self.max_memory_size_bytes = max_memory_size_mb * 1024 * 1024
        self.enable_compression = enable_compression
        
        # Storage tiers
        self.memory_cache: Dict[str, CacheEntry] = {}
        
        # Eviction strategy
        self.eviction_strategy = eviction_strategy or IntelligentLRUStrategy()
        
        # Prefetching
        self.prefetcher = PredictivePrefetcher(self) if enable_prefetching else None
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'prefetch_hits': 0,
            'total_requests': 0,
            'total_size_bytes': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background maintenance
        self.maintenance_thread: Optional[threading.Thread] = None
        self.is_running = False
        
    def start(self):
        """Start background maintenance."""
        if self.is_running:
            return
        
        self.is_running = True
        self.maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True
        )
        self.maintenance_thread.start()
        logger.info("Intelligent cache started")
    
    def stop(self):
        """Stop background maintenance."""
        self.is_running = False
        if self.maintenance_thread:
            self.maintenance_thread.join(timeout=5)
        logger.info("Intelligent cache stopped")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with intelligent access tracking."""
        with self._lock:
            self.stats['total_requests'] += 1
            
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                
                # Check expiration
                if entry.is_expired():
                    del self.memory_cache[key]
                    self.stats['total_size_bytes'] -= entry.size_bytes
                    self.stats['misses'] += 1
                    return default
                
                # Update access statistics
                entry.last_access = time.time()
                entry.access_count += 1
                
                # Record for prefetching
                if self.prefetcher:
                    self.prefetcher.record_access(key)
                
                self.stats['hits'] += 1
                return entry.value
            
            self.stats['misses'] += 1
            return default
    
    def put(self, key: str, value: Any, 
           ttl: Optional[float] = None,
           priority: int = 5,
           tags: Optional[List[str]] = None) -> bool:
        """Put value in cache with intelligent placement."""
        with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Check if we need to evict
            if self._needs_eviction(size_bytes):
                if not self._evict_entries(size_bytes):
                    logger.warning(f"Could not make space for {key}")
                    return False
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=self._compress_if_enabled(value),
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=size_bytes,
                priority=priority,
                tags=tags or []
            )
            
            # Store in memory cache
            if key in self.memory_cache:
                old_entry = self.memory_cache[key]
                self.stats['total_size_bytes'] -= old_entry.size_bytes
            
            self.memory_cache[key] = entry
            self.stats['total_size_bytes'] += size_bytes
            
            return True
    
    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            return key in self.memory_cache and not self.memory_cache[key].is_expired()
    
    def remove(self, key: str) -> bool:
        """Remove key from cache."""
        with self._lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                del self.memory_cache[key]
                self.stats['total_size_bytes'] -= entry.size_bytes
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.memory_cache.clear()
            self.stats['total_size_bytes'] = 0
    
    def get_with_loader(self, key: str, loader: Callable[[], Any],
                       ttl: Optional[float] = None,
                       priority: int = 5) -> Any:
        """Get value with automatic loading if not present."""
        value = self.get(key)
        
        if value is None:
            value = loader()
            self.put(key, value, ttl=ttl, priority=priority)
            
            # Trigger prefetching
            if self.prefetcher:
                self.prefetcher.prefetch(key, lambda k: loader() if k == key else None)
        
        return value
    
    def invalidate_by_tags(self, tags: List[str]):
        """Invalidate all entries with specified tags."""
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self.memory_cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self.remove(key)
            
            logger.info(f"Invalidated {len(keys_to_remove)} entries by tags: {tags}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = (self.stats['hits'] / max(1, self.stats['total_requests'])) * 100
            
            return {
                'memory_cache_size': len(self.memory_cache),
                'memory_usage_bytes': self.stats['total_size_bytes'],
                'memory_usage_mb': self.stats['total_size_bytes'] / (1024 * 1024),
                'hit_rate_percent': hit_rate,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'prefetch_hits': self.stats.get('prefetch_hits', 0),
                'total_requests': self.stats['total_requests'],
                'average_entry_size': self.stats['total_size_bytes'] / max(1, len(self.memory_cache))
            }
    
    def _needs_eviction(self, new_entry_size: int) -> bool:
        """Check if eviction is needed."""
        return (len(self.memory_cache) >= self.max_memory_entries or
                self.stats['total_size_bytes'] + new_entry_size > self.max_memory_size_bytes)
    
    def _evict_entries(self, required_space: int) -> bool:
        """Evict entries to make space."""
        victims = self.eviction_strategy.select_victims(self.memory_cache, required_space)
        
        if not victims:
            return False
        
        for key in victims:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                del self.memory_cache[key]
                self.stats['total_size_bytes'] -= entry.size_bytes
                self.stats['evictions'] += 1
        
        logger.debug(f"Evicted {len(victims)} entries to free space")
        return True
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            if isinstance(value, (str, bytes)):
                return len(value.encode('utf-8') if isinstance(value, str) else value)
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple, dict)):
                return len(json.dumps(value, default=str).encode('utf-8'))
            else:
                return len(str(value).encode('utf-8'))
        except:
            return 1024  # Default estimate
    
    def _compress_if_enabled(self, value: Any) -> Any:
        """Compress value if compression is enabled."""
        if not self.enable_compression:
            return value
        
        # Simple compression for large strings/bytes
        if isinstance(value, str) and len(value) > 1024:
            try:
                import zlib
                compressed = zlib.compress(value.encode('utf-8'))
                return {'__compressed__': True, 'data': compressed}
            except:
                pass
        
        return value
    
    def _decompress_if_needed(self, value: Any) -> Any:
        """Decompress value if needed."""
        if isinstance(value, dict) and value.get('__compressed__'):
            try:
                import zlib
                return zlib.decompress(value['data']).decode('utf-8')
            except:
                pass
        
        return value
    
    def _maintenance_loop(self):
        """Background maintenance tasks."""
        while self.is_running:
            try:
                # Clean expired entries
                self._cleanup_expired_entries()
                
                # Optimize data structures
                self._optimize_storage()
                
                time.sleep(60)  # Run maintenance every minute
                
            except Exception as e:
                logger.error(f"Cache maintenance error: {e}")
                time.sleep(60)
    
    def _cleanup_expired_entries(self):
        """Remove expired entries."""
        with self._lock:
            expired_keys = []
            
            for key, entry in self.memory_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.remove(key)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
    
    def _optimize_storage(self):
        """Optimize internal storage structures."""
        # This could include compaction, defragmentation, etc.
        pass


# Global intelligent cache instance
_global_intelligent_cache: Optional[IntelligentCache] = None


def get_intelligent_cache() -> IntelligentCache:
    """Get global intelligent cache instance."""
    global _global_intelligent_cache
    if _global_intelligent_cache is None:
        _global_intelligent_cache = IntelligentCache(
            max_memory_entries=10000,
            max_memory_size_mb=512,
            eviction_strategy=AdaptiveEvictionStrategy(),
            enable_prefetching=True,
            enable_compression=True
        )
        _global_intelligent_cache.start()
    
    return _global_intelligent_cache