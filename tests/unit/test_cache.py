"""Unit tests for caching system."""

import pytest
import time
from unittest.mock import Mock, patch

from moe_debugger.cache.manager import CacheManager
from moe_debugger.cache.memory_cache import MemoryCache
from moe_debugger.cache.redis_cache import RedisCache


class TestMemoryCache:
    """Test memory cache implementation."""
    
    def test_initialization(self):
        """Test memory cache initialization."""
        cache = MemoryCache(max_size=100, default_ttl=300)
        
        assert cache.max_size == 100
        assert cache.default_ttl == 300
        assert cache.size() == 0
        assert cache.is_available() is True
    
    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = MemoryCache(max_size=10)
        
        # Test set and get
        assert cache.set("key1", "value1") is True
        assert cache.get("key1") == "value1"
        
        # Test non-existent key
        assert cache.get("nonexistent") is None
        
        # Test exists
        assert cache.exists("key1") is True
        assert cache.exists("nonexistent") is False
        
        # Test delete
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("nonexistent") is False
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = MemoryCache()
        
        # Set with short TTL
        cache.set("temp_key", "temp_value", ttl=1)
        assert cache.get("temp_key") == "temp_value"
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("temp_key") is None
        assert cache.exists("temp_key") is False
    
    def test_default_ttl(self):
        """Test default TTL behavior."""
        cache = MemoryCache(default_ttl=1)
        
        cache.set("key1", "value1")  # Should use default TTL
        assert cache.get("key1") == "value1"
        
        time.sleep(1.1)
        assert cache.get("key1") is None
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = MemoryCache(max_size=3)
        
        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        assert cache.size() == 3
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new key, should evict key2 (least recently used)
        cache.set("key4", "value4")
        assert cache.size() == 3
        assert cache.get("key1") == "value1"  # Should still exist
        assert cache.get("key2") is None      # Should be evicted
        assert cache.get("key3") == "value3"  # Should still exist
        assert cache.get("key4") == "value4"  # Should exist
    
    def test_clear(self):
        """Test cache clearing."""
        cache = MemoryCache()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert cache.size() == 2
        
        cache.clear()
        assert cache.size() == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_keys(self):
        """Test key listing."""
        cache = MemoryCache()
        
        cache.set("user:1", "data1")
        cache.set("user:2", "data2")
        cache.set("session:1", "session_data")
        
        all_keys = cache.keys()
        assert len(all_keys) == 3
        assert "user:1" in all_keys
        assert "user:2" in all_keys
        assert "session:1" in all_keys
        
        # Test pattern matching
        user_keys = cache.keys("user:*")
        assert len(user_keys) == 2
        assert "user:1" in user_keys
        assert "user:2" in user_keys
        assert "session:1" not in user_keys
    
    def test_stats(self):
        """Test cache statistics."""
        cache = MemoryCache(max_size=10, default_ttl=300)
        
        # Initial stats
        stats = cache.get_stats()
        assert stats["backend"] == "memory"
        assert stats["size"] == 0
        assert stats["max_size"] == 10
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0
        assert stats["default_ttl"] == 300
        
        # Add some data and check stats
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["sets"] == 1


class TestRedisCache:
    """Test Redis cache implementation."""
    
    @patch('moe_debugger.cache.redis_cache.redis')
    def test_initialization_success(self, mock_redis):
        """Test successful Redis initialization."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client
        
        cache = RedisCache("redis://localhost:6379")
        
        assert cache.available is True
        assert cache.redis_client == mock_client
        mock_client.ping.assert_called_once()
    
    @patch('moe_debugger.cache.redis_cache.redis')
    def test_initialization_failure(self, mock_redis):
        """Test Redis initialization failure."""
        mock_redis.from_url.side_effect = Exception("Connection failed")
        
        cache = RedisCache("redis://localhost:6379")
        
        assert cache.available is False
        assert cache.redis_client is not None  # Client is created but marked unavailable
    
    @patch('moe_debugger.cache.redis_cache.REDIS_AVAILABLE', False)
    def test_redis_not_available(self):
        """Test behavior when Redis is not installed."""
        cache = RedisCache()
        
        assert cache.available is False
    
    @patch('moe_debugger.cache.redis_cache.redis')
    def test_basic_operations(self, mock_redis):
        """Test basic Redis operations."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = '"test_value"'
        mock_client.set.return_value = True
        mock_client.setex.return_value = True
        mock_client.delete.return_value = 1
        mock_client.exists.return_value = 1
        mock_redis.from_url.return_value = mock_client
        
        cache = RedisCache(prefix="test:")
        
        # Test set and get
        assert cache.set("key1", "test_value") is True
        mock_client.set.assert_called_with("test:key1", '"test_value"')
        
        assert cache.get("key1") == "test_value"
        mock_client.get.assert_called_with("test:key1")
        
        # Test set with TTL
        assert cache.set("key2", "value2", ttl=300) is True
        mock_client.setex.assert_called_with("test:key2", 300, '"value2"')
        
        # Test delete
        assert cache.delete("key1") is True
        mock_client.delete.assert_called_with("test:key1")
        
        # Test exists
        assert cache.exists("key1") is True
        mock_client.exists.assert_called_with("test:key1")
    
    @patch('moe_debugger.cache.redis_cache.redis')
    def test_serialization(self, mock_redis):
        """Test data serialization."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client
        
        cache = RedisCache(serializer="json")
        
        # Test JSON serialization
        test_data = {"key": "value", "number": 42}
        serialized = cache._serialize(test_data)
        assert serialized == '{"key": "value", "number": 42}'
        
        deserialized = cache._deserialize(serialized)
        assert deserialized == test_data
    
    @patch('moe_debugger.cache.redis_cache.redis')
    def test_unavailable_operations(self, mock_redis):
        """Test operations when Redis is unavailable."""
        cache = RedisCache()
        cache.available = False
        
        assert cache.get("key") is None
        assert cache.set("key", "value") is False
        assert cache.delete("key") is False
        assert cache.exists("key") is False
        assert cache.clear() is False
        assert cache.keys() == []


class TestCacheManager:
    """Test cache manager."""
    
    def test_memory_cache_creation(self):
        """Test memory cache creation."""
        manager = CacheManager(cache_type="memory", max_size=100)
        
        assert isinstance(manager.cache, MemoryCache)
        assert manager.cache_type == "memory"
    
    @patch('moe_debugger.cache.manager.RedisCache')
    def test_redis_cache_creation(self, mock_redis_class):
        """Test Redis cache creation."""
        mock_cache = Mock()
        mock_cache.is_available.return_value = True
        mock_redis_class.return_value = mock_cache
        
        manager = CacheManager(cache_type="redis", redis_url="redis://localhost:6379")
        
        assert manager.cache == mock_cache
        assert manager.cache_type == "redis"
        mock_redis_class.assert_called_once()
    
    @patch.dict('os.environ', {'REDIS_URL': 'redis://test:6379'})
    @patch('moe_debugger.cache.manager.RedisCache')
    def test_auto_cache_with_redis_env(self, mock_redis_class):
        """Test auto cache selection with Redis URL in environment."""
        mock_cache = Mock()
        mock_cache.is_available.return_value = True
        mock_redis_class.return_value = mock_cache
        
        manager = CacheManager(cache_type="auto")
        
        assert manager.cache == mock_cache
        mock_redis_class.assert_called_once()
    
    @patch('moe_debugger.cache.manager.RedisCache')
    def test_auto_cache_fallback_to_memory(self, mock_redis_class):
        """Test auto cache fallback to memory when Redis fails."""
        mock_redis_class.side_effect = Exception("Redis not available")
        
        manager = CacheManager(cache_type="auto")
        
        assert isinstance(manager.cache, MemoryCache)
    
    def test_key_generation_string(self):
        """Test key generation from string."""
        manager = CacheManager(cache_type="memory")
        
        key = manager._generate_key("simple_key")
        assert key == "simple_key"
    
    def test_key_generation_dict(self):
        """Test key generation from dictionary."""
        manager = CacheManager(cache_type="memory")
        
        key_dict = {"session_id": "123", "type": "routing"}
        key = manager._generate_key(key_dict)
        
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length
        
        # Same dict should generate same key
        key2 = manager._generate_key(key_dict)
        assert key == key2
        
        # Different dict should generate different key
        different_dict = {"session_id": "456", "type": "routing"}
        key3 = manager._generate_key(different_dict)
        assert key != key3
    
    def test_basic_operations(self):
        """Test basic cache manager operations."""
        manager = CacheManager(cache_type="memory")
        
        # Test set and get
        assert manager.set("test_key", {"data": "test"}) is True
        result = manager.get("test_key")
        assert result == {"data": "test"}
        
        # Test default value
        assert manager.get("nonexistent", "default") == "default"
        
        # Test exists
        assert manager.exists("test_key") is True
        assert manager.exists("nonexistent") is False
        
        # Test delete
        assert manager.delete("test_key") is True
        assert manager.get("test_key") is None
    
    def test_specialized_cache_methods(self):
        """Test specialized caching methods."""
        manager = CacheManager(cache_type="memory")
        
        # Test routing stats caching
        stats = {"total_events": 100, "avg_confidence": 0.8}
        assert manager.cache_routing_stats("session_123", stats) is True
        
        cached_stats = manager.get_routing_stats("session_123")
        assert cached_stats == stats
        
        # Test expert metrics caching
        metrics = {"expert_0": 0.5, "expert_1": 0.7}
        assert manager.cache_expert_metrics("session_123", metrics) is True
        
        cached_metrics = manager.get_expert_metrics("session_123")
        assert cached_metrics == metrics
        
        # Test analysis result caching
        result = {"dead_experts": [2, 5]}
        assert manager.cache_analysis_result("dead_experts", "session_123", result) is True
        
        cached_result = manager.get_analysis_result("dead_experts", "session_123")
        assert cached_result == result
        
        # Test visualization data caching
        viz_data = {"chart_type": "heatmap", "data": [[0.1, 0.9]]}
        assert manager.cache_visualization_data("session_123", "routing_heatmap", viz_data) is True
        
        cached_viz = manager.get_visualization_data("session_123", "routing_heatmap")
        assert cached_viz == viz_data
    
    def test_session_cache_invalidation(self):
        """Test invalidating cache for a session."""
        manager = CacheManager(cache_type="memory")
        
        # Add some cached data for session
        manager.cache_routing_stats("session_123", {"data": "test"})
        manager.cache_expert_metrics("session_123", {"metrics": "test"})
        manager.cache_analysis_result("test_analysis", "session_123", {"result": "test"})
        
        # Verify data exists
        assert manager.get_routing_stats("session_123") is not None
        assert manager.get_expert_metrics("session_123") is not None
        assert manager.get_analysis_result("test_analysis", "session_123") is not None
        
        # Invalidate session cache
        assert manager.invalidate_session_cache("session_123") is True
        
        # Verify data is cleared
        assert manager.get_routing_stats("session_123") is None
        assert manager.get_expert_metrics("session_123") is None
        assert manager.get_analysis_result("test_analysis", "session_123") is None
    
    def test_error_handling(self):
        """Test error handling in cache operations."""
        manager = CacheManager(cache_type="memory")
        
        # Mock cache to raise exceptions
        manager.cache.get = Mock(side_effect=Exception("Cache error"))
        manager.cache.set = Mock(side_effect=Exception("Cache error"))
        manager.cache.delete = Mock(side_effect=Exception("Cache error"))
        manager.cache.exists = Mock(side_effect=Exception("Cache error"))
        
        # Operations should not raise exceptions
        assert manager.get("key", "default") == "default"
        assert manager.set("key", "value") is False
        assert manager.delete("key") is False
        assert manager.exists("key") is False
    
    def test_stats(self):
        """Test cache statistics."""
        manager = CacheManager(cache_type="memory")
        
        stats = manager.get_stats()
        
        assert isinstance(stats, dict)
        assert "cache_type" in stats
        assert stats["cache_type"] == "memory"
    
    def test_availability(self):
        """Test cache availability check."""
        manager = CacheManager(cache_type="memory")
        
        assert manager.is_available() is True