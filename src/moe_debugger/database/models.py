"""SQLAlchemy database models for persistent storage."""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import relationship, Session
    from sqlalchemy.dialects.postgresql import UUID
    from sqlalchemy.types import TypeDecorator, TEXT
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    # Create dummy base for when SQLAlchemy is not available
    class declarative_base:
        pass
    Base = declarative_base()
    Session = None


if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    class JSONType(TypeDecorator):
        """Custom JSON type that works across different databases."""
        impl = TEXT
        cache_ok = True
        
        def process_bind_param(self, value, dialect):
            if value is not None:
                return json.dumps(value)
            return value
        
        def process_result_value(self, value, dialect):
            if value is not None:
                return json.loads(value)
            return value
    
    
    class DebugSessionDB(Base):
        """Database model for debugging sessions."""
        
        __tablename__ = "debug_sessions"
        
        id = Column(Integer, primary_key=True, index=True)
        session_id = Column(String(100), unique=True, index=True, nullable=False)
        model_name = Column(String(200), nullable=False)
        start_time = Column(DateTime, nullable=False)
        end_time = Column(DateTime, nullable=True)
        config = Column(JSONType, nullable=True)
        
        # Relationships
        routing_events = relationship("RoutingEventDB", back_populates="session", cascade="all, delete-orphan")
        expert_metrics = relationship("ExpertMetricsDB", back_populates="session", cascade="all, delete-orphan")
        
        # Metadata
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    
    class RoutingEventDB(Base):
        """Database model for routing events."""
        
        __tablename__ = "routing_events"
        
        id = Column(Integer, primary_key=True, index=True)
        session_id = Column(String(100), ForeignKey("debug_sessions.session_id"), index=True)
        sequence_id = Column(String(100), index=True)
        
        # Event data
        timestamp = Column(Float, nullable=False, index=True)
        layer_idx = Column(Integer, nullable=False)
        token_position = Column(Integer, nullable=False)
        token = Column(String(500), nullable=False)
        expert_weights = Column(JSONType, nullable=False)  # List[float]
        selected_experts = Column(JSONType, nullable=False)  # List[int]
        routing_confidence = Column(Float, nullable=False)
        
        # Relationships
        session = relationship("DebugSessionDB", back_populates="routing_events")
        
        # Metadata
        created_at = Column(DateTime, default=datetime.utcnow)
    
    
    class ExpertMetricsDB(Base):
        """Database model for expert performance metrics."""
        
        __tablename__ = "expert_metrics"
        
        id = Column(Integer, primary_key=True, index=True)
        session_id = Column(String(100), ForeignKey("debug_sessions.session_id"), index=True)
        
        # Expert identification
        expert_id = Column(Integer, nullable=False)
        layer_idx = Column(Integer, nullable=False)
        
        # Metrics
        utilization_rate = Column(Float, nullable=False)
        compute_time_ms = Column(Float, nullable=False)
        memory_usage_mb = Column(Float, nullable=False)
        parameter_count = Column(Integer, nullable=False)
        activation_count = Column(Integer, default=0)
        last_activated = Column(DateTime, nullable=True)
        
        # Additional metrics as JSON
        additional_metrics = Column(JSONType, nullable=True)
        
        # Relationships
        session = relationship("DebugSessionDB", back_populates="expert_metrics")
        
        # Metadata
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    
    class PerformanceProfileDB(Base):
        """Database model for performance profiles."""
        
        __tablename__ = "performance_profiles"
        
        id = Column(Integer, primary_key=True, index=True)
        session_id = Column(String(100), ForeignKey("debug_sessions.session_id"), index=True)
        
        # Performance metrics
        total_inference_time_ms = Column(Float, nullable=False)
        routing_overhead_ms = Column(Float, nullable=False)
        expert_compute_times = Column(JSONType, nullable=False)  # Dict[str, float]
        memory_peak_mb = Column(Float, nullable=False)
        cache_hit_rate = Column(Float, nullable=False)
        token_throughput = Column(Float, nullable=False)
        
        # Timestamp
        timestamp = Column(DateTime, nullable=False)
        created_at = Column(DateTime, default=datetime.utcnow)
    
    
    class AnalysisResultDB(Base):
        """Database model for analysis results."""
        
        __tablename__ = "analysis_results"
        
        id = Column(Integer, primary_key=True, index=True)
        session_id = Column(String(100), ForeignKey("debug_sessions.session_id"), index=True)
        
        # Analysis identification
        analysis_type = Column(String(100), nullable=False)  # load_balance, dead_experts, etc.
        
        # Results
        results = Column(JSONType, nullable=False)
        summary = Column(Text, nullable=True)
        
        # Diagnostics
        issues_found = Column(Integer, default=0)
        severity = Column(String(20), nullable=True)  # info, warning, error, critical
        
        # Metadata
        created_at = Column(DateTime, default=datetime.utcnow)
    
    
    class ModelArchitectureDB(Base):
        """Database model for model architecture information."""
        
        __tablename__ = "model_architectures"
        
        id = Column(Integer, primary_key=True, index=True)
        model_name = Column(String(200), unique=True, nullable=False)
        
        # Architecture details
        num_layers = Column(Integer, nullable=False)
        num_experts_per_layer = Column(Integer, nullable=False)
        hidden_size = Column(Integer, nullable=False)
        intermediate_size = Column(Integer, nullable=False)
        vocab_size = Column(Integer, nullable=False)
        max_sequence_length = Column(Integer, nullable=False)
        expert_capacity = Column(Float, nullable=False)
        router_type = Column(String(50), default="top_k")
        
        # Additional architecture details
        expert_types = Column(JSONType, nullable=True)
        additional_config = Column(JSONType, nullable=True)
        
        # Metadata
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    
    class CacheEntryDB(Base):
        """Database model for caching analysis results."""
        
        __tablename__ = "cache_entries"
        
        id = Column(Integer, primary_key=True, index=True)
        cache_key = Column(String(200), unique=True, nullable=False, index=True)
        
        # Cache data
        data = Column(JSONType, nullable=False)
        data_type = Column(String(100), nullable=False)  # routing_stats, analysis, etc.
        
        # Cache metadata
        expires_at = Column(DateTime, nullable=True)
        access_count = Column(Integer, default=0)
        last_accessed = Column(DateTime, default=datetime.utcnow)
        
        # Metadata
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

else:
    # Dummy models when SQLAlchemy is not available
    class DebugSessionDB:
        pass
    
    class RoutingEventDB:
        pass
    
    class ExpertMetricsDB:
        pass
    
    class PerformanceProfileDB:
        pass
    
    class AnalysisResultDB:
        pass
    
    class ModelArchitectureDB:
        pass
    
    class CacheEntryDB:
        pass