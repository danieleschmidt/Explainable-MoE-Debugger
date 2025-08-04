"""Repository pattern implementations for database operations."""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

try:
    from sqlalchemy.orm import Session
    from sqlalchemy import desc, and_, or_, func
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Session = None

from .models import (
    DebugSessionDB, RoutingEventDB, ExpertMetricsDB, 
    PerformanceProfileDB, AnalysisResultDB, CacheEntryDB
)
from ..models import RoutingEvent, ExpertMetrics, PerformanceProfile

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common database operations."""
    
    def __init__(self, session: Session):
        self.session = session
        self.enabled = SQLALCHEMY_AVAILABLE and session is not None
    
    def _check_enabled(self):
        """Check if database operations are enabled."""
        if not self.enabled:
            logger.warning("Database operations disabled - SQLAlchemy not available")
            return False
        return True


class SessionRepository(BaseRepository):
    """Repository for debug session operations."""
    
    def create_session(self, session_id: str, model_name: str, 
                      start_time: datetime, config: Dict[str, Any]) -> Optional[DebugSessionDB]:
        """Create a new debug session."""
        if not self._check_enabled():
            return None
        
        try:
            db_session = DebugSessionDB(
                session_id=session_id,
                model_name=model_name,
                start_time=start_time,
                config=config
            )
            
            self.session.add(db_session)
            self.session.commit()
            self.session.refresh(db_session)
            
            logger.info(f"Created debug session: {session_id}")
            return db_session
            
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            self.session.rollback()
            return None
    
    def end_session(self, session_id: str, end_time: datetime) -> bool:
        """Mark a session as ended."""
        if not self._check_enabled():
            return False
        
        try:
            session = self.session.query(DebugSessionDB).filter(
                DebugSessionDB.session_id == session_id
            ).first()
            
            if session:
                session.end_time = end_time
                self.session.commit()
                logger.info(f"Ended debug session: {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to end session {session_id}: {e}")
            self.session.rollback()
            return False
    
    def get_session(self, session_id: str) -> Optional[DebugSessionDB]:
        """Get a session by ID."""
        if not self._check_enabled():
            return None
        
        try:
            return self.session.query(DebugSessionDB).filter(
                DebugSessionDB.session_id == session_id
            ).first()
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    def list_sessions(self, limit: int = 100, active_only: bool = False) -> List[DebugSessionDB]:
        """List debug sessions."""
        if not self._check_enabled():
            return []
        
        try:
            query = self.session.query(DebugSessionDB)
            
            if active_only:
                query = query.filter(DebugSessionDB.end_time.is_(None))
            
            return query.order_by(desc(DebugSessionDB.start_time)).limit(limit).all()
            
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all related data."""
        if not self._check_enabled():
            return False
        
        try:
            session = self.session.query(DebugSessionDB).filter(
                DebugSessionDB.session_id == session_id
            ).first()
            
            if session:
                self.session.delete(session)
                self.session.commit()
                logger.info(f"Deleted debug session: {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            self.session.rollback()
            return False


class RoutingEventRepository(BaseRepository):
    """Repository for routing event operations."""
    
    def bulk_insert_events(self, events: List[RoutingEvent], session_id: str) -> bool:
        """Bulk insert routing events for efficiency."""
        if not self._check_enabled() or not events:
            return False
        
        try:
            db_events = []
            for event in events:
                db_event = RoutingEventDB(
                    session_id=session_id,
                    sequence_id=event.sequence_id,
                    timestamp=event.timestamp,
                    layer_idx=event.layer_idx,
                    token_position=event.token_position,
                    token=event.token,
                    expert_weights=event.expert_weights,
                    selected_experts=event.selected_experts,
                    routing_confidence=event.routing_confidence
                )
                db_events.append(db_event)
            
            self.session.bulk_save_objects(db_events)
            self.session.commit()
            
            logger.info(f"Inserted {len(events)} routing events for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert routing events: {e}")
            self.session.rollback()
            return False
    
    def get_events_by_session(self, session_id: str, limit: int = 1000, 
                             offset: int = 0) -> List[RoutingEventDB]:
        """Get routing events for a session."""
        if not self._check_enabled():
            return []
        
        try:
            return self.session.query(RoutingEventDB).filter(
                RoutingEventDB.session_id == session_id
            ).order_by(RoutingEventDB.timestamp).offset(offset).limit(limit).all()
            
        except Exception as e:
            logger.error(f"Failed to get events for session {session_id}: {e}")
            return []
    
    def get_events_by_layer(self, session_id: str, layer_idx: int) -> List[RoutingEventDB]:
        """Get routing events for a specific layer."""
        if not self._check_enabled():
            return []
        
        try:
            return self.session.query(RoutingEventDB).filter(
                and_(
                    RoutingEventDB.session_id == session_id,
                    RoutingEventDB.layer_idx == layer_idx
                )
            ).order_by(RoutingEventDB.timestamp).all()
            
        except Exception as e:
            logger.error(f"Failed to get events for layer {layer_idx}: {e}")
            return []
    
    def get_recent_events(self, session_id: str, minutes: int = 5) -> List[RoutingEventDB]:
        """Get recent routing events within specified time window."""
        if not self._check_enabled():
            return []
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
            cutoff_timestamp = cutoff_time.timestamp()
            
            return self.session.query(RoutingEventDB).filter(
                and_(
                    RoutingEventDB.session_id == session_id,
                    RoutingEventDB.timestamp >= cutoff_timestamp
                )
            ).order_by(RoutingEventDB.timestamp).all()
            
        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []
    
    def get_event_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics about routing events."""
        if not self._check_enabled():
            return {}
        
        try:
            stats = self.session.query(
                func.count(RoutingEventDB.id),
                func.min(RoutingEventDB.timestamp),
                func.max(RoutingEventDB.timestamp),
                func.count(func.distinct(RoutingEventDB.layer_idx)),
                func.count(func.distinct(RoutingEventDB.sequence_id))
            ).filter(RoutingEventDB.session_id == session_id).first()
            
            if stats:
                return {
                    "total_events": stats[0],
                    "first_event_time": stats[1],
                    "last_event_time": stats[2],
                    "unique_layers": stats[3],
                    "unique_sequences": stats[4]
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get event stats: {e}")
            return {}


class ExpertMetricsRepository(BaseRepository):
    """Repository for expert metrics operations."""
    
    def upsert_metrics(self, metrics: List[ExpertMetrics], session_id: str) -> bool:
        """Insert or update expert metrics."""
        if not self._check_enabled() or not metrics:
            return False
        
        try:
            for metric in metrics:
                # Check if metric exists
                existing = self.session.query(ExpertMetricsDB).filter(
                    and_(
                        ExpertMetricsDB.session_id == session_id,
                        ExpertMetricsDB.expert_id == metric.expert_id,
                        ExpertMetricsDB.layer_idx == metric.layer_idx
                    )
                ).first()
                
                if existing:
                    # Update existing metric
                    existing.utilization_rate = metric.utilization_rate
                    existing.compute_time_ms = metric.compute_time_ms
                    existing.memory_usage_mb = metric.memory_usage_mb
                    existing.activation_count = metric.activation_count
                    existing.last_activated = metric.last_activated
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new metric
                    db_metric = ExpertMetricsDB(
                        session_id=session_id,
                        expert_id=metric.expert_id,
                        layer_idx=metric.layer_idx,
                        utilization_rate=metric.utilization_rate,
                        compute_time_ms=metric.compute_time_ms,
                        memory_usage_mb=metric.memory_usage_mb,
                        parameter_count=metric.parameter_count,
                        activation_count=metric.activation_count,
                        last_activated=metric.last_activated
                    )
                    self.session.add(db_metric)
            
            self.session.commit()
            logger.info(f"Updated {len(metrics)} expert metrics for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert expert metrics: {e}")
            self.session.rollback()
            return False
    
    def get_metrics_by_session(self, session_id: str) -> List[ExpertMetricsDB]:
        """Get expert metrics for a session."""
        if not self._check_enabled():
            return []
        
        try:
            return self.session.query(ExpertMetricsDB).filter(
                ExpertMetricsDB.session_id == session_id
            ).order_by(ExpertMetricsDB.layer_idx, ExpertMetricsDB.expert_id).all()
            
        except Exception as e:
            logger.error(f"Failed to get metrics for session {session_id}: {e}")
            return []
    
    def get_expert_utilization(self, session_id: str) -> Dict[str, float]:
        """Get utilization rates for all experts."""
        if not self._check_enabled():
            return {}
        
        try:
            metrics = self.session.query(ExpertMetricsDB).filter(
                ExpertMetricsDB.session_id == session_id
            ).all()
            
            utilization = {}
            for metric in metrics:
                key = f"layer_{metric.layer_idx}_expert_{metric.expert_id}"
                utilization[key] = metric.utilization_rate
            
            return utilization
            
        except Exception as e:
            logger.error(f"Failed to get expert utilization: {e}")
            return {}


class CacheRepository(BaseRepository):
    """Repository for caching operations."""
    
    def get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get cached data by key."""
        if not self._check_enabled():
            return None
        
        try:
            cache_entry = self.session.query(CacheEntryDB).filter(
                CacheEntryDB.cache_key == cache_key
            ).first()
            
            if cache_entry:
                # Check if expired
                if cache_entry.expires_at and cache_entry.expires_at < datetime.utcnow():
                    self.session.delete(cache_entry)
                    self.session.commit()
                    return None
                
                # Update access statistics
                cache_entry.access_count += 1
                cache_entry.last_accessed = datetime.utcnow()
                self.session.commit()
                
                return cache_entry.data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached data for key {cache_key}: {e}")
            return None
    
    def set_cached_data(self, cache_key: str, data: Any, data_type: str, 
                       expires_at: Optional[datetime] = None) -> bool:
        """Set cached data."""
        if not self._check_enabled():
            return False
        
        try:
            # Check if entry exists
            existing = self.session.query(CacheEntryDB).filter(
                CacheEntryDB.cache_key == cache_key
            ).first()
            
            if existing:
                existing.data = data
                existing.data_type = data_type
                existing.expires_at = expires_at
                existing.updated_at = datetime.utcnow()
            else:
                cache_entry = CacheEntryDB(
                    cache_key=cache_key,
                    data=data,
                    data_type=data_type,
                    expires_at=expires_at
                )
                self.session.add(cache_entry)
            
            self.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cached data for key {cache_key}: {e}")
            self.session.rollback()
            return False
    
    def clear_expired_cache(self) -> int:
        """Clear expired cache entries."""
        if not self._check_enabled():
            return 0
        
        try:
            deleted = self.session.query(CacheEntryDB).filter(
                and_(
                    CacheEntryDB.expires_at.isnot(None),
                    CacheEntryDB.expires_at < datetime.utcnow()
                )
            ).delete()
            
            self.session.commit()
            logger.info(f"Cleared {deleted} expired cache entries")
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to clear expired cache: {e}")
            self.session.rollback()
            return 0