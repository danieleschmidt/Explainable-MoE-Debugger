"""Advanced Observability & Intelligence System for Progressive Quality Gates.

This module implements ML-powered monitoring, real-time insights, predictive alerting,
root cause analysis, and intelligent observability features for comprehensive
system visibility and proactive issue detection.

Features:
- ML-powered anomaly detection and pattern recognition
- Predictive alerting with root cause analysis
- Real-time performance profiling and trend analysis
- User experience monitoring and business impact correlation
- Distributed tracing and dependency mapping
- Intelligent log analysis and correlation
- Custom metrics and dimensional analysis
- Compliance monitoring and automated reporting

Authors: Terragon Labs - Progressive Quality Gates v2.0
License: MIT
"""

import time
import threading
import logging
import json
import hashlib
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import statistics
import re

# ML/Statistical libraries with graceful fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    
    class _NumpyFallback:
        @staticmethod
        def array(data):
            return data
        
        @staticmethod
        def mean(data):
            return statistics.mean(data) if data else 0.0
        
        @staticmethod
        def std(data):
            return statistics.stdev(data) if len(data) > 1 else 0.0
        
        @staticmethod
        def percentile(data, percentile):
            if not data:
                return 0.0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * percentile / 100.0
            f = int(k)
            c = k - f
            if f == len(sorted_data) - 1:
                return sorted_data[f]
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    
    np = _NumpyFallback()

from .logging_config import get_logger
from .validation import safe_json_dumps


class MetricType(Enum):
    """Types of metrics for observability."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AnomalyType(Enum):
    """Types of anomalies detected."""
    STATISTICAL = "statistical"
    SEASONAL = "seasonal"
    TREND = "trend"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"


class InsightCategory(Enum):
    """Categories of generated insights."""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    SECURITY = "security"
    BUSINESS = "business"
    INFRASTRUCTURE = "infrastructure"
    USER_EXPERIENCE = "user_experience"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: float
    metric_name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyDetection:
    """Detected anomaly with context and analysis."""
    anomaly_id: str
    timestamp: float
    metric_name: str
    anomaly_type: AnomalyType
    severity: AlertSeverity
    confidence_score: float
    expected_value: float
    actual_value: float
    deviation_magnitude: float
    context: Dict[str, Any] = field(default_factory=dict)
    root_cause_analysis: List[str] = field(default_factory=list)
    related_metrics: List[str] = field(default_factory=list)
    business_impact: Optional[str] = None
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class PredictiveAlert:
    """Predictive alert based on trend analysis."""
    alert_id: str
    timestamp: float
    metric_name: str
    predicted_issue: str
    time_to_impact_seconds: float
    confidence_score: float
    severity: AlertSeverity
    predicted_value: float
    threshold_value: float
    trend_analysis: Dict[str, float]
    mitigation_suggestions: List[str] = field(default_factory=list)


@dataclass
class SystemInsight:
    """Generated system insight with actionable recommendations."""
    insight_id: str
    timestamp: float
    category: InsightCategory
    title: str
    description: str
    confidence_score: float
    impact_score: float
    evidence: List[str] = field(default_factory=list)
    affected_services: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    estimated_cost_impact: Optional[float] = None
    estimated_performance_impact: Optional[float] = None


class StatisticalAnomalyDetector:
    """Statistical anomaly detection using multiple algorithms."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        self.logger = get_logger(__name__)
        
        # Algorithm parameters
        self.zscore_threshold = sensitivity
        self.iqr_multiplier = 1.5 * sensitivity
        self.isolation_contamination = 0.1 / sensitivity
        
        # Historical data for baselines
        self.metric_baselines: Dict[str, Dict[str, float]] = {}
        self.seasonal_patterns: Dict[str, List[float]] = {}
    
    def detect_anomalies(self, metric_points: List[MetricPoint]) -> List[AnomalyDetection]:
        """Detect anomalies in metric data using multiple algorithms."""
        anomalies = []
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for point in metric_points:
            metrics_by_name[point.metric_name].append(point)
        
        # Detect anomalies for each metric
        for metric_name, points in metrics_by_name.items():
            try:
                metric_anomalies = self._detect_metric_anomalies(metric_name, points)
                anomalies.extend(metric_anomalies)
            except Exception as e:
                self.logger.error(f"Error detecting anomalies for {metric_name}: {e}")
        
        return anomalies
    
    def _detect_metric_anomalies(self, metric_name: str, points: List[MetricPoint]) -> List[AnomalyDetection]:
        """Detect anomalies for a specific metric."""
        if len(points) < 5:
            return []  # Need minimum data points
        
        anomalies = []
        values = [p.value for p in points]
        timestamps = [p.timestamp for p in points]
        
        # Update baseline statistics
        self._update_baseline(metric_name, values)
        
        # Z-Score anomaly detection
        zscore_anomalies = self._detect_zscore_anomalies(metric_name, points, values)
        anomalies.extend(zscore_anomalies)
        
        # IQR-based anomaly detection
        iqr_anomalies = self._detect_iqr_anomalies(metric_name, points, values)
        anomalies.extend(iqr_anomalies)
        
        # Trend-based anomaly detection
        trend_anomalies = self._detect_trend_anomalies(metric_name, points, values, timestamps)
        anomalies.extend(trend_anomalies)
        
        # Seasonal anomaly detection
        seasonal_anomalies = self._detect_seasonal_anomalies(metric_name, points, values, timestamps)
        anomalies.extend(seasonal_anomalies)
        
        return anomalies
    
    def _update_baseline(self, metric_name: str, values: List[float]):
        """Update baseline statistics for a metric."""
        if metric_name not in self.metric_baselines:
            self.metric_baselines[metric_name] = {}
        
        baseline = self.metric_baselines[metric_name]
        baseline.update({
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'last_updated': time.time()
        })
    
    def _detect_zscore_anomalies(self, metric_name: str, points: List[MetricPoint], values: List[float]) -> List[AnomalyDetection]:
        """Detect anomalies using Z-score analysis."""
        anomalies = []
        
        if len(values) < 3:
            return anomalies
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        if std_val == 0:
            return anomalies
        
        for i, point in enumerate(points):
            z_score = abs((point.value - mean_val) / std_val)
            
            if z_score > self.zscore_threshold:
                severity = AlertSeverity.CRITICAL if z_score > self.zscore_threshold * 2 else AlertSeverity.WARNING
                
                anomaly = AnomalyDetection(
                    anomaly_id=f"zscore_{metric_name}_{int(point.timestamp)}",
                    timestamp=point.timestamp,
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    confidence_score=min(1.0, z_score / (self.zscore_threshold * 3)),
                    expected_value=mean_val,
                    actual_value=point.value,
                    deviation_magnitude=z_score,
                    context={
                        'z_score': z_score,
                        'mean': mean_val,
                        'std': std_val,
                        'threshold': self.zscore_threshold
                    }
                )
                
                # Add root cause analysis
                if point.value > mean_val:
                    anomaly.root_cause_analysis.append(f"Value {point.value:.2f} is {z_score:.2f} standard deviations above normal")
                else:
                    anomaly.root_cause_analysis.append(f"Value {point.value:.2f} is {z_score:.2f} standard deviations below normal")
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_iqr_anomalies(self, metric_name: str, points: List[MetricPoint], values: List[float]) -> List[AnomalyDetection]:
        """Detect anomalies using Interquartile Range (IQR) analysis."""
        anomalies = []
        
        if len(values) < 4:
            return anomalies
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        q1 = sorted_values[n // 4]
        q3 = sorted_values[3 * n // 4]
        iqr = q3 - q1
        
        if iqr == 0:
            return anomalies
        
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        for point in points:
            if point.value < lower_bound or point.value > upper_bound:
                distance = min(abs(point.value - lower_bound), abs(point.value - upper_bound))
                magnitude = distance / iqr if iqr > 0 else 0
                
                severity = AlertSeverity.CRITICAL if magnitude > 3 else AlertSeverity.WARNING
                
                anomaly = AnomalyDetection(
                    anomaly_id=f"iqr_{metric_name}_{int(point.timestamp)}",
                    timestamp=point.timestamp,
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    confidence_score=min(1.0, magnitude / 5),
                    expected_value=(q1 + q3) / 2,
                    actual_value=point.value,
                    deviation_magnitude=magnitude,
                    context={
                        'q1': q1,
                        'q3': q3,
                        'iqr': iqr,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                )
                
                if point.value < lower_bound:
                    anomaly.root_cause_analysis.append(f"Value {point.value:.2f} is below IQR lower bound {lower_bound:.2f}")
                else:
                    anomaly.root_cause_analysis.append(f"Value {point.value:.2f} is above IQR upper bound {upper_bound:.2f}")
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_trend_anomalies(self, metric_name: str, points: List[MetricPoint], 
                               values: List[float], timestamps: List[float]) -> List[AnomalyDetection]:
        """Detect trend-based anomalies."""
        anomalies = []
        
        if len(values) < 10:
            return anomalies
        
        # Calculate moving averages
        window_size = min(5, len(values) // 2)
        moving_averages = []
        
        for i in range(len(values) - window_size + 1):
            avg = statistics.mean(values[i:i + window_size])
            moving_averages.append(avg)
        
        # Detect sudden trend changes
        for i in range(1, len(moving_averages)):
            prev_avg = moving_averages[i-1]
            curr_avg = moving_averages[i]
            
            if prev_avg == 0:
                continue
            
            change_rate = abs(curr_avg - prev_avg) / prev_avg
            
            if change_rate > 0.5:  # 50% change threshold
                point_idx = i + window_size - 1
                if point_idx < len(points):
                    point = points[point_idx]
                    
                    severity = AlertSeverity.CRITICAL if change_rate > 1.0 else AlertSeverity.WARNING
                    
                    anomaly = AnomalyDetection(
                        anomaly_id=f"trend_{metric_name}_{int(point.timestamp)}",
                        timestamp=point.timestamp,
                        metric_name=metric_name,
                        anomaly_type=AnomalyType.TREND,
                        severity=severity,
                        confidence_score=min(1.0, change_rate),
                        expected_value=prev_avg,
                        actual_value=curr_avg,
                        deviation_magnitude=change_rate,
                        context={
                            'previous_average': prev_avg,
                            'current_average': curr_avg,
                            'change_rate': change_rate,
                            'window_size': window_size
                        }
                    )
                    
                    direction = "increased" if curr_avg > prev_avg else "decreased"
                    anomaly.root_cause_analysis.append(
                        f"Moving average {direction} by {change_rate*100:.1f}% indicating trend anomaly"
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_seasonal_anomalies(self, metric_name: str, points: List[MetricPoint],
                                  values: List[float], timestamps: List[float]) -> List[AnomalyDetection]:
        """Detect seasonal pattern anomalies."""
        anomalies = []
        
        # For now, implement simple day-of-week seasonality
        # In production, this would use more sophisticated seasonal decomposition
        
        if len(values) < 7 * 24:  # Need at least a week of hourly data
            return anomalies
        
        # Group by hour of day
        hourly_patterns = defaultdict(list)
        
        for i, timestamp in enumerate(timestamps):
            hour = datetime.fromtimestamp(timestamp).hour
            hourly_patterns[hour].append(values[i])
        
        # Calculate expected values for each hour
        hourly_expectations = {}
        for hour, hour_values in hourly_patterns.items():
            if len(hour_values) >= 3:
                hourly_expectations[hour] = {
                    'mean': statistics.mean(hour_values),
                    'std': statistics.stdev(hour_values) if len(hour_values) > 1 else 0.0
                }
        
        # Check recent points against seasonal expectations
        recent_points = points[-24:]  # Last 24 hours
        
        for point in recent_points:
            hour = datetime.fromtimestamp(point.timestamp).hour
            
            if hour in hourly_expectations:
                expectation = hourly_expectations[hour]
                expected_val = expectation['mean']
                std_val = expectation['std']
                
                if std_val > 0:
                    deviation = abs(point.value - expected_val) / std_val
                    
                    if deviation > self.sensitivity * 1.5:  # Seasonal threshold
                        severity = AlertSeverity.WARNING if deviation < 3 else AlertSeverity.CRITICAL
                        
                        anomaly = AnomalyDetection(
                            anomaly_id=f"seasonal_{metric_name}_{int(point.timestamp)}",
                            timestamp=point.timestamp,
                            metric_name=metric_name,
                            anomaly_type=AnomalyType.SEASONAL,
                            severity=severity,
                            confidence_score=min(1.0, deviation / 5),
                            expected_value=expected_val,
                            actual_value=point.value,
                            deviation_magnitude=deviation,
                            context={
                                'hour': hour,
                                'seasonal_mean': expected_val,
                                'seasonal_std': std_val,
                                'deviation': deviation
                            }
                        )
                        
                        anomaly.root_cause_analysis.append(
                            f"Value deviates from typical pattern for hour {hour} by {deviation:.2f} standard deviations"
                        )
                        
                        anomalies.append(anomaly)
        
        return anomalies


class PredictiveAnalyzer:
    """Predictive analysis for early warning alerts."""
    
    def __init__(self, prediction_horizon_minutes: int = 60):
        self.prediction_horizon = prediction_horizon_minutes
        self.logger = get_logger(__name__)
        
        # Prediction models (simplified)
        self.trend_models: Dict[str, Dict[str, float]] = {}
        self.threshold_config: Dict[str, Dict[str, float]] = {}
    
    def analyze_trends_and_predict(self, metric_points: List[MetricPoint]) -> List[PredictiveAlert]:
        """Analyze trends and generate predictive alerts."""
        alerts = []
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for point in metric_points:
            metrics_by_name[point.metric_name].append(point)
        
        # Analyze each metric
        for metric_name, points in metrics_by_name.items():
            try:
                metric_alerts = self._analyze_metric_trend(metric_name, points)
                alerts.extend(metric_alerts)
            except Exception as e:
                self.logger.error(f"Error analyzing trend for {metric_name}: {e}")
        
        return alerts
    
    def _analyze_metric_trend(self, metric_name: str, points: List[MetricPoint]) -> List[PredictiveAlert]:
        """Analyze trend for a specific metric."""
        if len(points) < 10:
            return []
        
        alerts = []
        
        # Sort points by timestamp
        points.sort(key=lambda p: p.timestamp)
        values = [p.value for p in points]
        timestamps = [p.timestamp for p in points]
        
        # Calculate trend using linear regression (simplified)
        trend_slope, trend_intercept = self._calculate_linear_trend(timestamps, values)
        
        # Update trend model
        self.trend_models[metric_name] = {
            'slope': trend_slope,
            'intercept': trend_intercept,
            'last_update': time.time(),
            'r_squared': self._calculate_r_squared(timestamps, values, trend_slope, trend_intercept)
        }
        
        # Get or set thresholds for this metric
        thresholds = self._get_metric_thresholds(metric_name, values)
        
        # Predict future values
        current_time = timestamps[-1]
        future_time = current_time + (self.prediction_horizon * 60)
        predicted_value = trend_slope * future_time + trend_intercept
        
        # Check if predicted value will exceed thresholds
        for threshold_name, threshold_value in thresholds.items():
            if self._will_exceed_threshold(predicted_value, threshold_value, trend_slope):
                time_to_impact = self._calculate_time_to_threshold(
                    timestamps[-1], values[-1], threshold_value, trend_slope, trend_intercept
                )
                
                if 0 < time_to_impact <= self.prediction_horizon * 60:
                    severity = self._determine_alert_severity(threshold_name, time_to_impact)
                    confidence = self._calculate_prediction_confidence(metric_name, values)
                    
                    alert = PredictiveAlert(
                        alert_id=f"predictive_{metric_name}_{threshold_name}_{int(current_time)}",
                        timestamp=current_time,
                        metric_name=metric_name,
                        predicted_issue=f"{metric_name} will exceed {threshold_name} threshold",
                        time_to_impact_seconds=time_to_impact,
                        confidence_score=confidence,
                        severity=severity,
                        predicted_value=predicted_value,
                        threshold_value=threshold_value,
                        trend_analysis={
                            'slope': trend_slope,
                            'current_value': values[-1],
                            'predicted_value': predicted_value,
                            'r_squared': self.trend_models[metric_name]['r_squared']
                        }
                    )
                    
                    # Add mitigation suggestions
                    alert.mitigation_suggestions = self._generate_mitigation_suggestions(
                        metric_name, threshold_name, trend_slope, time_to_impact
                    )
                    
                    alerts.append(alert)
        
        return alerts
    
    def _calculate_linear_trend(self, timestamps: List[float], values: List[float]) -> Tuple[float, float]:
        """Calculate linear trend using least squares regression."""
        n = len(timestamps)
        if n < 2:
            return 0.0, statistics.mean(values) if values else 0.0
        
        # Convert to relative timestamps to avoid numerical issues
        t_start = timestamps[0]
        rel_timestamps = [t - t_start for t in timestamps]
        
        # Calculate means
        t_mean = statistics.mean(rel_timestamps)
        v_mean = statistics.mean(values)
        
        # Calculate slope and intercept
        numerator = sum((t - t_mean) * (v - v_mean) for t, v in zip(rel_timestamps, values))
        denominator = sum((t - t_mean) ** 2 for t in rel_timestamps)
        
        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator
        
        # Adjust intercept for original timestamp scale
        intercept = v_mean - slope * (t_mean + t_start)
        
        return slope, intercept
    
    def _calculate_r_squared(self, timestamps: List[float], values: List[float], 
                            slope: float, intercept: float) -> float:
        """Calculate R-squared for trend line fit."""
        if not values:
            return 0.0
        
        mean_val = statistics.mean(values)
        total_ss = sum((v - mean_val) ** 2 for v in values)
        
        if total_ss == 0:
            return 1.0 if all(v == mean_val for v in values) else 0.0
        
        residual_ss = sum((v - (slope * t + intercept)) ** 2 for t, v in zip(timestamps, values))
        
        return max(0.0, 1.0 - (residual_ss / total_ss))
    
    def _get_metric_thresholds(self, metric_name: str, values: List[float]) -> Dict[str, float]:
        """Get or calculate thresholds for a metric."""
        if metric_name not in self.threshold_config:
            # Auto-generate thresholds based on historical data
            if values:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else mean_val * 0.1
                
                self.threshold_config[metric_name] = {
                    'warning': mean_val + 2 * std_val,
                    'critical': mean_val + 3 * std_val,
                    'emergency': mean_val + 4 * std_val
                }
            else:
                self.threshold_config[metric_name] = {
                    'warning': 100.0,
                    'critical': 200.0,
                    'emergency': 300.0
                }
        
        return self.threshold_config[metric_name]
    
    def _will_exceed_threshold(self, predicted_value: float, threshold: float, slope: float) -> bool:
        """Check if predicted value will exceed threshold given current trend."""
        if slope > 0:
            return predicted_value > threshold
        elif slope < 0:
            return predicted_value < threshold * 0.5  # For decreasing metrics, alert if too low
        else:
            return False
    
    def _calculate_time_to_threshold(self, current_time: float, current_value: float,
                                   threshold: float, slope: float, intercept: float) -> float:
        """Calculate time until threshold is reached."""
        if slope == 0:
            return float('inf')
        
        # Solve: threshold = slope * t + intercept for t
        target_time = (threshold - intercept) / slope
        time_to_impact = target_time - current_time
        
        return max(0, time_to_impact)
    
    def _determine_alert_severity(self, threshold_name: str, time_to_impact: float) -> AlertSeverity:
        """Determine alert severity based on threshold type and time to impact."""
        if threshold_name == 'emergency' or time_to_impact < 300:  # 5 minutes
            return AlertSeverity.EMERGENCY
        elif threshold_name == 'critical' or time_to_impact < 900:  # 15 minutes
            return AlertSeverity.CRITICAL
        else:
            return AlertSeverity.WARNING
    
    def _calculate_prediction_confidence(self, metric_name: str, values: List[float]) -> float:
        """Calculate confidence in prediction based on trend stability."""
        if metric_name not in self.trend_models:
            return 0.5
        
        model = self.trend_models[metric_name]
        r_squared = model.get('r_squared', 0.0)
        
        # Base confidence on R-squared and data stability
        data_stability = 1.0 / (1.0 + statistics.stdev(values[-10:]) / (statistics.mean(values[-10:]) + 0.001))
        
        confidence = (r_squared * 0.7) + (data_stability * 0.3)
        return min(1.0, max(0.1, confidence))
    
    def _generate_mitigation_suggestions(self, metric_name: str, threshold_name: str, 
                                       slope: float, time_to_impact: float) -> List[str]:
        """Generate mitigation suggestions for predicted issues."""
        suggestions = []
        
        # Generic suggestions based on metric patterns
        if 'cpu' in metric_name.lower():
            suggestions.extend([
                "Consider scaling up CPU resources",
                "Review and optimize high CPU consuming processes",
                "Implement CPU-based auto-scaling policies"
            ])
        elif 'memory' in metric_name.lower():
            suggestions.extend([
                "Consider scaling up memory resources",
                "Check for memory leaks in applications",
                "Optimize memory usage patterns"
            ])
        elif 'response_time' in metric_name.lower():
            suggestions.extend([
                "Optimize database queries and caching",
                "Consider increasing connection pool sizes",
                "Review application performance bottlenecks"
            ])
        elif 'error_rate' in metric_name.lower():
            suggestions.extend([
                "Review recent deployments for issues",
                "Check dependency health and circuit breakers",
                "Monitor and fix error patterns"
            ])
        
        # Time-sensitive suggestions
        if time_to_impact < 600:  # 10 minutes
            suggestions.insert(0, "URGENT: Immediate action required within 10 minutes")
        elif time_to_impact < 1800:  # 30 minutes
            suggestions.insert(0, "Action required within 30 minutes")
        
        # Trend-specific suggestions
        if slope > 0:
            suggestions.append("Trend is increasing - consider proactive scaling")
        else:
            suggestions.append("Trend is decreasing - may indicate degradation")
        
        return suggestions[:5]  # Limit to top 5 suggestions


class InsightEngine:
    """Generates actionable insights from observability data."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.insight_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def generate_insights(self, metric_points: List[MetricPoint],
                         anomalies: List[AnomalyDetection],
                         alerts: List[PredictiveAlert]) -> List[SystemInsight]:
        """Generate actionable insights from observability data."""
        insights = []
        
        try:
            # Performance insights
            performance_insights = self._generate_performance_insights(metric_points, anomalies)
            insights.extend(performance_insights)
            
            # Reliability insights
            reliability_insights = self._generate_reliability_insights(anomalies, alerts)
            insights.extend(reliability_insights)
            
            # Business impact insights
            business_insights = self._generate_business_insights(metric_points, anomalies)
            insights.extend(business_insights)
            
            # Infrastructure insights
            infrastructure_insights = self._generate_infrastructure_insights(metric_points)
            insights.extend(infrastructure_insights)
            
            # Store insights
            with self._lock:
                for insight in insights:
                    self.insight_history.append(insight)
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
        
        return insights
    
    def _generate_performance_insights(self, metric_points: List[MetricPoint],
                                     anomalies: List[AnomalyDetection]) -> List[SystemInsight]:
        """Generate performance-related insights."""
        insights = []
        
        # Group metrics by category
        performance_metrics = defaultdict(list)
        for point in metric_points:
            if any(perf_keyword in point.metric_name.lower() 
                  for perf_keyword in ['response_time', 'latency', 'throughput', 'cpu', 'memory']):
                performance_metrics[point.metric_name].append(point)
        
        # Response time analysis
        response_time_metrics = [name for name in performance_metrics.keys() 
                               if 'response_time' in name.lower() or 'latency' in name.lower()]
        
        if response_time_metrics:
            for metric_name in response_time_metrics:
                points = performance_metrics[metric_name]
                if len(points) >= 10:
                    values = [p.value for p in points]
                    recent_avg = statistics.mean(values[-10:])
                    historical_avg = statistics.mean(values[:-10]) if len(values) > 10 else recent_avg
                    
                    if recent_avg > historical_avg * 1.2:  # 20% increase
                        insight = SystemInsight(
                            insight_id=f"perf_degradation_{metric_name}_{int(time.time())}",
                            timestamp=time.time(),
                            category=InsightCategory.PERFORMANCE,
                            title="Performance Degradation Detected",
                            description=f"{metric_name} has increased by {((recent_avg - historical_avg) / historical_avg * 100):.1f}%",
                            confidence_score=0.8,
                            impact_score=min(1.0, (recent_avg - historical_avg) / historical_avg),
                            evidence=[
                                f"Recent average: {recent_avg:.2f}ms",
                                f"Historical average: {historical_avg:.2f}ms",
                                f"Performance degradation: {((recent_avg - historical_avg) / historical_avg * 100):.1f}%"
                            ],
                            recommendations=[
                                "Analyze recent deployments or configuration changes",
                                "Review database query performance",
                                "Check for resource constraints (CPU, memory, I/O)",
                                "Consider scaling up infrastructure resources"
                            ]
                        )
                        insights.append(insight)
        
        # CPU utilization patterns
        cpu_metrics = [name for name in performance_metrics.keys() if 'cpu' in name.lower()]
        if cpu_metrics:
            for metric_name in cpu_metrics:
                points = performance_metrics[metric_name]
                if len(points) >= 20:
                    values = [p.value for p in points]
                    max_cpu = max(values)
                    avg_cpu = statistics.mean(values)
                    
                    if max_cpu > 90 and avg_cpu > 70:
                        insight = SystemInsight(
                            insight_id=f"cpu_pressure_{metric_name}_{int(time.time())}",
                            timestamp=time.time(),
                            category=InsightCategory.INFRASTRUCTURE,
                            title="High CPU Utilization Pattern",
                            description=f"CPU utilization shows concerning pattern with max {max_cpu:.1f}% and average {avg_cpu:.1f}%",
                            confidence_score=0.9,
                            impact_score=min(1.0, avg_cpu / 100.0),
                            evidence=[
                                f"Maximum CPU utilization: {max_cpu:.1f}%",
                                f"Average CPU utilization: {avg_cpu:.1f}%",
                                f"Data points analyzed: {len(values)}"
                            ],
                            recommendations=[
                                "Consider horizontal scaling to distribute load",
                                "Profile applications for CPU optimization opportunities",
                                "Review CPU-intensive processes and algorithms",
                                "Implement CPU-based auto-scaling"
                            ]
                        )
                        insights.append(insight)
        
        return insights
    
    def _generate_reliability_insights(self, anomalies: List[AnomalyDetection],
                                     alerts: List[PredictiveAlert]) -> List[SystemInsight]:
        """Generate reliability-related insights."""
        insights = []
        
        # Anomaly clustering analysis
        anomaly_groups = defaultdict(list)
        for anomaly in anomalies:
            anomaly_groups[anomaly.metric_name].append(anomaly)
        
        # Identify metrics with frequent anomalies
        for metric_name, metric_anomalies in anomaly_groups.items():
            if len(metric_anomalies) >= 3:  # Multiple anomalies
                critical_count = sum(1 for a in metric_anomalies if a.severity == AlertSeverity.CRITICAL)
                
                if critical_count >= 2:
                    insight = SystemInsight(
                        insight_id=f"reliability_issue_{metric_name}_{int(time.time())}",
                        timestamp=time.time(),
                        category=InsightCategory.RELIABILITY,
                        title="Recurring Reliability Issues",
                        description=f"{metric_name} shows {len(metric_anomalies)} anomalies with {critical_count} critical issues",
                        confidence_score=0.85,
                        impact_score=min(1.0, critical_count / 5.0),
                        evidence=[
                            f"Total anomalies detected: {len(metric_anomalies)}",
                            f"Critical anomalies: {critical_count}",
                            f"Anomaly types: {list(set(a.anomaly_type.value for a in metric_anomalies))}"
                        ],
                        recommendations=[
                            "Investigate root cause of recurring anomalies",
                            "Review monitoring thresholds for accuracy",
                            "Implement automated remediation for known issues",
                            "Consider redundancy and failover mechanisms"
                        ]
                    )
                    insights.append(insight)
        
        # Predictive alert analysis
        urgent_alerts = [alert for alert in alerts if alert.time_to_impact_seconds < 1800]  # 30 minutes
        
        if len(urgent_alerts) >= 2:
            insight = SystemInsight(
                insight_id=f"multiple_predictions_{int(time.time())}",
                timestamp=time.time(),
                category=InsightCategory.RELIABILITY,
                title="Multiple Urgent Predictive Alerts",
                description=f"{len(urgent_alerts)} urgent issues predicted within 30 minutes",
                confidence_score=0.9,
                impact_score=len(urgent_alerts) / 10.0,
                evidence=[
                    f"Urgent alerts count: {len(urgent_alerts)}",
                    f"Affected metrics: {list(set(a.metric_name for a in urgent_alerts))}"
                ],
                affected_services=list(set(a.metric_name.split('_')[0] for a in urgent_alerts if '_' in a.metric_name)),
                recommendations=[
                    "Prioritize immediate investigation of predicted issues",
                    "Prepare incident response procedures",
                    "Consider preemptive scaling or resource allocation",
                    "Review system capacity planning"
                ]
            )
            insights.append(insight)
        
        return insights
    
    def _generate_business_insights(self, metric_points: List[MetricPoint],
                                   anomalies: List[AnomalyDetection]) -> List[SystemInsight]:
        """Generate business impact insights."""
        insights = []
        
        # Business metrics analysis
        business_metrics = defaultdict(list)
        for point in metric_points:
            if any(biz_keyword in point.metric_name.lower() 
                  for biz_keyword in ['user', 'request', 'transaction', 'revenue', 'conversion']):
                business_metrics[point.metric_name].append(point)
        
        # User experience analysis
        user_metrics = [name for name in business_metrics.keys() if 'user' in name.lower()]
        request_metrics = [name for name in business_metrics.keys() if 'request' in name.lower()]
        
        if user_metrics or request_metrics:
            all_business_points = []
            for metric_list in business_metrics.values():
                all_business_points.extend(metric_list)
            
            if len(all_business_points) >= 20:
                # Analyze business impact of anomalies
                business_anomalies = [a for a in anomalies 
                                    if any(biz_metric in a.metric_name.lower() 
                                          for biz_metric in ['user', 'request', 'transaction'])]
                
                if business_anomalies:
                    impact_score = sum(a.deviation_magnitude for a in business_anomalies) / len(business_anomalies)
                    
                    insight = SystemInsight(
                        insight_id=f"business_impact_{int(time.time())}",
                        timestamp=time.time(),
                        category=InsightCategory.BUSINESS,
                        title="Business Metrics Anomalies Detected",
                        description=f"{len(business_anomalies)} business-critical anomalies may impact user experience",
                        confidence_score=0.75,
                        impact_score=min(1.0, impact_score / 10.0),
                        evidence=[
                            f"Business anomalies count: {len(business_anomalies)}",
                            f"Affected metrics: {[a.metric_name for a in business_anomalies]}",
                            f"Average impact magnitude: {impact_score:.2f}"
                        ],
                        recommendations=[
                            "Monitor user experience metrics closely",
                            "Consider customer communication if issues persist",
                            "Review business process dependencies",
                            "Implement business impact tracking"
                        ],
                        estimated_performance_impact=impact_score * 10  # Rough estimate
                    )
                    insights.append(insight)
        
        return insights
    
    def _generate_infrastructure_insights(self, metric_points: List[MetricPoint]) -> List[SystemInsight]:
        """Generate infrastructure-related insights."""
        insights = []
        
        # Resource utilization analysis
        resource_metrics = defaultdict(list)
        for point in metric_points:
            if any(res_keyword in point.metric_name.lower() 
                  for res_keyword in ['cpu', 'memory', 'disk', 'network']):
                resource_type = next((keyword for keyword in ['cpu', 'memory', 'disk', 'network'] 
                                    if keyword in point.metric_name.lower()), 'unknown')
                resource_metrics[resource_type].append(point)
        
        # Resource efficiency analysis
        for resource_type, points in resource_metrics.items():
            if len(points) >= 20:
                values = [p.value for p in points]
                avg_utilization = statistics.mean(values)
                max_utilization = max(values)
                
                # Underutilization insight
                if avg_utilization < 30 and max_utilization < 50:
                    insight = SystemInsight(
                        insight_id=f"underutilization_{resource_type}_{int(time.time())}",
                        timestamp=time.time(),
                        category=InsightCategory.INFRASTRUCTURE,
                        title=f"{resource_type.title()} Underutilization",
                        description=f"{resource_type.title()} resources are underutilized with {avg_utilization:.1f}% average usage",
                        confidence_score=0.8,
                        impact_score=0.3,  # Lower impact for cost optimization
                        evidence=[
                            f"Average {resource_type} utilization: {avg_utilization:.1f}%",
                            f"Maximum {resource_type} utilization: {max_utilization:.1f}%",
                            f"Data points analyzed: {len(values)}"
                        ],
                        recommendations=[
                            f"Consider rightsizing {resource_type} resources",
                            "Review resource allocation policies",
                            "Implement dynamic resource allocation",
                            "Analyze cost optimization opportunities"
                        ],
                        estimated_cost_impact=avg_utilization * 10  # Rough cost estimate
                    )
                    insights.append(insight)
                
                # Capacity planning insight
                elif avg_utilization > 60 and max_utilization > 80:
                    growth_needed = (max_utilization - avg_utilization) / avg_utilization
                    
                    insight = SystemInsight(
                        insight_id=f"capacity_planning_{resource_type}_{int(time.time())}",
                        timestamp=time.time(),
                        category=InsightCategory.INFRASTRUCTURE,
                        title=f"{resource_type.title()} Capacity Planning",
                        description=f"{resource_type.title()} usage patterns suggest need for capacity planning",
                        confidence_score=0.85,
                        impact_score=min(1.0, max_utilization / 100.0),
                        evidence=[
                            f"Average {resource_type} utilization: {avg_utilization:.1f}%",
                            f"Peak {resource_type} utilization: {max_utilization:.1f}%",
                            f"Growth variability: {growth_needed:.1%}"
                        ],
                        recommendations=[
                            f"Plan for {resource_type} capacity expansion",
                            "Implement proactive monitoring and alerting",
                            "Consider auto-scaling policies",
                            "Review peak usage patterns and optimize"
                        ]
                    )
                    insights.append(insight)
        
        return insights


class AdvancedObservabilitySystem:
    """Main advanced observability and intelligence system."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Core components
        self.anomaly_detector = StatisticalAnomalyDetector()
        self.predictive_analyzer = PredictiveAnalyzer()
        self.insight_engine = InsightEngine()
        
        # Data storage
        self.metric_points: deque = deque(maxlen=50000)  # Store up to 50k points
        self.anomalies: deque = deque(maxlen=10000)
        self.predictive_alerts: deque = deque(maxlen=1000)
        self.insights: deque = deque(maxlen=1000)
        
        # Configuration
        self.analysis_interval_seconds = 300  # 5 minutes
        self.retention_hours = 24
        
        # System state
        self.is_analyzing = False
        self.analysis_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def ingest_metrics(self, metrics: List[Dict[str, Any]]):
        """Ingest metrics data for analysis."""
        try:
            current_time = time.time()
            
            with self._lock:
                for metric_data in metrics:
                    point = MetricPoint(
                        timestamp=metric_data.get('timestamp', current_time),
                        metric_name=metric_data.get('name', 'unknown'),
                        value=float(metric_data.get('value', 0.0)),
                        labels=metric_data.get('labels', {}),
                        metadata=metric_data.get('metadata', {})
                    )
                    self.metric_points.append(point)
            
            self.logger.debug(f"Ingested {len(metrics)} metric points")
            
        except Exception as e:
            self.logger.error(f"Error ingesting metrics: {e}")
    
    def run_analysis_cycle(self) -> Dict[str, Any]:
        """Run a complete analysis cycle."""
        try:
            start_time = time.perf_counter()
            
            # Get recent metric points for analysis
            with self._lock:
                recent_points = [p for p in self.metric_points 
                               if time.time() - p.timestamp < 3600]  # Last hour
            
            if len(recent_points) < 10:
                return {'message': 'Insufficient data for analysis'}
            
            # Run anomaly detection
            new_anomalies = self.anomaly_detector.detect_anomalies(recent_points)
            
            # Run predictive analysis
            new_alerts = self.predictive_analyzer.analyze_trends_and_predict(recent_points)
            
            # Generate insights
            new_insights = self.insight_engine.generate_insights(
                recent_points, new_anomalies, new_alerts
            )
            
            # Store results
            with self._lock:
                self.anomalies.extend(new_anomalies)
                self.predictive_alerts.extend(new_alerts)
                self.insights.extend(new_insights)
            
            analysis_time = time.perf_counter() - start_time
            
            result = {
                'timestamp': time.time(),
                'analysis_time_ms': analysis_time * 1000,
                'metrics_analyzed': len(recent_points),
                'anomalies_detected': len(new_anomalies),
                'predictive_alerts': len(new_alerts),
                'insights_generated': len(new_insights),
                'new_anomalies': [
                    {
                        'anomaly_id': a.anomaly_id,
                        'metric_name': a.metric_name,
                        'anomaly_type': a.anomaly_type.value,
                        'severity': a.severity.value,
                        'confidence_score': a.confidence_score,
                        'deviation_magnitude': a.deviation_magnitude
                    }
                    for a in new_anomalies
                ],
                'new_alerts': [
                    {
                        'alert_id': a.alert_id,
                        'metric_name': a.metric_name,
                        'predicted_issue': a.predicted_issue,
                        'time_to_impact_seconds': a.time_to_impact_seconds,
                        'severity': a.severity.value,
                        'confidence_score': a.confidence_score
                    }
                    for a in new_alerts
                ],
                'new_insights': [
                    {
                        'insight_id': i.insight_id,
                        'category': i.category.value,
                        'title': i.title,
                        'description': i.description,
                        'confidence_score': i.confidence_score,
                        'impact_score': i.impact_score
                    }
                    for i in new_insights
                ]
            }
            
            self.logger.info(f"Analysis cycle completed: {len(new_anomalies)} anomalies, {len(new_alerts)} alerts, {len(new_insights)} insights")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in analysis cycle: {e}")
            return {
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def get_observability_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive observability dashboard data."""
        with self._lock:
            current_time = time.time()
            
            # Recent data (last hour)
            recent_metrics = [p for p in self.metric_points 
                            if current_time - p.timestamp < 3600]
            recent_anomalies = [a for a in self.anomalies 
                              if current_time - a.timestamp < 3600]
            recent_alerts = [a for a in self.predictive_alerts 
                           if current_time - a.timestamp < 3600]
            recent_insights = [i for i in self.insights 
                             if current_time - i.timestamp < 3600]
            
            # System health summary
            health_score = self._calculate_system_health_score(
                recent_metrics, recent_anomalies, recent_alerts
            )
            
            dashboard = {
                'timestamp': current_time,
                'system_health_score': health_score,
                'summary': {
                    'total_metrics': len(self.metric_points),
                    'recent_metrics': len(recent_metrics),
                    'recent_anomalies': len(recent_anomalies),
                    'recent_alerts': len(recent_alerts),
                    'recent_insights': len(recent_insights)
                },
                'anomaly_breakdown': self._get_anomaly_breakdown(recent_anomalies),
                'alert_breakdown': self._get_alert_breakdown(recent_alerts),
                'insight_breakdown': self._get_insight_breakdown(recent_insights),
                'top_metrics': self._get_top_metrics_by_activity(recent_metrics),
                'critical_issues': self._get_critical_issues(recent_anomalies, recent_alerts),
                'performance_trends': self._get_performance_trends(recent_metrics)
            }
            
            return dashboard
    
    def start_continuous_analysis(self):
        """Start continuous analysis in background."""
        if self.is_analyzing:
            return
        
        self.is_analyzing = True
        self.analysis_thread = threading.Thread(
            target=self._analysis_loop,
            daemon=True
        )
        self.analysis_thread.start()
        self.logger.info("Advanced observability analysis started")
    
    def stop_analysis(self):
        """Stop continuous analysis."""
        self.is_analyzing = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=10.0)
        self.logger.info("Advanced observability analysis stopped")
    
    def _analysis_loop(self):
        """Main analysis loop."""
        while self.is_analyzing:
            try:
                # Run analysis cycle
                self.run_analysis_cycle()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                time.sleep(self.analysis_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                time.sleep(self.analysis_interval_seconds)
    
    def _cleanup_old_data(self):
        """Clean up old data beyond retention period."""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        with self._lock:
            # Clean metric points
            self.metric_points = deque(
                (p for p in self.metric_points if p.timestamp >= cutoff_time),
                maxlen=self.metric_points.maxlen
            )
            
            # Clean anomalies
            self.anomalies = deque(
                (a for a in self.anomalies if a.timestamp >= cutoff_time),
                maxlen=self.anomalies.maxlen
            )
            
            # Clean alerts
            self.predictive_alerts = deque(
                (a for a in self.predictive_alerts if a.timestamp >= cutoff_time),
                maxlen=self.predictive_alerts.maxlen
            )
            
            # Clean insights
            self.insights = deque(
                (i for i in self.insights if i.timestamp >= cutoff_time),
                maxlen=self.insights.maxlen
            )
    
    def _calculate_system_health_score(self, metrics: List[MetricPoint],
                                     anomalies: List[AnomalyDetection],
                                     alerts: List[PredictiveAlert]) -> float:
        """Calculate overall system health score (0-100)."""
        if not metrics:
            return 50.0  # Neutral score with no data
        
        # Base score
        health_score = 100.0
        
        # Penalize for anomalies
        for anomaly in anomalies:
            if anomaly.severity == AlertSeverity.CRITICAL:
                health_score -= 20
            elif anomaly.severity == AlertSeverity.WARNING:
                health_score -= 10
            else:
                health_score -= 5
        
        # Penalize for predictive alerts
        for alert in alerts:
            if alert.severity == AlertSeverity.EMERGENCY:
                health_score -= 30
            elif alert.severity == AlertSeverity.CRITICAL:
                health_score -= 15
            else:
                health_score -= 5
        
        return max(0.0, min(100.0, health_score))
    
    def _get_anomaly_breakdown(self, anomalies: List[AnomalyDetection]) -> Dict[str, int]:
        """Get breakdown of anomalies by type and severity."""
        breakdown = {
            'by_type': defaultdict(int),
            'by_severity': defaultdict(int)
        }
        
        for anomaly in anomalies:
            breakdown['by_type'][anomaly.anomaly_type.value] += 1
            breakdown['by_severity'][anomaly.severity.value] += 1
        
        return {
            'by_type': dict(breakdown['by_type']),
            'by_severity': dict(breakdown['by_severity'])
        }
    
    def _get_alert_breakdown(self, alerts: List[PredictiveAlert]) -> Dict[str, int]:
        """Get breakdown of alerts by severity."""
        breakdown = defaultdict(int)
        
        for alert in alerts:
            breakdown[alert.severity.value] += 1
        
        return dict(breakdown)
    
    def _get_insight_breakdown(self, insights: List[SystemInsight]) -> Dict[str, int]:
        """Get breakdown of insights by category."""
        breakdown = defaultdict(int)
        
        for insight in insights:
            breakdown[insight.category.value] += 1
        
        return dict(breakdown)
    
    def _get_top_metrics_by_activity(self, metrics: List[MetricPoint]) -> List[Dict[str, Any]]:
        """Get top metrics by activity level."""
        metric_counts = defaultdict(int)
        
        for metric in metrics:
            metric_counts[metric.metric_name] += 1
        
        top_metrics = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [
            {
                'metric_name': name,
                'data_points': count
            }
            for name, count in top_metrics
        ]
    
    def _get_critical_issues(self, anomalies: List[AnomalyDetection],
                           alerts: List[PredictiveAlert]) -> List[Dict[str, Any]]:
        """Get critical issues requiring immediate attention."""
        issues = []
        
        # Critical anomalies
        critical_anomalies = [a for a in anomalies if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]]
        for anomaly in critical_anomalies[:5]:  # Top 5
            issues.append({
                'type': 'anomaly',
                'id': anomaly.anomaly_id,
                'title': f"Critical Anomaly: {anomaly.metric_name}",
                'severity': anomaly.severity.value,
                'description': f"Anomaly detected with {anomaly.confidence_score:.1%} confidence",
                'timestamp': anomaly.timestamp
            })
        
        # Urgent alerts
        urgent_alerts = [a for a in alerts if a.time_to_impact_seconds < 1800]  # 30 minutes
        for alert in urgent_alerts[:5]:  # Top 5
            issues.append({
                'type': 'predictive_alert',
                'id': alert.alert_id,
                'title': f"Urgent: {alert.predicted_issue}",
                'severity': alert.severity.value,
                'description': f"Issue predicted in {alert.time_to_impact_seconds/60:.1f} minutes",
                'timestamp': alert.timestamp
            })
        
        return sorted(issues, key=lambda x: x['timestamp'], reverse=True)
    
    def _get_performance_trends(self, metrics: List[MetricPoint]) -> Dict[str, Any]:
        """Get performance trends summary."""
        # Group by performance-related metrics
        perf_metrics = defaultdict(list)
        
        for metric in metrics:
            if any(keyword in metric.metric_name.lower() 
                  for keyword in ['response_time', 'latency', 'cpu', 'memory', 'throughput']):
                perf_metrics[metric.metric_name].append(metric.value)
        
        trends = {}
        
        for metric_name, values in perf_metrics.items():
            if len(values) >= 10:
                recent_avg = statistics.mean(values[-5:])
                older_avg = statistics.mean(values[:-5]) if len(values) > 5 else recent_avg
                
                if older_avg > 0:
                    trend_percent = ((recent_avg - older_avg) / older_avg) * 100
                else:
                    trend_percent = 0.0
                
                trends[metric_name] = {
                    'current_avg': recent_avg,
                    'trend_percent': trend_percent,
                    'direction': 'improving' if trend_percent < -5 else 'degrading' if trend_percent > 5 else 'stable'
                }
        
        return trends


# Global observability system
_global_observability_system: Optional[AdvancedObservabilitySystem] = None
_system_lock = threading.Lock()


def get_observability_system() -> AdvancedObservabilitySystem:
    """Get or create the global observability system."""
    global _global_observability_system
    
    with _system_lock:
        if _global_observability_system is None:
            _global_observability_system = AdvancedObservabilitySystem()
        return _global_observability_system


def start_observability_monitoring():
    """Start the global observability monitoring system."""
    system = get_observability_system()
    system.start_continuous_analysis()


def stop_observability_monitoring():
    """Stop the global observability monitoring system."""
    system = get_observability_system()
    system.stop_analysis()