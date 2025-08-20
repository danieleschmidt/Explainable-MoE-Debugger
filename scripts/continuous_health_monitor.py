#!/usr/bin/env python3
"""
Continuous health monitoring service for progressive quality gates.
"""

import time
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List
import signal
from pathlib import Path
import logging

# Import our health checker
from deployment_health_check import DeploymentHealthChecker, HealthCheckResult


class ContinuousHealthMonitor:
    """Continuous health monitoring service with alerting and rollback."""
    
    def __init__(self):
        self.api_url = os.getenv('API_URL', 'http://localhost:8080')
        self.frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
        self.check_interval = int(os.getenv('CHECK_INTERVAL', '30'))  # seconds
        self.rollback_threshold = int(os.getenv('ROLLBACK_THRESHOLD', '3'))  # consecutive failures
        
        self.running = True
        self.consecutive_failures = 0
        self.last_healthy_time = datetime.now()
        self.health_history = []
        self.alerts_sent = set()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('health-monitor')
        
        # Health checker instance
        self.health_checker = DeploymentHealthChecker(
            base_url=self.api_url,
            timeout=30
        )
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.shutdown_handler)
        signal.signal(signal.SIGINT, self.shutdown_handler)
    
    def shutdown_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run a single health check cycle."""
        try:
            self.logger.info("Running health check cycle...")
            
            # Run all health checks
            results = await self.health_checker.run_all_health_checks()
            
            # Analyze results
            analysis = self.health_checker.analyze_health_results(results)
            
            # Store in history
            health_record = {
                'timestamp': datetime.now(),
                'analysis': analysis,
                'results': results
            }
            
            self.health_history.append(health_record)
            
            # Keep only last 100 records
            if len(self.health_history) > 100:
                self.health_history = self.health_history[-100:]
            
            return health_record
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'timestamp': datetime.now(),
                'analysis': {
                    'health_status': 'CRITICAL',
                    'success_rate': 0,
                    'error': str(e)
                },
                'results': []
            }
    
    def evaluate_health_trend(self) -> Dict[str, Any]:
        """Evaluate health trends over time."""
        if len(self.health_history) < 3:
            return {'trend': 'INSUFFICIENT_DATA', 'confidence': 0}
        
        recent_records = self.health_history[-10:]  # Last 10 checks
        
        # Calculate trend metrics
        healthy_count = sum(1 for r in recent_records if r['analysis']['health_status'] == 'HEALTHY')
        warning_count = sum(1 for r in recent_records if r['analysis']['health_status'] == 'WARNING')
        degraded_count = sum(1 for r in recent_records if r['analysis']['health_status'] == 'DEGRADED')
        critical_count = sum(1 for r in recent_records if r['analysis']['health_status'] == 'CRITICAL')
        
        total_checks = len(recent_records)
        
        # Determine trend
        if critical_count >= total_checks * 0.5:
            trend = 'CRITICAL_TREND'
        elif critical_count > 0 or degraded_count >= total_checks * 0.3:
            trend = 'DEGRADING'
        elif warning_count >= total_checks * 0.5:
            trend = 'UNSTABLE'
        elif healthy_count >= total_checks * 0.8:
            trend = 'HEALTHY'
        else:
            trend = 'MIXED'
        
        confidence = total_checks / 10.0  # Confidence based on data points
        
        return {
            'trend': trend,
            'confidence': confidence,
            'healthy_percentage': (healthy_count / total_checks) * 100,
            'critical_percentage': (critical_count / total_checks) * 100,
            'recent_checks': total_checks
        }
    
    def should_send_alert(self, health_record: Dict[str, Any], trend: Dict[str, Any]) -> List[str]:
        """Determine if alerts should be sent."""
        alerts_to_send = []
        
        analysis = health_record['analysis']
        health_status = analysis['health_status']
        
        # Critical status alert
        if health_status == 'CRITICAL' and 'critical_status' not in self.alerts_sent:
            alerts_to_send.append('critical_status')
            self.alerts_sent.add('critical_status')
        
        # Degrading trend alert
        if trend['trend'] == 'DEGRADING' and 'degrading_trend' not in self.alerts_sent:
            alerts_to_send.append('degrading_trend')
            self.alerts_sent.add('degrading_trend')
        
        # Consecutive failures alert
        if self.consecutive_failures >= self.rollback_threshold and 'consecutive_failures' not in self.alerts_sent:
            alerts_to_send.append('consecutive_failures')
            self.alerts_sent.add('consecutive_failures')
        
        # Performance degradation alert
        if analysis.get('average_response_time_ms', 0) > 2000 and 'slow_response' not in self.alerts_sent:
            alerts_to_send.append('slow_response')
            self.alerts_sent.add('slow_response')
        
        # Clear alerts if health improves
        if health_status == 'HEALTHY':
            self.alerts_sent.clear()
            self.consecutive_failures = 0
            self.last_healthy_time = datetime.now()
        
        return alerts_to_send
    
    def send_alert(self, alert_type: str, health_record: Dict[str, Any], trend: Dict[str, Any]):
        """Send alert notification."""
        timestamp = health_record['timestamp']
        analysis = health_record['analysis']
        
        alert_messages = {
            'critical_status': f"🚨 CRITICAL: System health is critical - immediate attention required",
            'degrading_trend': f"📉 WARNING: System health is degrading - {trend['critical_percentage']:.1f}% critical checks",
            'consecutive_failures': f"⚠️ ALERT: {self.consecutive_failures} consecutive health check failures",
            'slow_response': f"🐌 PERFORMANCE: Average response time {analysis.get('average_response_time_ms', 0):.0f}ms"
        }
        
        message = alert_messages.get(alert_type, f"UNKNOWN ALERT: {alert_type}")
        
        self.logger.warning(f"ALERT [{alert_type}]: {message}")
        
        # Here you would integrate with your alerting system:
        # - Send to Slack/Teams
        # - Send email notifications
        # - Trigger PagerDuty
        # - etc.
        
        # For now, just log and save to file
        alert_data = {
            'timestamp': timestamp.isoformat(),
            'alert_type': alert_type,
            'message': message,
            'health_status': analysis['health_status'],
            'success_rate': analysis['success_rate'],
            'trend': trend
        }
        
        # Save alert to file
        alerts_file = Path('health-alerts.jsonl')
        with open(alerts_file, 'a') as f:
            f.write(json.dumps(alert_data) + '\n')
    
    def should_trigger_rollback(self, health_record: Dict[str, Any], trend: Dict[str, Any]) -> bool:
        """Determine if automatic rollback should be triggered."""
        analysis = health_record['analysis']
        
        # Rollback criteria
        if self.consecutive_failures >= self.rollback_threshold:
            self.logger.critical(f"Triggering rollback: {self.consecutive_failures} consecutive failures")
            return True
        
        if trend['trend'] == 'CRITICAL_TREND' and trend['confidence'] > 0.7:
            self.logger.critical("Triggering rollback: Critical trend detected")
            return True
        
        if analysis['health_status'] == 'CRITICAL' and analysis.get('critical_failures', 0) > 2:
            self.logger.critical("Triggering rollback: Multiple critical failures")
            return True
        
        # Time-based rollback - if unhealthy for too long
        time_since_healthy = datetime.now() - self.last_healthy_time
        if time_since_healthy > timedelta(minutes=10) and analysis['health_status'] != 'HEALTHY':
            self.logger.critical(f"Triggering rollback: Unhealthy for {time_since_healthy}")
            return True
        
        return False
    
    def execute_emergency_rollback(self) -> bool:
        """Execute emergency rollback procedure."""
        self.logger.critical("🚨 EXECUTING EMERGENCY ROLLBACK")
        
        try:
            # Use the deployment health checker's rollback method
            rollback_success = self.health_checker.execute_rollback()
            
            if rollback_success:
                self.logger.info("✅ Emergency rollback completed successfully")
                
                # Send rollback notification
                rollback_alert = {
                    'timestamp': datetime.now().isoformat(),
                    'alert_type': 'emergency_rollback',
                    'message': '🔄 Emergency rollback executed due to health check failures',
                    'consecutive_failures': self.consecutive_failures,
                    'time_since_healthy': str(datetime.now() - self.last_healthy_time)
                }
                
                alerts_file = Path('health-alerts.jsonl')
                with open(alerts_file, 'a') as f:
                    f.write(json.dumps(rollback_alert) + '\n')
                
                return True
            else:
                self.logger.error("❌ Emergency rollback failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Emergency rollback error: {e}")
            return False
    
    def generate_health_summary(self) -> Dict[str, Any]:
        """Generate health summary for reporting."""
        if not self.health_history:
            return {'status': 'NO_DATA'}
        
        latest_record = self.health_history[-1]
        trend = self.evaluate_health_trend()
        
        # Calculate uptime
        total_checks = len(self.health_history)
        healthy_checks = sum(1 for r in self.health_history if r['analysis']['health_status'] == 'HEALTHY')
        uptime_percentage = (healthy_checks / total_checks * 100) if total_checks > 0 else 0
        
        # Calculate average response time
        response_times = [
            r['analysis'].get('average_response_time_ms', 0)
            for r in self.health_history
            if r['analysis'].get('average_response_time_ms', 0) > 0
        ]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            'current_status': latest_record['analysis']['health_status'],
            'uptime_percentage': uptime_percentage,
            'consecutive_failures': self.consecutive_failures,
            'last_healthy': self.last_healthy_time.isoformat(),
            'trend': trend,
            'total_checks': total_checks,
            'average_response_time_ms': avg_response_time,
            'monitoring_duration': str(datetime.now() - (self.health_history[0]['timestamp'] if self.health_history else datetime.now()))
        }
    
    async def monitoring_loop(self):
        """Main monitoring loop."""
        self.logger.info(f"Starting continuous health monitoring...")
        self.logger.info(f"API URL: {self.api_url}")
        self.logger.info(f"Check interval: {self.check_interval}s")
        self.logger.info(f"Rollback threshold: {self.rollback_threshold} consecutive failures")
        
        while self.running:
            try:
                # Run health check
                health_record = await self.run_health_check()
                analysis = health_record['analysis']
                
                # Update failure counter
                if analysis['health_status'] in ['CRITICAL', 'DEGRADED']:
                    self.consecutive_failures += 1
                else:
                    self.consecutive_failures = 0
                
                # Evaluate trends
                trend = self.evaluate_health_trend()
                
                # Check for alerts
                alerts = self.should_send_alert(health_record, trend)
                for alert_type in alerts:
                    self.send_alert(alert_type, health_record, trend)
                
                # Check for rollback conditions
                if self.should_trigger_rollback(health_record, trend):
                    rollback_success = self.execute_emergency_rollback()
                    if rollback_success:
                        # After rollback, reset counters and wait longer
                        self.consecutive_failures = 0
                        self.alerts_sent.clear()
                        await asyncio.sleep(self.check_interval * 3)  # Wait 3x longer after rollback
                        continue
                
                # Log current status
                self.logger.info(
                    f"Health Status: {analysis['health_status']} | "
                    f"Success Rate: {analysis['success_rate']:.1f}% | "
                    f"Consecutive Failures: {self.consecutive_failures} | "
                    f"Trend: {trend['trend']}"
                )
                
                # Save health summary
                summary = self.generate_health_summary()
                with open('health-summary.json', 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.check_interval)
        
        self.logger.info("Health monitoring stopped")
    
    async def run(self):
        """Run the health monitoring service."""
        try:
            await self.monitoring_loop()
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring service error: {e}")
            sys.exit(1)


async def main():
    """Main entry point."""
    monitor = ContinuousHealthMonitor()
    await monitor.run()


if __name__ == "__main__":
    asyncio.run(main())