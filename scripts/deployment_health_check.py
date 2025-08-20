#!/usr/bin/env python3
"""
Deployment health check and rollback system for progressive quality gates.
"""

import time
import sys
import json
import requests
import subprocess
import asyncio
import websockets
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import psutil


@dataclass
class HealthCheckResult:
    """Health check result."""
    name: str
    passed: bool
    response_time_ms: float = 0.0
    details: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DeploymentHealthChecker:
    """Comprehensive deployment health checker with rollback capabilities."""
    
    def __init__(self, base_url: str = "http://localhost:8080", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.health_results = []
        self.deployment_start_time = datetime.now()
        self.rollback_triggered = False
    
    def check_api_health(self) -> HealthCheckResult:
        """Check API health endpoint."""
        print("🔍 Checking API health endpoint...")
        
        start_time = time.time()
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate health response structure
                required_fields = ['status', 'timestamp']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return HealthCheckResult(
                        name="API Health",
                        passed=False,
                        response_time_ms=response_time,
                        details=f"Missing required fields: {missing_fields}"
                    )
                
                if data.get('status') == 'healthy':
                    return HealthCheckResult(
                        name="API Health",
                        passed=True,
                        response_time_ms=response_time,
                        details=f"API is healthy, response time: {response_time:.2f}ms"
                    )
                else:
                    return HealthCheckResult(
                        name="API Health",
                        passed=False,
                        response_time_ms=response_time,
                        details=f"API status: {data.get('status', 'unknown')}"
                    )
            else:
                return HealthCheckResult(
                    name="API Health",
                    passed=False,
                    response_time_ms=response_time,
                    details=f"HTTP {response.status_code}: {response.text[:100]}"
                )
                
        except requests.exceptions.ConnectionError:
            return HealthCheckResult(
                name="API Health",
                passed=False,
                details="Connection refused - service may not be running"
            )
        except requests.exceptions.Timeout:
            return HealthCheckResult(
                name="API Health",
                passed=False,
                details=f"Request timeout after {self.timeout}s"
            )
        except Exception as e:
            return HealthCheckResult(
                name="API Health",
                passed=False,
                details=f"Unexpected error: {str(e)}"
            )
    
    def check_api_endpoints(self) -> List[HealthCheckResult]:
        """Check critical API endpoints."""
        print("🔍 Checking critical API endpoints...")
        
        endpoints = [
            ("/api/v1/sessions", "POST", {"model_name": "test-model"}),
            ("/docs", "GET", None),
            ("/openapi.json", "GET", None)
        ]
        
        results = []
        
        for endpoint, method, payload in endpoints:
            start_time = time.time()
            
            try:
                url = f"{self.base_url}{endpoint}"
                
                if method == "GET":
                    response = requests.get(url, timeout=self.timeout)
                elif method == "POST":
                    response = requests.post(url, json=payload, timeout=self.timeout)
                else:
                    continue
                
                response_time = (time.time() - start_time) * 1000
                
                # Check if response is acceptable
                acceptable_codes = [200, 201, 422]  # 422 for validation errors is OK
                
                if response.status_code in acceptable_codes:
                    results.append(HealthCheckResult(
                        name=f"API {method} {endpoint}",
                        passed=True,
                        response_time_ms=response_time,
                        details=f"HTTP {response.status_code}, {response_time:.2f}ms"
                    ))
                else:
                    results.append(HealthCheckResult(
                        name=f"API {method} {endpoint}",
                        passed=False,
                        response_time_ms=response_time,
                        details=f"HTTP {response.status_code}: {response.text[:100]}"
                    ))
                    
            except Exception as e:
                results.append(HealthCheckResult(
                    name=f"API {method} {endpoint}",
                    passed=False,
                    details=f"Error: {str(e)}"
                ))
        
        return results
    
    async def check_websocket_health(self) -> HealthCheckResult:
        """Check WebSocket connectivity and functionality."""
        print("🔍 Checking WebSocket health...")
        
        ws_url = self.base_url.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws'
        
        start_time = time.time()
        
        try:
            async with websockets.connect(ws_url, timeout=self.timeout) as websocket:
                # Send ping message
                ping_message = {
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send(json.dumps(ping_message))
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=5.0
                    )
                    
                    response_time = (time.time() - start_time) * 1000
                    response_data = json.loads(response)
                    
                    if response_data.get('type') == 'pong':
                        return HealthCheckResult(
                            name="WebSocket Health",
                            passed=True,
                            response_time_ms=response_time,
                            details=f"WebSocket ping-pong successful, {response_time:.2f}ms"
                        )
                    else:
                        return HealthCheckResult(
                            name="WebSocket Health",
                            passed=True,  # Connection works, just different response
                            response_time_ms=response_time,
                            details=f"WebSocket connected, got: {response_data.get('type', 'unknown')}"
                        )
                        
                except asyncio.TimeoutError:
                    response_time = (time.time() - start_time) * 1000
                    return HealthCheckResult(
                        name="WebSocket Health",
                        passed=False,
                        response_time_ms=response_time,
                        details="WebSocket timeout waiting for response"
                    )
                    
        except websockets.exceptions.ConnectionRefused:
            return HealthCheckResult(
                name="WebSocket Health",
                passed=False,
                details="WebSocket connection refused"
            )
        except Exception as e:
            return HealthCheckResult(
                name="WebSocket Health",
                passed=False,
                details=f"WebSocket error: {str(e)}"
            )
    
    def check_database_connectivity(self) -> HealthCheckResult:
        """Check database connectivity if applicable."""
        print("🔍 Checking database connectivity...")
        
        # Check if database is configured
        db_configured = False
        
        # Try to check for database health through API
        try:
            response = requests.get(
                f"{self.base_url}/health/database",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('database_status') == 'connected':
                    return HealthCheckResult(
                        name="Database Health",
                        passed=True,
                        details="Database connection healthy"
                    )
                else:
                    return HealthCheckResult(
                        name="Database Health",
                        passed=False,
                        details=f"Database status: {data.get('database_status', 'unknown')}"
                    )
            else:
                # Database health endpoint not available - assume no database or it's optional
                return HealthCheckResult(
                    name="Database Health",
                    passed=True,
                    details="Database health endpoint not available (may be optional)"
                )
                
        except requests.exceptions.RequestException:
            # Database health check not available - not a failure
            return HealthCheckResult(
                name="Database Health",
                passed=True,
                details="Database health check not implemented (optional)"
            )
    
    def check_cache_connectivity(self) -> HealthCheckResult:
        """Check cache (Redis) connectivity."""
        print("🔍 Checking cache connectivity...")
        
        try:
            response = requests.get(
                f"{self.base_url}/health/cache",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('cache_status') == 'connected':
                    return HealthCheckResult(
                        name="Cache Health",
                        passed=True,
                        details="Cache connection healthy"
                    )
                else:
                    return HealthCheckResult(
                        name="Cache Health",
                        passed=False,
                        details=f"Cache status: {data.get('cache_status', 'unknown')}"
                    )
            else:
                # Cache is optional for basic functionality
                return HealthCheckResult(
                    name="Cache Health",
                    passed=True,
                    details="Cache health endpoint not available (degraded performance mode)"
                )
                
        except requests.exceptions.RequestException:
            return HealthCheckResult(
                name="Cache Health",
                passed=True,
                details="Cache not available (running without caching)"
            )
    
    def check_performance_metrics(self) -> HealthCheckResult:
        """Check performance metrics and resource usage."""
        print("🔍 Checking performance metrics...")
        
        # Check system resources
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage_percent = memory.percent
        
        # Check API response time
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            api_response_time = (time.time() - start_time) * 1000
        except:
            api_response_time = 999999  # Very high if failed
        
        # Performance thresholds
        performance_issues = []
        
        if cpu_usage > 80:
            performance_issues.append(f"High CPU usage: {cpu_usage:.1f}%")
        
        if memory_usage_percent > 85:
            performance_issues.append(f"High memory usage: {memory_usage_percent:.1f}%")
        
        if api_response_time > 1000:  # > 1 second
            performance_issues.append(f"Slow API response: {api_response_time:.0f}ms")
        
        if performance_issues:
            return HealthCheckResult(
                name="Performance Metrics",
                passed=False,
                response_time_ms=api_response_time,
                details=f"Performance issues: {'; '.join(performance_issues)}"
            )
        else:
            return HealthCheckResult(
                name="Performance Metrics",
                passed=True,
                response_time_ms=api_response_time,
                details=f"Performance healthy: CPU {cpu_usage:.1f}%, Memory {memory_usage_percent:.1f}%, API {api_response_time:.0f}ms"
            )
    
    def check_frontend_availability(self) -> HealthCheckResult:
        """Check frontend availability."""
        print("🔍 Checking frontend availability...")
        
        frontend_url = self.base_url.replace(':8080', ':3000')  # Assume frontend on 3000
        
        start_time = time.time()
        try:
            response = requests.get(frontend_url, timeout=self.timeout)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                # Check if it looks like the correct frontend
                content = response.text.lower()
                if any(keyword in content for keyword in ['moe', 'debugger', 'mixture', 'experts']):
                    return HealthCheckResult(
                        name="Frontend Availability",
                        passed=True,
                        response_time_ms=response_time,
                        details=f"Frontend accessible, {response_time:.2f}ms"
                    )
                else:
                    return HealthCheckResult(
                        name="Frontend Availability",
                        passed=False,
                        response_time_ms=response_time,
                        details="Frontend accessible but content doesn't match expected application"
                    )
            else:
                return HealthCheckResult(
                    name="Frontend Availability",
                    passed=False,
                    response_time_ms=response_time,
                    details=f"Frontend HTTP {response.status_code}"
                )
                
        except requests.exceptions.ConnectionError:
            # Frontend might not be running - not critical for API-only deployment
            return HealthCheckResult(
                name="Frontend Availability",
                passed=True,  # Non-critical
                details="Frontend not accessible (API-only deployment)"
            )
        except Exception as e:
            return HealthCheckResult(
                name="Frontend Availability",
                passed=True,  # Non-critical
                details=f"Frontend check error: {str(e)} (non-critical)"
            )
    
    def run_functional_test(self) -> HealthCheckResult:
        """Run a basic functional test."""
        print("🔍 Running functional test...")
        
        start_time = time.time()
        
        try:
            # Create a session
            session_response = requests.post(
                f"{self.base_url}/api/v1/sessions",
                json={"model_name": "test-model", "config": {"sampling_rate": 0.1}},
                timeout=self.timeout
            )
            
            if session_response.status_code != 201:
                return HealthCheckResult(
                    name="Functional Test",
                    passed=False,
                    details=f"Failed to create session: HTTP {session_response.status_code}"
                )
            
            session_data = session_response.json()
            session_id = session_data.get('session_id')
            
            if not session_id:
                return HealthCheckResult(
                    name="Functional Test",
                    passed=False,
                    details="No session ID returned"
                )
            
            # Send some routing data
            routing_data = {
                "events": [
                    {
                        "expert_id": 0,
                        "token_id": 123,
                        "routing_weight": 0.8,
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            }
            
            routing_response = requests.post(
                f"{self.base_url}/api/v1/sessions/{session_id}/routing-data",
                json=routing_data,
                timeout=self.timeout
            )
            
            if routing_response.status_code != 200:
                return HealthCheckResult(
                    name="Functional Test",
                    passed=False,
                    details=f"Failed to send routing data: HTTP {routing_response.status_code}"
                )
            
            # Get session stats
            stats_response = requests.get(
                f"{self.base_url}/api/v1/sessions/{session_id}/stats",
                timeout=self.timeout
            )
            
            if stats_response.status_code != 200:
                return HealthCheckResult(
                    name="Functional Test",
                    passed=False,
                    details=f"Failed to get stats: HTTP {stats_response.status_code}"
                )
            
            # Clean up - delete session
            delete_response = requests.delete(
                f"{self.base_url}/api/v1/sessions/{session_id}",
                timeout=self.timeout
            )
            
            test_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="Functional Test",
                passed=True,
                response_time_ms=test_time,
                details=f"Full workflow test passed in {test_time:.0f}ms"
            )
            
        except Exception as e:
            test_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="Functional Test",
                passed=False,
                response_time_ms=test_time,
                details=f"Functional test error: {str(e)}"
            )
    
    async def run_all_health_checks(self) -> List[HealthCheckResult]:
        """Run all health checks."""
        print("🔍 Running comprehensive deployment health checks...")
        
        results = []
        
        # Basic health checks
        results.append(self.check_api_health())
        results.extend(self.check_api_endpoints())
        
        # WebSocket check (async)
        ws_result = await self.check_websocket_health()
        results.append(ws_result)
        
        # Infrastructure checks
        results.append(self.check_database_connectivity())
        results.append(self.check_cache_connectivity())
        
        # Performance and resource checks
        results.append(self.check_performance_metrics())
        
        # Frontend check
        results.append(self.check_frontend_availability())
        
        # Functional test
        results.append(self.run_functional_test())
        
        return results
    
    def analyze_health_results(self, results: List[HealthCheckResult]) -> Dict[str, Any]:
        """Analyze health check results."""
        total_checks = len(results)
        passed_checks = sum(1 for r in results if r.passed)
        failed_checks = [r for r in results if not r.passed]
        
        # Calculate success rate
        success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        # Identify critical failures
        critical_failures = [
            r for r in failed_checks
            if r.name in ["API Health", "Functional Test", "Performance Metrics"]
        ]
        
        # Calculate average response time
        response_times = [r.response_time_ms for r in results if r.response_time_ms > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Determine overall health status
        if len(critical_failures) > 0:
            health_status = "CRITICAL"
        elif success_rate < 80:
            health_status = "DEGRADED"
        elif success_rate < 95:
            health_status = "WARNING"
        else:
            health_status = "HEALTHY"
        
        return {
            'health_status': health_status,
            'success_rate': success_rate,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': len(failed_checks),
            'critical_failures': len(critical_failures),
            'average_response_time_ms': avg_response_time,
            'failed_check_details': [
                {'name': r.name, 'details': r.details} for r in failed_checks
            ],
            'critical_failure_details': [
                {'name': r.name, 'details': r.details} for r in critical_failures
            ]
        }
    
    def should_trigger_rollback(self, analysis: Dict[str, Any]) -> bool:
        """Determine if rollback should be triggered."""
        # Rollback criteria
        if analysis['health_status'] == 'CRITICAL':
            print("🚨 CRITICAL health status detected - triggering rollback")
            return True
        
        if analysis['critical_failures'] > 0:
            print(f"🚨 {analysis['critical_failures']} critical failures detected - triggering rollback")
            return True
        
        if analysis['success_rate'] < 60:
            print(f"🚨 Success rate {analysis['success_rate']:.1f}% below threshold - triggering rollback")
            return True
        
        return False
    
    def execute_rollback(self) -> bool:
        """Execute deployment rollback."""
        if self.rollback_triggered:
            print("⚠️  Rollback already in progress")
            return False
        
        self.rollback_triggered = True
        print("🔄 Executing deployment rollback...")
        
        try:
            # Try different rollback mechanisms
            rollback_commands = [
                # Docker Compose rollback
                ["docker-compose", "down"],
                # Kubernetes rollback
                ["kubectl", "rollout", "undo", "deployment/moe-debugger"],
                # Docker service rollback
                ["docker", "service", "rollback", "moe-debugger"],
            ]
            
            for cmd in rollback_commands:
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        print(f"✅ Rollback successful using: {' '.join(cmd)}")
                        return True
                    else:
                        print(f"⚠️  Rollback command failed: {' '.join(cmd)}")
                        print(f"Error: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    print(f"⚠️  Rollback command timeout: {' '.join(cmd)}")
                except FileNotFoundError:
                    # Command not available, try next one
                    continue
            
            print("❌ All rollback mechanisms failed")
            return False
            
        except Exception as e:
            print(f"❌ Rollback execution error: {e}")
            return False
    
    def generate_health_report(self, results: List[HealthCheckResult], analysis: Dict[str, Any]) -> str:
        """Generate comprehensive health report."""
        report = []
        report.append("🏥 DEPLOYMENT HEALTH CHECK REPORT")
        report.append("=" * 60)
        
        # Summary
        report.append(f"\n📊 HEALTH SUMMARY:")
        report.append(f"   Overall Status: {analysis['health_status']}")
        report.append(f"   Success Rate: {analysis['success_rate']:.1f}% ({analysis['passed_checks']}/{analysis['total_checks']})")
        report.append(f"   Average Response Time: {analysis['average_response_time_ms']:.2f}ms")
        report.append(f"   Critical Failures: {analysis['critical_failures']}")
        
        # Deployment info
        deployment_duration = datetime.now() - self.deployment_start_time
        report.append(f"\n⏱️  DEPLOYMENT INFO:")
        report.append(f"   Deployment Duration: {deployment_duration}")
        report.append(f"   Check Timestamp: {datetime.now().isoformat()}")
        report.append(f"   Target URL: {self.base_url}")
        
        # Individual check results
        report.append(f"\n🔍 DETAILED CHECK RESULTS:")
        for result in results:
            status_icon = "✅" if result.passed else "❌"
            response_info = f" ({result.response_time_ms:.0f}ms)" if result.response_time_ms > 0 else ""
            report.append(f"   {status_icon} {result.name}{response_info}")
            if result.details:
                report.append(f"      {result.details}")
        
        # Failed checks
        if analysis['failed_check_details']:
            report.append(f"\n❌ FAILED CHECKS:")
            for failure in analysis['failed_check_details']:
                report.append(f"   • {failure['name']}: {failure['details']}")
        
        # Critical failures
        if analysis['critical_failure_details']:
            report.append(f"\n🚨 CRITICAL FAILURES:")
            for failure in analysis['critical_failure_details']:
                report.append(f"   • {failure['name']}: {failure['details']}")
        
        # Rollback status
        if self.rollback_triggered:
            report.append(f"\n🔄 ROLLBACK STATUS:")
            report.append(f"   Rollback Triggered: YES")
            report.append(f"   Reason: Health checks failed critical thresholds")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        if analysis['health_status'] == 'HEALTHY':
            report.append("   • Deployment is healthy and ready for production traffic ✅")
        elif analysis['health_status'] == 'WARNING':
            report.append("   • Monitor deployment closely, some non-critical issues detected")
            report.append("   • Consider investigating failed checks")
        elif analysis['health_status'] == 'DEGRADED':
            report.append("   • Deployment has significant issues but may be functional")
            report.append("   • Investigate and fix issues before promoting to production")
        elif analysis['health_status'] == 'CRITICAL':
            report.append("   • Deployment has critical issues - ROLLBACK RECOMMENDED")
            report.append("   • Do not promote to production")
            report.append("   • Investigate critical failures immediately")
        
        return "\n".join(report)
    
    async def run_deployment_health_check(self, rollback_on_failure: bool = True) -> bool:
        """Run complete deployment health check with optional rollback."""
        print(f"🏥 Starting deployment health check for: {self.base_url}")
        
        # Wait a moment for services to stabilize
        print("⏳ Waiting for services to stabilize...")
        time.sleep(5)
        
        try:
            # Run all health checks
            results = await self.run_all_health_checks()
            self.health_results = results
            
            # Analyze results
            analysis = self.analyze_health_results(results)
            
            # Generate report
            report = self.generate_health_report(results, analysis)
            print(report)
            
            # Save results
            health_data = {
                'analysis': analysis,
                'results': [
                    {
                        'name': r.name,
                        'passed': r.passed,
                        'response_time_ms': r.response_time_ms,
                        'details': r.details,
                        'timestamp': r.timestamp.isoformat()
                    }
                    for r in results
                ],
                'rollback_triggered': self.rollback_triggered
            }
            
            with open("deployment-health-check.json", "w") as f:
                json.dump(health_data, f, indent=2)
            
            # Determine if rollback needed
            if rollback_on_failure and self.should_trigger_rollback(analysis):
                rollback_success = self.execute_rollback()
                if not rollback_success:
                    print("❌ Rollback failed - manual intervention required")
                return False
            
            # Return overall success
            return analysis['health_status'] in ['HEALTHY', 'WARNING']
            
        except Exception as e:
            print(f"❌ Health check execution failed: {e}")
            if rollback_on_failure:
                print("🔄 Triggering rollback due to health check failure...")
                self.execute_rollback()
            return False


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deployment health check and rollback")
    parser.add_argument("--url", default="http://localhost:8080", help="Base URL to check")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--no-rollback", action="store_true", help="Disable automatic rollback")
    parser.add_argument("--wait-time", type=int, default=30, help="Time to wait before starting checks")
    
    args = parser.parse_args()
    
    # Wait for deployment to be ready
    if args.wait_time > 0:
        print(f"⏳ Waiting {args.wait_time} seconds for deployment to be ready...")
        time.sleep(args.wait_time)
    
    checker = DeploymentHealthChecker(
        base_url=args.url,
        timeout=args.timeout
    )
    
    try:
        success = await checker.run_deployment_health_check(
            rollback_on_failure=not args.no_rollback
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Deployment health check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())