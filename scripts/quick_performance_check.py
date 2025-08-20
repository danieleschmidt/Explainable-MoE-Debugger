#!/usr/bin/env python3
"""
Quick performance regression check for progressive quality gates.
"""

import time
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import argparse
from dataclasses import dataclass
import psutil
import threading


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    imports_time: float = 0.0
    test_execution_time: float = 0.0


class QuickPerformanceChecker:
    """Quick performance regression checker for quality gates."""
    
    def __init__(self, baseline_file: str = "performance-baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.current_metrics = PerformanceMetrics()
        self.monitoring = False
        self.resource_samples = []
    
    def monitor_resources(self, duration: float = 10.0):
        """Monitor system resources during execution."""
        def resource_monitor():
            process = psutil.Process()
            start_time = time.time()
            
            while self.monitoring and (time.time() - start_time) < duration:
                try:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()
                    
                    self.resource_samples.append({
                        'memory_mb': memory_mb,
                        'cpu_percent': cpu_percent,
                        'timestamp': time.time()
                    })
                    
                    time.sleep(0.1)  # Sample every 100ms
                except psutil.NoSuchProcess:
                    break
        
        self.monitoring = True
        monitor_thread = threading.Thread(target=resource_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return monitor_thread
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        
        if self.resource_samples:
            # Calculate averages
            avg_memory = sum(s['memory_mb'] for s in self.resource_samples) / len(self.resource_samples)
            avg_cpu = sum(s['cpu_percent'] for s in self.resource_samples) / len(self.resource_samples)
            max_memory = max(s['memory_mb'] for s in self.resource_samples)
            
            self.current_metrics.memory_usage_mb = max_memory
            self.current_metrics.cpu_usage_percent = avg_cpu
    
    def measure_import_performance(self) -> float:
        """Measure module import performance."""
        print("⚡ Measuring import performance...")
        
        start_time = time.time()
        
        try:
            # Test importing the main modules
            import_code = """
import sys
sys.path.insert(0, 'src')

# Time critical imports
import moe_debugger
from moe_debugger import MoEDebugger
from moe_debugger.enhanced_debugger import EnhancedMoEDebugger
from moe_debugger.analyzer import MoEAnalyzer
from moe_debugger.cache.manager import CacheManager
from moe_debugger.performance_optimization import PerformanceOptimizer
            """
            
            exec(import_code)
            import_time = time.time() - start_time
            
            print(f"   Import time: {import_time:.3f}s")
            return import_time
            
        except Exception as e:
            print(f"   ❌ Import error: {e}")
            return 999.0  # High penalty for import failures
    
    def measure_basic_functionality(self) -> float:
        """Measure basic functionality performance."""
        print("⚡ Measuring basic functionality performance...")
        
        start_time = time.time()
        
        try:
            # Start resource monitoring
            monitor_thread = self.monitor_resources(duration=30.0)
            
            # Test basic functionality
            exec("""
import sys
sys.path.insert(0, 'src')
from unittest.mock import Mock
from moe_debugger import MoEDebugger

# Create mock model
mock_model = Mock()
mock_model.config = Mock()
mock_model.config.num_experts = 8

# Create debugger
debugger = MoEDebugger(model=mock_model)

# Process some mock routing events
for i in range(1000):
    event = {
        "expert_id": i % 8,
        "token_id": i,
        "routing_weight": 0.1 + (i % 10) * 0.1
    }
    debugger.process_routing_event(event)

# Get some stats
stats = debugger.get_routing_stats()
            """)
            
            execution_time = time.time() - start_time
            
            # Stop monitoring
            self.stop_monitoring()
            monitor_thread.join(timeout=1)
            
            print(f"   Execution time: {execution_time:.3f}s")
            return execution_time
            
        except Exception as e:
            print(f"   ❌ Functionality test error: {e}")
            self.stop_monitoring()
            return 999.0  # High penalty for functionality failures
    
    def run_quick_tests(self) -> float:
        """Run a subset of quick tests to measure performance."""
        print("⚡ Running quick performance tests...")
        
        start_time = time.time()
        
        try:
            # Run only fast unit tests
            cmd = [
                "python", "-m", "pytest",
                "tests/unit/",
                "-x",  # Stop on first failure
                "--tb=no",  # No traceback
                "-q",  # Quiet mode
                "--maxfail=3",  # Stop after 3 failures
                "--timeout=30",  # 30 second timeout per test
                "-k", "not slow and not integration"  # Skip slow tests
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute total timeout
            )
            
            test_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"   Quick tests passed in {test_time:.3f}s")
            else:
                print(f"   ⚠️  Some quick tests failed (time: {test_time:.3f}s)")
                # Don't fail the performance check for test failures
            
            return test_time
            
        except subprocess.TimeoutExpired:
            print("   ❌ Quick tests timed out")
            return 999.0
        except Exception as e:
            print(f"   ❌ Quick test error: {e}")
            return 999.0
    
    def measure_startup_performance(self) -> float:
        """Measure application startup performance."""
        print("⚡ Measuring startup performance...")
        
        start_time = time.time()
        
        try:
            # Test server startup time (without actually starting)
            startup_code = """
import sys
sys.path.insert(0, 'src')

# Import server components
from moe_debugger.server import create_app
from moe_debugger.logging_config import setup_logging

# Setup logging
setup_logging()

# Create app instance
app = create_app()
            """
            
            exec(startup_code)
            startup_time = time.time() - start_time
            
            print(f"   Startup time: {startup_time:.3f}s")
            return startup_time
            
        except Exception as e:
            print(f"   ❌ Startup test error: {e}")
            return 999.0
    
    def load_baseline(self) -> Dict[str, Any]:
        """Load performance baseline if it exists."""
        if not self.baseline_file.exists():
            print("ℹ️  No performance baseline found")
            return {}
        
        try:
            with open(self.baseline_file, 'r') as f:
                baseline = json.load(f)
            print(f"📊 Loaded performance baseline from {self.baseline_file}")
            return baseline
        except Exception as e:
            print(f"⚠️  Could not load baseline: {e}")
            return {}
    
    def save_baseline(self, metrics: Dict[str, Any]):
        """Save current metrics as new baseline."""
        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"💾 Saved performance baseline to {self.baseline_file}")
        except Exception as e:
            print(f"⚠️  Could not save baseline: {e}")
    
    def compare_with_baseline(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current metrics with baseline."""
        comparison = {
            'regression_detected': False,
            'improvements': [],
            'regressions': [],
            'changes': {}
        }
        
        # Performance thresholds (percentage change)
        regression_threshold = 20.0  # 20% slower is a regression
        improvement_threshold = 10.0  # 10% faster is an improvement
        
        metrics_to_compare = [
            'imports_time',
            'execution_time',
            'test_execution_time',
            'startup_time'
        ]
        
        for metric in metrics_to_compare:
            if metric in current and metric in baseline:
                current_value = current[metric]
                baseline_value = baseline[metric]
                
                if baseline_value > 0:
                    change_percent = ((current_value - baseline_value) / baseline_value) * 100
                    comparison['changes'][metric] = {
                        'current': current_value,
                        'baseline': baseline_value,
                        'change_percent': change_percent
                    }
                    
                    if change_percent > regression_threshold:
                        comparison['regressions'].append({
                            'metric': metric,
                            'change_percent': change_percent,
                            'current': current_value,
                            'baseline': baseline_value
                        })
                        comparison['regression_detected'] = True
                    
                    elif change_percent < -improvement_threshold:
                        comparison['improvements'].append({
                            'metric': metric,
                            'change_percent': abs(change_percent),
                            'current': current_value,
                            'baseline': baseline_value
                        })
        
        return comparison
    
    def generate_performance_report(self, metrics: Dict[str, Any], comparison: Dict[str, Any] = None) -> str:
        """Generate performance analysis report."""
        report = []
        report.append("⚡ QUICK PERFORMANCE CHECK REPORT")
        report.append("=" * 50)
        
        # Current metrics
        report.append(f"\n📊 CURRENT PERFORMANCE METRICS:")
        report.append(f"   Import Time: {metrics.get('imports_time', 0):.3f}s")
        report.append(f"   Execution Time: {metrics.get('execution_time', 0):.3f}s")
        report.append(f"   Test Execution: {metrics.get('test_execution_time', 0):.3f}s")
        report.append(f"   Startup Time: {metrics.get('startup_time', 0):.3f}s")
        report.append(f"   Memory Usage: {metrics.get('memory_usage_mb', 0):.1f}MB")
        report.append(f"   CPU Usage: {metrics.get('cpu_usage_percent', 0):.1f}%")
        
        # Performance assessment
        total_time = (
            metrics.get('imports_time', 0) +
            metrics.get('execution_time', 0) +
            metrics.get('startup_time', 0)
        )
        
        performance_score = 100
        if total_time > 10:  # > 10 seconds total
            performance_score -= 30
        elif total_time > 5:  # > 5 seconds total
            performance_score -= 15
        
        if metrics.get('memory_usage_mb', 0) > 500:  # > 500MB
            performance_score -= 20
        
        if metrics.get('test_execution_time', 0) > 30:  # > 30 seconds
            performance_score -= 20
        
        status = "EXCELLENT" if performance_score >= 90 else "GOOD" if performance_score >= 75 else "ACCEPTABLE" if performance_score >= 60 else "POOR"
        
        report.append(f"\n🎯 PERFORMANCE SCORE: {performance_score}/100 - {status}")
        
        # Baseline comparison
        if comparison:
            report.append(f"\n📈 BASELINE COMPARISON:")
            
            if comparison['regression_detected']:
                report.append(f"   ❌ REGRESSION DETECTED!")
                for regression in comparison['regressions']:
                    report.append(f"      • {regression['metric']}: {regression['change_percent']:+.1f}% slower")
            
            if comparison['improvements']:
                report.append(f"   🚀 IMPROVEMENTS:")
                for improvement in comparison['improvements']:
                    report.append(f"      • {improvement['metric']}: {improvement['change_percent']:.1f}% faster")
            
            if not comparison['regression_detected'] and not comparison['improvements']:
                report.append(f"   ✅ Performance stable (no significant changes)")
        
        # Quality gates
        report.append(f"\n🚪 PERFORMANCE QUALITY GATES:")
        
        import_gate = "✅ PASS" if metrics.get('imports_time', 0) < 2.0 else "❌ FAIL"
        report.append(f"   Import Time (< 2s): {import_gate}")
        
        execution_gate = "✅ PASS" if metrics.get('execution_time', 0) < 5.0 else "❌ FAIL"
        report.append(f"   Execution Time (< 5s): {execution_gate}")
        
        memory_gate = "✅ PASS" if metrics.get('memory_usage_mb', 0) < 500 else "❌ FAIL"
        report.append(f"   Memory Usage (< 500MB): {memory_gate}")
        
        regression_gate = "✅ PASS" if not (comparison and comparison['regression_detected']) else "❌ FAIL"
        report.append(f"   No Regression: {regression_gate}")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        if metrics.get('imports_time', 0) > 2.0:
            report.append("   • Optimize module imports and dependencies")
        if metrics.get('execution_time', 0) > 5.0:
            report.append("   • Profile and optimize critical code paths")
        if metrics.get('memory_usage_mb', 0) > 500:
            report.append("   • Investigate memory usage and potential leaks")
        if comparison and comparison['regression_detected']:
            report.append("   • Investigate performance regressions before merging")
        if total_time < 3.0 and metrics.get('memory_usage_mb', 0) < 200:
            report.append("   • Performance is excellent ✅")
        
        return "\n".join(report)
    
    def check_performance_gates(self, metrics: Dict[str, Any], comparison: Dict[str, Any] = None) -> bool:
        """Check if performance meets quality gates."""
        gates_passed = True
        
        # Import time gate
        if metrics.get('imports_time', 0) > 2.0:
            print(f"❌ Import time gate FAILED: {metrics['imports_time']:.3f}s > 2.0s")
            gates_passed = False
        
        # Execution time gate
        if metrics.get('execution_time', 0) > 5.0:
            print(f"❌ Execution time gate FAILED: {metrics['execution_time']:.3f}s > 5.0s")
            gates_passed = False
        
        # Memory usage gate
        if metrics.get('memory_usage_mb', 0) > 500:
            print(f"❌ Memory usage gate FAILED: {metrics['memory_usage_mb']:.1f}MB > 500MB")
            gates_passed = False
        
        # Regression gate
        if comparison and comparison['regression_detected']:
            print(f"❌ Performance regression gate FAILED: {len(comparison['regressions'])} regressions detected")
            gates_passed = False
        
        if gates_passed:
            print("✅ All performance quality gates PASSED")
        
        return gates_passed
    
    def run_performance_check(self, update_baseline: bool = False) -> bool:
        """Run complete quick performance check."""
        print("⚡ Starting quick performance check...")
        
        # Measure performance metrics
        metrics = {
            'imports_time': self.measure_import_performance(),
            'execution_time': self.measure_basic_functionality(),
            'test_execution_time': self.run_quick_tests(),
            'startup_time': self.measure_startup_performance(),
            'memory_usage_mb': self.current_metrics.memory_usage_mb,
            'cpu_usage_percent': self.current_metrics.cpu_usage_percent,
            'timestamp': time.time()
        }
        
        # Load baseline for comparison
        baseline = self.load_baseline()
        comparison = None
        
        if baseline:
            comparison = self.compare_with_baseline(metrics, baseline)
        
        # Generate report
        report = self.generate_performance_report(metrics, comparison)
        print(report)
        
        # Update baseline if requested
        if update_baseline:
            self.save_baseline(metrics)
        
        # Save detailed results
        with open("quick-performance-check.json", "w") as f:
            json.dump({
                'metrics': metrics,
                'baseline': baseline,
                'comparison': comparison,
                'quality_gates_passed': self.check_performance_gates(metrics, comparison)
            }, f, indent=2)
        
        # Check quality gates
        return self.check_performance_gates(metrics, comparison)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Quick performance regression check")
    parser.add_argument("--update-baseline", action="store_true", help="Update performance baseline")
    parser.add_argument("--baseline-file", default="performance-baseline.json", help="Baseline file path")
    
    args = parser.parse_args()
    
    checker = QuickPerformanceChecker(baseline_file=args.baseline_file)
    
    try:
        success = checker.run_performance_check(update_baseline=args.update_baseline)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Performance check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()