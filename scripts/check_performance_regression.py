#!/usr/bin/env python3
"""
Performance regression detection script for progressive quality gates.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import statistics
from dataclasses import dataclass


@dataclass
class PerformanceThresholds:
    """Performance threshold configuration."""
    max_response_time_ms: float = 200.0
    min_throughput_rps: float = 100.0
    max_memory_usage_mb: float = 1024.0
    max_cpu_usage_percent: float = 80.0
    regression_tolerance_percent: float = 10.0


class PerformanceRegressionChecker:
    """Checks for performance regressions in benchmark results."""
    
    def __init__(self, thresholds: PerformanceThresholds):
        self.thresholds = thresholds
        self.baseline_file = Path("benchmark-baseline.json")
    
    def load_benchmark_results(self, results_file: Path) -> Dict[str, Any]:
        """Load benchmark results from JSON file."""
        try:
            with open(results_file) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Benchmark results file not found: {results_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in benchmark results: {e}")
            sys.exit(1)
    
    def load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline benchmark results if available."""
        if not self.baseline_file.exists():
            print("ℹ️  No baseline benchmark results found. Creating baseline...")
            return None
        
        try:
            with open(self.baseline_file) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("⚠️  Invalid baseline file. Ignoring baseline comparison.")
            return None
    
    def save_baseline(self, results: Dict[str, Any]) -> None:
        """Save current results as new baseline."""
        with open(self.baseline_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Baseline saved to {self.baseline_file}")
    
    def extract_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract key performance metrics from benchmark results."""
        metrics = {}
        
        if "benchmarks" in results:
            # pytest-benchmark format
            for benchmark in results["benchmarks"]:
                name = benchmark["name"]
                stats = benchmark["stats"]
                
                # Extract timing metrics (convert to milliseconds)
                metrics[f"{name}_mean_ms"] = stats["mean"] * 1000
                metrics[f"{name}_min_ms"] = stats["min"] * 1000
                metrics[f"{name}_max_ms"] = stats["max"] * 1000
                metrics[f"{name}_median_ms"] = stats["median"] * 1000
                metrics[f"{name}_stddev_ms"] = stats["stddev"] * 1000
        
        # Add custom metrics if available
        if "custom_metrics" in results:
            metrics.update(results["custom_metrics"])
        
        return metrics
    
    def check_absolute_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Check if metrics meet absolute performance thresholds."""
        violations = []
        
        # Check response time thresholds
        for metric_name, value in metrics.items():
            if "mean_ms" in metric_name and value > self.thresholds.max_response_time_ms:
                violations.append(
                    f"Response time {metric_name}: {value:.2f}ms > {self.thresholds.max_response_time_ms}ms"
                )
        
        # Check throughput thresholds
        for metric_name, value in metrics.items():
            if "throughput_rps" in metric_name and value < self.thresholds.min_throughput_rps:
                violations.append(
                    f"Throughput {metric_name}: {value:.2f} RPS < {self.thresholds.min_throughput_rps} RPS"
                )
        
        # Check memory usage
        for metric_name, value in metrics.items():
            if "memory_mb" in metric_name and value > self.thresholds.max_memory_usage_mb:
                violations.append(
                    f"Memory usage {metric_name}: {value:.2f}MB > {self.thresholds.max_memory_usage_mb}MB"
                )
        
        # Check CPU usage
        for metric_name, value in metrics.items():
            if "cpu_percent" in metric_name and value > self.thresholds.max_cpu_usage_percent:
                violations.append(
                    f"CPU usage {metric_name}: {value:.2f}% > {self.thresholds.max_cpu_usage_percent}%"
                )
        
        if violations:
            print("❌ Absolute performance threshold violations:")
            for violation in violations:
                print(f"   • {violation}")
            return False
        
        print("✅ All absolute performance thresholds met")
        return True
    
    def check_regression(self, current_metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> bool:
        """Check for performance regression against baseline."""
        regressions = []
        improvements = []
        
        for metric_name in current_metrics:
            if metric_name not in baseline_metrics:
                continue
            
            current_value = current_metrics[metric_name]
            baseline_value = baseline_metrics[metric_name]
            
            if baseline_value == 0:
                continue
            
            # Calculate percentage change
            change_percent = ((current_value - baseline_value) / baseline_value) * 100
            
            # For timing metrics, increase is bad
            if any(suffix in metric_name for suffix in ["_ms", "_time", "_latency"]):
                if change_percent > self.thresholds.regression_tolerance_percent:
                    regressions.append(
                        f"{metric_name}: {current_value:.2f} vs {baseline_value:.2f} "
                        f"({change_percent:+.1f}%)"
                    )
                elif change_percent < -5:  # Improvement threshold
                    improvements.append(
                        f"{metric_name}: {current_value:.2f} vs {baseline_value:.2f} "
                        f"({change_percent:+.1f}%)"
                    )
            
            # For throughput metrics, decrease is bad
            elif any(suffix in metric_name for suffix in ["_rps", "_throughput", "_qps"]):
                if change_percent < -self.thresholds.regression_tolerance_percent:
                    regressions.append(
                        f"{metric_name}: {current_value:.2f} vs {baseline_value:.2f} "
                        f"({change_percent:+.1f}%)"
                    )
                elif change_percent > 5:  # Improvement threshold
                    improvements.append(
                        f"{metric_name}: {current_value:.2f} vs {baseline_value:.2f} "
                        f"({change_percent:+.1f}%)"
                    )
        
        if improvements:
            print("🚀 Performance improvements detected:")
            for improvement in improvements:
                print(f"   • {improvement}")
        
        if regressions:
            print("❌ Performance regressions detected:")
            for regression in regressions:
                print(f"   • {regression}")
            return False
        
        print("✅ No significant performance regressions detected")
        return True
    
    def analyze_trends(self, metrics: Dict[str, float]) -> None:
        """Analyze performance trends and provide insights."""
        print("\n📊 Performance Analysis:")
        
        # Group metrics by category
        timing_metrics = {k: v for k, v in metrics.items() if any(suffix in k for suffix in ["_ms", "_time"])}
        throughput_metrics = {k: v for k, v in metrics.items() if any(suffix in k for suffix in ["_rps", "_throughput"])}
        resource_metrics = {k: v for k, v in metrics.items() if any(suffix in k for suffix in ["_mb", "_percent"])}
        
        if timing_metrics:
            print(f"   Response Times:")
            for name, value in timing_metrics.items():
                status = "🟢" if value <= self.thresholds.max_response_time_ms else "🔴"
                print(f"     {status} {name}: {value:.2f}ms")
        
        if throughput_metrics:
            print(f"   Throughput:")
            for name, value in throughput_metrics.items():
                status = "🟢" if value >= self.thresholds.min_throughput_rps else "🔴"
                print(f"     {status} {name}: {value:.2f}")
        
        if resource_metrics:
            print(f"   Resource Usage:")
            for name, value in resource_metrics.items():
                if "_mb" in name:
                    status = "🟢" if value <= self.thresholds.max_memory_usage_mb else "🔴"
                    print(f"     {status} {name}: {value:.2f}MB")
                elif "_percent" in name:
                    status = "🟢" if value <= self.thresholds.max_cpu_usage_percent else "🔴"
                    print(f"     {status} {name}: {value:.1f}%")
    
    def run_check(self, results_file: Path, update_baseline: bool = False) -> bool:
        """Run complete performance regression check."""
        print(f"🔍 Checking performance regression for: {results_file}")
        
        # Load current results
        current_results = self.load_benchmark_results(results_file)
        current_metrics = self.extract_metrics(current_results)
        
        if not current_metrics:
            print("❌ No performance metrics found in results")
            return False
        
        # Check absolute thresholds
        absolute_check_passed = self.check_absolute_thresholds(current_metrics)
        
        # Load baseline and check regression
        baseline_results = self.load_baseline()
        regression_check_passed = True
        
        if baseline_results:
            baseline_metrics = self.extract_metrics(baseline_results)
            regression_check_passed = self.check_regression(current_metrics, baseline_metrics)
        
        # Analyze trends
        self.analyze_trends(current_metrics)
        
        # Update baseline if requested
        if update_baseline:
            self.save_baseline(current_results)
        
        # Overall result
        overall_passed = absolute_check_passed and regression_check_passed
        
        if overall_passed:
            print("\n✅ Performance quality gate PASSED")
        else:
            print("\n❌ Performance quality gate FAILED")
        
        return overall_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check for performance regressions")
    parser.add_argument("results_file", type=Path, help="Benchmark results JSON file")
    parser.add_argument("--update-baseline", action="store_true", 
                       help="Update baseline with current results")
    parser.add_argument("--max-response-time", type=float, default=200.0,
                       help="Maximum allowed response time in ms")
    parser.add_argument("--min-throughput", type=float, default=100.0,
                       help="Minimum required throughput in RPS")
    parser.add_argument("--regression-tolerance", type=float, default=10.0,
                       help="Regression tolerance percentage")
    
    args = parser.parse_args()
    
    thresholds = PerformanceThresholds(
        max_response_time_ms=args.max_response_time,
        min_throughput_rps=args.min_throughput,
        regression_tolerance_percent=args.regression_tolerance
    )
    
    checker = PerformanceRegressionChecker(thresholds)
    
    try:
        success = checker.run_check(args.results_file, args.update_baseline)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Performance check failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()