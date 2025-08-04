#!/usr/bin/env python3
"""Test runner script with comprehensive testing options."""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle output."""
    if description:
        print(f"\n📋 {description}")
    print(f"🔧 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run tests for MoE Debugger")
    
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--slow", action="store_true", help="Include slow tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--html-coverage", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--fast", action="store_true", help="Run fast tests only (exclude slow and integration)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-n", type=int, help="Number of parallel workers")
    parser.add_argument("--memory", action="store_true", help="Test with memory cache only")
    parser.add_argument("--redis", action="store_true", help="Test with Redis cache (requires Redis)")
    parser.add_argument("--no-install", action="store_true", help="Skip installation check")
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Check installation unless skipped
    if not args.no_install:
        print("🔍 Checking installation...")
        if not run_command([sys.executable, "-c", "import moe_debugger; print('MoE Debugger installed successfully')"]):
            print("❌ MoE Debugger not installed. Run: pip install -e .")
            return 1
    
    # Build pytest command
    pytest_cmd = [sys.executable, "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        pytest_cmd.append("-v")
    else:
        pytest_cmd.append("-q")
    
    # Add parallel execution
    if args.parallel:
        pytest_cmd.extend(["-n", str(args.parallel)])
    
    # Add coverage options
    if args.coverage or args.html_coverage:
        pytest_cmd.extend([
            "--cov=src/moe_debugger",
            "--cov-report=term-missing"
        ])
        
        if args.html_coverage:
            pytest_cmd.extend(["--cov-report=html"])
    
    # Determine test selection
    test_paths = []
    markers = []
    
    if args.unit:
        test_paths.append("tests/unit/")
        markers.append("unit")
    elif args.integration:
        test_paths.append("tests/integration/")
        markers.append("integration")
    elif args.fast:
        markers.append("not slow and not integration")
    elif args.benchmark:
        test_paths.append("tests/performance/")
        markers.append("benchmark")
    else:
        test_paths.append("tests/")
    
    # Handle slow tests
    if not args.slow and not args.benchmark:
        if "not slow" not in " ".join(markers):
            markers.append("not slow")
    
    # Add test paths
    pytest_cmd.extend(test_paths)
    
    # Add markers
    if markers:
        marker_expr = " and ".join(markers)
        pytest_cmd.extend(["-m", marker_expr])
    
    # Set environment variables for cache testing
    env = os.environ.copy()
    
    if args.memory:
        env["CACHE_TYPE"] = "memory"
        print("🔧 Testing with memory cache")
    elif args.redis:
        env["CACHE_TYPE"] = "redis"
        env["REDIS_URL"] = "redis://localhost:6379"
        print("🔧 Testing with Redis cache")
        
        # Check Redis availability
        try:
            import redis
            r = redis.Redis.from_url(env["REDIS_URL"])
            r.ping()
            print("✅ Redis connection verified")
        except Exception as e:
            print(f"❌ Redis not available: {e}")
            print("💡 Install Redis or use --memory flag")
            return 1
    
    # Run tests
    print(f"\n🧪 Running tests with command: {' '.join(pytest_cmd)}")
    
    try:
        result = subprocess.run(pytest_cmd, env=env)
        exit_code = result.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        exit_code = 130
    
    # Summary
    if exit_code == 0:
        print("\n✅ All tests passed!")
        
        if args.html_coverage:
            print("📊 HTML coverage report available at: htmlcov/index.html")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())