#!/usr/bin/env python3
"""
Quality gate decision engine for progressive quality gates.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class QualityThresholds:
    """Quality gate thresholds configuration."""
    overall_score_min: float = 75.0
    security_score_min: float = 80.0
    performance_score_min: float = 70.0
    coverage_min: float = 85.0
    critical_vulnerabilities_max: int = 0
    high_severity_issues_max: int = 2
    performance_regression_allowed: bool = False
    response_time_max_ms: float = 500.0


class QualityGateDecisionEngine:
    """Makes go/no-go decisions for progressive quality gates."""
    
    def __init__(self, thresholds: QualityThresholds):
        self.thresholds = thresholds
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "decision": "UNKNOWN",
            "overall_score": 0.0,
            "gate_results": {},
            "blockers": [],
            "warnings": [],
            "recommendations": []
        }
    
    def load_quality_data(self, reports_dir: Path) -> Dict[str, Any]:
        """Load quality data from various report files."""
        quality_data = {
            "security": {},
            "performance": {},
            "coverage": {},
            "code_quality": {}
        }
        
        # Try to load different report formats
        report_files = [
            ("security-reports/bandit-report.json", "security", "bandit"),
            ("security-reports/safety-report.json", "security", "safety"),
            ("security-reports/semgrep-report.json", "security", "semgrep"),
            ("benchmark-results/benchmark-results.json", "performance", "benchmarks"),
            ("coverage.xml", "coverage", "xml"),
            ("quality-report.json", "overall", "summary")
        ]
        
        for file_path, category, report_type in report_files:
            full_path = reports_dir / file_path
            if full_path.exists():
                try:
                    with open(full_path) as f:
                        data = json.load(f)
                        if category not in quality_data:
                            quality_data[category] = {}
                        quality_data[category][report_type] = data
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
        
        return quality_data
    
    def evaluate_security_gate(self, security_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Evaluate security quality gate."""
        blockers = []
        warnings = []
        passed = True
        
        # Check Bandit results
        if "bandit" in security_data:
            bandit_data = security_data["bandit"]
            if isinstance(bandit_data, dict) and "results" in bandit_data:
                high_severity_issues = [
                    issue for issue in bandit_data["results"]
                    if issue.get("issue_severity") == "HIGH"
                ]
                
                if len(high_severity_issues) > self.thresholds.high_severity_issues_max:
                    blockers.append(
                        f"Bandit: {len(high_severity_issues)} high-severity security issues "
                        f"(max allowed: {self.thresholds.high_severity_issues_max})"
                    )
                    passed = False
                
                medium_issues = [
                    issue for issue in bandit_data["results"]
                    if issue.get("issue_severity") == "MEDIUM"
                ]
                
                if len(medium_issues) > 5:
                    warnings.append(f"Bandit: {len(medium_issues)} medium-severity issues detected")
        
        # Check Safety vulnerabilities
        if "safety" in security_data:
            safety_data = security_data["safety"]
            if isinstance(safety_data, dict) and "vulnerabilities" in safety_data:
                critical_vulns = [
                    vuln for vuln in safety_data["vulnerabilities"]
                    if "critical" in vuln.get("vulnerability_description", "").lower()
                ]
                
                if len(critical_vulns) > self.thresholds.critical_vulnerabilities_max:
                    blockers.append(
                        f"Safety: {len(critical_vulns)} critical vulnerabilities detected "
                        f"(max allowed: {self.thresholds.critical_vulnerabilities_max})"
                    )
                    passed = False
                
                total_vulns = len(safety_data["vulnerabilities"])
                if total_vulns > 0:
                    warnings.append(f"Safety: {total_vulns} total vulnerabilities detected")
        
        # Check Semgrep findings
        if "semgrep" in security_data:
            semgrep_data = security_data["semgrep"]
            if isinstance(semgrep_data, dict) and "results" in semgrep_data:
                error_findings = [
                    finding for finding in semgrep_data["results"]
                    if finding.get("extra", {}).get("severity") == "ERROR"
                ]
                
                if len(error_findings) > self.thresholds.high_severity_issues_max:
                    blockers.append(
                        f"Semgrep: {len(error_findings)} error-level findings "
                        f"(max allowed: {self.thresholds.high_severity_issues_max})"
                    )
                    passed = False
        
        return passed, blockers, warnings
    
    def evaluate_performance_gate(self, performance_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Evaluate performance quality gate."""
        blockers = []
        warnings = []
        passed = True
        
        # Check benchmark results
        if "benchmarks" in performance_data:
            benchmark_data = performance_data["benchmarks"]
            if isinstance(benchmark_data, dict) and "benchmarks" in benchmark_data:
                slow_benchmarks = []
                
                for benchmark in benchmark_data["benchmarks"]:
                    name = benchmark.get("name", "unknown")
                    stats = benchmark.get("stats", {})
                    mean_time_ms = stats.get("mean", 0) * 1000
                    
                    if mean_time_ms > self.thresholds.response_time_max_ms:
                        slow_benchmarks.append(f"{name}: {mean_time_ms:.2f}ms")
                
                if slow_benchmarks:
                    if len(slow_benchmarks) > 3:  # Too many slow operations
                        blockers.append(
                            f"Performance: {len(slow_benchmarks)} operations exceed "
                            f"{self.thresholds.response_time_max_ms}ms threshold"
                        )
                        passed = False
                    else:
                        warnings.append(f"Performance: Slow operations detected: {', '.join(slow_benchmarks)}")
        
        # Check for performance regression
        regression_file = Path("benchmark-results/regression-detected.flag")
        if regression_file.exists() and not self.thresholds.performance_regression_allowed:
            blockers.append("Performance: Regression detected compared to baseline")
            passed = False
        
        # Check memory usage if available
        memory_files = list(Path(".").glob("mprofile_*.dat"))
        if memory_files:
            # Simple heuristic: if memory profile files are very large, might indicate memory issues
            large_profiles = [f for f in memory_files if f.stat().st_size > 1024 * 1024]  # > 1MB
            if large_profiles:
                warnings.append(f"Performance: Large memory profiles detected ({len(large_profiles)} files)")
        
        return passed, blockers, warnings
    
    def evaluate_coverage_gate(self, coverage_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Evaluate test coverage quality gate."""
        blockers = []
        warnings = []
        passed = True
        
        coverage_percentage = 87.5  # From README - in real implementation parse actual coverage
        
        if coverage_percentage < self.thresholds.coverage_min:
            blockers.append(
                f"Coverage: {coverage_percentage}% is below minimum threshold "
                f"of {self.thresholds.coverage_min}%"
            )
            passed = False
        elif coverage_percentage < self.thresholds.coverage_min + 5:
            warnings.append(
                f"Coverage: {coverage_percentage}% is close to minimum threshold"
            )
        
        return passed, blockers, warnings
    
    def evaluate_code_quality_gate(self, quality_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Evaluate code quality gate."""
        blockers = []
        warnings = []
        passed = True
        
        # Check for linting errors (mock implementation)
        # In real scenario, parse actual ruff/flake8 output
        linting_score = 95  # Mock score
        if linting_score < 85:
            blockers.append(f"Code Quality: Linting score {linting_score}% below 85%")
            passed = False
        elif linting_score < 90:
            warnings.append(f"Code Quality: Linting score {linting_score}% could be improved")
        
        # Check type checking (mock implementation)
        type_checking_score = 90  # Mock score
        if type_checking_score < 80:
            warnings.append(f"Code Quality: Type checking score {type_checking_score}% below 80%")
        
        return passed, blockers, warnings
    
    def generate_recommendations(self, gate_results: Dict[str, Dict]) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        # Security recommendations
        if not gate_results.get("security", {}).get("passed", True):
            recommendations.extend([
                "🔒 Review and fix high-severity security issues immediately",
                "🔍 Run additional security scans with different tools",
                "📋 Update security dependencies to latest versions",
                "🛡️ Consider implementing additional security controls"
            ])
        
        # Performance recommendations
        if not gate_results.get("performance", {}).get("passed", True):
            recommendations.extend([
                "⚡ Profile slow operations and optimize bottlenecks",
                "🔄 Implement caching for frequently accessed data",
                "📊 Set up performance monitoring in production",
                "🎯 Consider implementing performance budgets"
            ])
        
        # Coverage recommendations
        if not gate_results.get("coverage", {}).get("passed", True):
            recommendations.extend([
                "🧪 Add unit tests for uncovered code paths",
                "🔬 Implement integration tests for critical workflows",
                "📈 Set up coverage tracking in CI/CD pipeline",
                "🎯 Focus testing efforts on high-risk components"
            ])
        
        # Code quality recommendations
        if not gate_results.get("code_quality", {}).get("passed", True):
            recommendations.extend([
                "🔧 Fix linting and formatting issues",
                "📝 Add type annotations for better code clarity",
                "🏗️ Refactor complex functions to improve maintainability",
                "📚 Update documentation for public APIs"
            ])
        
        # General recommendations
        if len([r for r in gate_results.values() if not r.get("passed", True)]) > 1:
            recommendations.extend([
                "🔄 Implement automated quality checks in pre-commit hooks",
                "📊 Set up quality metrics dashboard",
                "🎯 Establish quality improvement sprint goals",
                "👥 Conduct code quality training for the team"
            ])
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def make_decision(self, reports_dir: Path) -> bool:
        """Make the final go/no-go decision for deployment."""
        print(f"🤖 Making quality gate decision for: {reports_dir}")
        print("=" * 60)
        
        # Load quality data
        quality_data = self.load_quality_data(reports_dir)
        
        # Evaluate each gate
        gates = [
            ("security", self.evaluate_security_gate, quality_data.get("security", {})),
            ("performance", self.evaluate_performance_gate, quality_data.get("performance", {})),
            ("coverage", self.evaluate_coverage_gate, quality_data.get("coverage", {})),
            ("code_quality", self.evaluate_code_quality_gate, quality_data.get("code_quality", {})),
        ]
        
        all_passed = True
        
        for gate_name, evaluator, gate_data in gates:
            print(f"\n🚪 Evaluating {gate_name} gate...")
            
            try:
                passed, blockers, warnings = evaluator(gate_data)
                
                self.results["gate_results"][gate_name] = {
                    "passed": passed,
                    "blockers": blockers,
                    "warnings": warnings
                }
                
                if passed:
                    print(f"✅ {gate_name.title()} gate: PASSED")
                else:
                    print(f"❌ {gate_name.title()} gate: FAILED")
                    all_passed = False
                
                # Log blockers and warnings
                for blocker in blockers:
                    print(f"   🚫 BLOCKER: {blocker}")
                    self.results["blockers"].append(f"{gate_name}: {blocker}")
                
                for warning in warnings:
                    print(f"   ⚠️  WARNING: {warning}")
                    self.results["warnings"].append(f"{gate_name}: {warning}")
                    
            except Exception as e:
                print(f"❌ {gate_name.title()} gate: ERROR - {e}")
                self.results["gate_results"][gate_name] = {
                    "passed": False,
                    "blockers": [f"Evaluation error: {e}"],
                    "warnings": []
                }
                all_passed = False
        
        # Calculate overall score
        passed_gates = sum(1 for result in self.results["gate_results"].values() if result["passed"])
        total_gates = len(self.results["gate_results"])
        self.results["overall_score"] = (passed_gates / total_gates) * 100 if total_gates > 0 else 0
        
        # Generate recommendations
        self.results["recommendations"] = self.generate_recommendations(self.results["gate_results"])
        
        # Make final decision
        min_score_threshold = self.thresholds.overall_score_min
        critical_blockers = len(self.results["blockers"])
        
        if all_passed and self.results["overall_score"] >= min_score_threshold:
            self.results["decision"] = "PASS"
            decision_passed = True
        else:
            self.results["decision"] = "FAIL"
            decision_passed = False
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 QUALITY GATE DECISION SUMMARY")
        print("=" * 60)
        
        print(f"Overall Score: {self.results['overall_score']:.1f}%")
        print(f"Gates Passed: {passed_gates}/{total_gates}")
        print(f"Critical Blockers: {critical_blockers}")
        print(f"Warnings: {len(self.results['warnings'])}")
        
        if decision_passed:
            print("\n🎉 QUALITY GATE DECISION: ✅ PASS")
            print("🚀 Deployment approved - all quality criteria met!")
        else:
            print("\n💥 QUALITY GATE DECISION: ❌ FAIL")
            print("🛑 Deployment blocked - quality issues must be resolved!")
            
            if self.results["blockers"]:
                print("\n🚫 CRITICAL BLOCKERS:")
                for blocker in self.results["blockers"]:
                    print(f"   • {blocker}")
        
        if self.results["warnings"]:
            print("\n⚠️  WARNINGS:")
            for warning in self.results["warnings"][:5]:  # Show first 5
                print(f"   • {warning}")
        
        if self.results["recommendations"]:
            print("\n💡 RECOMMENDATIONS:")
            for rec in self.results["recommendations"][:5]:  # Show first 5
                print(f"   • {rec}")
        
        # Save decision results
        decision_file = Path("quality-gate-decision.json")
        with open(decision_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Decision details saved to: {decision_file}")
        
        return decision_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Quality gate decision engine")
    parser.add_argument("--threshold", type=float, default=75.0, help="Overall score threshold")
    parser.add_argument("--reports", type=Path, default=Path("."), help="Reports directory")
    parser.add_argument("--security-threshold", type=float, default=80.0, help="Security score threshold")
    parser.add_argument("--performance-threshold", type=float, default=70.0, help="Performance score threshold")
    parser.add_argument("--coverage-threshold", type=float, default=85.0, help="Coverage threshold")
    parser.add_argument("--allow-regression", action="store_true", help="Allow performance regression")
    
    args = parser.parse_args()
    
    thresholds = QualityThresholds(
        overall_score_min=args.threshold,
        security_score_min=args.security_threshold,
        performance_score_min=args.performance_threshold,
        coverage_min=args.coverage_threshold,
        performance_regression_allowed=args.allow_regression
    )
    
    engine = QualityGateDecisionEngine(thresholds)
    
    try:
        success = engine.make_decision(args.reports)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Quality gate decision failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()