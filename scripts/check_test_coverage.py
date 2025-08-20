#!/usr/bin/env python3
"""
Test coverage verification for progressive quality gates.
"""

import subprocess
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
from dataclasses import dataclass


@dataclass
class CoverageMetrics:
    """Test coverage metrics."""
    line_rate: float = 0.0
    branch_rate: float = 0.0
    lines_covered: int = 0
    lines_valid: int = 0
    branches_covered: int = 0
    branches_valid: int = 0


class CoverageAnalyzer:
    """Test coverage analyzer for progressive quality gates."""
    
    def __init__(self, min_coverage: float = 85.0, min_branch_coverage: float = 75.0):
        self.min_coverage = min_coverage
        self.min_branch_coverage = min_branch_coverage
        self.coverage_file = Path("coverage.xml")
        self.coverage_json = Path(".coverage")
    
    def run_coverage_analysis(self) -> bool:
        """Run pytest with coverage analysis."""
        print("🧪 Running test coverage analysis...")
        
        try:
            # Run pytest with coverage
            cmd = [
                "python", "-m", "pytest",
                "--cov=src/moe_debugger",
                "--cov-report=xml",
                "--cov-report=json",
                "--cov-report=term-missing",
                "--cov-branch",
                "tests/",
                "-v"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Test coverage analysis completed successfully")
                return True
            else:
                print(f"❌ Test coverage analysis failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Test coverage analysis timed out")
            return False
        except Exception as e:
            print(f"❌ Test coverage analysis error: {e}")
            return False
    
    def parse_coverage_xml(self) -> Optional[Dict[str, Any]]:
        """Parse XML coverage report."""
        if not self.coverage_file.exists():
            print(f"❌ Coverage XML file not found: {self.coverage_file}")
            return None
        
        try:
            tree = ET.parse(self.coverage_file)
            root = tree.getroot()
            
            # Overall coverage metrics
            coverage_element = root.find('.')
            line_rate = float(coverage_element.get('line-rate', 0)) * 100
            branch_rate = float(coverage_element.get('branch-rate', 0)) * 100
            lines_covered = int(coverage_element.get('lines-covered', 0))
            lines_valid = int(coverage_element.get('lines-valid', 0))
            branches_covered = int(coverage_element.get('branches-covered', 0))
            branches_valid = int(coverage_element.get('branches-valid', 0))
            
            overall_metrics = CoverageMetrics(
                line_rate=line_rate,
                branch_rate=branch_rate,
                lines_covered=lines_covered,
                lines_valid=lines_valid,
                branches_covered=branches_covered,
                branches_valid=branches_valid
            )
            
            # Package-level coverage
            packages = []
            for package in root.findall('.//package'):
                package_name = package.get('name', 'unknown')
                package_line_rate = float(package.get('line-rate', 0)) * 100
                package_branch_rate = float(package.get('branch-rate', 0)) * 100
                
                # Class-level coverage
                classes = []
                for class_elem in package.findall('.//class'):
                    class_name = class_elem.get('name', 'unknown')
                    class_filename = class_elem.get('filename', 'unknown')
                    class_line_rate = float(class_elem.get('line-rate', 0)) * 100
                    class_branch_rate = float(class_elem.get('branch-rate', 0)) * 100
                    
                    # Missing lines
                    missing_lines = []
                    for line in class_elem.findall('.//line'):
                        if line.get('hits') == '0':
                            missing_lines.append(int(line.get('number')))
                    
                    classes.append({
                        'name': class_name,
                        'filename': class_filename,
                        'line_rate': class_line_rate,
                        'branch_rate': class_branch_rate,
                        'missing_lines': missing_lines
                    })
                
                packages.append({
                    'name': package_name,
                    'line_rate': package_line_rate,
                    'branch_rate': package_branch_rate,
                    'classes': classes
                })
            
            return {
                'overall': overall_metrics,
                'packages': packages,
                'timestamp': self.coverage_file.stat().st_mtime
            }
            
        except Exception as e:
            print(f"❌ Error parsing coverage XML: {e}")
            return None
    
    def parse_coverage_json(self) -> Optional[Dict[str, Any]]:
        """Parse JSON coverage report if available."""
        json_file = Path("coverage.json")
        if not json_file.exists():
            return None
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            summary = data.get('totals', {})
            files = data.get('files', {})
            
            # File-level coverage details
            file_coverage = []
            for filename, file_data in files.items():
                if 'summary' in file_data:
                    file_summary = file_data['summary']
                    missing_lines = file_data.get('missing_lines', [])
                    excluded_lines = file_data.get('excluded_lines', [])
                    
                    file_coverage.append({
                        'filename': filename,
                        'num_statements': file_summary.get('num_statements', 0),
                        'missing_lines': len(missing_lines),
                        'covered_lines': file_summary.get('covered_lines', 0),
                        'percent_covered': file_summary.get('percent_covered', 0),
                        'missing_line_numbers': missing_lines,
                        'excluded_lines': excluded_lines
                    })
            
            return {
                'summary': summary,
                'files': file_coverage
            }
            
        except Exception as e:
            print(f"⚠️  Error parsing coverage JSON: {e}")
            return None
    
    def identify_coverage_gaps(self, coverage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify critical coverage gaps."""
        gaps = []
        
        if 'packages' not in coverage_data:
            return gaps
        
        for package in coverage_data['packages']:
            for class_info in package['classes']:
                if class_info['line_rate'] < self.min_coverage:
                    gaps.append({
                        'type': 'low_coverage',
                        'package': package['name'],
                        'class': class_info['name'],
                        'filename': class_info['filename'],
                        'coverage': class_info['line_rate'],
                        'missing_lines': class_info['missing_lines']
                    })
                
                if class_info['branch_rate'] < self.min_branch_coverage:
                    gaps.append({
                        'type': 'low_branch_coverage',
                        'package': package['name'],
                        'class': class_info['name'],
                        'filename': class_info['filename'],
                        'branch_coverage': class_info['branch_rate']
                    })
        
        return gaps
    
    def generate_coverage_report(self, coverage_data: Dict[str, Any], gaps: List[Dict[str, Any]]) -> str:
        """Generate coverage analysis report."""
        overall = coverage_data['overall']
        
        report = []
        report.append("🧪 TEST COVERAGE ANALYSIS REPORT")
        report.append("=" * 50)
        
        # Overall metrics
        report.append(f"\n📊 OVERALL COVERAGE:")
        report.append(f"   Line Coverage: {overall.line_rate:.2f}%")
        report.append(f"   Branch Coverage: {overall.branch_rate:.2f}%")
        report.append(f"   Lines Covered: {overall.lines_covered:,} / {overall.lines_valid:,}")
        if overall.branches_valid > 0:
            report.append(f"   Branches Covered: {overall.branches_covered:,} / {overall.branches_valid:,}")
        
        # Quality assessment
        quality_score = 100
        if overall.line_rate < self.min_coverage:
            quality_score -= 30
        if overall.branch_rate < self.min_branch_coverage:
            quality_score -= 20
        if len(gaps) > 5:
            quality_score -= 20
        
        status = "EXCELLENT" if quality_score >= 90 else "GOOD" if quality_score >= 75 else "NEEDS_IMPROVEMENT" if quality_score >= 60 else "CRITICAL"
        
        report.append(f"\n🎯 COVERAGE QUALITY SCORE: {quality_score}/100 - {status}")
        
        # Quality gates status
        line_gate = "✅ PASS" if overall.line_rate >= self.min_coverage else "❌ FAIL"
        branch_gate = "✅ PASS" if overall.branch_rate >= self.min_branch_coverage else "❌ FAIL"
        
        report.append(f"\n🚪 QUALITY GATES:")
        report.append(f"   Line Coverage Gate ({self.min_coverage}%): {line_gate}")
        report.append(f"   Branch Coverage Gate ({self.min_branch_coverage}%): {branch_gate}")
        
        # Coverage gaps
        if gaps:
            report.append(f"\n🔍 COVERAGE GAPS ({len(gaps)}):")
            
            low_coverage_gaps = [g for g in gaps if g['type'] == 'low_coverage']
            if low_coverage_gaps:
                report.append(f"\n   📉 Low Line Coverage:")
                for gap in low_coverage_gaps[:10]:  # Show first 10
                    report.append(f"      • {gap['filename']} - {gap['coverage']:.2f}% "
                                 f"(missing: {len(gap['missing_lines'])} lines)")
                if len(low_coverage_gaps) > 10:
                    report.append(f"      ... and {len(low_coverage_gaps) - 10} more files")
            
            low_branch_gaps = [g for g in gaps if g['type'] == 'low_branch_coverage']
            if low_branch_gaps:
                report.append(f"\n   🌿 Low Branch Coverage:")
                for gap in low_branch_gaps[:10]:  # Show first 10
                    report.append(f"      • {gap['filename']} - {gap['branch_coverage']:.2f}%")
                if len(low_branch_gaps) > 10:
                    report.append(f"      ... and {len(low_branch_gaps) - 10} more files")
        
        # Package breakdown
        if 'packages' in coverage_data:
            report.append(f"\n📦 PACKAGE COVERAGE:")
            for package in coverage_data['packages']:
                status_icon = "✅" if package['line_rate'] >= self.min_coverage else "❌"
                report.append(f"   {status_icon} {package['name']}: {package['line_rate']:.2f}% lines, "
                             f"{package['branch_rate']:.2f}% branches")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        if overall.line_rate < self.min_coverage:
            report.append(f"   • Increase line coverage to at least {self.min_coverage}%")
        if overall.branch_rate < self.min_branch_coverage:
            report.append(f"   • Improve branch coverage to at least {self.min_branch_coverage}%")
        if len(gaps) > 0:
            report.append("   • Focus testing efforts on files with low coverage")
            report.append("   • Add integration tests for complex workflows")
        if len([g for g in gaps if g['type'] == 'low_branch_coverage']) > 0:
            report.append("   • Add tests for error handling and edge cases")
        if overall.line_rate >= self.min_coverage and overall.branch_rate >= self.min_branch_coverage:
            report.append("   • Coverage meets quality standards ✅")
            report.append("   • Consider adding property-based tests for robustness")
        
        return "\n".join(report)
    
    def check_quality_gates(self, coverage_data: Dict[str, Any]) -> bool:
        """Check if coverage meets quality gates."""
        overall = coverage_data['overall']
        
        line_coverage_pass = overall.line_rate >= self.min_coverage
        branch_coverage_pass = overall.branch_rate >= self.min_branch_coverage
        
        if not line_coverage_pass:
            print(f"❌ Line coverage quality gate FAILED: {overall.line_rate:.2f}% < {self.min_coverage}%")
        
        if not branch_coverage_pass:
            print(f"❌ Branch coverage quality gate FAILED: {overall.branch_rate:.2f}% < {self.min_branch_coverage}%")
        
        if line_coverage_pass and branch_coverage_pass:
            print(f"✅ Test coverage quality gates PASSED")
            print(f"   Line Coverage: {overall.line_rate:.2f}%")
            print(f"   Branch Coverage: {overall.branch_rate:.2f}%")
        
        return line_coverage_pass and branch_coverage_pass
    
    def run_analysis(self) -> bool:
        """Run complete coverage analysis."""
        # Run tests with coverage
        if not self.run_coverage_analysis():
            return False
        
        # Parse coverage results
        coverage_data = self.parse_coverage_xml()
        if not coverage_data:
            print("❌ Could not parse coverage data")
            return False
        
        # Parse additional JSON data if available
        json_data = self.parse_coverage_json()
        if json_data:
            coverage_data['json_details'] = json_data
        
        # Identify coverage gaps
        gaps = self.identify_coverage_gaps(coverage_data)
        
        # Generate report
        report = self.generate_coverage_report(coverage_data, gaps)
        print(report)
        
        # Save detailed results
        with open("coverage-analysis.json", "w") as f:
            json.dump({
                'coverage_data': coverage_data,
                'gaps': gaps,
                'quality_gates_passed': self.check_quality_gates(coverage_data)
            }, f, indent=2, default=str)
        
        # Check quality gates
        return self.check_quality_gates(coverage_data)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test coverage verification")
    parser.add_argument("--min-coverage", type=float, default=85.0, help="Minimum line coverage percentage")
    parser.add_argument("--min-branch-coverage", type=float, default=75.0, help="Minimum branch coverage percentage")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests, analyze existing coverage")
    
    args = parser.parse_args()
    
    analyzer = CoverageAnalyzer(
        min_coverage=args.min_coverage,
        min_branch_coverage=args.min_branch_coverage
    )
    
    try:
        if args.skip_tests:
            # Just analyze existing coverage data
            coverage_data = analyzer.parse_coverage_xml()
            if not coverage_data:
                print("❌ No existing coverage data found")
                sys.exit(1)
            
            gaps = analyzer.identify_coverage_gaps(coverage_data)
            report = analyzer.generate_coverage_report(coverage_data, gaps)
            print(report)
            
            success = analyzer.check_quality_gates(coverage_data)
        else:
            # Run full analysis
            success = analyzer.run_analysis()
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Coverage analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()