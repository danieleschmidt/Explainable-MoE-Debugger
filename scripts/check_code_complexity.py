#!/usr/bin/env python3
"""
Code complexity analysis for progressive quality gates.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
from dataclasses import dataclass


@dataclass
class ComplexityMetrics:
    """Code complexity metrics."""
    cyclomatic: int = 0
    cognitive: int = 0
    lines_of_code: int = 0
    maintainability_index: float = 0.0


class ComplexityAnalyzer(ast.NodeVisitor):
    """AST-based code complexity analyzer."""
    
    def __init__(self):
        self.complexity = 1  # Base complexity
        self.cognitive_complexity = 0
        self.nesting_level = 0
        self.lines_of_code = 0
    
    def visit_FunctionDef(self, node):
        """Visit function definition."""
        self.complexity += 1
        self.lines_of_code += len(node.body)
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition."""
        self.complexity += 1
        self.lines_of_code += len(node.body)
        self.generic_visit(node)
    
    def visit_If(self, node):
        """Visit if statement."""
        self.complexity += 1
        self.cognitive_complexity += 1 + self.nesting_level
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1
    
    def visit_While(self, node):
        """Visit while loop."""
        self.complexity += 1
        self.cognitive_complexity += 1 + self.nesting_level
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1
    
    def visit_For(self, node):
        """Visit for loop."""
        self.complexity += 1
        self.cognitive_complexity += 1 + self.nesting_level
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1
    
    def visit_ExceptHandler(self, node):
        """Visit exception handler."""
        self.complexity += 1
        self.cognitive_complexity += 1 + self.nesting_level
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1
    
    def visit_With(self, node):
        """Visit with statement."""
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_Assert(self, node):
        """Visit assert statement."""
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_BoolOp(self, node):
        """Visit boolean operation (and/or)."""
        self.complexity += len(node.values) - 1
        self.generic_visit(node)
    
    def visit_ListComp(self, node):
        """Visit list comprehension."""
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_DictComp(self, node):
        """Visit dict comprehension."""
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_SetComp(self, node):
        """Visit set comprehension."""
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_GeneratorExp(self, node):
        """Visit generator expression."""
        self.complexity += 1
        self.generic_visit(node)


class QualityGateComplexityChecker:
    """Progressive quality gates complexity checker."""
    
    def __init__(self, max_cyclomatic=10, max_cognitive=15, max_function_lines=50):
        self.max_cyclomatic = max_cyclomatic
        self.max_cognitive = max_cognitive
        self.max_function_lines = max_function_lines
        self.violations = []
        self.warnings = []
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            # Overall file metrics
            analyzer = ComplexityAnalyzer()
            analyzer.visit(tree)
            
            file_metrics = ComplexityMetrics(
                cyclomatic=analyzer.complexity,
                cognitive=analyzer.cognitive_complexity,
                lines_of_code=content.count('\n') + 1
            )
            
            # Function-level analysis
            function_metrics = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_analyzer = ComplexityAnalyzer()
                    func_analyzer.visit(node)
                    
                    func_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else len(node.body)
                    
                    func_metrics = {
                        'name': node.name,
                        'line': node.lineno,
                        'cyclomatic': func_analyzer.complexity,
                        'cognitive': func_analyzer.cognitive_complexity,
                        'lines': func_lines
                    }
                    
                    # Check for violations
                    if func_metrics['cyclomatic'] > self.max_cyclomatic:
                        self.violations.append(
                            f"{file_path}:{node.lineno} - Function '{node.name}' has cyclomatic "
                            f"complexity {func_metrics['cyclomatic']} (max: {self.max_cyclomatic})"
                        )
                    
                    if func_metrics['cognitive'] > self.max_cognitive:
                        self.violations.append(
                            f"{file_path}:{node.lineno} - Function '{node.name}' has cognitive "
                            f"complexity {func_metrics['cognitive']} (max: {self.max_cognitive})"
                        )
                    
                    if func_metrics['lines'] > self.max_function_lines:
                        self.warnings.append(
                            f"{file_path}:{node.lineno} - Function '{node.name}' has "
                            f"{func_metrics['lines']} lines (recommended max: {self.max_function_lines})"
                        )
                    
                    function_metrics.append(func_metrics)
            
            # Calculate maintainability index (simplified version)
            # MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)
            # Simplified: MI = 100 - (CC * 2) - (LOC / 10)
            halstead_volume = file_metrics.lines_of_code * 0.5  # Simplified
            maintainability_index = max(0, 100 - (file_metrics.cyclomatic * 2) - (file_metrics.lines_of_code / 10))
            file_metrics.maintainability_index = maintainability_index
            
            return {
                'file_path': str(file_path),
                'file_metrics': file_metrics,
                'function_metrics': function_metrics
            }
            
        except SyntaxError as e:
            self.violations.append(f"{file_path} - Syntax error: {e}")
            return None
        except Exception as e:
            self.warnings.append(f"{file_path} - Analysis error: {e}")
            return None
    
    def analyze_project(self, project_path: Path) -> Dict[str, Any]:
        """Analyze entire project for complexity."""
        print("🧠 Analyzing code complexity...")
        
        python_files = list(project_path.rglob("*.py"))
        
        # Filter out certain directories
        exclude_patterns = ['__pycache__', '.git', '.venv', 'venv', 'env', 'node_modules']
        python_files = [
            f for f in python_files 
            if not any(pattern in str(f) for pattern in exclude_patterns)
        ]
        
        results = []
        total_complexity = 0
        total_lines = 0
        high_complexity_functions = []
        
        for file_path in python_files:
            file_result = self.analyze_file(file_path)
            if file_result:
                results.append(file_result)
                
                file_metrics = file_result['file_metrics']
                total_complexity += file_metrics.cyclomatic
                total_lines += file_metrics.lines_of_code
                
                # Track high complexity functions
                for func in file_result['function_metrics']:
                    if func['cyclomatic'] > 8 or func['cognitive'] > 12:
                        high_complexity_functions.append({
                            'file': str(file_path),
                            'function': func['name'],
                            'line': func['line'],
                            'cyclomatic': func['cyclomatic'],
                            'cognitive': func['cognitive']
                        })
        
        # Project-level metrics
        avg_complexity = total_complexity / len(results) if results else 0
        avg_maintainability = sum(r['file_metrics'].maintainability_index for r in results) / len(results) if results else 0
        
        return {
            'summary': {
                'files_analyzed': len(results),
                'total_lines': total_lines,
                'average_complexity': avg_complexity,
                'average_maintainability': avg_maintainability,
                'high_complexity_functions': len(high_complexity_functions)
            },
            'files': results,
            'high_complexity_functions': high_complexity_functions,
            'violations': self.violations,
            'warnings': self.warnings
        }
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate complexity analysis report."""
        summary = analysis_results['summary']
        violations = analysis_results['violations']
        warnings = analysis_results['warnings']
        high_complexity = analysis_results['high_complexity_functions']
        
        report = []
        report.append("🧠 CODE COMPLEXITY ANALYSIS REPORT")
        report.append("=" * 50)
        
        # Summary
        report.append(f"\n📊 SUMMARY:")
        report.append(f"   Files Analyzed: {summary['files_analyzed']}")
        report.append(f"   Total Lines: {summary['total_lines']:,}")
        report.append(f"   Average Complexity: {summary['average_complexity']:.2f}")
        report.append(f"   Average Maintainability: {summary['average_maintainability']:.2f}")
        report.append(f"   High Complexity Functions: {summary['high_complexity_functions']}")
        
        # Quality assessment
        quality_score = 100
        if summary['average_complexity'] > 8:
            quality_score -= 20
        if summary['average_maintainability'] < 70:
            quality_score -= 20
        if len(violations) > 0:
            quality_score -= 30
        
        status = "EXCELLENT" if quality_score >= 90 else "GOOD" if quality_score >= 75 else "NEEDS_IMPROVEMENT" if quality_score >= 60 else "CRITICAL"
        
        report.append(f"\n🎯 QUALITY SCORE: {quality_score}/100 - {status}")
        
        # Violations
        if violations:
            report.append(f"\n❌ COMPLEXITY VIOLATIONS ({len(violations)}):")
            for violation in violations[:10]:  # Show first 10
                report.append(f"   • {violation}")
            if len(violations) > 10:
                report.append(f"   ... and {len(violations) - 10} more")
        
        # Warnings
        if warnings:
            report.append(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for warning in warnings[:5]:  # Show first 5
                report.append(f"   • {warning}")
            if len(warnings) > 5:
                report.append(f"   ... and {len(warnings) - 5} more")
        
        # High complexity functions
        if high_complexity:
            report.append(f"\n🔍 HIGH COMPLEXITY FUNCTIONS:")
            for func in high_complexity[:10]:  # Show first 10
                report.append(f"   • {func['file']}:{func['line']} - {func['function']} "
                             f"(CC: {func['cyclomatic']}, Cognitive: {func['cognitive']})")
            if len(high_complexity) > 10:
                report.append(f"   ... and {len(high_complexity) - 10} more")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        if summary['average_complexity'] > 8:
            report.append("   • Refactor complex functions to reduce cyclomatic complexity")
        if summary['average_maintainability'] < 70:
            report.append("   • Improve code maintainability by simplifying logic")
        if len(violations) > 0:
            report.append("   • Address complexity violations before merging")
        if summary['high_complexity_functions'] > 5:
            report.append("   • Consider breaking down large functions into smaller ones")
        if len(violations) == 0 and len(warnings) == 0:
            report.append("   • Code complexity is within acceptable limits ✅")
        
        return "\n".join(report)
    
    def check_quality_gates(self, analysis_results: Dict[str, Any]) -> bool:
        """Check if code complexity meets quality gates."""
        violations = analysis_results['violations']
        summary = analysis_results['summary']
        
        # Quality gate criteria
        gate_passed = True
        
        # No critical complexity violations
        if len(violations) > 0:
            print(f"❌ Complexity quality gate FAILED: {len(violations)} violations")
            gate_passed = False
        
        # Average complexity within limits
        if summary['average_complexity'] > 12:
            print(f"❌ Complexity quality gate FAILED: Average complexity {summary['average_complexity']:.2f} > 12")
            gate_passed = False
        
        # Maintainability above threshold
        if summary['average_maintainability'] < 60:
            print(f"❌ Complexity quality gate FAILED: Maintainability {summary['average_maintainability']:.2f} < 60")
            gate_passed = False
        
        # Too many high complexity functions
        if summary['high_complexity_functions'] > 10:
            print(f"❌ Complexity quality gate FAILED: {summary['high_complexity_functions']} high complexity functions > 10")
            gate_passed = False
        
        if gate_passed:
            print("✅ Code complexity quality gate PASSED")
        
        return gate_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Code complexity analysis")
    parser.add_argument("--path", type=Path, default=Path("."), help="Project path to analyze")
    parser.add_argument("--max-cyclomatic", type=int, default=10, help="Maximum cyclomatic complexity")
    parser.add_argument("--max-cognitive", type=int, default=15, help="Maximum cognitive complexity")
    parser.add_argument("--max-function-lines", type=int, default=50, help="Maximum function lines")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    
    args = parser.parse_args()
    
    checker = QualityGateComplexityChecker(
        max_cyclomatic=args.max_cyclomatic,
        max_cognitive=args.max_cognitive,
        max_function_lines=args.max_function_lines
    )
    
    try:
        # Analyze project
        analysis_results = checker.analyze_project(args.path)
        
        # Generate report
        if args.report:
            report = checker.generate_report(analysis_results)
            print(report)
        
        # Check quality gates
        success = checker.check_quality_gates(analysis_results)
        
        # Save results
        import json
        with open("complexity-analysis.json", "w") as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Complexity analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()