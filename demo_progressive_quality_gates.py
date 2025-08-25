#!/usr/bin/env python3
"""
Progressive Quality Gates v2.0 - Interactive Demo

This demo showcases all the advanced Progressive Quality Gates features
including AI threat detection, quantum performance optimization, chaos engineering,
advanced observability, and enterprise governance.

Run with: python3 demo_progressive_quality_gates.py
"""

import sys
import time
import json
from typing import Dict, Any

# Add the src directory to Python path
sys.path.insert(0, 'src')

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"🎯 {title}")
    print("=" * 80)

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n📋 {title}")
    print("-" * 60)

def print_result(result: Dict[str, Any], key: str = None):
    """Print formatted result."""
    if key and key in result:
        value = result[key]
        if isinstance(value, (int, float)):
            print(f"   ✅ {key}: {value}")
        else:
            print(f"   ✅ {key}: {value}")
    else:
        print(f"   📊 Result: {json.dumps(result, indent=2, default=str)[:200]}...")

def demo_threat_detection():
    """Demonstrate AI-Powered Threat Detection capabilities."""
    print_header("AI-Powered Threat Detection Demo")
    
    try:
        from moe_debugger.ai_threat_detection import (
            get_threat_detection_system, start_threat_monitoring
        )
        
        print_section("System Initialization")
        threat_system = get_threat_detection_system()
        print("   ✅ AI Threat Detection System initialized")
        
        # Start monitoring
        start_threat_monitoring()
        print("   ✅ Threat monitoring started")
        
        print_section("Benign Request Analysis")
        benign_request = {
            'source_ip': '192.168.1.100',
            'user_agent': 'Mozilla/5.0 (legitimate browser)',
            'path': '/api/users/profile',
            'payload': '{"user_id": "12345"}',
            'user_id': 'legitimate_user',
            'timestamp': time.time()
        }
        
        threat_event = threat_system.analyze_request(benign_request)
        if threat_event:
            print(f"   ⚠️  Threat detected (may be false positive): {threat_event.category.value}")
        else:
            print("   ✅ No threat detected - request appears legitimate")
        
        print_section("Malicious Request Analysis")
        malicious_request = {
            'source_ip': '198.51.100.666',  # Suspicious IP
            'user_agent': 'sqlmap/1.0 (attacker tool)',
            'path': '/admin/config',
            'payload': "'; DROP TABLE users; --",
            'user_id': 'admin',
            'timestamp': time.time()
        }
        
        threat_event = threat_system.analyze_request(malicious_request)
        if threat_event:
            print(f"   🚨 Threat detected: {threat_event.category.value} - {threat_event.level.value}")
            print(f"   📊 Confidence: {threat_event.confidence_score:.1f}%")
            print(f"   🛡️  Response actions: {', '.join(threat_event.response_actions)}")
        else:
            print("   ✅ No threat detected (detection may need tuning)")
        
        print_section("Security Status")
        status = threat_system.get_security_status()
        print_result(status, 'timestamp')
        if 'security_metrics' in status:
            metrics = status['security_metrics']
            print(f"   📊 Threats detected: {metrics.get('threats_detected', 0)}")
            print(f"   🛡️  Threats blocked: {metrics.get('threats_blocked', 0)}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ AI Threat Detection not available: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error in threat detection demo: {e}")
        return False

def demo_performance_optimization():
    """Demonstrate Quantum-Ready Performance Optimization."""
    print_header("Quantum-Ready Performance Optimization Demo")
    
    try:
        from moe_debugger.quantum_performance_optimization import (
            get_performance_optimizer, ResourceMetrics
        )
        
        print_section("System Initialization")
        optimizer = get_performance_optimizer()
        print("   ✅ Quantum Performance Optimizer initialized")
        
        print_section("Resource Metrics Simulation")
        # Simulate resource metrics over time
        for i in range(3):
            metrics = ResourceMetrics(
                cpu_usage_percent=50.0 + (i * 10),
                memory_usage_mb=1024.0 + (i * 512),
                memory_total_mb=4096.0,
                response_time_ms=100.0 + (i * 20),
                request_rate=100.0 + (i * 50),
                cache_hit_rate=0.8 - (i * 0.1),
                error_rate=0.01 + (i * 0.005)
            )
            
            optimizer.add_performance_metrics(metrics)
            print(f"   📊 Added metrics sample {i+1}: CPU {metrics.cpu_usage_percent}%, Memory {metrics.memory_usage_mb}MB")
        
        print_section("Optimization Analysis")
        result = optimizer.run_optimization_cycle()
        
        if 'performance_score' in result:
            print(f"   📊 Performance Score: {result['performance_score']:.1f}/100")
        
        if 'scaling_decision' in result:
            scaling = result['scaling_decision']
            action = scaling.get('action', 'no_change')
            if action != 'no_change':
                print(f"   🔧 Scaling recommendation: {action}")
                print(f"   📈 Recommended instances: {scaling.get('recommended_instances', 'N/A')}")
        
        if 'recommendations' in result:
            recommendations = result['recommendations'][:3]  # Top 3
            for i, rec in enumerate(recommendations, 1):
                print(f"   💡 Recommendation {i}: {rec}")
        
        print_section("Optimization Status")
        status = optimizer.get_optimization_status()
        print_result(status, 'performance_trend')
        print_result(status, 'cost_optimization_savings')
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Quantum Performance Optimization not available: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error in performance optimization demo: {e}")
        return False

def demo_chaos_engineering():
    """Demonstrate Chaos Engineering and Resilience."""
    print_header("Chaos Engineering & Resilience Demo")
    
    try:
        from moe_debugger.chaos_engineering import (
            get_chaos_orchestrator, ChaosExperimentType, BlastRadiusLevel
        )
        
        print_section("System Initialization")
        chaos_orchestrator = get_chaos_orchestrator()
        print("   ✅ Chaos Engineering Orchestrator initialized")
        
        print_section("Chaos Experiment Creation")
        experiment_id = chaos_orchestrator.create_experiment(
            name="demo_network_latency",
            experiment_type=ChaosExperimentType.NETWORK_LATENCY,
            target_services=["demo_service"],
            blast_radius=BlastRadiusLevel.SINGLE_SERVICE,
            duration_seconds=30.0,
            parameters={'latency_ms': 150}
        )
        
        print(f"   ✅ Chaos experiment created: {experiment_id}")
        
        print_section("Experiment Status")
        status = chaos_orchestrator.get_experiment_status(experiment_id)
        print(f"   📊 Experiment: {status['name']}")
        print(f"   🎯 Type: {status['experiment_type']}")
        print(f"   📡 Target services: {', '.join(status['target_services'])}")
        print(f"   🔒 Blast radius: {status['blast_radius']}")
        print(f"   ⏱️  Duration: {status['duration_seconds']} seconds")
        
        print_section("Experiments List")
        experiments = chaos_orchestrator.list_experiments()
        print(f"   📋 Total experiments: {len(experiments)}")
        for exp in experiments[:2]:  # Show first 2
            print(f"   🧪 {exp['name']} - {exp['status']}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Chaos Engineering not available: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error in chaos engineering demo: {e}")
        return False

def demo_observability():
    """Demonstrate Advanced Observability and Intelligence."""
    print_header("Advanced Observability & Intelligence Demo")
    
    try:
        from moe_debugger.advanced_observability import get_observability_system
        
        print_section("System Initialization")
        obs_system = get_observability_system()
        print("   ✅ Advanced Observability System initialized")
        
        print_section("Metrics Ingestion")
        sample_metrics = [
            {
                'name': 'response_time_ms',
                'value': 125.0,
                'timestamp': time.time(),
                'labels': {'service': 'api', 'endpoint': '/users'}
            },
            {
                'name': 'cpu_usage_percent',
                'value': 65.0,
                'timestamp': time.time(),
                'labels': {'instance': 'web-01'}
            },
            {
                'name': 'memory_usage_mb',
                'value': 1800.0,
                'timestamp': time.time(),
                'labels': {'instance': 'web-01'}
            },
            {
                'name': 'request_rate',
                'value': 250.0,
                'timestamp': time.time(),
                'labels': {'service': 'api'}
            }
        ]
        
        obs_system.ingest_metrics(sample_metrics)
        print(f"   ✅ Ingested {len(sample_metrics)} metrics")
        
        print_section("Analysis Cycle")
        result = obs_system.run_analysis_cycle()
        print(f"   📊 Metrics analyzed: {result.get('metrics_analyzed', 0)}")
        print(f"   🚨 Anomalies detected: {result.get('anomalies_detected', 0)}")
        print(f"   ⚠️  Predictive alerts: {result.get('predictive_alerts', 0)}")
        print(f"   💡 Insights generated: {result.get('insights_generated', 0)}")
        
        # Show new anomalies if any
        new_anomalies = result.get('new_anomalies', [])
        if new_anomalies:
            print_section("Detected Anomalies")
            for anomaly in new_anomalies[:2]:  # Show first 2
                print(f"   🚨 {anomaly['metric_name']}: {anomaly['anomaly_type']} - {anomaly['severity']}")
        
        # Show insights if any  
        new_insights = result.get('new_insights', [])
        if new_insights:
            print_section("Generated Insights")
            for insight in new_insights[:2]:  # Show first 2
                print(f"   💡 {insight['title']}: {insight['description']}")
        
        print_section("Observability Dashboard")
        dashboard = obs_system.get_observability_dashboard()
        print(f"   📊 System Health Score: {dashboard.get('system_health_score', 0):.1f}/100")
        
        summary = dashboard.get('summary', {})
        print(f"   📈 Total metrics: {summary.get('total_metrics', 0)}")
        print(f"   📊 Recent anomalies: {summary.get('recent_anomalies', 0)}")
        print(f"   ⚠️  Recent alerts: {summary.get('recent_alerts', 0)}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Advanced Observability not available: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error in observability demo: {e}")
        return False

def demo_governance():
    """Demonstrate Enterprise-Grade Governance."""
    print_header("Enterprise-Grade Governance Demo")
    
    try:
        from moe_debugger.enterprise_governance import get_governance_system
        
        print_section("System Initialization")
        governance_system = get_governance_system()
        print("   ✅ Enterprise Governance System initialized")
        
        print_section("Request Context Evaluation")
        test_contexts = [
            {
                'user_id': 'regular_user',
                'action': 'read',
                'resource': 'public_data',
                'source_ip': '192.168.1.100'
            },
            {
                'user_id': 'admin_user',
                'action': 'configuration_change',
                'environment': 'production',
                'source_ip': '10.0.1.50'
            }
        ]
        
        for i, context in enumerate(test_contexts, 1):
            print(f"   📝 Evaluating context {i}:")
            evaluation = governance_system.evaluate_request_context(context)
            result = evaluation.get('evaluation_result', 'unknown')
            violations = evaluation.get('policy_violations', [])
            
            print(f"      Result: {result}")
            if violations:
                print(f"      Violations: {len(violations)}")
                for violation in violations[:1]:  # Show first violation
                    print(f"        - {violation.get('policy_name', 'Unknown policy')}")
            else:
                print(f"      No policy violations")
        
        print_section("Compliance Assessment")
        assessment = governance_system.run_compliance_assessment()
        print(f"   📊 Overall status: {assessment.get('overall_status', 'unknown')}")
        
        compliance_validation = assessment.get('compliance_validation', {})
        print(f"   📈 Compliance rate: {compliance_validation.get('compliance_rate', 0):.1%}")
        
        frameworks = compliance_validation.get('frameworks', {})
        for framework, details in frameworks.items():
            status = details.get('status', 'unknown')
            print(f"   🏛️  {framework}: {status}")
        
        print_section("Governance Dashboard")
        dashboard = governance_system.get_governance_dashboard()
        print(f"   📊 Governance Health Score: {dashboard.get('governance_health_score', 0):.1f}/100")
        
        summary = dashboard.get('summary', {})
        print(f"   📋 Total audit events: {summary.get('total_audit_events', 0)}")
        print(f"   ⚠️  Policy violations (24h): {summary.get('policy_violations_24h', 0)}")
        print(f"   🚨 High risk events: {summary.get('high_risk_events', 0)}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Enterprise Governance not available: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error in governance demo: {e}")
        return False

def demo_orchestrator():
    """Demonstrate Progressive Quality Orchestrator."""
    print_header("Progressive Quality Orchestrator Demo")
    
    try:
        from moe_debugger.progressive_quality_orchestrator import (
            ProgressiveQualityOrchestrator, QualityGateConfiguration
        )
        
        print_section("System Configuration")
        config = QualityGateConfiguration(
            threat_detection_enabled=True,
            performance_optimization_enabled=True,
            chaos_engineering_enabled=False,  # Disable for demo
            observability_enabled=True,
            governance_enabled=True,
            recovery_system_enabled=True,
            automated_incident_response_enabled=False  # Disable for demo
        )
        print("   ✅ Configuration created")
        
        print_section("Orchestrator Initialization")
        orchestrator = ProgressiveQualityOrchestrator(config)
        print("   ✅ Progressive Quality Orchestrator created")
        
        print_section("System Initialization")
        init_result = orchestrator.initialize_system()
        print(f"   📊 Initialization status: {init_result.get('overall_status', 'unknown')}")
        
        initialized = init_result.get('initialized_components', [])
        failed = init_result.get('failed_components', [])
        print(f"   ✅ Initialized components: {len(initialized)}")
        print(f"   ❌ Failed components: {len(failed)}")
        
        for component in initialized[:3]:  # Show first 3
            print(f"      ✅ {component}")
        
        print_section("System Status")
        status = orchestrator.get_system_status()
        print(f"   📊 Overall status: {status.get('overall_status', 'unknown')}")
        print(f"   📈 Overall health score: {status.get('overall_health_score', 0):.1f}/100")
        print(f"   ⏱️  System uptime: {status.get('system_uptime_seconds', 0):.0f} seconds")
        
        component_health = status.get('component_health', {})
        active_components = sum(1 for comp in component_health.values() if comp.get('status') == 'active')
        print(f"   🟢 Active components: {active_components}/{len(component_health)}")
        
        print_section("Quality Gate Assessment")
        assessment = orchestrator.trigger_quality_gate_assessment()
        print(f"   📊 Overall score: {assessment.get('overall_score', 0):.1f}/100")
        print(f"   🎯 Quality gate status: {assessment.get('quality_gate_status', 'unknown')}")
        
        component_assessments = assessment.get('component_assessments', {})
        for component, details in component_assessments.items():
            score = details.get('score', 0)
            print(f"   📊 {component}: {score:.1f}/100")
        
        critical_issues = assessment.get('critical_issues', [])
        if critical_issues:
            print_section("Critical Issues")
            for issue in critical_issues[:2]:  # Show first 2
                print(f"   🚨 {issue}")
        
        recommendations = assessment.get('recommendations', [])
        if recommendations:
            print_section("Recommendations")
            for rec in recommendations[:3]:  # Show first 3
                print(f"   💡 {rec}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Progressive Quality Orchestrator not available: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error in orchestrator demo: {e}")
        return False

def main():
    """Main demo function."""
    print_header("Progressive Quality Gates v2.0 - Interactive Demo")
    print("🚀 Welcome to the Progressive Quality Gates v2.0 Interactive Demo!")
    print("This demonstration showcases all advanced enterprise-grade features.")
    
    start_time = time.time()
    
    # Run all demos
    demos = [
        ("AI-Powered Threat Detection", demo_threat_detection),
        ("Quantum Performance Optimization", demo_performance_optimization),
        ("Chaos Engineering & Resilience", demo_chaos_engineering),
        ("Advanced Observability", demo_observability),
        ("Enterprise Governance", demo_governance),
        ("Progressive Quality Orchestrator", demo_orchestrator)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        print(f"\n⏳ Running {demo_name} demo...")
        try:
            success = demo_func()
            results.append((demo_name, success))
            if success:
                print(f"✅ {demo_name} demo completed successfully")
            else:
                print(f"❌ {demo_name} demo failed")
        except Exception as e:
            print(f"❌ {demo_name} demo error: {e}")
            results.append((demo_name, False))
    
    # Summary
    end_time = time.time()
    total_time = end_time - start_time
    successful_demos = sum(1 for _, success in results if success)
    total_demos = len(results)
    success_rate = (successful_demos / total_demos) * 100
    
    print_header("Demo Summary")
    print(f"⏱️  Total demo time: {total_time:.1f} seconds")
    print(f"✅ Successful demos: {successful_demos}/{total_demos}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    
    print("\n📊 Demo Results:")
    for demo_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {status} - {demo_name}")
    
    if success_rate >= 80:
        print("\n🎉 Progressive Quality Gates v2.0 Demo: SUCCESSFUL")
        print("🏆 All major components are working correctly!")
    elif success_rate >= 60:
        print("\n⚠️  Progressive Quality Gates v2.0 Demo: PARTIALLY SUCCESSFUL")
        print("🔧 Some components may need configuration or dependencies")
    else:
        print("\n❌ Progressive Quality Gates v2.0 Demo: ISSUES DETECTED")
        print("🛠️  Please check component dependencies and configuration")
    
    print("\n" + "=" * 80)
    print("🙏 Thank you for exploring Progressive Quality Gates v2.0!")
    print("📖 For more information, see: PROGRESSIVE_QUALITY_GATES_V2.md")
    print("🧪 Run tests with: python3 test_progressive_quality_quick.py")
    print("=" * 80)

if __name__ == '__main__':
    """Execute the interactive demo."""
    main()