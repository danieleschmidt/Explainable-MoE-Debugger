#!/usr/bin/env python3
"""
Quick Validation Test for Progressive Quality Gates

This test performs basic validation of Progressive Quality Gates components
to ensure they can be imported and initialized correctly.

Run with: python test_progressive_quality_quick.py
"""

import sys
import time
sys.path.insert(0, 'src')

def test_component_imports():
    """Test that all Progressive Quality Gates components can be imported."""
    print("🔍 Testing Progressive Quality Gates component imports...")
    
    results = {
        'threat_detection': False,
        'performance_optimization': False,
        'chaos_engineering': False,
        'observability': False,
        'governance': False,
        'orchestrator': False,
        'recovery': False
    }
    
    # Test AI Threat Detection
    try:
        from moe_debugger.ai_threat_detection import AIThreatDetectionSystem
        system = AIThreatDetectionSystem()
        results['threat_detection'] = True
        print("✅ AI Threat Detection: OK")
    except Exception as e:
        print(f"❌ AI Threat Detection: {e}")
    
    # Test Quantum Performance Optimization
    try:
        from moe_debugger.quantum_performance_optimization import QuantumPerformanceOptimizer
        system = QuantumPerformanceOptimizer()
        results['performance_optimization'] = True
        print("✅ Quantum Performance Optimization: OK")
    except Exception as e:
        print(f"❌ Quantum Performance Optimization: {e}")
    
    # Test Chaos Engineering
    try:
        from moe_debugger.autonomous_recovery import AutonomousRecoverySystem
        from moe_debugger.chaos_engineering import ChaosEngineeringOrchestrator
        recovery = AutonomousRecoverySystem()
        system = ChaosEngineeringOrchestrator(recovery)
        results['chaos_engineering'] = True
        print("✅ Chaos Engineering: OK")
    except Exception as e:
        print(f"❌ Chaos Engineering: {e}")
    
    # Test Advanced Observability
    try:
        from moe_debugger.advanced_observability import AdvancedObservabilitySystem
        system = AdvancedObservabilitySystem()
        results['observability'] = True
        print("✅ Advanced Observability: OK")
    except Exception as e:
        print(f"❌ Advanced Observability: {e}")
    
    # Test Enterprise Governance
    try:
        from moe_debugger.enterprise_governance import EnterpriseGovernanceSystem
        system = EnterpriseGovernanceSystem()
        results['governance'] = True
        print("✅ Enterprise Governance: OK")
    except Exception as e:
        print(f"❌ Enterprise Governance: {e}")
    
    # Test Autonomous Recovery
    try:
        from moe_debugger.autonomous_recovery import AutonomousRecoverySystem
        system = AutonomousRecoverySystem()
        results['recovery'] = True
        print("✅ Autonomous Recovery: OK")
    except Exception as e:
        print(f"❌ Autonomous Recovery: {e}")
    
    # Test Progressive Quality Orchestrator
    try:
        from moe_debugger.progressive_quality_orchestrator import ProgressiveQualityOrchestrator
        system = ProgressiveQualityOrchestrator()
        results['orchestrator'] = True
        print("✅ Progressive Quality Orchestrator: OK")
    except Exception as e:
        print(f"❌ Progressive Quality Orchestrator: {e}")
    
    return results


def test_basic_functionality():
    """Test basic functionality of available components."""
    print("\n🧪 Testing basic functionality...")
    
    functional_tests = 0
    total_tests = 0
    
    # Test threat detection basic functionality
    try:
        from moe_debugger.ai_threat_detection import get_threat_detection_system
        system = get_threat_detection_system()
        
        # Test basic request analysis
        test_data = {
            'source_ip': '192.168.1.100',
            'user_agent': 'TestAgent/1.0',
            'path': '/test',
            'payload': 'test'
        }
        
        result = system.analyze_request(test_data)
        # Should return None for benign request or a threat event
        print("✅ Threat Detection: Basic analysis working")
        functional_tests += 1
        total_tests += 1
        
    except Exception as e:
        print(f"❌ Threat Detection: Basic functionality failed - {e}")
        total_tests += 1
    
    # Test performance optimization basic functionality  
    try:
        from moe_debugger.quantum_performance_optimization import get_performance_optimizer
        system = get_performance_optimizer()
        
        # Test basic status
        status = system.get_optimization_status()
        if isinstance(status, dict):
            print("✅ Performance Optimization: Status retrieval working")
            functional_tests += 1
        total_tests += 1
        
    except Exception as e:
        print(f"❌ Performance Optimization: Basic functionality failed - {e}")
        total_tests += 1
    
    # Test observability basic functionality
    try:
        from moe_debugger.advanced_observability import get_observability_system
        system = get_observability_system()
        
        # Test basic metrics ingestion
        test_metrics = [{
            'name': 'test_metric',
            'value': 42.0,
            'timestamp': time.time()
        }]
        
        system.ingest_metrics(test_metrics)
        print("✅ Advanced Observability: Metrics ingestion working")
        functional_tests += 1
        total_tests += 1
        
    except Exception as e:
        print(f"❌ Advanced Observability: Basic functionality failed - {e}")
        total_tests += 1
    
    # Test governance basic functionality
    try:
        from moe_debugger.enterprise_governance import get_governance_system
        system = get_governance_system()
        
        # Test basic context evaluation
        context = {
            'user_id': 'test_user',
            'action': 'read',
            'resource': 'test_resource'
        }
        
        result = system.evaluate_request_context(context)
        if isinstance(result, dict) and 'evaluation_result' in result:
            print("✅ Enterprise Governance: Context evaluation working")
            functional_tests += 1
        total_tests += 1
        
    except Exception as e:
        print(f"❌ Enterprise Governance: Basic functionality failed - {e}")
        total_tests += 1
    
    return functional_tests, total_tests


def test_orchestrator_integration():
    """Test Progressive Quality Orchestrator integration."""
    print("\n🎛️  Testing orchestrator integration...")
    
    try:
        from moe_debugger.progressive_quality_orchestrator import (
            ProgressiveQualityOrchestrator, QualityGateConfiguration
        )
        
        # Create test configuration
        config = QualityGateConfiguration(
            threat_detection_enabled=True,
            performance_optimization_enabled=True,
            chaos_engineering_enabled=False,  # Disable for quick test
            observability_enabled=True,
            governance_enabled=True,
            recovery_system_enabled=True,
            automated_incident_response_enabled=False  # Disable for testing
        )
        
        # Test orchestrator creation
        orchestrator = ProgressiveQualityOrchestrator(config)
        print("✅ Orchestrator creation: OK")
        
        # Test system initialization
        init_result = orchestrator.initialize_system()
        if isinstance(init_result, dict) and 'overall_status' in init_result:
            print(f"✅ System initialization: {init_result['overall_status']}")
            
            # Test system status
            status = orchestrator.get_system_status()
            if isinstance(status, dict):
                print("✅ System status retrieval: OK")
                
                # Test quality assessment
                assessment = orchestrator.trigger_quality_gate_assessment()
                if isinstance(assessment, dict) and 'overall_score' in assessment:
                    score = assessment['overall_score']
                    gate_status = assessment.get('quality_gate_status', 'unknown')
                    print(f"✅ Quality assessment: {score:.1f}/100 ({gate_status})")
                    return True
                else:
                    print("❌ Quality assessment failed")
            else:
                print("❌ System status retrieval failed")
        else:
            print(f"❌ System initialization failed: {init_result}")
            
    except Exception as e:
        print(f"❌ Orchestrator integration failed: {e}")
        import traceback
        traceback.print_exc()
    
    return False


def run_quick_validation():
    """Run quick validation of Progressive Quality Gates."""
    print("🚀 Progressive Quality Gates - Quick Validation Test")
    print("=" * 60)
    
    start_time = time.time()
    
    # Test component imports
    import_results = test_component_imports()
    successful_imports = sum(import_results.values())
    total_components = len(import_results)
    
    print(f"\n📦 Component Import Results: {successful_imports}/{total_components} successful")
    
    # Test basic functionality
    functional_tests, total_functional = test_basic_functionality()
    print(f"📊 Functional Tests: {functional_tests}/{total_functional} passed")
    
    # Test orchestrator integration
    orchestrator_ok = test_orchestrator_integration()
    print(f"🎛️  Orchestrator Integration: {'✅ OK' if orchestrator_ok else '❌ Failed'}")
    
    end_time = time.time()
    
    # Calculate overall results
    total_score = 0
    max_score = 0
    
    # Import score (40% weight)
    import_score = (successful_imports / total_components) * 40
    total_score += import_score
    max_score += 40
    
    # Functional score (40% weight)  
    if total_functional > 0:
        functional_score = (functional_tests / total_functional) * 40
        total_score += functional_score
        max_score += 40
    
    # Orchestrator score (20% weight)
    if orchestrator_ok:
        total_score += 20
    max_score += 20
    
    final_percentage = (total_score / max_score * 100) if max_score > 0 else 0
    
    print("\n" + "=" * 60)
    print("📋 QUICK VALIDATION SUMMARY")
    print("=" * 60)
    print(f"⏱️  Validation time: {end_time - start_time:.2f} seconds")
    print(f"📦 Component imports: {successful_imports}/{total_components}")
    print(f"🧪 Functional tests: {functional_tests}/{total_functional}")
    print(f"🎛️  Orchestrator: {'✅' if orchestrator_ok else '❌'}")
    print(f"📈 Overall score: {final_percentage:.1f}%")
    
    if final_percentage >= 80:
        print("🎉 Progressive Quality Gates: VALIDATION PASSED")
        return True
    elif final_percentage >= 60:
        print("⚠️  Progressive Quality Gates: PARTIAL SUCCESS")
        return True
    else:
        print("❌ Progressive Quality Gates: VALIDATION FAILED")
        return False


if __name__ == '__main__':
    """Main validation execution."""
    success = run_quick_validation()
    sys.exit(0 if success else 1)