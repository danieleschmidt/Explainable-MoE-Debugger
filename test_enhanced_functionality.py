#!/usr/bin/env python3
"""Enhanced functionality tests for Generation 1 improvements."""

import sys
import time
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def test_enhanced_debugger_factory():
    """Test enhanced debugger factory functionality."""
    print("  ⏳ Enhanced debugger factory...")
    
    try:
        from moe_debugger.debugger_factory import MoEDebuggerFactory
        
        # Test factory creation
        debugger = MoEDebuggerFactory.create_debugger()
        assert debugger is not None, "Factory should create debugger instance"
        
        # Test session management
        session = debugger.start_session("test_session")
        assert session is not None, "Should create debug session"
        assert debugger.is_active, "Debugger should be active"
        
        # Test metrics collection
        routing_stats = debugger.get_routing_stats()
        expert_metrics = debugger.get_expert_metrics()
        performance_metrics = debugger.get_performance_metrics()
        
        assert isinstance(routing_stats, dict), "Should return routing stats"
        assert isinstance(expert_metrics, dict), "Should return expert metrics"  
        assert isinstance(performance_metrics, dict), "Should return performance metrics"
        
        # Test session end
        ended_session = debugger.end_session()
        assert ended_session is not None, "Should return ended session"
        assert not debugger.is_active, "Debugger should be inactive"
        
        return True
        
    except Exception as e:
        print(f"      Error: {e}")
        return False


def test_enhanced_mock_torch():
    """Test enhanced mock PyTorch functionality."""
    print("  ⏳ Enhanced mock PyTorch...")
    
    try:
        from moe_debugger.mock_torch import torch, nn
        
        # Test enhanced torch features
        with torch.no_grad():
            tensor = torch.tensor([1.0, 2.0, 3.0])
            assert tensor is not None, "Should create tensor"
        
        # Test model creation
        model = nn.Module()
        assert model is not None, "Should create module"
        
        # Test Linear layer
        linear = nn.Linear(10, 5)
        assert linear.in_features == 10, "Should set input features"
        assert linear.out_features == 5, "Should set output features"
        
        return True
        
    except Exception as e:
        print(f"      Error: {e}")
        return False


def test_architecture_analysis():
    """Test enhanced architecture analysis."""
    print("  ⏳ Architecture analysis...")
    
    try:
        from moe_debugger.debugger_factory import MoEDebuggerFactory
        
        debugger = MoEDebuggerFactory.create_debugger()
        arch = debugger.architecture
        
        assert arch.num_layers > 0, "Should detect layers"
        assert arch.num_experts_per_layer > 0, "Should detect experts"
        assert arch.hidden_size > 0, "Should detect hidden size"
        assert arch.vocab_size > 0, "Should have vocab size"
        
        return True
        
    except Exception as e:
        print(f"      Error: {e}")
        return False


def test_profiling_context():
    """Test enhanced profiling context manager."""
    print("  ⏳ Profiling context manager...")
    
    try:
        from moe_debugger.debugger_factory import MoEDebuggerFactory
        
        debugger = MoEDebuggerFactory.create_debugger()
        
        # Test profiling context
        with debugger.profile():
            # Simulate some work
            time.sleep(0.01)
        
        return True
        
    except Exception as e:
        print(f"      Error: {e}")
        return False


def main():
    """Run all enhanced functionality tests."""
    
    print("🚀 Running Enhanced Generation 1 Functionality Tests\n")
    print("="*60)
    print("🔧 Enhanced Core Functionality Tests")
    print("="*60)
    
    tests = [
        ("Enhanced debugger factory", test_enhanced_debugger_factory),
        ("Enhanced mock PyTorch", test_enhanced_mock_torch),
        ("Architecture analysis", test_architecture_analysis),
        ("Profiling context manager", test_profiling_context),
    ]
    
    passed = 0
    failed = 0
    
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"  ✅ ({time.time() - start_time:.3f}s)")
                passed += 1
            else:
                print(f"  ❌ ({time.time() - start_time:.3f}s)")
                failed += 1
        except Exception as e:
            print(f"  ❌ Error: {e} ({time.time() - start_time:.3f}s)")
            failed += 1
    
    total_duration = time.time() - start_time
    
    print("\n" + "="*60)
    print("🎯 ENHANCED FUNCTIONALITY TEST REPORT")  
    print("="*60)
    print(f"📊 Test Statistics:")
    print(f"   • Total Tests: {passed + failed}")
    print(f"   • Passed: {passed} ✅")
    print(f"   • Failed: {failed} ❌")
    print(f"   • Success Rate: {passed/(passed+failed)*100:.1f}%")
    print(f"   • Total Duration: {total_duration:.2f}s")
    
    if failed == 0:
        print("\n✅ ALL ENHANCED FUNCTIONALITY TESTS PASSED!")
        print("🚀 Generation 1 enhancements successfully implemented.")
    else:
        print(f"\n⚠️  {failed} tests failed - investigating...")
    
    print("="*60)


if __name__ == "__main__":
    main()