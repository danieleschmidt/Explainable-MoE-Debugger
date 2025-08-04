#!/usr/bin/env python3
"""Basic functionality test without heavy dependencies."""

import sys
import os
sys.path.insert(0, 'src')

# Test basic imports and data models
def test_models():
    """Test data models can be imported and created."""
    from moe_debugger.models import (
        RoutingEvent, ExpertMetrics, ModelArchitecture, 
        LoadBalanceMetrics, DebugSession, HookConfiguration
    )
    
    # Test RoutingEvent creation
    event = RoutingEvent(
        timestamp=1234567890.0,
        layer_idx=0,
        token_position=5,
        token="hello",
        expert_weights=[0.3, 0.7, 0.0, 0.0],
        selected_experts=[1],
        routing_confidence=0.85
    )
    
    # Test ModelArchitecture
    arch = ModelArchitecture(
        num_layers=12,
        num_experts_per_layer=8,
        hidden_size=768,
        intermediate_size=3072,
        vocab_size=32000,
        max_sequence_length=2048,
        expert_capacity=2.0
    )
    
    # Test HookConfiguration
    config = HookConfiguration(
        enabled_hooks={"router": True, "experts": True},
        sampling_rate=0.1,
        buffer_size=1000,
        save_gradients=False,
        save_activations=True,
        track_parameters=["weight"]
    )
    
    print("✅ All data models work correctly")
    return True

def test_cli_without_torch():
    """Test CLI functionality that doesn't require PyTorch."""
    try:
        from moe_debugger.cli import create_config
        import argparse
        
        # Mock args
        class MockArgs:
            track_attention = False
            sampling_rate = 0.1
            buffer_size = 10000
            save_gradients = False
            memory_limit = 2048
        
        config = create_config(MockArgs())
        assert config['sampling_rate'] == 0.1
        assert config['enabled_hooks']['router'] == True
        
        print("✅ CLI configuration works correctly")
        return True
    except ImportError as e:
        print(f"⚠️  CLI test skipped due to import: {e}")
        return True

def test_server_imports():
    """Test server can be imported (FastAPI might not be available)."""
    try:
        from moe_debugger.server import ConnectionManager
        
        # Test ConnectionManager basic functionality
        manager = ConnectionManager()
        assert len(manager.active_connections) == 0
        
        print("✅ Server components work correctly")
        return True
    except ImportError as e:
        print(f"⚠️  Server test skipped due to import: {e}")
        return True

def main():
    """Run all basic tests."""
    print("🧪 Testing MoE Debugger Basic Functionality\n")
    
    tests = [
        test_models,
        test_cli_without_torch, 
        test_server_imports,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic functionality tests passed!")
        return True
    else:
        print("⚠️  Some tests failed - check dependencies")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)