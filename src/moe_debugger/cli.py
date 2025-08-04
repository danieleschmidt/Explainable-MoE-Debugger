"""Command-line interface for MoE debugger."""

import argparse
import sys
import os
from typing import Optional, Dict, Any
import json

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import these only when available  
try:
    from .debugger import MoEDebugger
    from .server import DebugServer
    COMPONENTS_AVAILABLE = True
except ImportError:
    MoEDebugger = None
    DebugServer = None
    COMPONENTS_AVAILABLE = False

from .__about__ import __version__


def load_model(model_path: str, model_type: str = "auto") -> Optional[Any]:
    """Load a model for debugging."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Please install PyTorch to use the debugger.")
        return None
    
    try:
        if model_type == "huggingface" or model_path.startswith("huggingface:"):
            # Load from Hugging Face
            model_name = model_path.replace("huggingface:", "")
            try:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
                print(f"Loaded Hugging Face model: {model_name}")
                return model
            except ImportError:
                print("transformers package not installed. Install with: pip install transformers")
                return None
        
        elif model_path.endswith('.pt') or model_path.endswith('.pth'):
            # Load PyTorch checkpoint
            model = torch.load(model_path, map_location='cpu')
            print(f"Loaded PyTorch model from: {model_path}")
            return model
        
        elif os.path.isdir(model_path):
            # Try to load from directory (Hugging Face format)
            try:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(model_path)
                print(f"Loaded model from directory: {model_path}")
                return model
            except ImportError:
                print("transformers package not installed for directory loading")
                return None
        
        else:
            print(f"Unsupported model path or type: {model_path}")
            return None
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def create_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Create debugger configuration from command line arguments."""
    config = {
        "enabled_hooks": {
            "router": True,
            "experts": True,
            "attention": args.track_attention
        },
        "sampling_rate": args.sampling_rate,
        "buffer_size": args.buffer_size,
        "save_gradients": args.save_gradients,
        "save_activations": True,
        "track_parameters": ["weight", "bias"],
        "memory_limit_mb": args.memory_limit
    }
    
    return config


def run_interactive_mode(debugger: MoEDebugger):
    """Run interactive debugging session."""
    print("\n🔍 MoE Interactive Debugger")
    print("Commands: start, stop, status, analyze, export, clear, help, quit")
    
    session = None
    
    while True:
        try:
            command = input("\nmoe-debug> ").strip().lower()
            
            if command == "quit" or command == "exit":
                if session:
                    debugger.end_session()
                break
            
            elif command == "help":
                print("""
Available commands:
  start          - Start debugging session
  stop           - Stop current session
  status         - Show current status
  analyze        - Run analysis on current data
  export [file]  - Export session data
  clear          - Clear all data
  help           - Show this help
  quit           - Exit debugger
                """)
            
            elif command == "start":
                if debugger.is_active:
                    print("Session already active")
                else:
                    session = debugger.start_session()
                    print(f"Started session: {session.session_id}")
            
            elif command == "stop":
                if not debugger.is_active:
                    print("No active session")
                else:
                    session = debugger.end_session()
                    if session:
                        print(f"Ended session: {session.session_id}")
            
            elif command == "status":
                if debugger.is_active:
                    stats = debugger.get_routing_stats()
                    performance = debugger.get_performance_metrics()
                    print(f"Status: Active (Session: {debugger.current_session.session_id})")
                    print(f"Routing events: {stats.get('total_routing_decisions', 0)}")
                    print(f"Memory usage: {performance.get('current_memory_mb', 0):.1f} MB")
                else:
                    print("Status: Inactive")
            
            elif command == "analyze":
                if not debugger.is_active:
                    print("No active session")
                else:
                    issues = debugger.detect_issues()
                    if issues:
                        print("Detected issues:")
                        for issue in issues:
                            print(f"  {issue['severity'].upper()}: {issue['message']}")
                    else:
                        print("No issues detected")
            
            elif command.startswith("export"):
                if not debugger.current_session:
                    print("No session to export")
                else:
                    parts = command.split()
                    filename = parts[1] if len(parts) > 1 else f"session_{int(time.time())}.json"
                    debugger.export_session(filename)
                    print(f"Exported to: {filename}")
            
            elif command == "clear":
                debugger.clear_data()
                print("Data cleared")
            
            else:
                print(f"Unknown command: {command}")
        
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit")
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MoE Debugger - Chrome DevTools-style debugging for Mixture of Experts models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--version", action="version", version=f"moe-debugger {__version__}"
    )
    
    # Model arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Path to model or model identifier (e.g., 'mistralai/Mixtral-8x7B-v0.1')"
    )
    
    parser.add_argument(
        "--model-type",
        choices=["auto", "huggingface", "pytorch"],
        default="auto",
        help="Type of model to load"
    )
    
    # Server arguments
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Port for web server"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host for web server"
    )
    
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Run in interactive mode without web server"
    )
    
    # Debugging configuration
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=0.1,
        help="Sampling rate for data collection (0.0-1.0)"
    )
    
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=10000,
        help="Buffer size for storing events"
    )
    
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=2048,
        help="Memory limit in MB"
    )
    
    parser.add_argument(
        "--save-gradients",
        action="store_true",
        help="Enable gradient saving (high overhead)"
    )
    
    parser.add_argument(
        "--track-attention",
        action="store_true",
        help="Enable attention tracking"
    )
    
    # Utility arguments
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--export-config",
        type=str,
        help="Export default configuration to file"
    )
    
    args = parser.parse_args()
    
    # Handle configuration export
    if args.export_config:
        config = create_config(args)
        with open(args.export_config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration exported to: {args.export_config}")
        return
    
    # Load configuration from file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
            print(f"Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    else:
        file_config = {}
    
    # Create combined configuration
    config = create_config(args)
    config.update(file_config)
    
    # Load model if provided
    model = None
    if args.model:
        model = load_model(args.model, args.model_type)
        if model is None:
            print("Failed to load model")
            sys.exit(1)
    
    # Create debugger
    debugger = None
    if model and COMPONENTS_AVAILABLE and MoEDebugger:
        debugger = MoEDebugger(model, config)
        print(f"Created debugger for model: {model.__class__.__name__}")
        
        # Print model summary
        summary = debugger.get_model_summary()
        print(f"Model parameters: {summary['total_parameters']:,}")
        print(f"Detected architecture: {summary['architecture']['num_layers']} layers, "
              f"{summary['architecture']['num_experts_per_layer']} experts per layer")
    
    # Run in appropriate mode
    if args.no_server or model is None:
        if model is None:
            print("\nNo model provided. Use --model to specify a model.")
            print("Example: moe-debugger --model mistralai/Mixtral-8x7B-v0.1")
            print("Or run with --help for more options")
        else:
            run_interactive_mode(debugger)
    else:
        # Start web server
        if COMPONENTS_AVAILABLE and DebugServer:
            server = DebugServer(debugger, host=args.host, port=args.port)
        else:
            print("Server components not available")
            return
        print(f"\n🚀 Starting MoE Debugger server...")
        print(f"📱 Web interface: http://{args.host}:{args.port}")
        print(f"🔗 WebSocket endpoint: ws://{args.host}:{args.port}/ws")
        print(f"📊 API documentation: http://{args.host}:{args.port}/docs")
        print("\nPress Ctrl+C to stop")
        
        try:
            server.run()
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")


if __name__ == "__main__":
    main()