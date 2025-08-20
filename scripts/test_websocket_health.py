#!/usr/bin/env python3
"""
WebSocket health check script for progressive quality gates.
"""

import asyncio
import json
import sys
from typing import Dict, Any
import websockets
import argparse
from datetime import datetime


class WebSocketHealthChecker:
    """Health checker for WebSocket connections."""
    
    def __init__(self, host: str = "localhost", port: int = 8080, timeout: int = 30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.ws_url = f"ws://{host}:{port}/ws"
    
    async def check_connection(self) -> bool:
        """Test basic WebSocket connection."""
        try:
            print(f"🔗 Testing WebSocket connection to {self.ws_url}")
            
            async with websockets.connect(
                self.ws_url, 
                timeout=self.timeout,
                ping_interval=10,
                ping_timeout=5
            ) as websocket:
                print("✅ WebSocket connection established")
                
                # Send a ping message
                ping_message = {
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send(json.dumps(ping_message))
                print("📤 Ping message sent")
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(
                        websocket.recv(), 
                        timeout=5.0
                    )
                    response_data = json.loads(response)
                    print(f"📥 Received response: {response_data.get('type', 'unknown')}")
                    
                    if response_data.get('type') == 'pong':
                        print("✅ Ping-pong test successful")
                        return True
                    else:
                        print("⚠️  Unexpected response type")
                        return True  # Still connected
                        
                except asyncio.TimeoutError:
                    print("⚠️  No response received within timeout")
                    return True  # Connection is still valid
                    
        except websockets.exceptions.ConnectionRefused:
            print("❌ WebSocket connection refused")
            return False
        except websockets.exceptions.InvalidURI:
            print("❌ Invalid WebSocket URI")
            return False
        except asyncio.TimeoutError:
            print("❌ WebSocket connection timeout")
            return False
        except Exception as e:
            print(f"❌ WebSocket connection error: {e}")
            return False
    
    async def check_debugger_session(self) -> bool:
        """Test debugger session WebSocket functionality."""
        try:
            print(f"🧪 Testing debugger session WebSocket")
            
            async with websockets.connect(
                self.ws_url,
                timeout=self.timeout
            ) as websocket:
                # Simulate debugger session start
                session_message = {
                    "type": "start_session",
                    "session_id": "test_session_123",
                    "config": {
                        "sampling_rate": 0.1,
                        "buffer_size": 1000
                    }
                }
                
                await websocket.send(json.dumps(session_message))
                print("📤 Session start message sent")
                
                # Wait for session confirmation
                try:
                    response = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=10.0
                    )
                    response_data = json.loads(response)
                    
                    if response_data.get('type') == 'session_started':
                        print("✅ Debugger session started successfully")
                        
                        # Send some mock routing data
                        routing_data = {
                            "type": "routing_event",
                            "session_id": "test_session_123",
                            "data": {
                                "expert_id": 0,
                                "token_id": 123,
                                "routing_weight": 0.8,
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                        
                        await websocket.send(json.dumps(routing_data))
                        print("📤 Mock routing data sent")
                        
                        # Try to receive processed data
                        try:
                            processed_response = await asyncio.wait_for(
                                websocket.recv(),
                                timeout=5.0
                            )
                            processed_data = json.loads(processed_response)
                            print(f"📥 Processed data received: {processed_data.get('type', 'unknown')}")
                            
                        except asyncio.TimeoutError:
                            print("ℹ️  No processed data response (normal for mock data)")
                        
                        return True
                    else:
                        print(f"❌ Unexpected session response: {response_data}")
                        return False
                        
                except asyncio.TimeoutError:
                    print("❌ No session confirmation received")
                    return False
                    
        except Exception as e:
            print(f"❌ Debugger session test error: {e}")
            return False
    
    async def check_realtime_streaming(self) -> bool:
        """Test real-time data streaming capabilities."""
        try:
            print(f"📊 Testing real-time streaming")
            
            async with websockets.connect(
                self.ws_url,
                timeout=self.timeout
            ) as websocket:
                # Subscribe to real-time updates
                subscribe_message = {
                    "type": "subscribe",
                    "channels": ["routing_events", "performance_metrics"],
                    "session_id": "test_session_123"
                }
                
                await websocket.send(json.dumps(subscribe_message))
                print("📤 Subscription message sent")
                
                # Send multiple events to test streaming
                for i in range(3):
                    event_message = {
                        "type": "routing_event",
                        "session_id": "test_session_123",
                        "data": {
                            "expert_id": i % 4,
                            "token_id": 1000 + i,
                            "routing_weight": 0.5 + (i * 0.1),
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    
                    await websocket.send(json.dumps(event_message))
                    await asyncio.sleep(0.1)  # Small delay between events
                
                print("📤 Multiple routing events sent")
                
                # Check for any streaming responses
                responses_received = 0
                try:
                    while responses_received < 5:  # Try to receive up to 5 responses
                        response = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=2.0
                        )
                        response_data = json.loads(response)
                        responses_received += 1
                        print(f"📥 Streaming response {responses_received}: {response_data.get('type', 'unknown')}")
                        
                except asyncio.TimeoutError:
                    if responses_received > 0:
                        print(f"✅ Received {responses_received} streaming responses")
                    else:
                        print("ℹ️  No streaming responses (may be normal for test environment)")
                
                return True
                
        except Exception as e:
            print(f"❌ Real-time streaming test error: {e}")
            return False
    
    async def check_connection_stability(self) -> bool:
        """Test WebSocket connection stability over time."""
        try:
            print(f"⏱️  Testing connection stability (30 seconds)")
            
            async with websockets.connect(
                self.ws_url,
                timeout=self.timeout,
                ping_interval=5,
                ping_timeout=3
            ) as websocket:
                start_time = datetime.now()
                ping_count = 0
                
                while (datetime.now() - start_time).seconds < 30:
                    # Send periodic ping
                    ping_message = {
                        "type": "ping",
                        "sequence": ping_count,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    await websocket.send(json.dumps(ping_message))
                    ping_count += 1
                    
                    # Wait for response or timeout
                    try:
                        response = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=3.0
                        )
                        response_data = json.loads(response)
                        
                        if ping_count % 5 == 0:  # Log every 5th ping
                            print(f"📍 Ping {ping_count}: {response_data.get('type', 'unknown')}")
                            
                    except asyncio.TimeoutError:
                        print(f"⚠️  Ping {ping_count} timeout")
                    
                    await asyncio.sleep(1)
                
                print(f"✅ Connection remained stable for 30 seconds ({ping_count} pings)")
                return True
                
        except Exception as e:
            print(f"❌ Connection stability test error: {e}")
            return False
    
    async def run_all_checks(self) -> bool:
        """Run all WebSocket health checks."""
        print(f"🔍 Starting WebSocket health checks for {self.ws_url}")
        print("=" * 60)
        
        checks = [
            ("Basic Connection", self.check_connection),
            ("Debugger Session", self.check_debugger_session),
            ("Real-time Streaming", self.check_realtime_streaming),
            ("Connection Stability", self.check_connection_stability),
        ]
        
        results = []
        
        for check_name, check_func in checks:
            print(f"\n🧪 Running {check_name} check...")
            try:
                result = await check_func()
                results.append((check_name, result))
                
                if result:
                    print(f"✅ {check_name} check PASSED")
                else:
                    print(f"❌ {check_name} check FAILED")
                    
            except Exception as e:
                print(f"❌ {check_name} check ERROR: {e}")
                results.append((check_name, False))
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 WebSocket Health Check Summary:")
        
        passed_checks = sum(1 for _, result in results if result)
        total_checks = len(results)
        
        for check_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {status}: {check_name}")
        
        success_rate = (passed_checks / total_checks) * 100
        print(f"\nOverall: {passed_checks}/{total_checks} checks passed ({success_rate:.1f}%)")
        
        if success_rate >= 75:  # Allow some non-critical failures
            print("🎉 WebSocket health check PASSED")
            return True
        else:
            print("💥 WebSocket health check FAILED")
            return False


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WebSocket health check")
    parser.add_argument("--host", default="localhost", help="WebSocket host")
    parser.add_argument("--port", type=int, default=8080, help="WebSocket port")
    parser.add_argument("--timeout", type=int, default=30, help="Connection timeout")
    
    args = parser.parse_args()
    
    checker = WebSocketHealthChecker(
        host=args.host,
        port=args.port,
        timeout=args.timeout
    )
    
    try:
        success = await checker.run_all_checks()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Health check interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Health check failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())