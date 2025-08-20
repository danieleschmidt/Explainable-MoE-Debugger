"""
Load testing configuration for progressive quality gates.
"""

import json
import random
import time
from datetime import datetime
from locust import HttpUser, task, between, events
import websocket


class MoEDebuggerLoadTest(HttpUser):
    """Load test for MoE Debugger API endpoints."""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup before testing starts."""
        self.session_id = None
        self.expert_count = 8
        self.websocket_connection = None
    
    def on_stop(self):
        """Cleanup after testing."""
        if self.session_id:
            self.stop_debugging_session()
        if self.websocket_connection:
            self.websocket_connection.close()
    
    @task(1)
    def health_check(self):
        """Test health check endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed with status {response.status_code}")
    
    @task(2)
    def api_docs(self):
        """Test API documentation endpoint."""
        with self.client.get("/docs", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"API docs failed with status {response.status_code}")
    
    @task(5)
    def start_debugging_session(self):
        """Test starting a debugging session."""
        payload = {
            "model_name": "test-mixtral-8x7b",
            "config": {
                "sampling_rate": 0.1,
                "buffer_size": 1000,
                "enable_gradients": False
            }
        }
        
        with self.client.post("/api/v1/sessions", json=payload, catch_response=True) as response:
            if response.status_code == 201:
                data = response.json()
                self.session_id = data.get("session_id")
                response.success()
            else:
                response.failure(f"Session start failed with status {response.status_code}")
    
    @task(3)
    def stop_debugging_session(self):
        """Test stopping a debugging session."""
        if not self.session_id:
            return
        
        with self.client.delete(f"/api/v1/sessions/{self.session_id}", catch_response=True) as response:
            if response.status_code in [200, 404]:  # 404 is OK if session doesn't exist
                self.session_id = None
                response.success()
            else:
                response.failure(f"Session stop failed with status {response.status_code}")
    
    @task(10)
    def send_routing_data(self):
        """Test sending routing data to a session."""
        if not self.session_id:
            self.start_debugging_session()
        
        if not self.session_id:
            return
        
        # Generate realistic routing data
        routing_data = {
            "events": [
                {
                    "expert_id": random.randint(0, self.expert_count - 1),
                    "token_id": random.randint(0, 2048),
                    "routing_weight": random.uniform(0.0, 1.0),
                    "layer_id": random.randint(0, 32),
                    "sequence_position": random.randint(0, 128),
                    "timestamp": datetime.now().isoformat()
                }
                for _ in range(random.randint(10, 100))
            ]
        }
        
        with self.client.post(
            f"/api/v1/sessions/{self.session_id}/routing-data",
            json=routing_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Routing data failed with status {response.status_code}")
    
    @task(4)
    def get_session_stats(self):
        """Test getting session statistics."""
        if not self.session_id:
            return
        
        with self.client.get(f"/api/v1/sessions/{self.session_id}/stats", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                # Validate response structure
                required_fields = ["expert_utilization", "routing_stats", "performance_metrics"]
                if all(field in data for field in required_fields):
                    response.success()
                else:
                    response.failure("Invalid stats response structure")
            else:
                response.failure(f"Stats request failed with status {response.status_code}")
    
    @task(2)
    def get_expert_analysis(self):
        """Test expert analysis endpoint."""
        if not self.session_id:
            return
        
        with self.client.get(f"/api/v1/sessions/{self.session_id}/analysis/experts", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "expert_utilization" in data and "load_balance" in data:
                    response.success()
                else:
                    response.failure("Invalid expert analysis response")
            else:
                response.failure(f"Expert analysis failed with status {response.status_code}")
    
    @task(2)
    def get_performance_metrics(self):
        """Test performance metrics endpoint."""
        if not self.session_id:
            return
        
        with self.client.get(f"/api/v1/sessions/{self.session_id}/metrics", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "response_time" in data and "throughput" in data:
                    response.success()
                else:
                    response.failure("Invalid performance metrics response")
            else:
                response.failure(f"Performance metrics failed with status {response.status_code}")
    
    @task(1)
    def export_session_data(self):
        """Test exporting session data."""
        if not self.session_id:
            return
        
        params = {"format": "json", "include_raw_data": "false"}
        
        with self.client.get(
            f"/api/v1/sessions/{self.session_id}/export",
            params=params,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                # Check if response is valid JSON or other format
                try:
                    if params["format"] == "json":
                        json.loads(response.text)
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON export format")
            else:
                response.failure(f"Export failed with status {response.status_code}")


class WebSocketLoadTest(HttpUser):
    """Load test for WebSocket real-time functionality."""
    
    wait_time = between(2, 5)
    
    def on_start(self):
        """Setup WebSocket connection."""
        self.ws_url = f"ws://{self.host.replace('http://', '')}/ws"
        self.session_id = f"load_test_{random.randint(1000, 9999)}"
        self.connect_websocket()
    
    def on_stop(self):
        """Clean up WebSocket connection."""
        if hasattr(self, 'ws') and self.ws:
            self.ws.close()
    
    def connect_websocket(self):
        """Establish WebSocket connection."""
        try:
            self.ws = websocket.create_connection(self.ws_url, timeout=10)
            
            # Send initial connection message
            connect_msg = {
                "type": "connect",
                "session_id": self.session_id,
                "client_type": "load_test"
            }
            self.ws.send(json.dumps(connect_msg))
            
            # Wait for connection confirmation
            response = self.ws.recv()
            response_data = json.loads(response)
            
            if response_data.get("type") == "connected":
                events.request_success.fire(
                    request_type="WebSocket",
                    name="connect",
                    response_time=100,
                    response_length=len(response)
                )
            else:
                events.request_failure.fire(
                    request_type="WebSocket",
                    name="connect",
                    response_time=100,
                    response_length=len(response),
                    exception="Connection failed"
                )
                
        except Exception as e:
            events.request_failure.fire(
                request_type="WebSocket",
                name="connect",
                response_time=0,
                response_length=0,
                exception=str(e)
            )
            self.ws = None
    
    @task(10)
    def send_realtime_routing_data(self):
        """Send real-time routing data via WebSocket."""
        if not hasattr(self, 'ws') or not self.ws:
            self.connect_websocket()
            return
        
        start_time = time.time()
        
        try:
            # Generate routing event
            routing_event = {
                "type": "routing_event",
                "session_id": self.session_id,
                "data": {
                    "expert_id": random.randint(0, 7),
                    "token_id": random.randint(0, 2048),
                    "routing_weight": random.uniform(0.0, 1.0),
                    "layer_id": random.randint(0, 32),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            self.ws.send(json.dumps(routing_event))
            
            # Try to receive response (with timeout)
            self.ws.settimeout(1.0)
            try:
                response = self.ws.recv()
                response_time = (time.time() - start_time) * 1000
                
                events.request_success.fire(
                    request_type="WebSocket",
                    name="routing_event",
                    response_time=response_time,
                    response_length=len(response)
                )
            except websocket.timeout:
                # No immediate response is OK for routing events
                response_time = (time.time() - start_time) * 1000
                events.request_success.fire(
                    request_type="WebSocket",
                    name="routing_event",
                    response_time=response_time,
                    response_length=0
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            events.request_failure.fire(
                request_type="WebSocket",
                name="routing_event",
                response_time=response_time,
                response_length=0,
                exception=str(e)
            )
    
    @task(3)
    def subscribe_to_updates(self):
        """Subscribe to real-time updates."""
        if not hasattr(self, 'ws') or not self.ws:
            return
        
        start_time = time.time()
        
        try:
            subscribe_msg = {
                "type": "subscribe",
                "session_id": self.session_id,
                "channels": ["routing_events", "performance_metrics"]
            }
            
            self.ws.send(json.dumps(subscribe_msg))
            
            # Wait for subscription confirmation
            self.ws.settimeout(2.0)
            response = self.ws.recv()
            response_time = (time.time() - start_time) * 1000
            
            response_data = json.loads(response)
            if response_data.get("type") == "subscribed":
                events.request_success.fire(
                    request_type="WebSocket",
                    name="subscribe",
                    response_time=response_time,
                    response_length=len(response)
                )
            else:
                events.request_failure.fire(
                    request_type="WebSocket",
                    name="subscribe",
                    response_time=response_time,
                    response_length=len(response),
                    exception="Subscription failed"
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            events.request_failure.fire(
                request_type="WebSocket",
                name="subscribe",
                response_time=response_time,
                response_length=0,
                exception=str(e)
            )
    
    @task(1)
    def ping_connection(self):
        """Send ping to maintain connection."""
        if not hasattr(self, 'ws') or not self.ws:
            return
        
        start_time = time.time()
        
        try:
            ping_msg = {
                "type": "ping",
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat()
            }
            
            self.ws.send(json.dumps(ping_msg))
            
            # Wait for pong response
            self.ws.settimeout(3.0)
            response = self.ws.recv()
            response_time = (time.time() - start_time) * 1000
            
            response_data = json.loads(response)
            if response_data.get("type") == "pong":
                events.request_success.fire(
                    request_type="WebSocket",
                    name="ping",
                    response_time=response_time,
                    response_length=len(response)
                )
            else:
                events.request_failure.fire(
                    request_type="WebSocket",
                    name="ping",
                    response_time=response_time,
                    response_length=len(response),
                    exception="Invalid pong response"
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            events.request_failure.fire(
                request_type="WebSocket",
                name="ping",
                response_time=response_time,
                response_length=0,
                exception=str(e)
            )


# Load test configuration for different scenarios
class LightLoadTest(MoEDebuggerLoadTest):
    """Light load test - simulates normal usage."""
    weight = 3


class HeavyLoadTest(MoEDebuggerLoadTest):
    """Heavy load test - simulates peak usage."""
    weight = 2
    wait_time = between(0.5, 1.5)  # Faster requests


class WebSocketHeavyLoad(WebSocketLoadTest):
    """Heavy WebSocket load test."""
    weight = 1
    wait_time = between(0.1, 0.5)  # Very fast for real-time testing


# Event handlers for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize load test metrics."""
    print("🚀 Starting MoE Debugger load tests...")
    print(f"Target host: {environment.host}")
    print(f"Users: {environment.runner.target_user_count}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Finalize load test metrics."""
    print("📊 Load test completed!")
    
    # Calculate custom metrics
    stats = environment.runner.stats
    total_requests = stats.total.num_requests
    total_failures = stats.total.num_failures
    success_rate = ((total_requests - total_failures) / total_requests * 100) if total_requests > 0 else 0
    
    print(f"Total requests: {total_requests}")
    print(f"Total failures: {total_failures}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    
    # Load test quality gates
    quality_gates_passed = True
    
    if success_rate < 95:
        print("❌ Quality Gate FAILED: Success rate below 95%")
        quality_gates_passed = False
    
    if stats.total.avg_response_time > 500:
        print("❌ Quality Gate FAILED: Average response time above 500ms")
        quality_gates_passed = False
    
    if stats.total.get_response_time_percentile(0.95) > 1000:
        print("❌ Quality Gate FAILED: 95th percentile above 1000ms")
        quality_gates_passed = False
    
    if quality_gates_passed:
        print("✅ All load test quality gates PASSED")
    else:
        print("💥 Load test quality gates FAILED")
        exit(1)