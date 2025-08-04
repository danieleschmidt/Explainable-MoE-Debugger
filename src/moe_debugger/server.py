"""Web server for MoE debugger with real-time communication."""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import logging

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    # Mock WebSocket for type hints
    class WebSocket:
        pass
    class WebSocketDisconnect(Exception):
        pass
    class HTTPException(Exception):
        pass
    FASTAPI_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    from .mock_torch import torch, nn
    TORCH_AVAILABLE = False

from .models import VisualizationData, DebugSession


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_sessions: Dict[str, str] = {}  # connection_id -> session_id
    
    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if connection_id in self.connection_sessions:
            del self.connection_sessions[connection_id]
    
    async def send_personal_message(self, message: dict, connection_id: str):
        """Send a message to a specific connection."""
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_text(json.dumps(message))
            except:
                self.disconnect(connection_id)
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        disconnected = []
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except:
                disconnected.append(connection_id)
        
        # Clean up disconnected clients
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    async def broadcast_to_session(self, message: dict, session_id: str):
        """Broadcast message to all connections in a specific session."""
        disconnected = []
        for connection_id, websocket in self.active_connections.items():
            if self.connection_sessions.get(connection_id) == session_id:
                try:
                    await websocket.send_text(json.dumps(message))
                except:
                    disconnected.append(connection_id)
        
        for connection_id in disconnected:
            self.disconnect(connection_id)


class DebugServer:
    """FastAPI server for MoE debugging interface."""
    
    def __init__(self, debugger=None, host: str = "localhost", port: int = 8080):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI dependencies not installed. Run: pip install fastapi uvicorn websockets")
        
        self.debugger = debugger
        self.host = host
        self.port = port
        
        # Create FastAPI app
        self.app = FastAPI(
            title="MoE Debugger",
            description="Chrome DevTools-style GUI for debugging Mixture of Experts models",
            version="0.1.0"
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Connection manager for WebSockets
        self.connection_manager = ConnectionManager()
        
        # Active sessions
        self.active_sessions: Dict[str, DebugSession] = {}
        
        # Setup routes
        self._setup_routes()
        
        # Setup WebSocket handlers
        self._setup_websockets()
        
        # Setup event handlers
        if self.debugger:
            self._setup_debugger_callbacks()
    
    def set_debugger(self, debugger):
        """Set the debugger instance."""
        self.debugger = debugger
        self._setup_debugger_callbacks()
    
    def _setup_routes(self):
        """Setup REST API routes."""
        
        @self.app.get("/")
        async def root():
            """Serve the main debugging interface."""
            return HTMLResponse(self._get_html_interface())
        
        @self.app.get("/api/status")
        async def get_status():
            """Get server and debugger status."""
            status = {
                "server_running": True,
                "debugger_attached": self.debugger is not None,
                "active_sessions": len(self.active_sessions),
                "connected_clients": len(self.connection_manager.active_connections)
            }
            
            if self.debugger:
                status.update({
                    "debugger_active": self.debugger.is_active,
                    "current_session": self.debugger.current_session.session_id if self.debugger.current_session else None,
                })
            
            return status
        
        @self.app.post("/api/sessions")
        async def create_session(session_config: Optional[Dict[str, Any]] = None):
            """Create a new debugging session."""
            if not self.debugger:
                raise HTTPException(status_code=400, detail="No debugger attached")
            
            session = self.debugger.start_session()
            self.active_sessions[session.session_id] = session
            
            await self.connection_manager.broadcast({
                "type": "session_created",
                "session": {
                    "session_id": session.session_id,
                    "start_time": session.start_time,
                    "model_name": session.model_name
                }
            })
            
            return {"session_id": session.session_id}
        
        @self.app.delete("/api/sessions/{session_id}")
        async def end_session(session_id: str):
            """End a debugging session."""
            if not self.debugger or not self.debugger.current_session:
                raise HTTPException(status_code=400, detail="No active session")
            
            if self.debugger.current_session.session_id != session_id:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session = self.debugger.end_session()
            if session and session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
            
            await self.connection_manager.broadcast({
                "type": "session_ended",
                "session_id": session_id
            })
            
            return {"message": "Session ended successfully"}
    
    def _setup_websockets(self):
        """Setup WebSocket endpoints."""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Main WebSocket endpoint for real-time updates."""
            connection_id = await self.connection_manager.connect(websocket)
            
            try:
                while True:
                    # Receive messages from client
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle different message types
                    await self._handle_websocket_message(message, connection_id)
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect(connection_id)
    
    async def _handle_websocket_message(self, message: Dict[str, Any], connection_id: str):
        """Handle incoming WebSocket messages."""
        msg_type = message.get("type")
        
        if msg_type == "join_session":
            session_id = message.get("session_id")
            if session_id:
                self.connection_manager.connection_sessions[connection_id] = session_id
                await self.connection_manager.send_personal_message({
                    "type": "session_joined",
                    "session_id": session_id
                }, connection_id)
        
        elif msg_type == "request_update":
            # Send current state to client
            if self.debugger and self.debugger.is_active:
                update = await self._create_real_time_update()
                await self.connection_manager.send_personal_message(update, connection_id)
        
        elif msg_type == "ping":
            await self.connection_manager.send_personal_message({
                "type": "pong",
                "timestamp": time.time()
            }, connection_id)
    
    def _setup_debugger_callbacks(self):
        """Setup callbacks to receive debugger events."""
        # Note: In a real implementation, you'd need to modify the debugger
        # to support async callbacks or use a message queue
        pass
    
    async def _create_real_time_update(self) -> Dict[str, Any]:
        """Create a real-time update with current debugger state."""
        if not self.debugger or not self.debugger.is_active:
            return {"type": "no_data"}
        
        return {
            "type": "real_time_update",
            "timestamp": time.time(),
            "data": {
                "status": "active",
                "message": "Debugging session active"
            }
        }
    
    def _get_html_interface(self) -> str:
        """Get HTML for the debugging interface."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MoE Debugger</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: #f5f5f5; 
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
        }
        .header { 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 20px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }
        .panel { 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 20px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }
        .status { 
            padding: 10px; 
            border-radius: 4px; 
            margin-bottom: 10px; 
        }
        .status.connected { 
            background: #d4edda; 
            color: #155724; 
        }
        .status.disconnected { 
            background: #f8d7da; 
            color: #721c24; 
        }
        button { 
            background: #007bff; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 4px; 
            cursor: pointer; 
            margin: 5px;
        }
        button:hover { 
            background: #0056b3; 
        }
        button:disabled { 
            background: #6c757d; 
            cursor: not-allowed; 
        }
        .log { 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 4px; 
            height: 200px; 
            overflow-y: auto; 
            font-family: monospace; 
            font-size: 12px; 
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 MoE Debugger</h1>
            <p>Chrome DevTools-style debugging for Mixture of Experts models</p>
            <div id="connection-status" class="status disconnected">Disconnected</div>
        </div>
        
        <div class="panel">
            <h2>Session Control</h2>
            <button onclick="startSession()" id="start-btn">Start Session</button>
            <button onclick="endSession()" id="end-btn" disabled>End Session</button>
            <button onclick="clearData()">Clear Data</button>
        </div>
        
        <div class="panel">
            <h2>Activity Log</h2>
            <div id="log" class="log"></div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let currentSession = null;
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = function() {
                document.getElementById('connection-status').textContent = 'Connected';
                document.getElementById('connection-status').className = 'status connected';
                log('Connected to debugger server');
            };
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                handleMessage(message);
            };
            
            ws.onclose = function() {
                document.getElementById('connection-status').textContent = 'Disconnected';
                document.getElementById('connection-status').className = 'status disconnected';
                log('Disconnected from server');
                setTimeout(connect, 2000); // Reconnect after 2 seconds
            };
        }
        
        function handleMessage(message) {
            switch(message.type) {
                case 'session_created':
                    currentSession = message.session.session_id;
                    document.getElementById('start-btn').disabled = true;
                    document.getElementById('end-btn').disabled = false;
                    log(`Session started: ${currentSession}`);
                    break;
                    
                case 'session_ended':
                    currentSession = null;
                    document.getElementById('start-btn').disabled = false;
                    document.getElementById('end-btn').disabled = true;
                    log('Session ended');
                    break;
                    
                case 'real_time_update':
                    log('Real-time update received');
                    break;
            }
        }
        
        function startSession() {
            fetch('/api/sessions', {method: 'POST'})
                .then(response => response.json())
                .then(data => log('Session creation requested'))
                .catch(error => log(`Error: ${error}`));
        }
        
        function endSession() {
            if (currentSession) {
                fetch(`/api/sessions/${currentSession}`, {method: 'DELETE'})
                    .then(response => response.json())
                    .then(data => log('Session end requested'))
                    .catch(error => log(`Error: ${error}`));
            }
        }
        
        function clearData() {
            log('Clear data requested');
        }
        
        function log(message) {
            const logElement = document.getElementById('log');
            const timestamp = new Date().toLocaleTimeString();
            logElement.innerHTML += `[${timestamp}] ${message}\\n`;
            logElement.scrollTop = logElement.scrollHeight;
        }
        
        // Connect on page load
        connect();
    </script>
</body>
</html>
        '''
    
    def run(self, **kwargs):
        """Run the debug server."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            **kwargs
        )
    
    def start(self, background: bool = False):
        """Start the server."""
        self.run()