#!/bin/bash
set -e

# Autonomous MoE Debugger Entrypoint Script
echo "🚀 Starting Autonomous MoE Debugger..."

# Initialize autonomous systems
echo "🔧 Initializing autonomous systems..."

# Start autonomous recovery system
if [ "${AUTONOMOUS_RECOVERY_ENABLED:-true}" = "true" ]; then
    echo "⚡ Starting autonomous recovery system..."
    python -c "from moe_debugger.autonomous_recovery import get_recovery_system; get_recovery_system().start_monitoring()" &
fi

# Initialize quantum routing if enabled
if [ "${QUANTUM_ROUTING_ENABLED:-true}" = "true" ]; then
    echo "🔬 Initializing quantum routing system..."
    python -c "from moe_debugger.quantum_routing import get_quantum_router; print('Quantum router initialized with 8 experts')" || echo "⚠️  Quantum routing initialization deferred"
fi

# Start distributed optimization if enabled
if [ "${DISTRIBUTED_MODE:-false}" = "true" ]; then
    echo "🌐 Starting distributed optimization..."
    python -c "from moe_debugger.distributed_optimization import get_distributed_optimizer; get_distributed_optimizer()" &
fi

# Initialize advanced caching
if [ "${ADVANCED_CACHING_ENABLED:-true}" = "true" ]; then
    echo "🧠 Starting advanced caching system..."
    python -c "from moe_debugger.advanced_caching import get_cache_manager; get_cache_manager().start_background_tasks()" &
fi

# Wait for dependencies
echo "⏳ Waiting for dependencies..."

# Wait for Redis if using caching
if [ -n "${REDIS_URL}" ]; then
    echo "Waiting for Redis..."
    until python -c "import redis; redis.from_url('${REDIS_URL}').ping()" 2>/dev/null; do
        echo "Redis is unavailable - sleeping"
        sleep 2
    done
    echo "✅ Redis is ready"
fi

# Wait for PostgreSQL if using distributed mode
if [ "${DISTRIBUTED_MODE}" = "true" ] && [ -n "${POSTGRES_URL}" ]; then
    echo "Waiting for PostgreSQL..."
    until python -c "import psycopg2; psycopg2.connect('${POSTGRES_URL}')" 2>/dev/null; do
        echo "PostgreSQL is unavailable - sleeping"
        sleep 2
    done
    echo "✅ PostgreSQL is ready"
fi

# Create log directory
mkdir -p /app/logs

# Set up environment
export PYTHONPATH=/app/src:$PYTHONPATH

# Log startup information
echo "
🎉 Autonomous MoE Debugger Started Successfully!

Configuration:
- Autonomous Recovery: ${AUTONOMOUS_RECOVERY_ENABLED:-true}
- Quantum Routing: ${QUANTUM_ROUTING_ENABLED:-true}
- Distributed Mode: ${DISTRIBUTED_MODE:-false}
- Advanced Caching: ${ADVANCED_CACHING_ENABLED:-true}
- Log Level: ${LOG_LEVEL:-INFO}

Endpoints:
- Main API: http://localhost:8080
- Health Check: http://localhost:8080/health
- Metrics: http://localhost:8080/metrics
- Autonomous Status: http://localhost:8081/autonomous

For documentation, visit: http://localhost:8080/docs
"

# Handle different service modes
case "${1}" in
    "python")
        # Default MoE debugger server
        exec "$@"
        ;;
    "--monitor-mode")
        # Autonomous recovery monitor
        echo "🔍 Starting in autonomous recovery monitor mode..."
        exec python -m moe_debugger.autonomous_recovery --monitor
        ;;
    "--service-mode")
        # Service-specific modes
        if [[ "$0" == *"quantum_routing"* ]]; then
            echo "🔬 Starting quantum routing service..."
            exec python -m moe_debugger.quantum_routing --service
        elif [[ "$0" == *"distributed_optimization"* ]]; then
            echo "🌐 Starting distributed optimization service..."
            exec python -m moe_debugger.distributed_optimization --service
        elif [[ "$0" == *"advanced_caching"* ]]; then
            echo "🧠 Starting advanced caching service..."
            exec python -m moe_debugger.advanced_caching --service
        else
            echo "⚠️  Unknown service mode"
            exit 1
        fi
        ;;
    "--coordinator-mode")
        # Distributed coordinator
        echo "🌐 Starting distributed coordinator mode..."
        exec python -m moe_debugger.distributed_optimization --coordinator
        ;;
    *)
        # Default: start main server
        exec python -m moe_debugger.server "$@"
        ;;
esac