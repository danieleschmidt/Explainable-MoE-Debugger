#!/bin/bash

# Deployment script for MoE Debugger
set -e

echo "🚀 Deploying MoE Debugger..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[DEPLOY]${NC} $1"
}

# Parse command line arguments
DEPLOYMENT_TYPE="local"
NAMESPACE="default"
CLEANUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --type TYPE        Deployment type: local, docker, k8s (default: local)"
            echo "  -n, --namespace NS     Kubernetes namespace (default: default)"
            echo "  --cleanup              Clean up existing deployment first"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

print_header "Deploying MoE Debugger (Type: $DEPLOYMENT_TYPE)"

# Cleanup existing deployment if requested
if [ "$CLEANUP" = true ]; then
    print_status "Cleaning up existing deployment..."
    
    case $DEPLOYMENT_TYPE in
        "docker")
            docker-compose down -v 2>/dev/null || true
            ;;
        "k8s")
            kubectl delete -f k8s-deployment.yaml -n "$NAMESPACE" 2>/dev/null || true
            ;;
        "local")
            # Kill any running processes on ports 8000, 3000
            pkill -f "uvicorn.*8000" 2>/dev/null || true
            pkill -f "next.*3000" 2>/dev/null || true
            ;;
    esac
    
    print_status "Cleanup completed"
fi

# Deploy based on type
case $DEPLOYMENT_TYPE in
    "local")
        print_header "Local Development Deployment"
        
        # Check if build exists
        if [ ! -d "dist" ]; then
            print_status "No build found, running build first..."
            ./scripts/build.sh
        fi
        
        # Start Redis if not running
        if ! pgrep redis-server > /dev/null; then
            print_status "Starting Redis..."
            redis-server --daemonize yes --port 6379
        fi
        
        # Start backend in background
        print_status "Starting backend server..."
        cd "$PROJECT_ROOT"
        export PYTHONPATH="$PROJECT_ROOT/src"
        python -m moe_debugger.server --host 0.0.0.0 --port 8000 &
        BACKEND_PID=$!
        
        # Wait for backend to start
        print_status "Waiting for backend to start..."
        for i in {1..30}; do
            if curl -f http://localhost:8000/api/status 2>/dev/null; then
                break
            fi
            sleep 1
        done
        
        # Start frontend in background
        print_status "Starting frontend server..."
        cd frontend
        npm run dev &
        FRONTEND_PID=$!
        
        # Wait for frontend to start
        print_status "Waiting for frontend to start..."
        for i in {1..30}; do
            if curl -f http://localhost:3000 2>/dev/null; then
                break
            fi
            sleep 1
        done
        
        print_status "✅ Local deployment completed!"
        echo "🌐 Frontend: http://localhost:3000"
        echo "🔧 Backend API: http://localhost:8000"
        echo "📊 API Docs: http://localhost:8000/docs"
        echo ""
        echo "To stop the servers:"
        echo "  kill $BACKEND_PID $FRONTEND_PID"
        
        ;;
        
    "docker")
        print_header "Docker Deployment"
        
        # Check if Docker is available
        if ! command -v docker &> /dev/null; then
            print_error "Docker is not installed"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            print_error "Docker Compose is not installed"
            exit 1
        fi
        
        # Build images if they don't exist
        if ! docker images moe-debugger:latest --format "table {{.Repository}}:{{.Tag}}" | grep -q moe-debugger:latest; then
            print_status "Building Docker images..."
            ./scripts/build.sh
        fi
        
        # Deploy with Docker Compose
        print_status "Starting services with Docker Compose..."
        docker-compose up -d
        
        print_status "Waiting for services to start..."
        sleep 10
        
        # Health check
        for i in {1..30}; do
            if curl -f http://localhost:8080/api/status 2>/dev/null; then
                break
            fi
            sleep 2
        done
        
        print_status "✅ Docker deployment completed!"
        echo "🌐 Application: http://localhost:80"
        echo "🔧 Direct Backend: http://localhost:8080"
        echo "📊 API Docs: http://localhost:8080/docs"
        echo "📈 Monitoring: http://localhost:9090 (Prometheus)"
        echo "📊 Dashboards: http://localhost:3001 (Grafana)"
        echo ""
        echo "View logs: docker-compose logs -f"
        echo "Stop services: docker-compose down"
        
        ;;
        
    "k8s")
        print_header "Kubernetes Deployment"
        
        # Check if kubectl is available
        if ! command -v kubectl &> /dev/null; then
            print_error "kubectl is not installed"
            exit 1
        fi
        
        # Check cluster connection
        if ! kubectl cluster-info &> /dev/null; then
            print_error "Unable to connect to Kubernetes cluster"
            exit 1
        fi
        
        # Create namespace if it doesn't exist
        kubectl create namespace "$NAMESPACE" 2>/dev/null || true
        
        # Deploy to Kubernetes
        print_status "Deploying to Kubernetes namespace: $NAMESPACE"
        kubectl apply -f k8s-deployment.yaml -n "$NAMESPACE"
        
        # Wait for deployment to be ready
        print_status "Waiting for deployment to be ready..."
        kubectl rollout status deployment/moe-debugger -n "$NAMESPACE" --timeout=300s
        
        # Get service information
        SERVICE_TYPE=$(kubectl get service moe-debugger -n "$NAMESPACE" -o jsonpath='{.spec.type}')
        
        if [ "$SERVICE_TYPE" = "LoadBalancer" ]; then
            print_status "Waiting for LoadBalancer IP..."
            EXTERNAL_IP=""
            while [ -z $EXTERNAL_IP ]; do
                sleep 10
                EXTERNAL_IP=$(kubectl get service moe-debugger -n "$NAMESPACE" --template="{{range .status.loadBalancer.ingress}}{{.ip}}{{end}}")
            done
            
            print_status "✅ Kubernetes deployment completed!"
            echo "🌐 Application: http://$EXTERNAL_IP"
            echo "🔧 Backend API: http://$EXTERNAL_IP/api"
        elif [ "$SERVICE_TYPE" = "NodePort" ]; then
            NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[0].address}')
            NODE_PORT=$(kubectl get service moe-debugger -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].nodePort}')
            
            print_status "✅ Kubernetes deployment completed!"
            echo "🌐 Application: http://$NODE_IP:$NODE_PORT"
        else
            print_status "✅ Kubernetes deployment completed!"
            echo "🌐 Application: http://localhost:8080 (use kubectl port-forward)"
            echo ""
            echo "To access the application:"
            echo "  kubectl port-forward service/moe-debugger 8080:80 -n $NAMESPACE"
        fi
        
        echo ""
        echo "Useful commands:"
        echo "  View pods: kubectl get pods -n $NAMESPACE"
        echo "  View logs: kubectl logs -l app=moe-debugger -n $NAMESPACE"
        echo "  Delete deployment: kubectl delete -f k8s-deployment.yaml -n $NAMESPACE"
        
        ;;
        
    *)
        print_error "Unknown deployment type: $DEPLOYMENT_TYPE"
        echo "Supported types: local, docker, k8s"
        exit 1
        ;;
esac

# Final status
echo
echo "=================================="
echo "🎉 DEPLOYMENT COMPLETED"
echo "=================================="
echo "Type: $DEPLOYMENT_TYPE"
if [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
    echo "Namespace: $NAMESPACE"
fi
echo "Status: ✅ Success"
echo "Timestamp: $(date)"
echo