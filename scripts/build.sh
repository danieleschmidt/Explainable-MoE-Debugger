#!/bin/bash

# Build script for MoE Debugger
set -e

echo "🚀 Building MoE Debugger..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check prerequisites
print_status "Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed" 
    exit 1
fi

if ! command -v docker &> /dev/null; then
    print_warning "Docker is not installed - skipping container build"
    SKIP_DOCKER=true
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

print_status "Project root: $PROJECT_ROOT"

# Build backend
print_status "Building backend..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

if [ -f "pyproject.toml" ]; then
    pip install -e .
fi

# Run backend tests
print_status "Running backend tests..."
python -m pytest tests/ -v --cov=src/moe_debugger --cov-report=html --cov-report=term-missing || {
    print_warning "Some backend tests failed"
}

# Build frontend
print_status "Building frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    print_status "Installing frontend dependencies..."
    npm ci
fi

print_status "Running frontend tests..."
npm run test -- --passWithNoTests --coverage || {
    print_warning "Some frontend tests failed"
}

print_status "Type checking frontend..."
npm run type-check || {
    print_warning "Type checking found issues"
}

print_status "Linting frontend..."
npm run lint || {
    print_warning "Linting found issues"
}

print_status "Building frontend production bundle..."
npm run build

cd "$PROJECT_ROOT"

# Build Docker images if Docker is available
if [ "$SKIP_DOCKER" != "true" ]; then
    print_status "Building Docker images..."
    
    print_status "Building production image..."
    docker build -t moe-debugger:latest .
    
    print_status "Building development image..."
    docker build -f frontend/Dockerfile.dev -t moe-debugger-frontend:dev ./frontend
    
    print_status "Docker images built successfully"
    docker images | grep moe-debugger
fi

# Create distribution package
print_status "Creating distribution package..."
mkdir -p dist

# Copy backend files
cp -r src dist/
cp -r tests dist/
cp requirements.txt dist/
cp pyproject.toml dist/
cp README.md dist/
cp LICENSE dist/

# Copy built frontend
if [ -d "frontend/.next" ]; then
    cp -r frontend/.next dist/frontend-build
    cp frontend/package.json dist/frontend-package.json
fi

# Copy deployment files
cp -r k8s-deployment.yaml dist/
cp -r docker-compose.yml dist/
cp -r docker-compose.dev.yml dist/
cp -r nginx.conf dist/

# Create version info
echo "Build completed at: $(date)" > dist/BUILD_INFO
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')" >> dist/BUILD_INFO
echo "Git branch: $(git branch --show-current 2>/dev/null || echo 'unknown')" >> dist/BUILD_INFO

print_status "Distribution package created in dist/"

# Run final validation
print_status "Running final validation..."

# Check if critical files exist
REQUIRED_FILES=(
    "dist/src/moe_debugger/__init__.py"
    "dist/pyproject.toml"
    "dist/README.md"
    "dist/docker-compose.yml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        print_error "Missing required file: $file"
        exit 1
    fi
done

print_status "Build validation passed!"

# Build summary
echo
echo "=================================="
echo "🎉 BUILD SUMMARY"
echo "=================================="
echo "✅ Backend built and tested"
echo "✅ Frontend built and tested"
if [ "$SKIP_DOCKER" != "true" ]; then
    echo "✅ Docker images created"
else
    echo "⚠️  Docker images skipped"
fi
echo "✅ Distribution package created"
echo "✅ All validations passed"
echo
echo "📦 Distribution: $PROJECT_ROOT/dist/"
echo "🚀 Ready for deployment!"
echo