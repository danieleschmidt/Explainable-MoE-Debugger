# Multi-stage Production Dockerfile for MoE Debugger
# Stage 1: Build frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy frontend package files
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --only=production

# Copy frontend source
COPY frontend/ ./
RUN npm run build

# Stage 2: Python backend with built frontend
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Node.js for frontend serving
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r moe && useradd -r -g moe moe

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/.next ./frontend/.next
COPY --from=frontend-builder /app/frontend/public ./frontend/public
COPY --from=frontend-builder /app/frontend/package.json ./frontend/package.json

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache && \
    chown -R moe:moe /app

# Switch to non-root user
USER moe

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Environment variables
ENV PYTHONPATH=/app/src
ENV MOE_DEBUG_LOG_LEVEL=INFO
ENV MOE_DEBUG_PORT=8080
ENV MOE_DEBUG_HOST=0.0.0.0

# Start command
CMD ["python", "-m", "moe_debugger.server", "--host", "0.0.0.0", "--port", "8080"]