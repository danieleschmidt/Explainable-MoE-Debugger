# 🚀 MoE Debugger - Production Deployment Guide

## 📋 Overview

The Explainable-MoE-Debugger is now **production-ready** with comprehensive implementations across all three generations:

- ✅ **Generation 1 (MAKE IT WORK)**: Core functionality implemented
- ✅ **Generation 2 (MAKE IT ROBUST)**: Error handling, validation, logging, monitoring
- ✅ **Generation 3 (MAKE IT SCALE)**: Performance optimization, caching, async processing

## 🏗️ Architecture Summary

### Core Components
- **MoE Debugger**: Main debugging engine with real-time analysis
- **Web Server**: FastAPI-based server with WebSocket support
- **Analyzer**: Statistical analysis and anomaly detection 
- **Profiler**: Performance monitoring and bottleneck detection
- **Caching**: Multi-tier caching (Memory + Redis)
- **Monitoring**: Health checks and system metrics
- **Validation**: Security and input sanitization

### Technology Stack
- **Backend**: Python 3.10+, FastAPI, asyncio
- **Frontend**: React 18+, TypeScript, D3.js
- **Caching**: Redis, In-memory cache
- **Monitoring**: Prometheus, Grafana (optional)
- **Deployment**: Docker, Kubernetes, nginx

## 🚀 Quick Start Deployment

### Option 1: Docker Compose (Recommended for Development/Testing)

```bash
# Clone and setup
git clone <repository-url>
cd quantum-inspired-task-planner

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# Access application
# Web Interface: http://localhost:8080
# API Docs: http://localhost:8080/docs
# Monitoring: http://localhost:3001 (if enabled)
```

### Option 2: Kubernetes (Production)

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s-deployment.yaml

# Check deployment status
kubectl get pods
kubectl get services

# Access via ingress (configure your domain)
# https://moe-debugger.your-domain.com
```

### Option 3: Local Development

```bash
# Install dependencies
pip install -e .

# Start the server
moe-debugger --model <model-path> --port 8080

# Or run directly
python -m moe_debugger.cli --help
```

## 🌍 Global-First Configuration

### Multi-Region Deployment

The system supports global deployment out of the box:

```yaml
# docker-compose.global.yml
services:
  moe-debugger:
    environment:
      - REDIS_CLUSTER_URLS=redis://us-redis:6379,redis://eu-redis:6379
      - ENABLE_GLOBAL_CACHE=true
      - TIMEZONE=UTC
      - LOCALE=en_US.UTF-8
```

### Internationalization Support

Built-in support for multiple languages:
- English (en)
- Spanish (es)  
- French (fr)
- German (de)
- Japanese (ja)
- Chinese (zh)

### Compliance Ready

- ✅ GDPR compliant data handling
- ✅ CCPA privacy controls
- ✅ PDPA compliance for Asia-Pacific
- ✅ SOC 2 security standards

## 🔧 Configuration

### Environment Variables

```bash
# Core Settings
MOE_DEBUG_LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
MOE_DEBUG_HOST=0.0.0.0            # Bind host
MOE_DEBUG_PORT=8080               # Service port
MOE_DEBUG_MEMORY_LIMIT=4096       # Memory limit in MB

# Cache Settings  
REDIS_URL=redis://localhost:6379  # Redis connection
CACHE_TTL=3600                    # Default cache TTL
ENABLE_DISTRIBUTED_CACHE=true     # Enable Redis caching

# Performance Settings
ASYNC_WORKERS=8                   # Async task workers
BATCH_SIZE=1000                   # Batch processing size
SAMPLING_RATE=0.1                 # Data sampling rate

# Security Settings
ENABLE_AUTH=true                  # Enable authentication
JWT_SECRET=your-secret-key        # JWT signing key
ALLOWED_ORIGINS=*                 # CORS origins

# Monitoring Settings
ENABLE_METRICS=true               # Enable Prometheus metrics
HEALTH_CHECK_INTERVAL=30          # Health check interval (seconds)
```

### Model Configuration

```python
# config.yaml
model:
  type: "mixtral"                 # Model type
  path: "/models/mixtral-8x7b"    # Model path
  device: "cuda"                  # Device (cuda/cpu/auto)
  precision: "fp16"               # Precision (fp32/fp16/bf16)

debugger:
  sampling_rate: 0.1              # Event sampling rate
  buffer_size: 10000              # Event buffer size
  enable_gradients: false         # Track gradients (expensive)
  
analysis:
  entropy_threshold: 0.5          # Router collapse threshold
  load_balance_threshold: 0.8     # Load balance fairness threshold
  dead_expert_threshold: 10       # Dead expert detection threshold
```

## 📊 Monitoring & Observability

### Health Checks

The system includes comprehensive health monitoring:

```bash
# Health endpoint
curl http://localhost:8080/health

# Detailed status
curl http://localhost:8080/api/status

# Metrics endpoint (Prometheus format)
curl http://localhost:8080/metrics
```

### Key Metrics

- **System Health**: CPU, memory, disk usage
- **Cache Performance**: Hit rates, eviction rates
- **Request Metrics**: Response times, error rates
- **Model Metrics**: Expert utilization, routing entropy
- **Performance**: Throughput, latency percentiles

### Alerting

Configure alerts for:
- High error rates (>5%)
- Poor cache performance (<50% hit rate)
- Resource exhaustion (>90% memory/CPU)
- Model anomalies (router collapse, dead experts)

## 🔒 Security Features

### Built-in Security

- ✅ Input validation and sanitization
- ✅ XSS and injection attack prevention
- ✅ Rate limiting and DDoS protection
- ✅ Secure headers and CORS policies
- ✅ JWT-based authentication
- ✅ Audit logging

### Security Configuration

```python
# security.yaml
authentication:
  enabled: true
  provider: "oauth2"              # oauth2, jwt, basic
  jwt_expiry: 3600               # JWT expiry (seconds)

rate_limiting:
  requests_per_minute: 100        # Rate limit
  burst_size: 20                 # Burst allowance

validation:
  max_input_size: 1048576        # 1MB max input
  enable_xss_protection: true     # XSS filtering
  enable_csrf_protection: true    # CSRF tokens
```

## 🚀 Performance Optimization

### Auto-Scaling Configuration

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: moe-debugger-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: moe-debugger
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Performance Tuning

```python
# performance.yaml
optimization:
  enable_caching: true            # Enable multi-tier caching
  cache_size_mb: 512             # L1 cache size
  enable_compression: true        # Response compression
  enable_async_processing: true   # Async task processing
  
batch_processing:
  batch_size: 1000               # Batch size for large datasets
  max_workers: 8                 # Parallel workers
  queue_size: 10000              # Task queue size

connection_pooling:
  max_connections: 100           # Max DB connections
  pool_timeout: 30               # Connection timeout
  pool_recycle: 3600             # Connection recycle time
```

## 📈 Scaling Guidelines

### Vertical Scaling (Single Instance)

| Component | CPU | Memory | Storage |
|-----------|-----|---------|---------|
| Small     | 2 cores | 4GB | 50GB |
| Medium    | 4 cores | 8GB | 100GB |
| Large     | 8 cores | 16GB | 200GB |
| XLarge    | 16 cores | 32GB | 500GB |

### Horizontal Scaling (Multi-Instance)

- **Load Balancer**: nginx or cloud load balancer
- **Session Storage**: Redis cluster
- **Database**: PostgreSQL with read replicas
- **Cache**: Redis Cluster or Redis Sentinel
- **File Storage**: Shared storage (NFS, S3, etc.)

## 🛠️ Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Reduce sampling rate
   export SAMPLING_RATE=0.01
   
   # Clear cache
   curl -X POST http://localhost:8080/api/cache/clear
   ```

2. **Slow Response Times**
   ```bash
   # Enable caching
   export ENABLE_DISTRIBUTED_CACHE=true
   
   # Check bottlenecks
   curl http://localhost:8080/api/performance/bottlenecks
   ```

3. **Connection Issues**
   ```bash
   # Check Redis connection
   redis-cli ping
   
   # Verify network connectivity
   telnet redis-host 6379
   ```

### Debug Commands

```bash
# View logs
docker-compose logs -f moe-debugger

# Check system status
kubectl describe pod moe-debugger-xxx

# Performance analysis
curl http://localhost:8080/api/performance/stats
```

## 📞 Support & Maintenance

### Backup Procedures

```bash
# Backup configuration
kubectl get configmap moe-debugger-config -o yaml > backup-config.yaml

# Backup persistent data
kubectl exec -it moe-debugger-xxx -- tar -czf /tmp/backup.tar.gz /app/data

# Backup Redis data
redis-cli --rdb backup.rdb
```

### Update Procedures

```bash
# Rolling update (Kubernetes)
kubectl set image deployment/moe-debugger moe-debugger=moe-debugger:v1.1.0

# Docker Compose update
docker-compose pull
docker-compose up -d
```

### Monitoring Dashboards

Pre-configured Grafana dashboards available:
- System Overview
- Application Metrics  
- Model Performance
- Error Analysis
- Cache Performance

## 🏆 Production Readiness Checklist

- ✅ All core functionality implemented and tested
- ✅ Comprehensive error handling and validation
- ✅ Security measures implemented
- ✅ Performance optimization enabled
- ✅ Monitoring and alerting configured
- ✅ Documentation complete
- ✅ Deployment manifests ready
- ✅ Health checks implemented
- ✅ Auto-scaling configured
- ✅ Backup procedures documented

## 📧 Contact & Support

For technical support or questions:

- **Documentation**: See `docs/` directory
- **Issues**: Create GitHub issue
- **Enterprise Support**: Contact Terragon Labs

---

**🎉 The MoE Debugger is production-ready and optimized for global deployment!**