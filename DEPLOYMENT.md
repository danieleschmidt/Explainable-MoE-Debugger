# MoE Debugger - Production Deployment Guide

## Overview

This guide covers deploying the MoE Debugger in production environments using Docker, Docker Compose, and Kubernetes.

## Prerequisites

- Docker Engine 20.10+ and Docker Compose 2.0+
- Kubernetes cluster (for K8s deployment)
- SSL certificates (for HTTPS)
- Sufficient compute resources (minimum 4GB RAM, 2 CPU cores)

## Quick Start with Docker Compose

```bash
# Clone and build
git clone <repository-url>
cd Explainable-MoE-Debugger

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f moe-debugger
```

## Production Deployment Options

### 1. Docker Compose (Recommended for single-node deployments)

**Features:**
- Full stack deployment with Redis caching
- Nginx reverse proxy with SSL termination
- Health checks and automatic restarts
- Volume persistence for logs and data

**Configuration:**
```yaml
# docker-compose.yml includes:
# - MoE Debugger main service
# - Redis for caching
# - Nginx for load balancing
# - Monitoring with Prometheus/Grafana (optional)
```

**Start production stack:**
```bash
docker-compose -f docker-compose.yml up -d
```

### 2. Kubernetes (Recommended for multi-node deployments)

**Features:**
- Horizontal scaling with 3+ replicas
- Persistent storage for data and logs
- Ingress with SSL termination
- Resource limits and health checks
- Service discovery and load balancing

**Deploy to Kubernetes:**
```bash
# Apply all manifests
kubectl apply -f k8s-deployment.yaml

# Check deployment status
kubectl get pods -l app=moe-debugger
kubectl get services

# View logs
kubectl logs -l app=moe-debugger -f
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MOE_DEBUG_HOST` | `0.0.0.0` | Host to bind the server |
| `MOE_DEBUG_PORT` | `8080` | Port for the web server |
| `MOE_DEBUG_LOG_LEVEL` | `INFO` | Logging level |
| `MOE_DEBUG_MEMORY_LIMIT` | `4096` | Memory limit in MB |
| `REDIS_URL` | `redis://redis:6379` | Redis connection URL |

### Volume Mounts

- `/app/logs` - Application logs
- `/app/data` - Persistent debugging data
- `/app/cache` - Cache directory (can be ephemeral)

## Scaling and Performance

### Horizontal Scaling
```bash
# Docker Compose
docker-compose up -d --scale moe-debugger=3

# Kubernetes
kubectl scale deployment moe-debugger --replicas=5
```

### Resource Requirements

**Minimum (Development):**
- 2 CPU cores
- 4GB RAM
- 20GB storage

**Recommended (Production):**
- 4+ CPU cores
- 8GB+ RAM
- 100GB+ storage

**High Load (Large Models):**
- 8+ CPU cores
- 16GB+ RAM
- 500GB+ storage

## Monitoring and Health Checks

### Health Endpoints
- `GET /health` - Application health status
- `GET /metrics` - Prometheus metrics

### Monitoring Stack (Optional)
Enable with profile: `docker-compose --profile monitoring up -d`

- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Visualization dashboard (port 3001)

### Log Management
```bash
# View logs
docker-compose logs -f moe-debugger

# Log rotation is handled automatically
# Logs are stored in ./logs/ directory
```

## Security

### SSL/TLS Configuration
1. Obtain SSL certificates
2. Place certificates in `./ssl/` directory
3. Update nginx.conf with certificate paths
4. Restart nginx service

### Network Security
- All services run in isolated Docker network
- Only necessary ports are exposed
- Rate limiting configured in Nginx
- Security headers enabled

### User Security
- Services run as non-root user
- File permissions properly configured
- Secrets managed via environment variables

## Backup and Recovery

### Data Backup
```bash
# Backup data volumes
docker run --rm -v moe_data:/data -v $(pwd):/backup alpine tar czf /backup/moe-data-backup.tar.gz /data

# Backup Redis data
docker exec redis redis-cli BGSAVE
```

### Recovery
```bash
# Restore data
docker run --rm -v moe_data:/data -v $(pwd):/backup alpine sh -c "cd /data && tar xzf /backup/moe-data-backup.tar.gz --strip 1"
```

## Troubleshooting

### Common Issues

**Service won't start:**
```bash
# Check logs
docker-compose logs moe-debugger

# Check resource usage
docker stats
```

**High memory usage:**
- Increase `MOE_DEBUG_MEMORY_LIMIT`
- Scale horizontally
- Check for memory leaks in logs

**Performance issues:**
- Enable Redis caching
- Scale to multiple replicas
- Monitor resource usage

### Debug Commands
```bash
# Enter container
docker-compose exec moe-debugger bash

# Check service status
curl http://localhost:8080/health

# Monitor resource usage
docker-compose exec moe-debugger top
```

## Production Checklist

- [ ] SSL certificates configured
- [ ] Environment variables set
- [ ] Volume persistence configured
- [ ] Health checks passing
- [ ] Monitoring enabled
- [ ] Backup strategy implemented
- [ ] Security hardening applied
- [ ] Load testing completed
- [ ] Documentation updated

## Support

For deployment issues:
1. Check logs with `docker-compose logs`
2. Verify configuration files
3. Review resource requirements
4. Consult troubleshooting section

For development questions, see the main README.md and CONTRIBUTING.md files.