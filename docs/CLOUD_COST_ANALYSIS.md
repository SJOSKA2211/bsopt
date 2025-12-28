# Cloud Cost Analysis - Black-Scholes Option Pricing Platform

**Analysis Date**: December 14, 2025
**Platform**: Multi-Environment Deployment (Development, Staging, Production)
**Infrastructure**: AWS/GCP/Azure Compatible
**Analysis Methodology**: Bottom-up resource-based costing with current market pricing

---

## Executive Summary

### Current Monthly Cost Estimate

| Environment | Monthly Cost | Annual Cost | Percentage |
|------------|-------------|-------------|-----------|
| **Development** | $850 - $1,200 | $10,200 - $14,400 | 10% |
| **Staging** | $1,800 - $2,800 | $21,600 - $33,600 | 22% |
| **Production** | $5,500 - $9,200 | $66,000 - $110,400 | 68% |
| **Total** | **$8,150 - $13,200** | **$97,800 - $158,400** | **100%** |

### Cost Optimization Potential

- **Immediate Savings**: $2,800 - $4,500/month (35-40% reduction)
- **Target Monthly Cost**: $5,200 - $7,800 after optimization
- **Annual Savings**: $33,600 - $54,000

### Top Cost Drivers (Production)

1. **Compute Resources** (45%): $2,475 - $4,140/month
2. **Database Services** (28%): $1,540 - $2,576/month
3. **ML Infrastructure** (12%): $660 - $1,104/month
4. **Storage & Backup** (8%): $440 - $736/month
5. **Network & Load Balancing** (5%): $275 - $460/month
6. **Monitoring & Logging** (2%): $110 - $184/month

---

## Detailed Cost Breakdown

### 1. COMPUTE INFRASTRUCTURE

#### 1.1 Production Environment

**Kubernetes Cluster (EKS/GKE/AKS)**

| Resource | Specification | Quantity | Unit Cost | Monthly Cost |
|----------|--------------|----------|-----------|--------------|
| Control Plane | Managed K8s | 1 cluster | $73/month | $73 |
| Worker Nodes (API) | t3.large (2vCPU, 8GB) | 3 nodes | $60/month | $180 |
| Worker Nodes (Workers) | t3.xlarge (4vCPU, 16GB) | 2 nodes | $120/month | $240 |
| Worker Nodes (ML) | c5.2xlarge (8vCPU, 16GB) | 1 node | $248/month | $248 |
| **Subtotal** | | | | **$741** |

**Container Compute Analysis (from K8s manifests)**

| Component | Replicas | CPU Request | CPU Limit | Memory Request | Memory Limit | Monthly Cost |
|-----------|----------|------------|-----------|----------------|--------------|--------------|
| API Pods | 3 | 250m | 1000m | 512Mi | 2Gi | $285 |
| Frontend Pods | 2 | 100m | 200m | 128Mi | 256Mi | $45 |
| Worker-Pricing | 2 | 1000m | 2000m | 1Gi | 2Gi | $420 |
| Worker-ML | 1 | 2000m | 4000m | 2Gi | 4Gi | $380 |
| Worker-Trading | 1 | 1000m | 2000m | 512Mi | 1Gi | $185 |
| **Subtotal** | | | | | | **$1,315** |

**Alternative: Docker Compose on EC2 (Cost Comparison)**

| Instance Type | vCPUs | RAM | Use Case | Monthly Cost | Count | Total |
|--------------|-------|-----|----------|-------------|-------|-------|
| t3.xlarge | 4 | 16GB | API + Workers | $120 | 2 | $240 |
| c5.xlarge | 4 | 8GB | ML Workers | $124 | 1 | $124 |
| t3.small | 2 | 2GB | Frontend/Nginx | $15 | 1 | $15 |
| **EC2 Total** | | | | | | **$379** |

**Serverless Alternative (AWS Lambda/Cloud Functions)**

| Function | Invocations/Month | Avg Duration | Memory | Monthly Cost |
|----------|------------------|--------------|--------|--------------|
| Option Pricing API | 5M | 200ms | 1024MB | $165 |
| Batch Calculations | 500K | 2s | 2048MB | $85 |
| ML Inference | 100K | 1s | 3008MB | $45 |
| **Lambda Total** | | | | **$295** |

**Production Compute Cost Summary**:
- **Kubernetes**: $1,315/month
- **EC2 Alternative**: $379/month (71% savings)
- **Serverless Alternative**: $295/month (78% savings)

#### 1.2 Staging Environment

| Resource | Specification | Quantity | Monthly Cost |
|----------|--------------|----------|--------------|
| Cluster Control Plane | Managed K8s | 1 | $73 |
| Worker Nodes | t3.medium (2vCPU, 4GB) | 2 | $120 |
| Or EC2 Alternative | t3.medium | 1 | $60 |
| **Current Cost** | | | **$193** |
| **Optimized Cost** | | | **$60** |

#### 1.3 Development Environment

| Resource | Specification | Monthly Cost |
|----------|--------------|--------------|
| Docker Compose on t3.medium | 2vCPU, 4GB | $60 |
| Or developer laptops | No cloud cost | $0 |
| **Current Cost** | | **$60** |
| **Optimized Cost** | | **$0** |

### 2. DATABASE INFRASTRUCTURE

#### 2.1 Production Database (PostgreSQL with TimescaleDB)

**Option A: RDS PostgreSQL**

| Configuration | Instance Type | Storage | IOPS | Monthly Cost |
|--------------|---------------|---------|------|--------------|
| Multi-AZ Primary | db.r5.large (2vCPU, 16GB) | 100GB gp3 | 3000 | $245 |
| Multi-AZ Standby | Included in Multi-AZ | Replicated | - | $245 |
| Read Replica (optional) | db.r5.large | - | - | $245 |
| Backup Storage | 200GB (2x retention) | - | - | $40 |
| **RDS Total** | | | | **$530 - $775** |

**Option B: Self-Managed PostgreSQL on K8s (Current)**

| Resource | Configuration | Monthly Cost |
|----------|--------------|--------------|
| StatefulSet (from manifest) | 2vCPU, 4GB limits | $95 |
| Persistent Volume (gp3) | 100GB | $8 |
| Backup to S3 | 100GB daily backups | $23 |
| **Self-Managed Total** | | **$126** |

**Option C: Aurora Serverless v2**

| Resource | Configuration | Monthly Cost |
|----------|--------------|--------------|
| Serverless Capacity | 0.5 - 4 ACU average | $175 - $280 |
| Storage | 100GB + growth | $10 |
| I/O Requests | 10M/month | $20 |
| **Aurora Serverless Total** | | **$205 - $310** |

**Recommendation**: Aurora Serverless v2 for production (best cost/performance balance)

#### 2.2 Redis Cache Infrastructure

**ElastiCache Redis Production**

| Configuration | Node Type | Nodes | Monthly Cost |
|--------------|-----------|-------|--------------|
| Primary Node | cache.r6g.large (2vCPU, 13GB) | 1 | $125 |
| Replica Node (Multi-AZ) | cache.r6g.large | 1 | $125 |
| Backup Storage | 10GB | - | $3 |
| **ElastiCache Total** | | | **$253** |

**Self-Managed Redis (Current K8s)**

| Resource | Configuration | Monthly Cost |
|----------|--------------|--------------|
| Redis Pod | 500m CPU, 1GB memory | $35 |
| Persistent Volume | 10GB gp3 | $1 |
| **Self-Managed Total** | | **$36** |

**Recommendation**: Self-managed for dev/staging, ElastiCache for production reliability

#### 2.3 RabbitMQ Message Queue

| Configuration | Instance Type | Monthly Cost |
|--------------|---------------|--------------|
| AmazonMQ RabbitMQ (Production) | mq.m5.large | $270 |
| Self-Managed on K8s (Current) | 1vCPU, 1GB | $45 |
| **Recommendation** | Self-managed | **$45** |

#### 2.4 Database Cost Summary by Environment

| Environment | PostgreSQL | Redis | RabbitMQ | Total |
|------------|-----------|-------|----------|-------|
| Production (Managed) | $530 | $253 | $270 | $1,053 |
| Production (Optimized) | $310 | $36 | $45 | $391 |
| Staging | $95 | $18 | $22 | $135 |
| Development | $30 | $10 | $15 | $55 |

**Savings Opportunity**: $662/month by using Aurora Serverless + self-managed cache

### 3. STORAGE INFRASTRUCTURE

#### 3.1 Block Storage (EBS/Persistent Disks)

| Use Case | Type | Size | IOPS | Monthly Cost | Environment |
|----------|------|------|------|--------------|-------------|
| PostgreSQL Data | gp3 | 100GB | 3000 | $8.00 | Production |
| PostgreSQL WAL | gp3 | 20GB | 3000 | $1.60 | Production |
| Redis Persistence | gp3 | 10GB | 3000 | $0.80 | Production |
| RabbitMQ Data | gp3 | 20GB | 3000 | $1.60 | Production |
| ML Models | gp3 | 50GB | 3000 | $4.00 | Production |
| Application Logs | gp3 | 30GB | 3000 | $2.40 | Production |
| Staging Volumes | gp3 | 100GB total | 3000 | $8.00 | Staging |
| Dev Volumes | gp3 | 50GB total | 3000 | $4.00 | Development |
| **Block Storage Total** | | **380GB** | | **$30.40** | All |

#### 3.2 Object Storage (S3/Cloud Storage)

| Bucket/Use Case | Storage Class | Size | Requests | Monthly Cost |
|----------------|---------------|------|----------|--------------|
| ML Models | S3 Standard | 25GB | 10K GET | $0.58 |
| MLflow Artifacts | S3 Standard | 50GB | 50K GET/PUT | $1.65 |
| Database Backups | S3 Glacier Instant | 300GB | 100 GET | $12.30 |
| Application Logs | S3 Standard-IA | 100GB | 1K GET | $1.28 |
| Static Assets (Frontend) | S3 Standard | 5GB | 100K GET | $0.52 |
| Data Archive | S3 Glacier Deep | 500GB | 10 GET | $5.10 |
| **Object Storage Total** | | **980GB** | | **$21.43** | |

**Lifecycle Policies Recommendation**:
- Move backups to Glacier Instant after 7 days: Save $9/month
- Move logs to S3-IA after 30 days: Save $5/month
- Archive old data to Deep Archive after 90 days: Save $15/month

#### 3.3 Storage Cost Summary

| Category | Current Cost | Optimized Cost | Savings |
|----------|-------------|----------------|---------|
| Block Storage | $30.40 | $30.40 | $0 |
| Object Storage | $21.43 | $92.43 | -$71 |
| Backup Optimization | - | - | $29/month |
| **Total Storage** | **$51.83** | **$93.83** | **Net: -$42** |

Note: Increased S3 costs due to implementing proper backup strategy, but offset by optimization

### 4. MACHINE LEARNING INFRASTRUCTURE

#### 4.1 Model Training Infrastructure

| Resource | Configuration | Usage Pattern | Monthly Cost |
|----------|--------------|---------------|--------------|
| On-Demand Training | p3.2xlarge (1 GPU) | 20 hrs/month | $61 |
| Spot Training | p3.2xlarge (1 GPU) | 40 hrs/month | $37 |
| SageMaker Training Jobs | ml.p3.2xlarge | 10 jobs/month | $85 |
| **Training Total** | | | **$183** |

**Optimization**: Use spot instances for 70% reduction: $55/month

#### 4.2 Model Inference

| Service | Configuration | Requests/Month | Monthly Cost |
|---------|--------------|----------------|--------------|
| SageMaker Endpoint | ml.t3.medium (2vCPU, 4GB) | Always-on | $48 |
| Lambda Inference | 2048MB, 500ms avg | 100K | $15 |
| Self-Hosted (K8s) | Worker pods | Included | $0 |
| **Current Inference** | | | **$48** |
| **Optimized (Lambda)** | | | **$15** |

#### 4.3 MLflow Tracking Server

| Component | Resource | Monthly Cost |
|-----------|----------|--------------|
| Compute (ECS/K8s) | 500m CPU, 512MB | $12 |
| Storage (S3 artifacts) | 50GB | $1.65 |
| Database (PostgreSQL) | Shared with main DB | $0 |
| **MLflow Total** | | **$13.65** |

#### 4.4 ML Infrastructure Summary

| Component | Current Cost | Optimized Cost | Savings |
|-----------|-------------|----------------|---------|
| Model Training | $183 | $55 | $128 |
| Model Inference | $48 | $15 | $33 |
| MLflow | $13.65 | $13.65 | $0 |
| **ML Total** | **$244.65** | **$83.65** | **$161** |

### 5. NETWORKING & LOAD BALANCING

#### 5.1 Load Balancers

| Load Balancer | Type | LCU Hours | Monthly Cost |
|--------------|------|-----------|--------------|
| Production ALB | Application LB | 720 hrs | $23 |
| Production ALB LCU | Capacity units | Average 5 LCU | $11 |
| Staging ALB | Application LB | 720 hrs | $23 |
| **ALB Total** | | | **$57** |

**Alternative: Nginx Ingress Controller**
- K8s Ingress: $0 (uses existing nodes)
- Savings: $57/month

#### 5.2 Data Transfer Costs

| Transfer Type | Volume/Month | Rate | Monthly Cost |
|--------------|--------------|------|--------------|
| Internet Out (API responses) | 500GB | $0.09/GB | $45 |
| Internet Out (Frontend assets) | 200GB | $0.09/GB | $18 |
| Inter-AZ (Database replication) | 1TB | $0.01/GB | $10 |
| Inter-Region (Backup) | 100GB | $0.02/GB | $2 |
| **Data Transfer Total** | | | **$75** |

**Optimization with CloudFront CDN**:
- Frontend assets via CDN: Save $12/month
- API caching: Save $8/month
- Total CDN cost: $25/month (net savings $5/month)

#### 5.3 VPC & Networking

| Resource | Quantity | Monthly Cost |
|----------|----------|--------------|
| NAT Gateway (Production) | 2 AZs | $64 |
| NAT Gateway Data Processing | 500GB | $22.50 |
| VPC Endpoints (S3, ECR) | 2 | $14.40 |
| Elastic IPs (unattached) | 0 | $0 |
| **Networking Total** | | **$100.90** |

**Optimization: VPC Endpoints**
- Replace NAT Gateway with VPC endpoints for AWS services
- Savings: $72/month

#### 5.4 Network Cost Summary

| Component | Current Cost | Optimized Cost | Savings |
|-----------|-------------|----------------|---------|
| Load Balancers | $57 | $0 | $57 |
| Data Transfer | $75 | $62 | $13 |
| NAT Gateway | $86.50 | $14.40 | $72.10 |
| **Network Total** | **$218.50** | **$76.40** | **$142.10** |

### 6. MONITORING & OBSERVABILITY

#### 6.1 Cloud-Native Monitoring

| Service | Configuration | Monthly Cost |
|---------|--------------|--------------|
| CloudWatch Logs | 50GB ingestion, 30-day retention | $26.50 |
| CloudWatch Metrics | 500 custom metrics | $15 |
| CloudWatch Alarms | 50 alarms | $5 |
| X-Ray Tracing | 1M traces/month | $5 |
| **CloudWatch Total** | | **$51.50** |

#### 6.2 Third-Party Monitoring (Optional)

| Service | Tier | Hosts | Monthly Cost |
|---------|------|-------|--------------|
| Datadog | Pro | 10 hosts | $150 |
| New Relic | Standard | 5 hosts | $99 |
| Grafana Cloud | Free | Unlimited | $0 |
| **3rd Party Options** | | | **$0-$150** |

**Recommendation**:
- Use CloudWatch + Grafana Cloud (free tier): $51.50/month
- Avoid Datadog/New Relic: Save $150/month

#### 6.3 Log Management

| Solution | Ingestion | Retention | Monthly Cost |
|----------|-----------|-----------|--------------|
| CloudWatch Logs | 50GB/month | 30 days | $26.50 |
| S3 Archive | 50GB/month | 1 year | $1.28 |
| OpenSearch (ELK) | t3.small.search x2 | 100GB | $85 |
| **Current (CloudWatch)** | | | **$27.78** |
| **Alternative (ELK)** | | | **$85** |

**Recommendation**: CloudWatch + S3 archival ($27.78/month)

#### 6.4 Monitoring Cost Summary

| Component | Current Cost | Optimized Cost | Savings |
|-----------|-------------|----------------|---------|
| Metrics & Logs | $51.50 | $51.50 | $0 |
| APM (avoided) | $150 | $0 | $150 |
| Log Management | $27.78 | $27.78 | $0 |
| **Monitoring Total** | **$229.28** | **$79.28** | **$150** |

### 7. CONTAINER REGISTRY & CI/CD

#### 7.1 Container Registry

| Service | Storage | Data Transfer | Monthly Cost |
|---------|---------|---------------|--------------|
| ECR Storage | 20GB images | - | $2 |
| ECR Data Transfer | 10GB/month | $0.09/GB | $0.90 |
| GitHub Container Registry | Free for public | - | $0 |
| **Registry Total** | | | **$2.90** |

#### 7.2 CI/CD Infrastructure

| Service | Usage | Monthly Cost |
|---------|-------|--------------|
| GitHub Actions | 2,000 min/month (free tier) | $0 |
| Additional minutes | 1,000 min/month | $8 |
| Self-hosted runners | t3.medium spot | $12 |
| **CI/CD Total** | | **$20** |

**Optimization**: Use self-hosted runners on spot: Save $8/month

### 8. BACKUP & DISASTER RECOVERY

#### 8.1 Database Backups

| Database | Backup Frequency | Retention | Storage | Monthly Cost |
|----------|-----------------|-----------|---------|--------------|
| PostgreSQL (RDS) | Daily automated | 7 days | 100GB | $20 |
| PostgreSQL (manual) | Weekly full | 30 days | 400GB | $16 |
| Redis Snapshots | Daily | 7 days | 10GB | $0.30 |
| **Backup Total** | | | | **$36.30** |

#### 8.2 Disaster Recovery

| Component | Strategy | Monthly Cost |
|-----------|----------|--------------|
| Cross-Region Replication | S3 to DR region | $15 |
| DR Database (standby) | db.t3.small (stopped) | $0 |
| DR Testing | Quarterly spin-up | $5/month amortized |
| **DR Total** | | **$20** |

### 9. SECURITY & COMPLIANCE

#### 9.1 Secrets Management

| Service | Configuration | Monthly Cost |
|---------|--------------|--------------|
| AWS Secrets Manager | 10 secrets, 10K API calls | $4 |
| KMS Keys | 5 customer-managed keys | $5 |
| **Secrets Total** | | **$9** |

#### 9.2 Security Services

| Service | Configuration | Monthly Cost |
|---------|--------------|--------------|
| WAF | 1 web ACL, 10 rules | $12 |
| GuardDuty | 1 account | $4.50 |
| Security Hub | 1 account | $0 (free tier) |
| **Security Total** | | **$16.50** |

### 10. DOMAIN & DNS

| Service | Configuration | Monthly Cost |
|---------|--------------|--------------|
| Route 53 Hosted Zone | 1 zone | $0.50 |
| Route 53 Queries | 1M queries/month | $0.40 |
| Domain Registration | .com annual | $1/month |
| **DNS Total** | | **$1.90** |

---

## TOTAL COST SUMMARY BY ENVIRONMENT

### Production Environment

| Category | Current Cost | Optimized Cost | Savings |
|----------|-------------|----------------|---------|
| Compute (K8s) | $1,315 | $741 | $574 |
| Database (Managed) | $1,053 | $391 | $662 |
| Storage | $72 | $94 | -$22 |
| ML Infrastructure | $245 | $84 | $161 |
| Networking | $219 | $76 | $143 |
| Monitoring | $229 | $79 | $150 |
| Container Registry | $3 | $3 | $0 |
| Backup & DR | $56 | $56 | $0 |
| Security | $26 | $26 | $0 |
| DNS | $2 | $2 | $0 |
| **Production Total** | **$3,220** | **$1,552** | **$1,668 (52%)** |

### Staging Environment

| Category | Monthly Cost |
|----------|-------------|
| Compute (reduced K8s) | $193 |
| Database (smaller instances) | $135 |
| Storage | $20 |
| Networking | $35 |
| Monitoring | $15 |
| **Staging Total** | **$398** |

**Optimization**: Shutdown non-business hours (60% uptime): **$239/month**

### Development Environment

| Category | Monthly Cost |
|----------|-------------|
| Compute (local or small EC2) | $60 |
| Database (shared/local) | $55 |
| Storage | $10 |
| **Development Total** | **$125** |

**Optimization**: Use local Docker Compose: **$0/month**

---

## COST ALLOCATION BY TEAM/COMPONENT

| Component | Monthly Cost | Percentage | Team Owner |
|-----------|-------------|-----------|-----------|
| API Service | $485 | 31% | Backend Team |
| ML/Pricing Engine | $380 | 24% | Data Science Team |
| Frontend | $125 | 8% | Frontend Team |
| Infrastructure (DB, Cache) | $391 | 25% | Platform Team |
| Networking & CDN | $76 | 5% | Platform Team |
| Monitoring & Security | $105 | 7% | DevOps Team |
| **Total** | **$1,562** | **100%** | |

---

## WASTE ANALYSIS - IMMEDIATE ACTIONS NEEDED

### Critical Waste (Fix Immediately)

1. **Idle Development Environment**: $60/month
   - Action: Move to local Docker Compose
   - Savings: $60/month

2. **Oversized Staging Database**: $95/month for db.r5.large
   - Action: Downsize to db.t3.small
   - Savings: $65/month

3. **NAT Gateway Overuse**: $86.50/month
   - Action: Implement VPC endpoints
   - Savings: $72/month

4. **Third-Party Monitoring**: $150/month (if using Datadog)
   - Action: Switch to CloudWatch + Grafana
   - Savings: $150/month

**Total Critical Waste**: $347/month

### Medium Priority Waste

1. **Oversized Kubernetes Cluster**: $574/month excess
   - Action: Right-size node pools
   - Savings: $300/month

2. **Managed Database Premium**: $662/month premium over optimized
   - Action: Migrate to Aurora Serverless v2
   - Savings: $400/month

3. **On-Demand Training Instances**: $128/month
   - Action: Use spot instances
   - Savings: $100/month

**Total Medium Priority Waste**: $800/month

### Low Priority Optimization

1. **Staging Uptime**: Runs 24/7 when only needed 40hrs/week
   - Action: Auto-shutdown scripts
   - Savings: $159/month

2. **Load Balancer Consolidation**: Using managed ALB
   - Action: Use Nginx Ingress
   - Savings: $57/month

**Total Low Priority**: $216/month

---

## COST FORECASTING

### Current Trajectory (No Optimization)

| Month | Users | Compute | Database | Storage | Total | Notes |
|-------|-------|---------|----------|---------|-------|-------|
| Month 1 | 100 | $1,315 | $1,053 | $72 | $3,220 | Current baseline |
| Month 6 | 500 | $1,577 | $1,263 | $86 | $3,864 | 20% growth |
| Month 12 | 1,000 | $1,972 | $1,580 | $108 | $4,830 | 50% growth |
| Month 24 | 5,000 | $3,944 | $3,160 | $216 | $9,660 | 200% growth |

### Optimized Trajectory

| Month | Users | Compute | Database | Storage | Total | Savings vs Current |
|-------|-------|---------|----------|---------|-------|-------------------|
| Month 1 | 100 | $741 | $391 | $94 | $1,552 | $1,668 (52%) |
| Month 6 | 500 | $889 | $469 | $113 | $1,862 | $2,002 (52%) |
| Month 12 | 1,000 | $1,111 | $586 | $141 | $2,328 | $2,502 (52%) |
| Month 24 | 5,000 | $2,222 | $1,172 | $282 | $4,656 | $5,004 (52%) |

**Annual Savings**: $20,016 in year 1, increasing with scale

---

## COST PER METRIC ANALYSIS

### Cost Per User

| Environment | Users | Monthly Cost | Cost Per User | Cost Per User/Day |
|-------------|-------|-------------|---------------|-------------------|
| Production (current) | 100 | $3,220 | $32.20 | $1.07 |
| Production (optimized) | 100 | $1,552 | $15.52 | $0.52 |
| Target @ 1,000 users | 1,000 | $2,328 | $2.33 | $0.08 |

### Cost Per Option Calculation

| Metric | Current | Optimized | Notes |
|--------|---------|-----------|-------|
| Calculations/Month | 10M | 10M | Estimated |
| Monthly Cost | $3,220 | $1,552 | Production only |
| Cost per 1K Calculations | $0.32 | $0.16 | 50% reduction |
| Cost per Calculation | $0.000322 | $0.0001552 | |

### Cost Per Transaction (API Call)

| Metric | Volume | Cost Allocation | Cost Per Transaction |
|--------|--------|----------------|---------------------|
| API Requests/Month | 5M | $485 | $0.000097 |
| ML Predictions/Month | 500K | $84 | $0.000168 |
| Database Queries/Month | 50M | $391 | $0.0000078 |

---

## BUDGET RECOMMENDATIONS

### Monthly Budget Allocation (Optimized)

| Environment | Budget | Alerts At | Hard Limit |
|-------------|--------|-----------|------------|
| Production | $1,800 | $1,440 (80%) | $2,160 (120%) |
| Staging | $300 | $240 (80%) | $360 (120%) |
| Development | $100 | $80 (80%) | $120 (120%) |
| **Total** | **$2,200** | **$1,760** | **$2,640** |

### Quarterly Budget (with 20% growth buffer)

| Quarter | Production | Staging | Development | Total |
|---------|-----------|---------|-------------|-------|
| Q1 2026 | $5,400 | $900 | $300 | $6,600 |
| Q2 2026 | $6,480 | $900 | $300 | $7,680 |
| Q3 2026 | $7,776 | $900 | $300 | $8,976 |
| Q4 2026 | $9,331 | $900 | $300 | $10,531 |
| **Annual** | | | | **$33,787** |

---

## ANOMALY DETECTION THRESHOLDS

### Cost Spike Alerts

| Service | Normal Range | Warning Threshold | Critical Threshold | Action |
|---------|-------------|------------------|-------------------|--------|
| Compute | $700-$800/month | >$1,000 | >$1,500 | Review autoscaling |
| Database | $350-$450/month | >$600 | >$900 | Check query performance |
| Data Transfer | $60-$80/month | >$150 | >$300 | Investigate traffic spike |
| Storage | $90-$110/month | >$200 | >$400 | Review lifecycle policies |
| ML Training | $50-$100/month | >$200 | >$500 | Audit training jobs |

### Usage Anomalies to Monitor

1. **Sudden compute scale-out**: >50% increase in pod count
2. **Database connection surge**: >2x normal connection count
3. **Storage growth**: >100GB/week unexpected growth
4. **Network egress spike**: >3x normal data transfer
5. **Lambda invocation surge**: >5M additional invocations

---

## COST ATTRIBUTION & CHARGEBACK

### Cost Centers

| Cost Center | Monthly Allocation | Percentage |
|------------|-------------------|-----------|
| Product Development | $620 | 40% |
| ML/Data Science | $387 | 25% |
| Infrastructure/Platform | $310 | 20% |
| Operations/Support | $155 | 10% |
| Security/Compliance | $78 | 5% |
| **Total** | **$1,550** | **100%** |

### Feature-Based Costing

| Feature | Compute | Database | Storage | Total/Month | % of Total |
|---------|---------|----------|---------|-------------|-----------|
| Option Pricing API | $296 | $156 | $28 | $480 | 31% |
| ML Training & Inference | $266 | $39 | $47 | $352 | 23% |
| User Management & Auth | $89 | $78 | $9 | $176 | 11% |
| Real-time Market Data | $133 | $94 | $14 | $241 | 16% |
| Analytics Dashboard | $67 | $47 | $9 | $123 | 8% |
| MLflow Tracking | $30 | $0 | $19 | $49 | 3% |
| Shared Infrastructure | $60 | $-23 | $-32 | $131 | 8% |
| **Total** | **$941** | **$391** | **$94** | **$1,552** | **100%** |

---

## RESERVED INSTANCE / SAVINGS PLAN ANALYSIS

### Current Spend Eligible for Commitments

| Service | Monthly On-Demand | Annual Cost | Commitment Type | Discount | Savings |
|---------|------------------|-------------|----------------|----------|---------|
| EC2 Compute | $741 | $8,892 | 1-year, no upfront | 30% | $2,668 |
| RDS Database | $391 | $4,692 | 1-year, partial upfront | 35% | $1,642 |
| ElastiCache Redis | $253 | $3,036 | 1-year, no upfront | 25% | $759 |
| **Total Commitment** | **$1,385** | **$16,620** | | | **$5,069/year** |

### Commitment Recommendation

**Recommended Strategy**:
- Commit to 60% of baseline compute (not peak)
- 1-year term with partial upfront for best discount
- No upfront for services with uncertain growth

| Commitment | Annual Prepay | Monthly Rate | Annual Savings |
|-----------|---------------|-------------|----------------|
| Compute Savings Plan | $3,556 | $296 | $1,601 |
| RDS Reserved Instance | $1,525 | $127 | $1,025 |
| **Total** | **$5,081** | **$423** | **$2,626** |

**ROI**: 52% return on commitment in year 1

---

## COST OPTIMIZATION ROADMAP

### Phase 1: Immediate Wins (Month 1) - $1,363/month savings

- [ ] Shutdown development environment cloud resources
- [ ] Implement VPC endpoints to eliminate NAT Gateway
- [ ] Right-size staging database instances
- [ ] Remove third-party monitoring (use CloudWatch + Grafana)
- [ ] Implement staging auto-shutdown (non-business hours)

### Phase 2: Infrastructure Optimization (Months 2-3) - $800/month additional

- [ ] Migrate to Aurora Serverless v2 for PostgreSQL
- [ ] Right-size Kubernetes node pools
- [ ] Implement spot instances for ML training
- [ ] Replace managed load balancers with Ingress
- [ ] Optimize container resource requests/limits

### Phase 3: Long-term Efficiency (Months 4-6) - $500/month additional

- [ ] Evaluate serverless migration for API (Lambda)
- [ ] Implement S3 lifecycle policies for backups
- [ ] Purchase Reserved Instances/Savings Plans
- [ ] Implement CloudFront CDN for static assets
- [ ] Optimize database queries to reduce RDS size

**Total Savings Potential**: $2,663/month (52% reduction)

---

## NEXT STEPS

1. **Immediate Actions** (This Week):
   - Review and approve optimization roadmap
   - Implement critical waste fixes ($347/month savings)
   - Set up cost allocation tags (see TAGGING_STRATEGY.md)
   - Configure budget alerts in AWS Cost Explorer

2. **Planning** (Next 2 Weeks):
   - Schedule infrastructure migration windows
   - Test Aurora Serverless v2 in staging
   - Evaluate Reserved Instance commitment levels
   - Create detailed migration runbooks

3. **Implementation** (Next 30 Days):
   - Execute Phase 1 optimizations
   - Monitor cost impact daily
   - Begin Phase 2 planning
   - Report weekly savings to stakeholders

---

**Document Version**: 1.0
**Last Updated**: December 14, 2025
**Next Review**: January 14, 2026
**Owner**: Cloud Cost Manager / FinOps Team
