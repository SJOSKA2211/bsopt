# High Availability Architecture for Black-Scholes Option Pricing Platform

## Architecture Overview

This document describes the high-availability (HA) PostgreSQL/TimescaleDB architecture designed to achieve 99.99% uptime (52 minutes downtime/year) with automatic failover, read scaling, and disaster recovery capabilities.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Application Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │   FastAPI    │  │   Workers    │  │  Notebooks   │                 │
│  │   Services   │  │   (Celery)   │  │   (Jupyter)  │                 │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                 │
│         │                  │                  │                          │
└─────────┼──────────────────┼──────────────────┼──────────────────────────┘
          │                  │                  │
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼──────────────────────────┐
│                   HAProxy Load Balancer                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Port 5000: Primary Connection (Write)                            │ │
│  │  Port 5001: Read-Only Replicas (Round-robin)                      │ │
│  │  Port 5002: PgBouncer Connection Pooling                          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└───────┬──────────────────┬──────────────────────────┬────────────────────┘
        │                  │                          │
   ┌────▼─────┐       ┌────▼─────┐              ┌────▼─────┐
   │ PgBouncer│       │ PgBouncer│              │ PgBouncer│
   │ (Primary)│       │(Standby1)│              │(Standby2)│
   └────┬─────┘       └────┬─────┘              └────┬─────┘
        │                  │                          │
┌───────▼──────────────────▼──────────────────────────▼────────────────────┐
│                      Patroni Cluster                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    etcd Consensus Layer                          │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │   │
│  │  │ etcd-1   │◄──►│ etcd-2   │◄──►│ etcd-3   │                   │   │
│  │  └──────────┘    └──────────┘    └──────────┘                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │   PRIMARY    │    │  STANDBY-1   │    │  STANDBY-2   │             │
│  │  PostgreSQL  │───►│  PostgreSQL  │    │  PostgreSQL  │             │
│  │  TimescaleDB │    │  TimescaleDB │◄───│  TimescaleDB │             │
│  │              │    │              │    │              │             │
│  │  Patroni     │    │  Patroni     │    │  Patroni     │             │
│  │  (Leader)    │    │  (Replica)   │    │  (Replica)   │             │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘             │
│         │                   │                    │                      │
│    Streaming                │               Streaming                   │
│    Replication              │               Replication                 │
│    (Synchronous)            │               (Asynchronous)              │
└─────────┼───────────────────┼────────────────────┼──────────────────────┘
          │                   │                    │
┌─────────▼───────────────────▼────────────────────▼──────────────────────┐
│                      pgBackRest Backup Repository                       │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Full Backups (Weekly) + Incremental (Daily) + WAL Archive       │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │  │
│  │  │   Local     │───►│    S3/R2    │───►│   Offsite   │          │  │
│  │  │   Storage   │    │   Primary   │    │   Replica   │          │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      Monitoring & Alerting Stack                        │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐           │
│  │  Prometheus    │◄─│ postgres_      │  │  AlertManager  │           │
│  │  (Metrics)     │  │   exporter     │  │   (Alerts)     │           │
│  └────────┬───────┘  └────────────────┘  └────────┬───────┘           │
│           │                                        │                    │
│  ┌────────▼───────┐                       ┌───────▼────────┐           │
│  │    Grafana     │                       │   PagerDuty/   │           │
│  │  (Dashboards)  │                       │   Slack/Email  │           │
│  └────────────────┘                       └────────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Specifications

### 1. PostgreSQL/TimescaleDB Cluster

**Node Configuration (Per Server):**
- PostgreSQL Version: 16.x
- TimescaleDB Version: 2.14+
- CPU: 16 cores (32 vCPU with hyperthreading)
- RAM: 64GB
- Storage: 1TB NVMe SSD (RAID 10 for production)
- Network: 10Gbps

**Primary Node:**
- Handles all write operations
- Synchronous replication to Standby-1
- Asynchronous replication to Standby-2
- WAL archiving to pgBackRest repository

**Standby Nodes (2):**
- Standby-1: Synchronous replica (zero data loss on failover)
- Standby-2: Asynchronous replica (read scaling + disaster recovery)
- Both capable of promotion to primary
- Hot standby mode (accepts read-only queries)

### 2. Patroni Configuration

**Purpose:** Automated failover and cluster management

**Key Features:**
- Leader election via etcd
- Automatic health checks (every 10 seconds)
- Failover time: <30 seconds
- Split-brain prevention
- Rolling restarts with zero downtime

**Failover Triggers:**
- Primary node failure detection (3 consecutive health check failures)
- Network partition detection
- PostgreSQL service crash
- Manual failover via REST API

**Synchronous vs Asynchronous Replication:**
```yaml
synchronous_mode: true
synchronous_mode_strict: false  # Allow async if sync replica is down
synchronous_node_count: 1       # One synchronous replica required
```

### 3. HAProxy Load Balancer

**Purpose:** Connection routing and health-aware load balancing

**Listening Ports:**
- Port 5000: Primary (write) connections
- Port 5001: Read-only replica pool (round-robin)
- Port 5002: PgBouncer connection pool

**Health Checks:**
- HTTP check via Patroni REST API (port 8008)
- Interval: 2 seconds
- Timeout: 1 second
- Rise: 2 (consecutive successes to mark healthy)
- Fall: 3 (consecutive failures to mark unhealthy)

**Load Balancing Algorithm:**
- Primary: First available (single node)
- Replicas: Round-robin with least connections
- Sticky sessions: Not required for read replicas

### 4. PgBouncer Connection Pooling

**Purpose:** Connection multiplexing to support 10,000+ concurrent connections

**Pool Configuration:**
- Pool mode: Transaction (best for web applications)
- Max client connections: 10,000
- Default pool size: 25 per database
- Reserve pool: 5
- Max DB connections: 100 per node

**Connection String:**
```
postgresql://user:password@haproxy:5002/options_platform?pool_size=10&max_overflow=0
```

### 5. etcd Cluster

**Purpose:** Distributed configuration store and consensus

**Cluster Size:** 3 nodes (minimum for quorum)

**Key Parameters:**
- Heartbeat interval: 100ms
- Election timeout: 1000ms
- Snapshot count: 10,000

**Data Stored:**
- Patroni cluster state
- Leader lease
- Configuration changes
- Member metadata

### 6. pgBackRest Backup System

**Purpose:** Enterprise-grade backup and point-in-time recovery

**Backup Strategy:**
- Full backup: Weekly (Sunday 00:00 UTC)
- Differential backup: Daily (00:00 UTC)
- Incremental backup: Every 6 hours
- WAL archiving: Continuous (every 60 seconds or 16MB)

**Retention Policy:**
- Full backups: 4 weeks
- Differential backups: 2 weeks
- Incremental backups: 7 days
- WAL archives: 30 days

**Storage Locations:**
- Primary: Local SSD (/var/lib/pgbackrest)
- Secondary: S3-compatible object storage (Cloudflare R2)
- Tertiary: Offsite backup (different region)

**Recovery Objectives:**
- RTO (Recovery Time Objective): <15 minutes
- RPO (Recovery Point Objective): <5 minutes

## Performance Targets

### Availability
- Uptime SLA: 99.99% (52 minutes downtime/year)
- Planned maintenance: Zero downtime (rolling updates)
- Unplanned failover: <30 seconds

### Throughput
- Transactions per second: 50,000+ TPS
- Concurrent connections: 10,000+
- Query latency (p95): <100ms
- Replication lag: <10ms (synchronous), <100ms (asynchronous)

### Scalability
- Read scaling: Linear (add more replicas)
- Write scaling: Single primary (vertical scaling)
- Storage: Horizontal scaling via TimescaleDB partitioning
- Future: Sharding for multi-region deployment

## Network Architecture

### Internal Network (Private)
```
10.0.1.0/24  - Database cluster
10.0.2.0/24  - etcd cluster
10.0.3.0/24  - Backup infrastructure
10.0.4.0/24  - Monitoring stack
```

### Security Groups

**PostgreSQL Security Group:**
- Inbound 5432 from HAProxy only
- Inbound 8008 from HAProxy (Patroni API)
- Inbound 22 from bastion host (SSH)

**HAProxy Security Group:**
- Inbound 5000-5002 from application layer
- Inbound 80/443 from monitoring
- Outbound 5432 to PostgreSQL nodes

**etcd Security Group:**
- Inbound 2379-2380 from Patroni nodes
- Inbound 2379 from monitoring

## Deployment Topology

### Single Region (High Availability)
```
┌──────────────────────────────────────────────────────────┐
│                    Availability Zone 1                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   PRIMARY    │  │   etcd-1     │  │  HAProxy-1   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└──────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────┐
│                    Availability Zone 2                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  STANDBY-1   │  │   etcd-2     │  │  HAProxy-2   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└──────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────┐
│                    Availability Zone 3                   │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │  STANDBY-2   │  │   etcd-3     │                     │
│  └──────────────┘  └──────────────┘                     │
└──────────────────────────────────────────────────────────┘
```

### Multi-Region (Disaster Recovery)
```
┌────────────────────────────────────────┐
│          Primary Region (US-East)      │
│  ┌──────────┐  ┌──────────┐           │
│  │ PRIMARY  │  │STANDBY-1 │           │
│  │  (AZ-1)  │  │  (AZ-2)  │           │
│  └──────────┘  └──────────┘           │
│         │                              │
│         │ Async Replication            │
│         ▼                              │
└────────────────────────────────────────┘
         │
         │ Cross-Region Replication
         │ (Asynchronous, <5s lag)
         ▼
┌────────────────────────────────────────┐
│      Secondary Region (US-West)        │
│  ┌──────────┐  ┌──────────┐           │
│  │STANDBY-2 │  │ Backup   │           │
│  │  (AZ-1)  │  │  Repo    │           │
│  └──────────┘  └──────────┘           │
└────────────────────────────────────────┘
```

## Configuration Management

### Infrastructure as Code
- Terraform: Cloud resource provisioning
- Ansible: Configuration management
- Helm: Kubernetes deployment (optional)

### Version Control
- All configurations in Git repository
- Automated deployment via CI/CD
- Rollback capability for config changes

### Environment Isolation
- Development: Single node
- Staging: 2-node cluster
- Production: 3-node HA cluster

## Maintenance Procedures

### Rolling Updates (Zero Downtime)
1. Update standby replicas one at a time
2. Verify replication health after each update
3. Switchover to updated replica
4. Update old primary (now standby)
5. Switchover back if desired

### Planned Failover (Manual)
```bash
# Via Patroni REST API
curl -s http://patroni-node:8008/switchover \
  -XPOST \
  -d '{"leader":"pg-primary","candidate":"pg-standby-1"}'

# Via patronictl
patronictl switchover options-cluster \
  --master pg-primary \
  --candidate pg-standby-1
```

### Scaling Read Capacity
1. Provision new server
2. Install PostgreSQL + Patroni
3. Add to Patroni cluster configuration
4. Patroni automatically sets up replication
5. Add to HAProxy backend pool

## Monitoring Integration

### Key Metrics
- Replication lag (streaming_lag_bytes, replay_lag)
- Connection pool saturation (clients_waiting)
- Query performance (pg_stat_statements)
- Disk I/O (pg_stat_io)
- Lock contention (pg_locks)

### Critical Alerts
- Primary node down (PagerDuty)
- Replication lag >60 seconds
- Connection pool >80% saturated
- Disk space <20% free
- Failed backup jobs

## Disaster Recovery Scenarios

### Scenario 1: Primary Node Failure
**Detection:** Patroni health checks fail (30 seconds)
**Action:** Automatic failover to Standby-1 (synchronous replica)
**Downtime:** <30 seconds
**Data Loss:** Zero (synchronous replication)

### Scenario 2: Availability Zone Failure
**Detection:** All nodes in AZ unreachable
**Action:** Patroni promotes replica in healthy AZ
**Downtime:** <60 seconds
**Data Loss:** Minimal (<5 seconds of transactions if async replica promoted)

### Scenario 3: Region-Wide Disaster
**Detection:** Entire primary region unavailable
**Action:** Manual promotion of cross-region replica
**Downtime:** 5-15 minutes (manual intervention required)
**Data Loss:** Up to 5 seconds (async replication lag)

### Scenario 4: Data Corruption
**Detection:** Application errors, integrity check failures
**Action:** Point-in-time recovery from pgBackRest
**Downtime:** 5-15 minutes (depending on database size)
**Data Loss:** <5 minutes (last WAL archive interval)

## Cost Analysis

### Monthly Infrastructure Costs (AWS Example)

**Compute (EC2 Instances):**
- 3x r6i.4xlarge (Primary + Standbys): $2,400/month
- 2x t3.medium (HAProxy): $60/month
- 3x t3.small (etcd): $45/month

**Storage:**
- 3TB EBS gp3 (Database): $240/month
- 5TB S3 (Backups): $115/month

**Network:**
- Data transfer: $500/month (estimated)

**Total Estimated Cost:** $3,360/month ($40,320/year)

**Cost Optimization:**
- Reserved instances: -40% ($24,192/year)
- Spot instances for standby nodes: -60% on standbys
- Cloudflare R2 instead of S3: -$90/month

## Security Considerations

### Encryption
- At-rest: LUKS disk encryption + PostgreSQL TDE
- In-transit: SSL/TLS for all connections (required)
- Backup encryption: AES-256

### Access Control
- Network isolation (private subnets)
- Bastion host for SSH access
- Role-based access control (RBAC)
- Audit logging enabled

### Compliance
- SOC 2 Type II
- GDPR compliance (data residency controls)
- HIPAA compliance (optional BAA)

## Implementation Checklist

- [ ] Provision infrastructure (Terraform)
- [ ] Install PostgreSQL/TimescaleDB on all nodes
- [ ] Configure Patroni cluster
- [ ] Set up etcd cluster
- [ ] Configure HAProxy load balancer
- [ ] Install and configure PgBouncer
- [ ] Set up pgBackRest repository
- [ ] Configure backup schedules
- [ ] Install monitoring stack (Prometheus/Grafana)
- [ ] Configure alerting rules
- [ ] Test failover scenarios
- [ ] Perform disaster recovery drills
- [ ] Document runbooks
- [ ] Train operations team

## Operational Runbooks

See the following documents for detailed procedures:
- `/home/kamau/comparison/database/failover-playbook.md` - Failover procedures
- `/home/kamau/comparison/database/backup-policy.md` - Backup and recovery
- `/home/kamau/comparison/database/tuning.sql` - Performance tuning

## References

- PostgreSQL HA Documentation: https://www.postgresql.org/docs/current/high-availability.html
- Patroni Documentation: https://patroni.readthedocs.io/
- TimescaleDB Best Practices: https://docs.timescale.com/
- pgBackRest User Guide: https://pgbackrest.org/user-guide.html
- HAProxy Configuration Manual: https://www.haproxy.org/documentation.html

---

**Document Version:** 1.0
**Last Updated:** 2025-12-14
**Author:** Database Resilience Architect
**Review Cycle:** Quarterly
