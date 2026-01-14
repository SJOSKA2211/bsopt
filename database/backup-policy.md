# Backup and Recovery Policy - Black-Scholes Option Pricing Platform

## Executive Summary

This document defines the backup and disaster recovery strategy for the Black-Scholes Option Pricing Platform PostgreSQL/TimescaleDB database cluster. The policy is designed to achieve:

- **Recovery Time Objective (RTO):** <15 minutes
- **Recovery Point Objective (RPO):** <5 minutes
- **Backup Retention:** 30 days full, 2 years compliance archives
- **Compliance:** SOC 2, GDPR, HIPAA-ready

## Backup Architecture

### pgBackRest Overview

We use pgBackRest for enterprise-grade backup and recovery:

- Parallel backup and restore
- Full, differential, and incremental backups
- WAL archiving for point-in-time recovery (PITR)
- Built-in encryption and compression
- Multiple repository support (local + cloud)
- Backup verification and integrity checks

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│              PostgreSQL Cluster (Patroni Managed)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   PRIMARY    │  │  STANDBY-1   │  │  STANDBY-2   │          │
│  │              │  │              │  │              │          │
│  │  WAL Archive │  │  WAL Archive │  │  WAL Archive │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┴─────────────────┘                   │
│                           │                                     │
└───────────────────────────┼─────────────────────────────────────┘
                            │
                            │ WAL Streaming
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   pgBackRest Repository                         │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Local Repository (Primary)                   │ │
│  │  /var/lib/pgbackrest/                                     │ │
│  │  - Full backups (weekly)                                  │ │
│  │  - Differential backups (daily)                           │ │
│  │  - Incremental backups (6 hourly)                         │ │
│  │  - WAL archives (continuous)                              │ │
│  └───────────────────┬───────────────────────────────────────┘ │
└──────────────────────┼─────────────────────────────────────────┘
                       │
                       │ Replication
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│               Cloud Repository (S3/Cloudflare R2)               │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  s3://backups.options-platform.com/pgbackrest/            │ │
│  │  - Encrypted backups (AES-256)                            │ │
│  │  - Compressed (zstd level 3)                              │ │
│  │  - Versioned (S3 versioning enabled)                      │ │
│  │  - Lifecycle policies (30 days retention)                 │ │
│  └───────────────────┬───────────────────────────────────────┘ │
└──────────────────────┼─────────────────────────────────────────┘
                       │
                       │ Cross-Region Replication
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│           Offsite Repository (Different Region)                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  s3://backups-dr.options-platform.com/pgbackrest/         │ │
│  │  - Geographic redundancy                                  │ │
│  │  - Long-term compliance archives (2 years)                │ │
│  │  - Immutable storage (WORM compliance)                    │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Backup Strategy

### Backup Types and Schedule

| Backup Type    | Frequency  | Day/Time         | Retention | Size Estimate |
|----------------|------------|------------------|-----------|---------------|
| Full           | Weekly     | Sunday 00:00 UTC | 4 weeks   | 500GB         |
| Differential   | Daily      | 00:00 UTC        | 2 weeks   | 50GB          |
| Incremental    | 6-hourly   | 00/06/12/18      | 7 days    | 10GB          |
| WAL Archive    | Continuous | Every 60s/16MB   | 30 days   | 200GB/day     |

### Storage Requirements

```
Weekly Storage:
- Full backup: 500GB × 4 = 2TB
- Differential: 50GB × 14 = 700GB
- Incremental: 10GB × 7 × 4 = 280GB
- WAL archives: 200GB × 30 = 6TB

Total Primary Storage: ~9TB
Total Cloud Storage: ~9TB (replicated)
Total with Compression (3:1 ratio): ~3TB per repository
```

### Backup Configuration

**pgBackRest Configuration (`/etc/pgbackrest/pgbackrest.conf`):**

```ini
[global]
# Repository configuration
repo1-path=/var/lib/pgbackrest
repo1-retention-full=4
repo1-retention-diff=14
repo1-cipher-type=aes-256-cbc
repo1-cipher-pass=${PGBACKREST_REPO1_CIPHER_PASS}

# Cloud repository (S3-compatible)
repo2-type=s3
repo2-s3-bucket=backups-options-platform
repo2-s3-endpoint=s3.amazonaws.com
repo2-s3-region=us-east-1
repo2-s3-key=${AWS_ACCESS_KEY_ID}
repo2-s3-key-secret=${AWS_SECRET_ACCESS_KEY}
repo2-path=/pgbackrest
repo2-retention-full=4
repo2-retention-diff=14
repo2-cipher-type=aes-256-cbc
repo2-cipher-pass=${PGBACKREST_REPO2_CIPHER_PASS}

# Offsite repository (cross-region)
repo3-type=s3
repo3-s3-bucket=backups-dr-options-platform
repo3-s3-endpoint=s3.us-west-2.amazonaws.com
repo3-s3-region=us-west-2
repo3-s3-key=${AWS_ACCESS_KEY_ID}
repo3-s3-key-secret=${AWS_SECRET_ACCESS_KEY}
repo3-path=/pgbackrest
repo3-retention-full=52  # 1 year of weekly backups
repo3-cipher-type=aes-256-cbc
repo3-cipher-pass=${PGBACKREST_REPO3_CIPHER_PASS}

# Performance settings
process-max=8
compress-type=zst
compress-level=3
compress-level-network=3

# Logging
log-level-console=info
log-level-file=debug
log-path=/var/log/pgbackrest

[options-cluster]
# PostgreSQL cluster configuration
pg1-path=/var/lib/postgresql/16/main
pg1-port=5432
pg1-socket-path=/var/run/postgresql

# Patroni integration (backup from standby to reduce primary load)
pg2-host=pg-standby-1.internal
pg2-path=/var/lib/postgresql/16/main
pg2-port=5432

# Recovery settings
recovery-option=primary_conninfo=host=pg-primary.internal port=5432 user=replicator
recovery-option=restore_command='/usr/bin/pgbackrest --stanza=options-cluster archive-get %f "%p"'
```

## Backup Procedures

### 1. Full Backup (Weekly)

**Command:**
```bash
pgbackrest --stanza=options-cluster --type=full backup
```

**Automated via cron:**
```cron
# /etc/cron.d/pgbackrest-full
0 0 * * 0 postgres /usr/bin/pgbackrest --stanza=options-cluster --type=full backup
```

**Expected Duration:** 2-4 hours for 500GB database

**Validation:**
```bash
# Verify backup completed successfully
pgbackrest --stanza=options-cluster info

# Expected output:
# stanza: options-cluster
#     status: ok
#     cipher: aes-256-cbc
#
#     db (current)
#         wal archive min/max (16): 000000010000000000000001/000000010000000000000064
#
#         full backup: 20251214-000000F
#             timestamp start/stop: 2025-12-14 00:00:00 / 2025-12-14 02:34:12
#             wal start/stop: 000000010000000000000001 / 000000010000000000000001
#             database size: 500.0GB, database backup size: 500.0GB
#             repo1: backup set size: 167.3GB, backup size: 167.3GB
```

### 2. Differential Backup (Daily)

**Command:**
```bash
pgbackrest --stanza=options-cluster --type=diff backup
```

**Automated via cron:**
```cron
# /etc/cron.d/pgbackrest-diff
0 0 * * 1-6 postgres /usr/bin/pgbackrest --stanza=options-cluster --type=diff backup
```

**Expected Duration:** 20-60 minutes

### 3. Incremental Backup (6-Hourly)

**Command:**
```bash
pgbackrest --stanza=options-cluster --type=incr backup
```

**Automated via cron:**
```cron
# /etc/cron.d/pgbackrest-incr
0 */6 * * * postgres /usr/bin/pgbackrest --stanza=options-cluster --type=incr backup
```

**Expected Duration:** 5-15 minutes

### 4. WAL Archiving (Continuous)

**PostgreSQL Configuration (managed by Patroni):**
```ini
# postgresql.conf
archive_mode = on
archive_command = 'pgbackrest --stanza=options-cluster archive-push %p'
archive_timeout = 60  # Force archive every 60 seconds
```

**Monitoring WAL Archive:**
```bash
# Check WAL archive status
pgbackrest --stanza=options-cluster info

# View pending WAL files
ls -lh /var/lib/postgresql/16/main/pg_wal/archive_status/
```

## Recovery Procedures

### Scenario 1: Point-in-Time Recovery (PITR)

**Use Case:** Recover from accidental data deletion or corruption

**Procedure:**

1. **Stop PostgreSQL on all nodes**
```bash
# On each node
systemctl stop patroni
```

2. **Clear existing data directory**
```bash
# On primary node
rm -rf /var/lib/postgresql/16/main/*
```

3. **Restore from backup to specific timestamp**
```bash
# Restore to 2025-12-14 14:30:00
pgbackrest --stanza=options-cluster \
  --type=time \
  --target="2025-12-14 14:30:00" \
  --target-action=promote \
  restore

# Alternative: Restore to specific transaction ID
# pgbackrest --stanza=options-cluster \
#   --type=xid \
#   --target=12345678 \
#   --target-action=promote \
#   restore
```

4. **Start PostgreSQL**
```bash
systemctl start patroni
```

5. **Verify recovery**
```bash
# Check database is accessible
psql -U postgres -d options_platform -c "SELECT NOW();"

# Verify data at recovery point
psql -U postgres -d options_platform -c "SELECT MAX(time) FROM options_prices;"
```

6. **Rebuild standbys from new primary**
```bash
# On each standby node
systemctl stop patroni
rm -rf /var/lib/postgresql/16/main/*
patronictl reinit options-cluster pg-standby-1
systemctl start patroni
```

**Expected RTO:** 10-15 minutes

### Scenario 2: Full Cluster Recovery (Complete Disaster)

**Use Case:** All database nodes lost, restore from cloud backup

**Procedure:**

1. **Provision new infrastructure**
```bash
# Via Terraform or manual provisioning
terraform apply -var="environment=disaster-recovery"
```

2. **Install PostgreSQL and pgBackRest**
```bash
# On each new node
apt-get update
apt-get install postgresql-16 postgresql-16-timescaledb pgbackrest
```

3. **Configure pgBackRest to access cloud repository**
```bash
# /etc/pgbackrest/pgbackrest.conf
# (Use repo2 or repo3 cloud repository configuration)
```

4. **Restore latest backup**
```bash
# On new primary node
pgbackrest --stanza=options-cluster --delta restore

# --delta: Reuse any existing valid files (faster)
# Omit --type and --target to restore latest backup
```

5. **Start Patroni cluster**
```bash
# On primary
systemctl start patroni

# Verify primary is running
patronictl list options-cluster

# On standbys
systemctl start patroni
```

6. **Verify cluster health**
```bash
patronictl list options-cluster

# Expected output:
# + Cluster: options-cluster --------+----+-----------+
# | Member       | Host      | Role   | State   | TL | Lag in MB |
# +--------------+-----------+--------+---------+----+-----------+
# | pg-primary   | 10.0.1.10 | Leader | running |  1 |           |
# | pg-standby-1 | 10.0.1.11 | Replica| running |  1 |         0 |
# | pg-standby-2 | 10.0.1.12 | Replica| running |  1 |         0 |
# +--------------+-----------+--------+---------+----+-----------+
```

**Expected RTO:** 30-60 minutes (depending on cloud download speed)

### Scenario 3: Single Table Recovery

**Use Case:** Recover specific table without full database restore

**Procedure:**

1. **Restore to temporary database**
```bash
# Create temporary restore directory
mkdir /var/tmp/pg_restore
chown postgres:postgres /var/tmp/pg_restore

# Restore to temporary location
pgbackrest --stanza=options-cluster \
  --pg1-path=/var/tmp/pg_restore \
  --type=time \
  --target="2025-12-14 12:00:00" \
  restore
```

2. **Start temporary PostgreSQL instance**
```bash
# Start on alternate port
su - postgres -c "/usr/lib/postgresql/16/bin/postgres -D /var/tmp/pg_restore -p 5433" &
```

3. **Export specific table**
```bash
# Export table data
pg_dump -h localhost -p 5433 -U postgres \
  -d options_platform \
  -t positions \
  --data-only \
  -f /tmp/positions_recovery.sql
```

4. **Import to production**
```bash
# Truncate existing table (if needed)
psql -U postgres -d options_platform -c "TRUNCATE positions CASCADE;"

# Import recovered data
psql -U postgres -d options_platform -f /tmp/positions_recovery.sql
```

5. **Cleanup**
```bash
# Stop temporary instance
pkill -f "postgres -D /var/tmp/pg_restore"

# Remove temporary data
rm -rf /var/tmp/pg_restore
```

**Expected RTO:** 15-30 minutes

## Backup Verification

### Automated Verification (Weekly)

**Verification Script (`/usr/local/bin/verify-backups.sh`):**

```bash
#!/bin/bash
set -e

STANZA="options-cluster"
VERIFY_LOG="/var/log/pgbackrest/verify.log"

echo "$(date): Starting backup verification" >> ${VERIFY_LOG}

# 1. Verify backup integrity
pgbackrest --stanza=${STANZA} verify >> ${VERIFY_LOG} 2>&1

# 2. Test restore to temporary location
TEMP_DIR="/var/tmp/backup_verify_$(date +%Y%m%d)"
mkdir -p ${TEMP_DIR}

pgbackrest --stanza=${STANZA} \
  --pg1-path=${TEMP_DIR} \
  --type=latest \
  restore >> ${VERIFY_LOG} 2>&1

# 3. Verify PostgreSQL can start
su - postgres -c "/usr/lib/postgresql/16/bin/postgres -D ${TEMP_DIR} -p 5434" &
PG_PID=$!

sleep 10

# 4. Test database connectivity
psql -h localhost -p 5434 -U postgres -d options_platform -c "SELECT COUNT(*) FROM users;" >> ${VERIFY_LOG} 2>&1

# 5. Cleanup
kill ${PG_PID}
rm -rf ${TEMP_DIR}

echo "$(date): Backup verification completed successfully" >> ${VERIFY_LOG}

# 6. Send success notification
curl -X POST https://monitoring.internal/api/alerts \
  -d '{"event": "backup_verification_success", "timestamp": "'$(date -Iseconds)'"}'
```

**Cron Schedule:**
```cron
# /etc/cron.d/verify-backups
0 4 * * 0 postgres /usr/local/bin/verify-backups.sh
```

### Manual Verification

```bash
# Check backup information
pgbackrest --stanza=options-cluster info --output=json | jq '.'

# Verify all repositories
pgbackrest --stanza=options-cluster verify --repo=1
pgbackrest --stanza=options-cluster verify --repo=2
pgbackrest --stanza=options-cluster verify --repo=3

# Check for WAL archive gaps
psql -U postgres -d postgres -c "SELECT * FROM pg_ls_archive_statusdir();"
```

## Monitoring and Alerting

### Key Metrics

1. **Backup Success/Failure**
   - Alert: Failed backup job
   - Severity: Critical
   - Response: Investigate within 30 minutes

2. **WAL Archive Lag**
   - Threshold: >100 pending WAL files
   - Alert: WAL archive falling behind
   - Severity: Warning
   - Response: Check pgBackRest logs

3. **Repository Disk Space**
   - Threshold: <20% free space
   - Alert: Low backup storage space
   - Severity: Warning
   - Response: Cleanup old backups or expand storage

4. **Last Successful Backup Age**
   - Threshold: >25 hours since last successful backup
   - Alert: Backup not running
   - Severity: Critical
   - Response: Immediate investigation

### Prometheus Metrics

**pgBackRest Exporter:**
```yaml
# /etc/prometheus/pgbackrest-exporter.yml
metrics:
  - name: pgbackrest_last_backup_timestamp
    query: "pgbackrest info --stanza=options-cluster --output=json"
    labels: ["backup_type"]

  - name: pgbackrest_backup_size_bytes
    query: "pgbackrest info --stanza=options-cluster --output=json"
    labels: ["backup_type", "repo"]

  - name: pgbackrest_wal_archive_count
    query: "SELECT COUNT(*) FROM pg_ls_archive_statusdir()"
```

### Alert Rules (`/etc/prometheus/rules/pgbackrest.yml`)

```yaml
groups:
  - name: pgbackrest
    interval: 60s
    rules:
      - alert: BackupFailed
        expr: pgbackrest_last_backup_status != 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "pgBackRest backup failed"
          description: "Backup job failed for stanza {{ $labels.stanza }}"

      - alert: BackupTooOld
        expr: (time() - pgbackrest_last_backup_timestamp) > 90000  # 25 hours
        for: 1h
        labels:
          severity: critical
        annotations:
          summary: "No recent backup available"
          description: "Last backup was {{ $value | humanizeDuration }} ago"

      - alert: WALArchiveLag
        expr: pgbackrest_wal_archive_count > 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "WAL archive falling behind"
          description: "{{ $value }} WAL files pending archive"

      - alert: BackupRepositoryLowSpace
        expr: (node_filesystem_avail_bytes{mountpoint="/var/lib/pgbackrest"} / node_filesystem_size_bytes) < 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Backup repository low on disk space"
          description: "Only {{ $value | humanizePercentage }} space remaining"
```

## Compliance and Auditing

### Audit Log Requirements

**pgBackRest Logging:**
- All backup/restore operations logged
- Logs retained for 1 year
- Logs forwarded to SIEM (Security Information and Event Management)

**PostgreSQL Audit Logging:**
```sql
-- Enable pgaudit extension
CREATE EXTENSION pgaudit;

-- Audit all DDL and DML on sensitive tables
ALTER DATABASE options_platform SET pgaudit.log = 'ddl, write';
ALTER TABLE users SET (pgaudit.log = 'read, write');
```

### Compliance Reports

**Monthly Backup Report:**
```bash
#!/bin/bash
# /usr/local/bin/backup-report.sh

MONTH=$(date -d "last month" +%Y-%m)
REPORT_FILE="/var/reports/backup-report-${MONTH}.txt"

cat > ${REPORT_FILE} <<EOF
Backup Report for ${MONTH}
================================

Backup Statistics:
$(pgbackrest --stanza=options-cluster info)

Successful Backups:
$(grep "backup complete" /var/log/pgbackrest/*.log | grep ${MONTH} | wc -l)

Failed Backups:
$(grep "ERROR" /var/log/pgbackrest/*.log | grep ${MONTH} | wc -l)

Average Backup Size:
$(pgbackrest --stanza=options-cluster info --output=json | jq '.[] | .backup[] | .info.size' | awk '{sum+=$1} END {print sum/NR}')

Storage Usage:
Local: $(du -sh /var/lib/pgbackrest/)
Cloud: $(aws s3 ls --summarize --recursive s3://backups-options-platform/pgbackrest/ | grep "Total Size")

Recovery Tests Performed:
$(grep "verification completed successfully" /var/log/pgbackrest/verify.log | grep ${MONTH} | wc -l)

Compliance Status: ✓ COMPLIANT
EOF

# Email report to compliance team
mail -s "Backup Report ${MONTH}" compliance@example.com < ${REPORT_FILE}
```

## Disaster Recovery Drills

### Quarterly DR Test Schedule

| Quarter | Test Scenario | Expected Duration | Success Criteria |
|---------|--------------|-------------------|------------------|
| Q1 | Point-in-time recovery | 2 hours | RTO <15 min, RPO <5 min |
| Q2 | Full cluster rebuild | 4 hours | RTO <60 min |
| Q3 | Cross-region failover | 4 hours | RTO <60 min, data integrity verified |
| Q4 | Single table recovery | 1 hour | RTO <30 min, zero data loss |

### DR Test Procedure

1. **Schedule test window** (outside business hours)
2. **Notify stakeholders** (operations, engineering, compliance)
3. **Execute test scenario** (document all steps)
4. **Validate results** (verify data integrity, measure RTO/RPO)
5. **Document findings** (update runbooks, fix issues)
6. **Post-mortem meeting** (review lessons learned)

## Appendices

### A. pgBackRest Commands Reference

```bash
# Initialize stanza
pgbackrest --stanza=options-cluster stanza-create

# Perform backup
pgbackrest --stanza=options-cluster --type=full backup

# List backups
pgbackrest --stanza=options-cluster info

# Restore latest backup
pgbackrest --stanza=options-cluster restore

# Restore to point in time
pgbackrest --stanza=options-cluster --type=time --target="2025-12-14 12:00:00" restore

# Delete old backups
pgbackrest --stanza=options-cluster --repo=1 expire

# Verify backup integrity
pgbackrest --stanza=options-cluster verify
```

### B. PostgreSQL Archive Command

```sql
-- View current archive settings
SHOW archive_mode;
SHOW archive_command;
SHOW archive_timeout;

-- Check archiver status
SELECT * FROM pg_stat_archiver;

-- Manually archive current WAL
SELECT pg_switch_wal();
```

### C. Emergency Contact Information

| Role | Name | Phone | Email |
|------|------|-------|-------|
| DBA Lead | [Name] | [Phone] | [Email] |
| Backup Administrator | [Name] | [Phone] | [Email] |
| On-Call Engineer | [Rotation] | [PagerDuty] | [Email] |
| Infrastructure Lead | [Name] | [Phone] | [Email] |

---

**Document Version:** 1.0
**Last Updated:** 2025-12-14
**Next Review:** 2026-03-14
**Approved By:** Database Resilience Architect
