# EquaFlow Multi-Cloud Infrastructure Blueprint (AWS/GCP)
# This Terraform manifest provides the core resource definitions for an institutional deployment.

provider "aws" {
  region = var.aws_region
}

resource "aws_vpc" "equaflow_vpc" {
  cidr_block = "10.0.0.0/16"
  enable_dns_support = true
  enable_dns_hostnames = true
  tags = { Name = "EquaFlow-VPC" }
}

resource "aws_db_instance" "timescaledb" {
  allocated_storage = 500
  engine = "postgres"
  engine_version = "15.3"
  instance_class = "db.m6g.xlarge"
  db_name = "equaflow_prod"
  username = "admin"
  password = var.db_password
  storage_type = "gp3"
  multi_az = true
  skip_final_snapshot = false
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id = "equaflow-redis"
  engine = "redis"
  node_type = "cache.m6g.large"
  num_cache_nodes = 3
  parameter_group_name = "default.redis7"
  port = 6379
}

resource "aws_ecs_cluster" "equaflow_cluster" {
  name = "EquaFlow-Cluster"
  setting {
    name = "containerInsights"
    value = "enabled"
  }
}

# ECS Service Definitions would follow for each microservice (API, Scraper, Backtester, etc.)
