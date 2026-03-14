# BS-Opt (EquaFlow) Infrastructure as Code
# Terraform Scaffolding for AWS Deployment

provider "aws" {
  region = var.aws_region
}

resource "aws_vpc" "bsopt_vpc" {
  cidr_block = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = { Name = "bsopt-vpc" }
}

# ECS Cluster for Services
resource "aws_ecs_cluster" "bsopt_cluster" {
  name = "bsopt-cluster"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# TimescaleDB on RDS (PostgreSQL compatible)
resource "aws_db_instance" "timescaledb" {
  allocated_storage    = 100
  engine               = "postgres"
  engine_version       = "16"
  instance_class       = "db.t3.large"
  name                 = "bsopt"
  username             = "admin"
  password             = var.db_password
  parameter_group_name = "default.postgres16"
  skip_final_snapshot  = true
}

# RabbitMQ on MQ
resource "aws_mq_broker" "rabbitmq" {
  broker_name = "bsopt-rabbitmq"
  engine_type = "RabbitMQ"
  engine_version = "3.11.20"
  host_instance_type = "mq.t3.micro"
  user {
    username = "admin"
    password = var.rmq_password
  }
}
