# Terraform Configuration for BS-Opt (EquaFlow)
# Blue-Green Deployment Strategy on Kubernetes (EKS/GKE)

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket = "bsopt-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# Assume an existing EKS cluster
data "aws_eks_cluster" "cluster" {
  name = var.cluster_name
}

data "aws_eks_cluster_auth" "cluster" {
  name = var.cluster_name
}

provider "kubernetes" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

# --- Security Groups ---
resource "aws_security_group" "bsopt_api_sg" {
  name        = "bsopt-api-sg"
  description = "Security group for BS-Opt API"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # Restricted in prod
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- IAM Roles for Service Accounts (IRSA) ---
resource "aws_iam_role" "bsopt_scraper_role" {
  name = "bsopt-scraper-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = "arn:aws:iam::${var.account_id}:oidc-provider/${var.oidc_provider}"
        }
        Condition = {
          StringEquals = {
            "${var.oidc_provider}:sub" : "system:serviceaccount:default:bsopt-scraper"
          }
        }
      }
    ]
  })
}

# --- ElastiCache (Redis) for Rate Limiting ---
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "bsopt-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
}

# --- Blue/Green Namespaces ---
resource "kubernetes_namespace" "blue" {
  metadata {
    name = "bsopt-blue"
    labels = {
      env   = "prod"
      color = "blue"
    }
  }
}

resource "kubernetes_namespace" "green" {
  metadata {
    name = "bsopt-green"
    labels = {
      env   = "prod"
      color = "green"
    }
  }
}

# --- Core Service Routing (Active Environment) ---
resource "kubernetes_service" "bsopt_active" {
  metadata {
    name      = "bsopt-active-service"
    namespace = "default"
  }
  spec {
    selector = {
      color = var.active_environment
      app   = "bsopt-api"
    }
    port {
      port        = 80
      target_port = 8000
    }
    type = "LoadBalancer"
  }
}
