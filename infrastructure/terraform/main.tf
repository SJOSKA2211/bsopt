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

# --- Blue/Green Namespaces ---
resource "kubernetes_namespace" "blue" {
  metadata {
    name = "bsopt-blue"
    labels = {
      env = "prod"
      color = "blue"
    }
  }
}

resource "kubernetes_namespace" "green" {
  metadata {
    name = "bsopt-green"
    labels = {
      env = "prod"
      color = "green"
    }
  }
}

# --- Core Service Routing (Active Environment) ---
# This service points to the active color environment
resource "kubernetes_service" "bsopt_active" {
  metadata {
    name      = "bsopt-active-service"
    namespace = "default"
  }
  spec {
    selector = {
      # Change this to "green" to swap traffic
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
