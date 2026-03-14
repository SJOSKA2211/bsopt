variable "aws_region" {
  description = "AWS Region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS Cluster Name"
  type        = string
  default     = "bsopt-prod-cluster"
}

variable "active_environment" {
  description = "The active color for the Blue-Green deployment (blue or green)"
  type        = string
  default     = "blue"
}

variable "docker_image_tag" {
  description = "The docker image tag to deploy"
  type        = string
  default     = "latest"
}
