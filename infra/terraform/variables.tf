variable "aws_region" {
  description = "The AWS region to deploy to"
  type        = "string"
  default     = "us-east-1"
}

variable "project_name" {
  description = "The name of the project"
  type        = "string"
  default     = "bsopt"
}

variable "environment" {
  description = "The deployment environment"
  type        = "string"
  default     = "dev"
}

variable "vpc_cidr" {
  description = "The CIDR block for the VPC"
  type        = "string"
  default     = "10.0.0.0/16"
}
