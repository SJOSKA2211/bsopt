variable "aws_region" {
  default = "us-east-1"
}

variable "db_password" {
  description = "Database password for TimescaleDB"
  sensitive   = true
}

variable "rmq_password" {
  description = "RabbitMQ password"
  sensitive   = true
}
