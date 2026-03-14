output "active_load_balancer_hostname" {
  description = "The hostname of the active environment load balancer"
  value       = kubernetes_service.bsopt_active.status.0.load_balancer.0.ingress.0.hostname
}

output "active_environment" {
  description = "The currently active color"
  value       = var.active_environment
}
