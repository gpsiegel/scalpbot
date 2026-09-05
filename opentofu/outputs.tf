# =============================================================================
# outputs.tf
# =============================================================================

output "droplet_id" {
  description = "DigitalOcean droplet ID."
  value       = digitalocean_droplet.app.id
}

output "droplet_public_ipv4" {
  description = "Ephemeral public IPv4 assigned to the droplet."
  value       = digitalocean_droplet.app.ipv4_address
}

output "reserved_ipv4" {
  description = "Stable reserved IPv4 (maps to the RESERVED_IPV4 / DEPLOY_HOST secret)."
  value       = digitalocean_reserved_ip.app.ip_address
}

output "region" {
  description = "Region the infrastructure is deployed to."
  value       = var.region
}

output "ssh_key_fingerprint" {
  description = "Fingerprint of the registered deploy SSH key (DO_SSH_KEY_FINGERPRINT)."
  value       = digitalocean_ssh_key.deploy.fingerprint
}
