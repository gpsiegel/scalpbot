# =============================================================================
# variables.tf -- input variables for the scalpbot DigitalOcean infrastructure
# =============================================================================

variable "do_token" {
  description = "DigitalOcean API token (from DIGITALOCEAN_TOKEN secret). Needs read/write for droplets, reserved IPs, firewalls, and SSH keys."
  type        = string
  sensitive   = true
}

variable "environment" {
  description = "Deployment environment: nonprod | staging | prod."
  type        = string
  validation {
    condition     = contains(["nonprod", "staging", "prod"], var.environment)
    error_message = "environment must be one of: nonprod, staging, prod."
  }
}

variable "region" {
  description = "DigitalOcean region slug. nyc3 chosen for low latency to Alpaca (NY-based)."
  type        = string
  default     = "nyc3"
}

variable "droplet_size" {
  description = "DigitalOcean droplet size slug. s-1vcpu-2gb == 2 GB RAM (equivalent to the old Linode 2 GB instance)."
  type        = string
  default     = "s-1vcpu-2gb"
}

variable "droplet_image" {
  description = "Base image slug for the droplet."
  type        = string
  default     = "ubuntu-24-04-x64"
}

variable "enable_backups" {
  description = "Enable DigitalOcean automated droplet backups (matches the prior Linode backups setting)."
  type        = bool
  default     = true
}

variable "ssh_public_key" {
  description = "SSH public key material used for deploy access (from DEPLOY_SSH_PUBKEY secret). Registered as a digitalocean_ssh_key and attached to the droplet."
  type        = string
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks permitted to reach the droplet on TCP 22. Lock this down to your admin/CI egress ranges in prod."
  type        = list(string)
  default     = ["0.0.0.0/0", "::/0"]
}

variable "tags" {
  description = "Extra tags applied to created resources."
  type        = list(string)
  default     = []
}

# --- Spaces (optional; only if using DigitalOcean Spaces via the provider) ---
variable "spaces_access_id" {
  description = "DigitalOcean Spaces access key ID (optional; leave empty if unused)."
  type        = string
  default     = ""
  sensitive   = true
}

variable "spaces_secret_key" {
  description = "DigitalOcean Spaces secret access key (optional; leave empty if unused)."
  type        = string
  default     = ""
  sensitive   = true
}
