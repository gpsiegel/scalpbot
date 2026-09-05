# =============================================================================
# providers.tf -- DigitalOcean provider configuration
# =============================================================================
# The API token is read from the `do_token` variable, which is populated from
# the DIGITALOCEAN_TOKEN environment/CI secret (see variables.tf). The
# DigitalOcean provider also natively honors the DIGITALOCEAN_TOKEN env var, so
# passing -var is optional when the env var is exported.

provider "digitalocean" {
  token = var.do_token

  # Spaces (S3-compatible object storage) credentials -- only needed if you use
  # DigitalOcean Spaces for backups/artifacts via this provider. Reserved-IP,
  # droplet, firewall, and SSH-key resources do NOT require these.
  spaces_access_id  = var.spaces_access_id
  spaces_secret_key = var.spaces_secret_key
}
