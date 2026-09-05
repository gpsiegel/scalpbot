# =============================================================================
# main.tf -- scalpbot core compute on DigitalOcean (migrated from Linode)
# =============================================================================
# Resource mapping (Linode -> DigitalOcean):
#   linode_instance          -> digitalocean_droplet
#   linode reserved/extra IP -> digitalocean_reserved_ip + _assignment
#   linode_firewall          -> digitalocean_firewall
#   linode SSH key           -> digitalocean_ssh_key
# Region: us-ord (Chicago) -> nyc3 (New York 3, low latency to Alpaca).
# =============================================================================

locals {
  name        = "scalpbot-${var.environment}"
  common_tags = distinct(concat(["scalpbot", var.environment], var.tags))
}

# -----------------------------------------------------------------------------
# SSH key (deploy access) -- was a Linode-managed SSH key
# -----------------------------------------------------------------------------
resource "digitalocean_ssh_key" "deploy" {
  name       = "${local.name}-deploy"
  public_key = var.ssh_public_key
}

# -----------------------------------------------------------------------------
# Droplet -- was linode_instance (Linode 2 GB, us-ord)
# Postgres is co-hosted on this box initially (same approach as on Linode).
# -----------------------------------------------------------------------------
resource "digitalocean_droplet" "app" {
  name     = local.name
  region   = var.region
  size     = var.droplet_size
  image    = var.droplet_image
  backups  = var.enable_backups
  ipv6     = true
  ssh_keys = [digitalocean_ssh_key.deploy.fingerprint]
  tags     = local.common_tags

  # Root password is NOT set here: SSH key auth only (LINODE_ROOT_PASS removed).
  lifecycle {
    ignore_changes = [image] # avoid rebuilds if the base image slug moves
  }
}

# -----------------------------------------------------------------------------
# Reserved IP -- was the Linode reserved/extra IPv4 (RESERVED_IPV4)
# Split into the address + an assignment so the IP survives droplet rebuilds.
# -----------------------------------------------------------------------------
resource "digitalocean_reserved_ip" "app" {
  region = var.region
}

resource "digitalocean_reserved_ip_assignment" "app" {
  ip_address = digitalocean_reserved_ip.app.ip_address
  droplet_id = digitalocean_droplet.app.id
}

# -----------------------------------------------------------------------------
# Firewall -- was linode_firewall
# Inbound: SSH (restrictable), HTTP/HTTPS. The API server binds 127.0.0.1 only,
# so its port is intentionally NOT exposed. Outbound: allow all.
# -----------------------------------------------------------------------------
resource "digitalocean_firewall" "app" {
  name        = "${local.name}-fw"
  droplet_ids = [digitalocean_droplet.app.id]
  tags        = local.common_tags

  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = var.allowed_ssh_cidrs
  }

  inbound_rule {
    protocol         = "tcp"
    port_range       = "80"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  inbound_rule {
    protocol         = "tcp"
    port_range       = "443"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  # ICMP (ping) for basic reachability checks.
  inbound_rule {
    protocol         = "icmp"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  # Allow all outbound traffic (API calls to Alpaca, Doppler, sentiment sources).
  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "icmp"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}
