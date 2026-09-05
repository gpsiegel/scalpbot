# =============================================================================
# versions.tf -- OpenTofu + provider version constraints and state backend
# =============================================================================
# NOTE: This project uses OpenTofu 1.10 (the `tofu` CLI), NOT Terraform.
#   Init/plan/apply with:  tofu init / tofu plan / tofu apply
# =============================================================================

terraform {
  required_version = ">= 1.10.0"

  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.43"
    }
  }

  # ---------------------------------------------------------------------------
  # Remote state backend (S3-compatible object storage)
  # ---------------------------------------------------------------------------
  # State lives in a DigitalOcean Spaces bucket (S3-compatible).
  #
  # The non-secret values below (bucket, key, endpoint, region) are supplied at
  # `tofu init` time via `-backend-config=backend-<env>.hcl`, keeping the state
  # location environment-specific. Credentials (TFSTATE_S3_ACCESS_KEY /
  # TFSTATE_S3_SECRET_KEY) are passed via env / CI secrets, never committed.
  backend "s3" {
    # Values provided at init time:
    #   -backend-config=backend-<env>.hcl
    #
    #   bucket                      = "scalpbot-tfstate"
    #   key                         = "<env>/terraform.tfstate"
    #   endpoints                   = { s3 = "https://<region>.<provider>.com" }
    #   region                      = "us-east-1"   # dummy for non-AWS S3
    #   skip_credentials_validation = true
    #   skip_metadata_api_check     = true
    #   skip_region_validation      = true
    #   skip_requesting_account_id  = true
    #   use_path_style              = true
  }
}
