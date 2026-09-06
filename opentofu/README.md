# scalpbot infrastructure (OpenTofu -> DigitalOcean)

Infrastructure-as-code for the scalpbot host, managed with **OpenTofu 1.10**
(the `tofu` CLI — **not** Terraform).

> Runs on **DigitalOcean** (`s-1vcpu-2gb`, `nyc3`). `nyc3` was chosen for low
> latency to Alpaca (NY-based).

## What this provisions

| Resource | Purpose |
| --- | --- |
| `digitalocean_droplet.app` | 2 GB app + co-hosted Postgres box, backups on |
| `digitalocean_reserved_ip.app` + `_assignment` | Stable public IP (`RESERVED_IPV4`) |
| `digitalocean_firewall.app` | SSH/HTTP/HTTPS in, all out |
| `digitalocean_ssh_key.deploy` | Deploy SSH key |

The API server binds to `127.0.0.1` only, so its port is intentionally **not**
opened in the firewall.

## Prerequisites

- OpenTofu >= 1.10 (`tofu version`)
- `doctl` (DigitalOcean CLI) for manual ops
- A DigitalOcean API token exported as `DIGITALOCEAN_TOKEN`

## Environments

Two environments — `nonprod`, `prod` — are separated by OpenTofu
workspaces and per-env backend keys / tfvars.

```bash
export TF_VAR_do_token="$DIGITALOCEAN_TOKEN"
export TF_VAR_ssh_public_key="$DEPLOY_SSH_PUBKEY"
export AWS_ACCESS_KEY_ID="$TFSTATE_S3_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$TFSTATE_S3_SECRET_KEY"

cp backend-nonprod.hcl.example backend-nonprod.hcl   # edit if needed
cp terraform.tfvars.example nonprod.tfvars           # edit if needed

tofu init -backend-config=backend-nonprod.hcl
tofu workspace select nonprod || tofu workspace new nonprod
tofu plan  -var-file=nonprod.tfvars
tofu apply -var-file=nonprod.tfvars
```

## Reserved IP (manual, if ever needed)

```bash
doctl compute reserved-ip create --region nyc3
```

## State backend

State lives in a **DigitalOcean Spaces** bucket (S3-compatible), region `nyc3`.

1. Create a Spaces bucket in `nyc3` and generate Spaces access keys.
2. Set `endpoints` in `backend-<env>.hcl` to
   `https://nyc3.digitaloceanspaces.com` and set the `TFSTATE_S3_*` secrets to
   the Spaces keys.
3. Init: `tofu init -backend-config=backend-<env>.hcl`

## Secrets used

| Secret | Use |
| --- | --- |
| `DIGITALOCEAN_TOKEN` | DO API token (`TF_VAR_do_token`) |
| `DEPLOY_SSH_PUBKEY` | Registered as the deploy SSH key |
| `TFSTATE_S3_ACCESS_KEY` / `TFSTATE_S3_SECRET_KEY` | State bucket creds |
| `TF_ENCRYPTION` | OpenTofu state encryption passphrase (see CI) |
