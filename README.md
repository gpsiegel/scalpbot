# scalpbot

Multi-avenue trading bot (Crypto + Stocks + Options), Python, deployed on
**DigitalOcean** (`nyc3`, migrated from Linode). Secrets via Doppler;
non-secret operational config via plain `.env`.

## Infrastructure

- **IaC:** OpenTofu 1.10 (the `tofu` CLI, not Terraform) in
  [`opentofu/`](opentofu/README.md).
- **Compute:** one DigitalOcean droplet `s-1vcpu-2gb` (2 GB) in `nyc3`, backups
  enabled, Postgres co-hosted initially. The API server binds `127.0.0.1` only.
- **Networking:** a DigitalOcean reserved IP + a firewall (SSH/HTTP/HTTPS in).
- **CI/CD:** GitHub Actions — `infra.yml` (OpenTofu) and `deploy.yml` (app
  deploy over SSH), both authenticating to DigitalOcean via `DIGITALOCEAN_TOKEN`
  and `doctl`. These currently live in [`ci-workflows/`](ci-workflows/README.md)
  and must be moved into `.github/workflows/` (see that README — the PR
  automation lacked the GitHub `workflows` permission).
- **State:** S3-compatible backend (existing Linode Object Storage bucket still
  works; optional migration to DigitalOcean Spaces documented in the infra
  README).

Reserve a floating/reserved IP manually with:

```bash
# Old (Linode):  linode-cli networking ip-reserve --region us-ord
doctl compute reserved-ip create --region nyc3
```

See [`python_space/sentiment/README.md`](python_space/sentiment/README.md) for
the crypto sentiment stack.
