# CI workflows (staged) — MOVE to `.github/workflows/`

These two GitHub Actions workflows are the DigitalOcean-migrated CI/CD pipeline:

- `infra.yml` — provisions/updates infra with OpenTofu (`tofu` 1.10) on
  DigitalOcean (`nyc3`).
- `deploy.yml` — deploys the app to the droplet over SSH.

> **Why they live here instead of `.github/workflows/`:** the automation that
> opened this PR authenticates as a GitHub App that was **not** granted the
> `workflows` permission, so it is not allowed to create files under
> `.github/workflows/`. The workflow content is complete and validated — it just
> needs to be relocated by someone with that permission.

## To activate

```bash
mkdir -p .github/workflows
git mv ci-workflows/infra.yml  .github/workflows/infra.yml
git mv ci-workflows/deploy.yml .github/workflows/deploy.yml
git rm ci-workflows/README.md
git commit -m "ci: move DigitalOcean workflows into .github/workflows"
```

(Or grant the Abacus AI GitHub App the **workflows** permission and re-run the
migration so it can commit them directly.)

## Required GitHub Environment secrets

Add for each environment (`nonprod` / `staging` / `prod`):

- `DIGITALOCEAN_TOKEN` (replaces `LINODE_TOKEN`)
- `DEPLOY_SSH_PUBKEY`, `DEPLOY_SSH_KEY`, `DEPLOY_HOST`, `RESERVED_IPV4`
- `DOPPLER_TOKEN`
- `TFSTATE_S3_ACCESS_KEY`, `TFSTATE_S3_SECRET_KEY`, `TF_ENCRYPTION`

Remove: `LINODE_TOKEN`, `LINODE_ROOT_PASS`.
