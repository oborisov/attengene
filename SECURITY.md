# Security

## Reporting a vulnerability

If you find a security issue in AttenGene - anything that lets an attacker
read or modify data they shouldn't, escalate privileges, or compromise the
host - please email **oleg@attengene.de** rather than opening a public
issue. I will acknowledge within a few working days and coordinate a fix
before any public disclosure.

For non-security bugs and feature requests, open an issue on the
[Codeberg tracker](https://codeberg.org/oborisov/attengene/issues).

## Scope

In scope:

- The FastAPI application (`app/`), including authentication, query
  routing, retrieval, and audit logging
- The indexing scripts (`scripts/`)
- The bundled Docker Compose stack and Dockerfiles
- The bundled embeddings server

Out of scope:

- Vulnerabilities in third-party dependencies (please report upstream;
  AttenGene will pick up fixed versions in its own release cycle)
- Misconfigurations of self-hosted deployments (e.g. exposing the API
  without `ATTENGENE_API_KEY` set, weak `DB_PASSWORD`, leaving
  PostgreSQL on a public interface). The defaults in
  [`.env.example`](.env.example) are documented; operators are
  responsible for hardening them before exposing the service.

## Clinical safety reminder

AttenGene is a research prototype, **not a medical device**, and **not
intended for any clinical or diagnostic use**. Any report involving
"the model gave wrong clinical advice" is a clinical-safety report,
not a security report, and should also be raised with the operator of
the deployment - not just the upstream project.
