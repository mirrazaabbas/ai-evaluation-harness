# Security Policy

## Supported branch

The `main` branch is the actively maintained version of this portfolio project.

## Reporting a vulnerability

Do not publish credentials, private evaluation data, or actionable exploit details in a public issue. Use GitHub private vulnerability reporting when available. Otherwise, open a minimal issue that identifies a security concern without sensitive reproduction details until a private channel is established.

## Security principles

- Never commit API keys, tokens, passwords, or private benchmark data.
- Treat benchmark inputs and generated model outputs as untrusted data.
- Validate dataset structure and numeric operational metrics before reporting.
- Escape untrusted text before rendering HTML reports.
- Do not infer or invent latency, token, cost, citation, or factuality measurements that were not actually collected.
