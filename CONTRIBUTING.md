# Contributing

Contributions should preserve deterministic behavior, reproducibility, and clear separation between measured data and inferred claims.

1. Create a focused branch from `main`.
2. Add tests for new metrics, validation rules, or report behavior.
3. Run `ruff check .`, the coverage-gated unit tests, and JSON/HTML smoke commands before opening a pull request.
4. Never commit credentials, private benchmark data, or generated reports containing sensitive information.
5. Document metric assumptions and limitations rather than presenting proxies as ground truth.
6. In the pull request, explain what changed, why it changed, and how it was validated.
