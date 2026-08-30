# Changelog

All notable changes to this project are documented here. Tagged releases follow semantic versioning.

## 1.0.0 - 2026-08-30

### Added
- Installable `ai-evaluation-harness` package and `ai-eval` CLI.
- Deterministic RAG and agent metrics: token F1, recall@k, reciprocal rank, nDCG, citation precision/recall and tool-call accuracy.
- Optional embedding-provider semantic similarity interface.
- Transparent sentence-level factual-support proxy.
- Opt-in structured LLM-judge adapter with versioned rubric metadata and injectable transport for credential-free CI tests.
- Heterogeneous benchmark suite runner covering retrieval, citations, routing, prompt injection, tool calls and failure boundaries.
- SQLite evaluation history plus HTML trend reporting.
- Cross-project `portfolio-evidence/v1` bridge.
- Package build verification, CodeQL, dependency auditing and CycloneDX SBOM generation.
- Tagged release workflow with build provenance attestation.

### Scope
The deterministic metrics remain the default evidence layer. External embedding and LLM-judge providers are opt-in and are not presented as credential-free deterministic measurements.
