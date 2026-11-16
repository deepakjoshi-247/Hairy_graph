# Contributing to HairyGraph

Thank you for contributing to HairyGraph. This guide outlines the workflow for setting up your development environment, running tests, and following project conventions.

---

## Development Setup

1. **Fork and Clone**
   Fork the repository on GitHub, then clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/hairy-graph.git
   cd hairy-graph
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv .venv
   ```
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```cmd
     .venv\Scripts\activate
     ```

3. **Install Editable Package**
   ```bash
   pip install -e .
   ```

4. **Configure Environment Variables**
   Copy the example environment configuration:
   - On Linux/macOS:
     ```bash
     cp .env.example .env
     ```
   - On Windows:
     ```cmd
     copy .env.example .env
     ```

---

## Running Tests

Install test dependencies and run the pytest suite from the repository root:

```bash
pip install pytest
pytest -v
```

All tests execute against temporary SQLite instances and verify cold start behavior, graph transitions, semantic routing, cross-cluster bridging, strict token capping, supersede mechanics, auto-promotion, and API authentication.

---

## Contribution Conventions

- **Backward Compatibility**: Keep all public API endpoints (`/get_context`, `/save_memory`, `/archive_last`, `/graph_stats`, `/health`) strictly backward-compatible.
- **Security & Clean Git**: Never commit SQLite `.db`, `.db-wal`, or `.db-shm` files, and never commit secrets or live API keys.
- **Unit Testing**: Add or update unit tests in `tests/` for any graph rule, scoring logic, or threshold modification.
- **Documentation**: Update [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) whenever data schemas, traversal mechanics, or system rules change.
