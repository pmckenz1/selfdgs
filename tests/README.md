# Tests

The tests cover VCF/BCF integration, DGS utilities, model probabilities,
likelihood fitting, CLI behavior, plotting, SLiM wrappers, and validation
workflows.

Run the default suite from the package root:

```bash
pytest -q
```

Registered markers:

- `plot`: plotting tests
- `slim`: integration tests that require an external SLiM executable

Useful targeted runs:

```bash
pytest -q -m "not plot and not slim"
pytest -q -m plot
```
