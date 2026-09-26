# Tests

The tests cover VCF/BCF integration, DGS utilities, model probabilities,
likelihood fitting, CLI behavior, plotting, SLiM wrappers, and validation
workflows. Fixed-individual validation tests check sample identity, simulation
design metadata, pooled-VCF identities, and invalid configurations. SLiM 5.2
integration tests exercise outcrossing, partial selfing, and complete selfing
with the same individuals across multiple loci.

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
pytest -q -m slim
```
