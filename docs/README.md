# docs/

Reference documents for the PRIMED-AI pipeline.

| Document | Contents |
|----------|----------|
| [embeddings.md](embeddings.md) | Where the pre-extracted echo/ECG embeddings live (HuggingFace + ORCD), loading code, model weights, and how to re-run extraction |
| [echo_hubert_loader.md](echo_hubert_loader.md) | Synchronizing EchoJEPA + HuBERT embeddings against the paired cohort |
| [echo_hubert_results.md](echo_hubert_results.md) | Manifest join counts and local reproduction steps |

For higher-level context see the root-level docs:

| Document | Contents |
|----------|----------|
| [README.md](../README.md) | Project scope, motivation, and status |
| [TECHNICAL.md](../TECHNICAL.md) | Full pipeline architecture — cohort, encoders, probes, evaluation |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Local setup, testing, linting, data access, reproducibility |
