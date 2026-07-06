# Model-feed data

Place the final cleaned modelling dataset here:

```text
data/model_feed/model_dataset_clean.csv
```

This CSV is ignored by Git by default to avoid accidentally publishing data. See `docs/DATA.md` for guidance on whether to add the clean dataset to the repository, GitHub Releases, or Git LFS.

The evaluation scripts expect this file unless another path is passed with `--dataset` or `--data`.
