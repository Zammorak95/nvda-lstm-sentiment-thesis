# Legacy helpers

This folder contains scripts that were replaced by the generic end-to-end stock pipeline.

Active entrypoint:

```bash
scripts/run_stock_full_pipeline.sh
```

Archived files are kept only for traceability. They should not be used for the main thesis reproduction workflow.

The active pipeline is configured through environment variables such as `SYMBOL`, `KEYWORD`, `SCAN_START`, `END`, `TRIALS`, and `GPU`.
