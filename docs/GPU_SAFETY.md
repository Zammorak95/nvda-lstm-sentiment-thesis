# GPU safety notes

If the screen goes black, the desktop freezes, or the AMD/ROCm driver resets during LSTM training, treat it as a GPU stability/load issue first. The pipeline is checkpointed, so it is better to rerun in a safer mode than to push the GPU at full load.

## Recommended safer command

Use the safe wrapper for the first full run:

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 GPU=0 \
PYTHON="$PWD/.venv/bin/python" \
bash scripts/run_stock_full_pipeline_safe.sh main
```

This wrapper sets conservative defaults:

```text
TRIALS=10
RANDOM_EPOCHS=20
WALK_EPOCHS=15
TF_FORCE_GPU_ALLOW_GROWTH=true
OMP_NUM_THREADS=4
TF_NUM_INTRAOP_THREADS=4
TF_NUM_INTEROP_THREADS=2
```

For a longer but still safer run:

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 GPU=0 \
TRIALS=25 RANDOM_EPOCHS=30 WALK_EPOCHS=20 \
PYTHON="$PWD/.venv/bin/python" \
bash scripts/run_stock_full_pipeline_safe.sh main
```

## About an 80% GPU cap

TensorFlow/ROCm does not provide a simple portable `use at most 80% of GPU compute` switch. In practice, the safer options are:

1. reduce trials and epochs;
2. reduce batch size or avoid the largest random-search settings;
3. use `TF_FORCE_GPU_ALLOW_GROWTH=true` to avoid eager full-memory allocation where supported;
4. use a hardware/driver-level power cap with ROCm tools if your system supports it;
5. run the LSTM steps on CPU for maximum stability, at the cost of speed.

## Optional hardware-level AMD cap

Some AMD setups support a power cap through `rocm-smi`. This is outside the Python pipeline and may require sudo privileges. Example pattern:

```bash
rocm-smi
sudo rocm-smi --setpoweroverdrive <watts>
```

Use a conservative watt value appropriate for your GPU. Do not guess a high value.

## Resume behaviour

The pipeline reuses existing raw, processed and model outputs where possible. If a run stops during model training, rerun the same command. Existing data checkpoints should be reused; missing model/report stages will continue from the relevant phase.

## Practical advice

Start with `run_stock_full_pipeline_safe.sh main`. Once that completes, run the full version only if needed:

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 TRIALS=50 GPU=0 \
PYTHON="$PWD/.venv/bin/python" \
bash scripts/run_stock_full_pipeline.sh main
```

For the historical 0.5506-style check, safer run:

```bash
SYMBOL=NVDA KEYWORD="NVIDIA stock" END=2026-03-01 GPU=0 \
PYTHON="$PWD/.venv/bin/python" \
bash scripts/run_stock_full_pipeline_safe.sh legacy_05506
```
