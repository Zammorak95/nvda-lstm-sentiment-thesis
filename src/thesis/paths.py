from pathlib import Path
import os


PROJECT_ROOT = Path(
    os.getenv("THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[2])
)

DATA_DIR = Path(os.getenv("THESIS_DATA_DIR", PROJECT_ROOT / "data")).resolve()
ARTIFACTS_DIR = Path(os.getenv("THESIS_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")).resolve()
MLRUNS_DIR = Path(os.getenv("THESIS_MLRUNS_DIR", PROJECT_ROOT / "mlruns")).resolve()

RAW = DATA_DIR / "raw"
EXTERNAL = DATA_DIR / "external"
INTERIM = DATA_DIR / "interim"
PROCESSED = DATA_DIR / "processed"

MODELS = ARTIFACTS_DIR / "models"
FEATURES = ARTIFACTS_DIR / "features"
FIGURES = ARTIFACTS_DIR / "figures"
RUNS = ARTIFACTS_DIR / "runs"

def ensure_dirs():
    for d in [RAW, EXTERNAL, INTERIM, PROCESSED, MODELS, FEATURES, FIGURES, RUNS, MLRUNS_DIR]:
        d.mkdir(parents=True, exist_ok=True)