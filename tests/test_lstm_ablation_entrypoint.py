from pathlib import Path


def test_lstm_ablation_entrypoint_exists():
    root = Path(__file__).resolve().parents[1]
    script = root / "src" / "thesis" / "eval" / "run_lstm_feature_ablation.py"
    wrapper = root / "scripts" / "run_lstm_feature_ablation.sh"
    assert script.exists()
    assert wrapper.exists()
