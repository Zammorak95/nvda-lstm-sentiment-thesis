"""Smoke tests for the public thesis entry points.

These tests do not require the private/large thesis dataset and therefore can run
in GitHub Actions on every push.
"""


def test_import_public_entry_points() -> None:
    import thesis.data_acquisition.stockdata_api  # noqa: F401
    import thesis.preprocessing.data_pipeline  # noqa: F401
    import thesis.eval.make_scientific_outputs  # noqa: F401
    import thesis.eval.run_baseline_models_linear_svm  # noqa: F401
    import thesis.eval.make_model_comparison_table  # noqa: F401


def test_cli_builders_are_available() -> None:
    from thesis.data_acquisition.stockdata_api import build_parser as build_fetch_parser
    from thesis.preprocessing.data_pipeline import build_parser as build_preprocess_parser

    assert build_fetch_parser() is not None
    assert build_preprocess_parser() is not None
