#!/usr/bin/env python3
"""Canonical entry point for LSTM hyperparameter tuning.

This wrapper keeps backwards compatibility with the original thesis script
`random_search_lstm_direction_v2.py`, but exposes a clearer name for
reproduction and documentation.
"""

from thesis.model_training.optimalisation.random_search_lstm_direction_v2 import main


if __name__ == "__main__":
    main()
