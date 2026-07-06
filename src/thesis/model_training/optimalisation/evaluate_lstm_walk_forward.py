#!/usr/bin/env python3
"""Canonical entry point for final LSTM walk-forward evaluation.

This wrapper keeps backwards compatibility with the original thesis script
`walk_forward_lstm_direction_rocm.py`, but exposes a clearer name for
reproduction and documentation.
"""

from thesis.model_training.optimalisation.walk_forward_lstm_direction_rocm import main


if __name__ == "__main__":
    main()
