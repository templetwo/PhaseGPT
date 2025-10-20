"""
Data module adapter for Phase A training.

This module wraps the data preparation functions from data/prepare_data.py
to provide the interface expected by train.py.
"""

import importlib.util
from pathlib import Path

# Load data/prepare_data.py directly using importlib
_data_module_path = Path(__file__).parent.parent / "data" / "prepare_data.py"

spec = importlib.util.spec_from_file_location("prepare_data_module", _data_module_path)
_prepare_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_prepare_data)

# Get the functions from the loaded module
get_dataloaders = _prepare_data.get_dataloaders
CharacterDataset = _prepare_data.CharacterDataset

# Alias for train.py compatibility
CharDataset = CharacterDataset

__all__ = ['get_dataloaders', 'CharDataset', 'CharacterDataset']
