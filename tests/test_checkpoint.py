from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from gpt_from_scratch.checkpoint import load_checkpoint


def test_load_checkpoint_rejects_non_mapping_payload():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.pt"
        torch.save(["not", "a", "mapping"], path)

        with pytest.raises(ValueError, match="mapping payload"):
            load_checkpoint(path)


def test_load_checkpoint_reports_missing_required_keys():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.pt"
        torch.save({"model_state": {}}, path)

        with pytest.raises(ValueError, match="missing required keys"):
            load_checkpoint(path)
