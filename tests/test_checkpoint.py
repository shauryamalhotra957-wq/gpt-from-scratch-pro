from __future__ import annotations

import inspect
import pickle
import tempfile
from pathlib import Path
import unittest

import torch

from gpt_from_scratch.checkpoint import load_checkpoint


class _UnsafePayload:
    def __reduce__(self):
        return eval, ("40 + 2",)


class CheckpointTests(unittest.TestCase):
    def test_loader_rejects_untrusted_pickle_globals(self) -> None:
        if "weights_only" not in inspect.signature(torch.load).parameters:
            self.skipTest("This torch version does not provide safe weights-only loading")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unsafe.pt"
            torch.save({"payload": _UnsafePayload()}, path)

            with self.assertRaises(pickle.UnpicklingError):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
