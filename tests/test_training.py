from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from gpt_from_scratch.checkpoint import load_model_from_checkpoint
from gpt_from_scratch.config import GPTConfig, TrainConfig
from gpt_from_scratch.data import prepare_text_corpus
from gpt_from_scratch.train import evaluate_checkpoint, train_model


class TrainingTests(unittest.TestCase):
    def test_tiny_training_run_saves_reloadable_checkpoint(self) -> None:
        text = (
            "first engineer builds a model.\n"
            "second engineer tests the model.\n"
            "the model predicts the next token.\n"
        ) * 40

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.txt"
            corpus.write_text(text, encoding="utf-8")
            prepared = root / "prepared"
            info = prepare_text_corpus(
                input_path=corpus,
                out_dir=prepared,
                tokenizer_type="char",
                train_fraction=0.8,
            )
            model_config = GPTConfig(
                vocab_size=info.vocab_size,
                block_size=8,
                n_layer=1,
                n_head=2,
                n_embd=16,
                dropout=0.0,
            )
            train_config = TrainConfig(
                batch_size=4,
                max_iters=1,
                eval_interval=1,
                eval_iters=1,
                learning_rate=0.001,
                min_lr=0.0001,
                warmup_iters=1,
                lr_decay_iters=3,
                weight_decay=0.01,
                device="cpu",
            )
            summary = train_model(
                data_dir=prepared,
                out_dir=root / "run",
                model_config=model_config,
                train_config=train_config,
            )
            checkpoint_path = Path(summary["best_checkpoint"])
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(Path(summary["last_checkpoint"]).exists())
            self.assertTrue(torch.isfinite(torch.tensor(summary["best_val_loss"])))

            metrics = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                data_dir=prepared,
                batch_size=4,
                eval_iters=1,
                device="cpu",
            )
            self.assertIn("val_perplexity", metrics)

            resumed_config = TrainConfig(
                batch_size=4,
                max_iters=3,
                eval_interval=1,
                eval_iters=1,
                learning_rate=0.001,
                min_lr=0.0001,
                warmup_iters=1,
                lr_decay_iters=3,
                weight_decay=0.01,
                device="cpu",
            )
            resumed_summary = train_model(
                data_dir=prepared,
                out_dir=root / "run",
                model_config=model_config,
                train_config=resumed_config,
                resume_from=root / "run" / "last.pt",
            )
            self.assertEqual(len(resumed_summary["history"]), 4)

            model, tokenizer, checkpoint = load_model_from_checkpoint(checkpoint_path)
            prompt = "first"
            idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
            out = model.generate(idx, max_new_tokens=2, temperature=0)
            self.assertEqual(out.shape[1], len(prompt) + 2)
            self.assertIn("best_val_loss", checkpoint)
            _, _, last_checkpoint = load_model_from_checkpoint(root / "run" / "last.pt")
            self.assertEqual(last_checkpoint["iter_num"], 3)


if __name__ == "__main__":
    unittest.main()
