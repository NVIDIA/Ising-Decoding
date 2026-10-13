# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint files are loaded with ``torch.load(..., weights_only=True)``.

Every ``.pt`` loader in the repo must accept the checkpoints we write (tensors,
builtin scalars/containers) and must refuse a checkpoint that carries arbitrary
pickled Python objects instead of executing it.
"""

import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from export.checkpoint_to_safetensors import _load_checkpoint_state_dict
from training.utils import (
    load_checkpoint,
    load_latest_best_checkpoint_metadata,
    save_checkpoint,
)
from workflows.run import _load_state_dict_from_pt


class _ArbitraryObject:
    """Stand-in for a non-tensor object that only full pickle can rebuild."""

    def __init__(self):
        self.value = 1


UNSAFE_LOAD_ERRORS = (pickle.UnpicklingError, RuntimeError)


def _tiny_model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


def _write_unsafe_training_checkpoint(dirpath, name="checkpoint.0.1.pt"):
    path = os.path.join(dirpath, name)
    torch.save(
        {
            "epoch": 1,
            "global_step": 10,
            "metadata": {
                "best_vloss": 0.5,
                "payload": _ArbitraryObject()
            },
        },
        path,
    )
    return path


class TestTrainingCheckpointLoading(unittest.TestCase):

    def test_round_trip_with_numpy_metadata(self):
        """save_checkpoint stores builtin types so load_checkpoint succeeds under weights_only."""
        model = _tiny_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as d:
            save_checkpoint(
                path=d,
                models=model,
                optimizer=opt,
                epoch=2,
                global_step=20,
                metadata={
                    "best_vloss": np.float64(0.25),
                    "epochs_since_best": np.int64(1),
                    "using_ler": np.bool_(True),
                },
            )
            meta = {}
            epoch, global_step = load_checkpoint(
                d, models=_tiny_model(), optimizer=opt, metadata_dict=meta, device="cpu"
            )
        self.assertEqual((epoch, global_step), (2, 20))
        self.assertEqual(meta["best_vloss"], 0.25)
        self.assertIs(type(meta["best_vloss"]), float)
        self.assertIs(type(meta["epochs_since_best"]), int)
        self.assertIs(type(meta["using_ler"]), bool)

    def test_load_checkpoint_rejects_arbitrary_objects(self):
        with tempfile.TemporaryDirectory() as d:
            _write_unsafe_training_checkpoint(d)
            with self.assertRaises(UNSAFE_LOAD_ERRORS):
                load_checkpoint(d, device="cpu")

    def test_best_metadata_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            save_checkpoint(path=d, epoch=3, metadata={"best_vloss": 0.1, "using_ler": False})
            meta = load_latest_best_checkpoint_metadata(d)
        self.assertEqual(meta, {"best_vloss": 0.1, "using_ler": False})

    def test_best_metadata_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(load_latest_best_checkpoint_metadata(d))
            torch.save({"epoch": 1}, os.path.join(d, "checkpoint.0.1.pt"))
            self.assertIsNone(load_latest_best_checkpoint_metadata(d))

    def test_best_metadata_rejects_arbitrary_objects(self):
        with tempfile.TemporaryDirectory() as d:
            _write_unsafe_training_checkpoint(d)
            with self.assertRaises(UNSAFE_LOAD_ERRORS):
                load_latest_best_checkpoint_metadata(d)


class TestStateDictLoaders(unittest.TestCase):
    """Inference (_load_state_dict_from_pt) and the SafeTensors converter."""

    LOADERS = {
        "inference": _load_state_dict_from_pt,
        "converter": _load_checkpoint_state_dict,
    }

    def _save(self, obj):
        f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        f.close()
        self.addCleanup(os.unlink, f.name)
        torch.save(obj, f.name)
        return f.name

    def test_accepts_bare_wrapped_and_ddp_state_dicts(self):
        model = _tiny_model()
        sd = model.state_dict()
        variants = {
            "bare": sd,
            "model_state_dict": {
                "model_state_dict": sd
            },
            "state_dict": {
                "state_dict": sd
            },
            "ddp": {
                "module." + k: v for k, v in sd.items()
            },
        }
        for name, loader in self.LOADERS.items():
            for variant, obj in variants.items():
                with self.subTest(loader=name, variant=variant):
                    loaded = loader(self._save(obj), "cpu")
                    self.assertEqual(set(loaded), set(sd))
                    for k in sd:
                        torch.testing.assert_close(loaded[k], sd[k])

    def test_rejects_arbitrary_objects(self):
        path = self._save(
            {
                "model_state_dict": _tiny_model().state_dict(),
                "cfg": _ArbitraryObject()
            }
        )
        for name, loader in self.LOADERS.items():
            with self.subTest(loader=name):
                with self.assertRaises(UNSAFE_LOAD_ERRORS):
                    loader(path, "cpu")

    def test_rejects_non_dict_payload(self):
        path = self._save(torch.zeros(3))
        for name, loader in self.LOADERS.items():
            with self.subTest(loader=name):
                with self.assertRaises(ValueError):
                    loader(path, "cpu")


if __name__ == "__main__":
    unittest.main()
