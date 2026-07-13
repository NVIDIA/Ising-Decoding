# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Focused tests for color-code threshold metric plumbing.
"""

import json
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation import logical_error_rate_color as color_ler
from evaluation.color_chromobius_timing_results import append_chromobius_timing_results
from evaluation.color_sdr_results import append_sdr_results
from evaluation.color_threshold_results import append_threshold_results, threshold_results_path
from qec.color_code.detector_input import ColorDetectorInputTransform
from workflows import run as run_module


class _Dist:
    rank = 0
    world_size = 1


class _DummyDistributedManager:

    @staticmethod
    def initialize():
        return None

    def __init__(self):
        self.rank = 0
        self.world_size = 1
        self.device = torch.device("cpu")


def test_count_logical_errors_color_includes_chromobius_timing(monkeypatch):

    def fake_run_inference_and_decode_color(
        model, device, dist, cfg, return_diagnostics=False, log_summary=True
    ):
        assert return_diagnostics is True
        assert log_summary is True
        return (
            10,
            100,
            25,
            {
                "total_speedup": 1.75,
                "baseline_decode_time_us_per_round": 12.5,
                "predecoder_decode_time_us_per_round": 7.1,
            },
        )

    monkeypatch.setattr(
        color_ler,
        "run_inference_and_decode_color",
        fake_run_inference_and_decode_color,
    )

    cfg = SimpleNamespace(
        n_rounds=5,
        test=SimpleNamespace(
            n_rounds=5,
            meas_basis_test="X",
        ),
    )

    result = color_ler.count_logical_errors_color(
        model=object(),
        device=torch.device("cpu"),
        dist=_Dist(),
        cfg=cfg,
        include_diagnostics=True,
    )

    assert result["X"]["logical_error_rate (mean)"] == pytest.approx(10 / 100 / 5)
    assert result["X"]["chromobius_error_rate (mean)"] == pytest.approx(25 / 100 / 5)
    assert result["X"]["chromobius_timing"]["total_speedup"] == pytest.approx(1.75)
    assert result["X"]["chromobius_timing"]["baseline_decode_time_us_per_round"] == pytest.approx(
        12.5
    )


def test_run_color_inference_dispatches_to_chromobius(monkeypatch, tmp_path):

    def fake_load_model(cfg, dist):
        cfg.resolved_model_checkpoint_path = str(
            tmp_path / "models" / "best_model" / "PreDecoderModelMemory_v1.0.1.pt"
        )
        return object()

    calls = []

    def fake_count_logical_errors_color(
        model, device, dist, cfg, include_diagnostics=False, log_summary=True
    ):
        calls.append(
            {
                "device": device,
                "include_diagnostics": include_diagnostics,
                "log_summary": log_summary,
                "num_samples": int(cfg.test.num_samples),
                "latency_num_samples": int(cfg.test.latency_num_samples),
                "basis": cfg.test.meas_basis_test,
            }
        )
        return {
            "X":
                {
                    "num_shots": int(cfg.test.num_samples),
                    "n_rounds": int(cfg.test.n_rounds),
                    "logical_errors": 1,
                    "chromobius_errors": 2,
                    "logical_error_rate (mean)": 0.01,
                    "logical_error_rate (stderr)": 0.001,
                    "chromobius_error_rate (mean)": 0.02,
                    "chromobius_error_rate (stderr)": 0.002,
                }
        }

    monkeypatch.setattr(run_module, "_load_model", fake_load_model)
    monkeypatch.setattr(run_module, "DistributedManager", _DummyDistributedManager)
    monkeypatch.setattr(
        color_ler,
        "count_logical_errors_color",
        fake_count_logical_errors_color,
    )
    monkeypatch.setenv("PREDECODER_INFERENCE_NUM_SAMPLES", "17")
    monkeypatch.setenv("PREDECODER_INFERENCE_LATENCY_SAMPLES", "3")
    monkeypatch.setenv("PREDECODER_INFERENCE_MEAS_BASIS", "X")

    cfg = SimpleNamespace(
        code="color",
        output=str(tmp_path),
        distance=5,
        n_rounds=5,
        workflow=SimpleNamespace(task="inference"),
        model=SimpleNamespace(version="predecoder_memory_v1"),
        test=SimpleNamespace(
            distance=5,
            n_rounds=5,
            p_error=0.001,
            meas_basis_test="both",
            num_samples=100,
            latency_num_samples=11,
            use_model_checkpoint=-1,
        ),
    )

    run_module.run_color(cfg)

    assert calls == [
        {
            "device": torch.device("cpu"),
            "include_diagnostics": True,
            "log_summary": True,
            "num_samples": 17,
            "latency_num_samples": 3,
            "basis": "X",
        }
    ]


def test_load_model_safetensors_records_checkpoint_path(monkeypatch, tmp_path):
    safetensors_path = tmp_path / "color_model.safetensors"
    safetensors_path.write_bytes(b"placeholder")

    safetensors_module = ModuleType("export.safetensors_utils")

    def fake_load_safetensors(path, model_id=None, device="cpu"):
        assert path == str(safetensors_path)
        assert model_id is None
        assert device == "cpu"
        return torch.nn.Linear(1, 1), {
            "model_id": "1",
            "quant_format": "fp16",
            "receptive_field": "5",
        }

    safetensors_module.load_safetensors = fake_load_safetensors
    monkeypatch.setitem(sys.modules, "export.safetensors_utils", safetensors_module)
    monkeypatch.setenv("PREDECODER_SAFETENSORS_CHECKPOINT", str(safetensors_path))

    cfg = SimpleNamespace(
        workflow=SimpleNamespace(task="threshold"),
        model=SimpleNamespace(
            out_channels=4,
            input_channels=4,
            num_filters=[16, 4],
        ),
        enable_fp16=False,
    )

    model = run_module._load_model(cfg, _DummyDistributedManager())

    assert isinstance(model, torch.nn.Linear)
    assert cfg.enable_fp16 is True
    assert cfg.resolved_model_checkpoint_path == str(safetensors_path)


def test_record_resolved_model_path_force_adds_structured_key(tmp_path):
    cfg = OmegaConf.create({"output": str(tmp_path)})
    OmegaConf.set_struct(cfg, True)
    checkpoint_path = str(tmp_path / "model.safetensors")

    run_module._record_resolved_model_path(cfg, checkpoint_path)

    assert cfg.resolved_model_checkpoint_path == checkpoint_path


def test_chromobius_timing_summary_includes_pre_post_breakdown():
    summary = color_ler._build_chromobius_timing_summary(
        detector_shape=(10, 20),
        packed_detector_shape=(10, 3),
        total_samples=10,
        num_batches_processed=1,
        n_rounds=5,
        baseline_decode_time=0.010,
        predecoder_decode_time=0.020,
        baseline_density_sum=0.2,
        residual_density_sum=0.1,
        floor_time_per_round=0.0,
        baseline_batch_us_per_round=[200.0],
        predecoder_batch_us_per_round=[400.0],
        single_shot_latency=None,
        inclusive_timing_totals={
            "dataloader_batch_time": 0.001,
            "batch_to_device_time": 0.002,
            "baseline_pack_time": 0.003,
            "model_forward_time": 0.004,
            "prediction_sampling_time": 0.005,
            "syndrome_reconstruction_time": 0.006,
            "residual_assembly_time": 0.007,
            "residual_pack_time": 0.008,
        },
    )

    inclusive = summary["inclusive_timing"]
    assert inclusive["input_preprocess_time_us_per_round"] == pytest.approx(60.0)
    assert inclusive["output_postprocess_time_us_per_round"] == pytest.approx(520.0)
    assert inclusive["baseline_inclusive_time_us_per_round"] == pytest.approx(260.0)
    assert inclusive["predecoder_inclusive_time_us_per_round"] == pytest.approx(1000.0)
    assert inclusive["breakdown_us_per_round"]["residual_pack_time"] == pytest.approx(160.0)
    assert "production" in inclusive["measurement_note"]
    assert "worker" in inclusive["dataloader_batch_time_note"]


def test_run_color_threshold_writes_count_only_aggregate(monkeypatch, tmp_path):

    def fake_load_model(cfg, dist):
        cfg.resolved_model_checkpoint_path = str(
            tmp_path / "models" / "best_model" / "PreDecoderModelMemory_v1.0.1.pt"
        )
        return object()

    def fake_count_logical_errors_color(
        model, device, dist, cfg, include_diagnostics=False, log_summary=True
    ):
        assert include_diagnostics is False
        assert log_summary is False
        return {
            cfg.test.meas_basis_test:
                {
                    "num_shots": 100,
                    "n_rounds": int(cfg.test.n_rounds),
                    "logical_errors": 10 if cfg.test.meas_basis_test == "X" else 12,
                    "chromobius_errors": 20 if cfg.test.meas_basis_test == "X" else 24,
                    "logical_error_rate (mean)": 0.02,
                    "logical_error_rate (stderr)": 0.001,
                    "chromobius_error_rate (mean)": 0.04,
                    "chromobius_error_rate (stderr)": 0.002,
                }
        }

    monkeypatch.setattr(run_module, "_load_model", fake_load_model)
    monkeypatch.setattr(run_module, "DistributedManager", _DummyDistributedManager)
    monkeypatch.setattr(
        color_ler,
        "count_logical_errors_color",
        fake_count_logical_errors_color,
    )

    cfg = SimpleNamespace(
        code="color",
        output=str(tmp_path),
        distance=5,
        n_rounds=5,
        workflow=SimpleNamespace(task="threshold"),
        model=SimpleNamespace(version="predecoder_memory_v1"),
        threshold=SimpleNamespace(
            p_values=[0.001],
            distances=[5],
            n_rounds=[5],
            num_samples=100,
            basis="both",
        ),
        test=SimpleNamespace(
            distance=5,
            n_rounds=5,
            p_error=0.001,
            meas_basis_test="both",
            num_samples=100,
        ),
    )

    run_module.run_color(cfg)

    result_path = tmp_path / "models" / "best_model" / "threshold_results_n_rounds_eq_d.json"
    payload = json.loads(result_path.read_text())
    x_payload = payload["points"]["5"]["0.001"]["X"]
    z_payload = payload["points"]["5"]["0.001"]["Z"]

    assert x_payload["pd_chromobius"]["logical_errors"] == 10
    assert x_payload["pd_chromobius"]["shots"] == 100
    assert x_payload["chromobius"]["logical_errors"] == 20
    assert "chromobius_timing" not in x_payload
    assert "syndrome_density" not in x_payload

    assert z_payload["pd_chromobius"]["logical_errors"] == 12
    assert z_payload["chromobius"]["logical_errors"] == 24


def test_threshold_aggregation_appends_and_rejects_round_mismatch(tmp_path):
    cfg = SimpleNamespace(
        model=SimpleNamespace(version="predecoder_memory_v1"),
        test=SimpleNamespace(use_model_checkpoint=-1),
    )
    model_path = str(tmp_path / "models" / "best_model" / "PreDecoderModelMemory_v1.0.1.pt")
    row = {
        "distance": 5,
        "n_rounds": 5,
        "p": 0.001,
        "basis": "X",
        "logical_errors": 10,
        "num_shots": 100,
        "chromobius_errors": 20,
    }

    result_path, _ = append_threshold_results(cfg, [row], model_path)
    append_threshold_results(cfg, [row], model_path)
    payload = json.loads(
        (tmp_path / "models" / "best_model" / "threshold_results_n_rounds_eq_d.json").read_text()
    )
    point = payload["points"]["5"]["0.001"]["X"]
    assert result_path.endswith("threshold_results_n_rounds_eq_d.json")
    assert point["pd_chromobius"]["logical_errors"] == 20
    assert point["pd_chromobius"]["shots"] == 200
    assert point["chromobius"]["logical_errors"] == 40

    payload["points"]["5"]["0.001"]["X"]["n_rounds"] = 99
    result_file = tmp_path / "models" / "best_model" / "threshold_results_n_rounds_eq_d.json"
    result_file.write_text(json.dumps(payload, indent=2))
    before = result_file.read_text()
    with pytest.raises(ValueError, match="n_rounds"):
        append_threshold_results(cfg, [row], model_path)
    assert result_file.read_text() == before


def test_threshold_explicit_checkpoint_path_includes_checkpoint_and_rounds_mode(tmp_path):
    cfg = SimpleNamespace(
        model=SimpleNamespace(version="predecoder_memory_v1"),
        test=SimpleNamespace(use_model_checkpoint=12),
    )
    model_path = str(tmp_path / "models" / "PreDecoderModelMemory_v1.0.12.pt")
    row = {
        "distance": 5,
        "n_rounds": 20,
        "p": 0.001,
        "basis": "X",
        "logical_errors": 10,
        "num_shots": 100,
        "chromobius_errors": 20,
    }

    result_path = threshold_results_path(cfg, model_path, [row])

    assert result_path.endswith(
        "models/PreDecoderModelMemory_v1.0.12_threshold_results_n_rounds_eq_4d.json"
    )


def test_sdr_aggregation_appends_raw_counts_and_rejects_round_mismatch(tmp_path):
    cfg = SimpleNamespace(
        model=SimpleNamespace(version="predecoder_memory_v1"),
        test=SimpleNamespace(use_model_checkpoint=-1),
    )
    model_path = str(tmp_path / "models" / "best_model" / "PreDecoderModelMemory_v1.0.1.pt")
    row = {
        "distance": 5,
        "n_rounds": 5,
        "p": 0.001,
        "basis": "X",
        "input_syndrome_ones": 30,
        "residual_syndrome_ones": 10,
        "syndrome_elements": 100,
    }

    result_path, _ = append_sdr_results(cfg, [row], model_path)
    append_sdr_results(cfg, [row], model_path)
    result_file = tmp_path / "models" / "best_model" / "sdr_results_n_rounds_eq_d.json"
    payload = json.loads(result_file.read_text())
    point = payload["points"]["5"]["0.001"]["X"]

    assert result_path.endswith("sdr_results_n_rounds_eq_d.json")
    assert point["input_syndrome_ones"] == 60
    assert point["residual_syndrome_ones"] == 20
    assert point["syndrome_elements"] == 200
    assert point["input_syndrome_density"] == pytest.approx(0.3)
    assert point["residual_syndrome_density"] == pytest.approx(0.1)
    assert point["reduction_factor"] == pytest.approx(3.0)

    payload["points"]["5"]["0.001"]["X"]["n_rounds"] = 99
    result_file.write_text(json.dumps(payload, indent=2))
    before = result_file.read_text()
    with pytest.raises(ValueError, match="n_rounds"):
        append_sdr_results(cfg, [row], model_path)
    assert result_file.read_text() == before


def test_chromobius_timing_aggregation_merges_sufficient_statistics(tmp_path):
    cfg = SimpleNamespace(
        model=SimpleNamespace(version="predecoder_memory_v1"),
        test=SimpleNamespace(use_model_checkpoint=-1),
    )
    model_path = str(tmp_path / "models" / "best_model" / "PreDecoderModelMemory_v1.0.1.pt")
    row_a = {
        "distance": 5,
        "n_rounds": 5,
        "p": 0.001,
        "basis": "X",
        "original_syndromes":
            {
                "shots": 2,
                "sum_us_per_round": 4.0,
                "sum_sq_us_per_round": 10.0,
                "avg_us_per_round": 2.0,
                "min_us_per_round": 1.0,
                "max_us_per_round": 3.0,
            },
        "residual_syndromes":
            {
                "shots": 2,
                "sum_us_per_round": 8.0,
                "sum_sq_us_per_round": 34.0,
                "avg_us_per_round": 4.0,
                "min_us_per_round": 3.0,
                "max_us_per_round": 5.0,
            },
    }
    row_b = {
        **row_a,
        "original_syndromes":
            {
                "shots": 1,
                "sum_us_per_round": 5.0,
                "sum_sq_us_per_round": 25.0,
                "min_us_per_round": 5.0,
                "max_us_per_round": 5.0,
            },
        "residual_syndromes":
            {
                "shots": 1,
                "sum_us_per_round": 7.0,
                "sum_sq_us_per_round": 49.0,
                "min_us_per_round": 7.0,
                "max_us_per_round": 7.0,
            },
    }

    result_path, _ = append_chromobius_timing_results(cfg, [row_a], model_path)
    append_chromobius_timing_results(cfg, [row_b], model_path)
    result_file = tmp_path / "models" / "best_model" / "chromobius_timing_results_n_rounds_eq_d.json"
    payload = json.loads(result_file.read_text())
    point = payload["points"]["5"]["0.001"]["X"]
    original = point["original_syndromes"]
    residual = point["residual_syndromes"]

    assert result_path.endswith("chromobius_timing_results_n_rounds_eq_d.json")
    assert original["shots"] == 3
    assert original["avg_us_per_round"] == pytest.approx(3.0)
    assert original["variance_us_per_round_sq"] == pytest.approx(4.0)
    assert original["min_us_per_round"] == pytest.approx(1.0)
    assert original["max_us_per_round"] == pytest.approx(5.0)
    assert residual["shots"] == 3
    assert residual["avg_us_per_round"] == pytest.approx(5.0)
    assert residual["variance_us_per_round_sq"] == pytest.approx(4.0)
    assert residual["min_us_per_round"] == pytest.approx(3.0)
    assert residual["max_us_per_round"] == pytest.approx(7.0)


def test_run_color_sdr_writes_model_local_aggregate(monkeypatch, tmp_path):

    def fake_load_model(cfg, dist):
        cfg.resolved_model_checkpoint_path = str(
            tmp_path / "models" / "best_model" / "PreDecoderModelMemory_v1.0.1.pt"
        )
        return object()

    calls = []

    def fake_compute_sdr(model, device, dist, cfg):
        calls.append(
            (
                int(cfg.test.distance), int(cfg.test.n_rounds), float(cfg.test.p_error),
                cfg.test.meas_basis_test
            )
        )
        return {
            "input_syndrome_ones": 30 if cfg.test.meas_basis_test == "X" else 60,
            "residual_syndrome_ones": 10 if cfg.test.meas_basis_test == "X" else 20,
            "syndrome_elements": 100,
            "input_syndrome_density": 0.3 if cfg.test.meas_basis_test == "X" else 0.6,
            "residual_syndrome_density": 0.1 if cfg.test.meas_basis_test == "X" else 0.2,
            "reduction_factor": 3.0,
            "basis": cfg.test.meas_basis_test,
        }

    monkeypatch.setattr(run_module, "_load_model", fake_load_model)
    monkeypatch.setattr(run_module, "DistributedManager", _DummyDistributedManager)
    monkeypatch.setattr(color_ler, "compute_syndrome_density_reduction_color", fake_compute_sdr)

    cfg = SimpleNamespace(
        code="color",
        output=str(tmp_path),
        distance=5,
        n_rounds=5,
        workflow=SimpleNamespace(task="sdr"),
        model=SimpleNamespace(version="predecoder_memory_v1"),
        threshold=SimpleNamespace(
            p_values=[0.001],
            distances=[5],
            n_rounds=[5],
            num_samples=100,
            basis="both",
        ),
        test=SimpleNamespace(
            distance=5,
            n_rounds=5,
            p_error=0.001,
            meas_basis_test="both",
            num_samples=100,
            use_model_checkpoint=-1,
        ),
    )

    run_module.run_color(cfg)

    result_path = tmp_path / "models" / "best_model" / "sdr_results_n_rounds_eq_d.json"
    payload = json.loads(result_path.read_text())
    assert calls == [(5, 5, 0.001, "X"), (5, 5, 0.001, "Z")]
    assert payload["points"]["5"]["0.001"]["X"]["input_syndrome_ones"] == 30
    assert payload["points"]["5"]["0.001"]["Z"]["residual_syndrome_ones"] == 20


def test_run_color_chromobius_timing_writes_model_local_aggregate(monkeypatch, tmp_path):

    def fake_load_model(cfg, dist):
        cfg.resolved_model_checkpoint_path = str(
            tmp_path / "models" / "best_model" / "PreDecoderModelMemory_v1.0.1.pt"
        )
        return object()

    calls = []

    def fake_timing(model, device, dist, cfg):
        calls.append(
            (
                int(cfg.test.distance), int(cfg.test.n_rounds), float(cfg.test.p_error),
                cfg.test.meas_basis_test
            )
        )
        return {
            "basis": cfg.test.meas_basis_test,
            "n_rounds": int(cfg.test.n_rounds),
            "samples_timed": 2,
            "original_syndromes":
                {
                    "shots": 2,
                    "sum_us_per_round": 4.0,
                    "sum_sq_us_per_round": 10.0,
                    "avg_us_per_round": 2.0,
                    "min_us_per_round": 1.0,
                    "max_us_per_round": 3.0,
                },
            "residual_syndromes":
                {
                    "shots": 2,
                    "sum_us_per_round": 8.0,
                    "sum_sq_us_per_round": 34.0,
                    "avg_us_per_round": 4.0,
                    "min_us_per_round": 3.0,
                    "max_us_per_round": 5.0,
                },
        }

    monkeypatch.setattr(run_module, "_load_model", fake_load_model)
    monkeypatch.setattr(run_module, "DistributedManager", _DummyDistributedManager)
    monkeypatch.setattr(color_ler, "compute_chromobius_single_shot_timing_color", fake_timing)

    cfg = SimpleNamespace(
        code="color",
        output=str(tmp_path),
        distance=5,
        n_rounds=5,
        workflow=SimpleNamespace(task="chromobius_timing"),
        model=SimpleNamespace(version="predecoder_memory_v1"),
        threshold=SimpleNamespace(
            p_values=[0.001],
            distances=[5],
            n_rounds=[5],
            num_samples=100,
            basis="X",
        ),
        test=SimpleNamespace(
            distance=5,
            n_rounds=5,
            p_error=0.001,
            meas_basis_test="X",
            num_samples=100,
            use_model_checkpoint=-1,
        ),
    )

    run_module.run_color(cfg)

    result_path = tmp_path / "models" / "best_model" / "chromobius_timing_results_n_rounds_eq_d.json"
    payload = json.loads(result_path.read_text())
    point = payload["points"]["5"]["0.001"]["X"]
    assert calls == [(5, 5, 0.001, "X")]
    assert point["original_syndromes"]["avg_us_per_round"] == pytest.approx(2.0)
    assert point["residual_syndromes"]["variance_us_per_round_sq"] == pytest.approx(2.0)


def test_predecoder_color_eval_module_assembles_residual_and_logical():

    class StaticModel(torch.nn.Module):

        def __init__(self, logits):
            super().__init__()
            self.register_buffer("logits", logits)

        def forward(self, trainX):
            return self.logits.expand(trainX.shape[0], -1, -1, -1, -1)

    def logits_from_bits(bits):
        return torch.where(
            bits.bool(),
            torch.ones_like(bits, dtype=torch.float32),
            -torch.ones_like(bits, dtype=torch.float32),
        )

    z_data = torch.tensor(
        [[
            [[1, 0], [1, 0]],
            [[0, 1], [0, 1]],
            [[1, 1], [0, 0]],
        ]],
        dtype=torch.int32,
    )
    zeros = torch.zeros_like(z_data)
    logits = torch.stack(
        [
            logits_from_bits(z_data),
            logits_from_bits(zeros),
            logits_from_bits(zeros),
            logits_from_bits(zeros),
        ],
        dim=1,
    )
    maps = {
        "H_idx": torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        "H_mask": torch.ones(2, 2, dtype=torch.bool),
        "stab_to_grid": torch.tensor([0, 3], dtype=torch.long),
        "data_to_grid": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "num_plaq": 2,
        "num_data": 4,
        "n_rows": 2,
        "n_cols": 2,
        "K": 2,
    }
    cfg = SimpleNamespace(
        enable_fp16=False,
        test=SimpleNamespace(
            th_data=0.0,
            th_syn=0.0,
            sampling_mode="threshold",
            temperature=1.0,
        ),
    )
    module = color_ler.PreDecoderColorEvalModule(
        StaticModel(logits),
        cfg,
        maps,
        basis="X",
        obs_support=torch.tensor([1, 0, 1, 0], dtype=torch.float32),
        num_boundary_dets=2,
    )

    trainX = torch.zeros(1, 4, 3, 2, 2)
    x_syn_diff = torch.zeros(1, 2, 3, dtype=torch.int32)
    z_syn_diff = torch.zeros(1, 2, 3, dtype=torch.int32)
    boundary = torch.tensor([[1, 0]], dtype=torch.uint8)

    pre_L, residual = module.forward_parts(trainX, x_syn_diff, z_syn_diff, boundary)
    combined = module(trainX, x_syn_diff, z_syn_diff, boundary)

    assert pre_L.tolist() == [1]
    assert residual.tolist() == [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]]
    assert combined.tolist() == [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]]


def test_predecoder_color_eval_module_null_temperatures_fall_back():
    """Public configs set test.temperature_data/temperature_syn to null.

    OmegaConf getattr returns None for an existing-but-null key (the fallback
    only applies to missing keys), so the eval module must treat None as
    "use the main temperature" instead of crashing on float(None).
    """
    from omegaconf import OmegaConf

    maps = {
        "H_idx": torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        "H_mask": torch.ones(2, 2, dtype=torch.bool),
        "stab_to_grid": torch.tensor([0, 3], dtype=torch.long),
        "data_to_grid": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "num_plaq": 2,
        "num_data": 4,
        "n_rows": 2,
        "n_cols": 2,
        "K": 2,
    }

    def build_module(cfg):
        return color_ler.PreDecoderColorEvalModule(
            torch.nn.Identity(),
            cfg,
            maps,
            basis="X",
            obs_support=torch.tensor([1, 0, 1, 0], dtype=torch.float32),
            num_boundary_dets=2,
        )

    cfg = OmegaConf.create(
        {
            "enable_fp16": False,
            "test":
                {
                    "th_data": 0.0,
                    "th_syn": 0.0,
                    "sampling_mode": "threshold",
                    "temperature": 0.7,
                    "temperature_data": None,
                    "temperature_syn": None,
                },
        }
    )
    module = build_module(cfg)
    assert module.temperature_data == pytest.approx(0.7)
    assert module.temperature_syn == pytest.approx(0.7)

    cfg.test.temperature_data = 0.3
    cfg.test.temperature_syn = 0.9
    module = build_module(cfg)
    assert module.temperature_data == pytest.approx(0.3)
    assert module.temperature_syn == pytest.approx(0.9)


def test_color_detector_input_transform_dense_and_gather_match():
    gather_transform = ColorDetectorInputTransform(
        distance=3,
        rounds=3,
        basis="X",
        preprocess_strategy="gather",
    )
    dense_transform = ColorDetectorInputTransform(
        distance=3,
        rounds=3,
        basis="X",
        preprocess_strategy="dense_matmul",
    )
    dets = (torch.arange(gather_transform.detector_width).view(1, -1) % 2).to(torch.float32)

    gather_train_x, gather_x, gather_z, gather_boundary = gather_transform.build_train_x(dets)
    dense_train_x, dense_x, dense_z, dense_boundary = dense_transform.build_train_x(dets)

    assert torch.equal(gather_train_x, dense_train_x)
    assert torch.equal(gather_x, dense_x)
    assert torch.equal(gather_z, dense_z)
    assert torch.equal(gather_boundary, dense_boundary)
    assert gather_x.dtype == torch.int32
    assert gather_train_x.shape == (1, 4, 3, gather_transform.height, gather_transform.width)
