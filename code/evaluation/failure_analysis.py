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
Decoder ablation study: apply multiple global decoders of varying complexity
to the same pre-decoder residual syndromes and compare logical error rates.
"""
import os
import random

import numpy as np
import torch

from evaluation.logical_error_rate import (
    _build_stab_maps,
    _decode_batch,
    map_grid_to_stabilizer_tensor,
    sample_predictions,
)

# LDPC-based decoders built by _build_ldpc_decoders.
LDPC_DECODER_NAMES = ("Union-Find", "BP-only", "BP+LSD-0")

# Ordered names of all decoders run by decoder_ablation_study.
DECODER_NAMES = ("No-op",) + LDPC_DECODER_NAMES + ("Uncorr-PM", "Corr-PM")


def _build_cudaq_decoders(det_model):
    """
    Build GPU-accelerated cudaq-qec nv-qldpc-decoder instances from a Stim DEM.
    Returns dict of {name: (decoder, L_dense)} mirroring _build_ldpc_decoders.

    Decoder variants:
      - "cudaq-BP":        sum-product BP (bp_method=0), no OSD
      - "cudaq-MinSum":    min-sum BP (bp_method=1), no OSD
      - "cudaq-BP+OSD-0":  sum-product BP + OSD order 0
      - "cudaq-BP+OSD-7":  sum-product BP + OSD order 7
      - "cudaq-MemBP":     min-sum+mem BP (bp_method=2, uniform gamma)
      - "cudaq-MemBP+OSD": min-sum+mem BP + OSD order 7
      - "cudaq-RelayBP":   sequential relay (composition=1, bp_method=3)
    """
    import cudaq_qec
    import scipy.sparse as sp
    from beliefmatching.belief_matching import detector_error_model_to_check_matrices

    matrices = detector_error_model_to_check_matrices(det_model)
    H_sparse = sp.csc_matrix(matrices.check_matrix)
    L = matrices.observables_matrix
    priors = np.array(matrices.priors, dtype=np.float64)
    L_dense = np.asarray(L.toarray(), dtype=np.uint8)

    # cudaq-qec expects a dense row-major (C-contiguous) H matrix (uint8)
    H_dense = np.ascontiguousarray(H_sparse.toarray(), dtype=np.uint8)

    # Per-edge priors clamped for numerical stability
    priors_list = np.clip(priors, 1e-9, 1.0 - 1e-9).tolist()

    # Enable num_iter reporting in opt_results for all decoders
    opt_res = {"num_iter": True}

    # max_iterations=50 for standard BP/MinSum/OSD
    bp_kwargs = dict(max_iterations=50, error_rate_vec=priors_list, opt_results=opt_res)
    # max_iterations=100 for MemBP and RelayBP (need more iterations to converge)
    mem_kwargs = dict(max_iterations=100, error_rate_vec=priors_list, opt_results=opt_res)

    decoders = {}
    # list of cudaq decoder names that failed to initialize
    unavailable = []

    # --- Standard BP variants (max_iterations=10) ---
    try:
        # Sum-product BP (no OSD)
        decoders["cudaq-BP"] = (
            cudaq_qec.get_decoder("nv-qldpc-decoder", H_dense, bp_method=0, use_osd=0, **bp_kwargs),
            L_dense,
        )
        # Min-sum BP (no OSD)
        decoders["cudaq-MinSum"] = (
            cudaq_qec.get_decoder("nv-qldpc-decoder", H_dense, bp_method=1, use_osd=0, **bp_kwargs),
            L_dense,
        )
        # Sum-product BP + OSD-0
        decoders["cudaq-BP+OSD-0"] = (
            cudaq_qec.get_decoder(
                "nv-qldpc-decoder", H_dense, bp_method=0, use_osd=1, osd_order=0, **bp_kwargs
            ),
            L_dense,
        )
        # Sum-product BP + OSD-7
        decoders["cudaq-BP+OSD-7"] = (
            cudaq_qec.get_decoder(
                "nv-qldpc-decoder", H_dense, bp_method=0, use_osd=1, osd_order=7, **bp_kwargs
            ),
            L_dense,
        )
    except Exception as e:
        import warnings
        warnings.warn(f"cudaq-qec BP unavailable: {e}")
        unavailable.extend(["cudaq-BP", "cudaq-MinSum", "cudaq-BP+OSD-0", "cudaq-BP+OSD-7"])

    # --- Memory BP variants (max_iterations=100) ---
    try:
        decoders["cudaq-MemBP"] = (
            cudaq_qec.get_decoder(
                "nv-qldpc-decoder",
                H_dense,
                bp_method=2,
                use_sparsity=True,
                gamma0=0.5,
                use_osd=0,
                **mem_kwargs
            ),
            L_dense,
        )
        decoders["cudaq-MemBP+OSD"] = (
            cudaq_qec.get_decoder(
                "nv-qldpc-decoder",
                H_dense,
                bp_method=2,
                use_sparsity=True,
                gamma0=0.5,
                use_osd=1,
                osd_order=7,
                **mem_kwargs
            ),
            L_dense,
        )
    except Exception as e:
        import warnings
        warnings.warn(f"cudaq-qec MemBP unavailable: {e}")
        unavailable.extend(["cudaq-MemBP", "cudaq-MemBP+OSD"])

    # --- RelayBP (max_iterations=100) ---
    # composition=1 (sequential relay), bp_method=3 (min-sum+dmem)
    # gamma_dist=[-0.254, 0.985] optimized for surface codes
    try:
        srelay_cfg = {
            "pre_iter": 10,
            "num_sets": 5,
            "stopping_criterion": "FirstConv",
        }
        # Note: opt_results num_iter not supported for composition=1 per docs
        relay_kwargs = dict(max_iterations=100, error_rate_vec=priors_list)
        decoders["cudaq-RelayBP"] = (
            cudaq_qec.get_decoder(
                "nv-qldpc-decoder",
                H_dense,
                composition=1,
                bp_method=3,
                use_sparsity=True,
                gamma0=0.5,
                gamma_dist=[-0.254, 0.985],
                srelay_config=srelay_cfg,
                **relay_kwargs
            ),
            L_dense,
        )
    except Exception as e:
        import warnings
        warnings.warn(f"cudaq-qec RelayBP unavailable: {e}")
        unavailable.append("cudaq-RelayBP")

    return decoders, unavailable


def _decode_cudaq_batch(decoder, L_dense, syndromes_np):
    """
    Decode a batch of syndromes with a cudaq-qec nv-qldpc-decoder (single-shot loop).
    Returns (obs, stats) where:
      - obs: observable predictions as np.ndarray of shape (B,)
      - stats: dict with per-sample convergence flags, iteration counts
    The decoder.decode() takes list[float] and returns DecoderResult with .result (list[float]).
    """
    B = syndromes_np.shape[0]
    obs = np.zeros(B, dtype=np.uint8)
    converged_flags = np.zeros(B, dtype=bool)
    iter_counts = np.zeros(B, dtype=np.int32)
    for i in range(B):
        syndrome_list = syndromes_np[i].astype(np.float64).tolist()
        result = decoder.decode(syndrome_list)
        correction = np.array(result.result, dtype=np.uint8)
        obs[i] = int((L_dense @ correction).item() %
                     2) if L_dense.shape[0] == 1 else int((L_dense @ correction)[0] % 2)
        converged_flags[i] = result.converged
        # Collect iteration count if available via opt_results
        opt = getattr(result, 'opt_results', None)
        if opt and isinstance(opt, dict) and 'num_iter' in opt:
            iter_counts[i] = opt['num_iter']
    return obs, {"converged_flags": converged_flags, "iter_counts": iter_counts}


def _build_ldpc_decoders(det_model):
    """
    Convert a Stim DetectorErrorModel to an H matrix and build ldpc decoders.
    Returns dict of {name: (decoder, L_dense)} where L_dense is (num_obs, num_mechanisms).
    """
    import scipy.sparse as sp
    from beliefmatching.belief_matching import detector_error_model_to_check_matrices
    from ldpc.bp_decoder import BpDecoder
    from ldpc.bplsd_decoder import BpLsdDecoder
    from ldpc.union_find_decoder import UnionFindDecoder

    matrices = detector_error_model_to_check_matrices(det_model)
    H = sp.csc_matrix(matrices.check_matrix)
    L = matrices.observables_matrix
    priors = np.array(matrices.priors, dtype=np.float64)
    L_dense = np.asarray(L.toarray(), dtype=np.uint8)

    # Clamp priors away from 0/1 for BP stability
    priors = np.clip(priors, 1e-9, 1.0 - 1e-9)

    _uf, _bp, _bplsd = LDPC_DECODER_NAMES
    decoders = {}
    decoders[_uf] = (UnionFindDecoder(H, uf_method="peeling"), L_dense)
    decoders[_bp] = (
        BpDecoder(
            H, error_channel=priors, bp_method="product_sum", max_iter=10, schedule="parallel"
        ),
        L_dense,
    )
    decoders[_bplsd] = (
        BpLsdDecoder(
            H,
            error_channel=priors,
            bp_method="product_sum",
            max_iter=10,
            schedule="parallel",
            lsd_method="lsd_cs",
            lsd_order=0,
        ),
        L_dense,
    )
    return decoders


def _decode_ldpc_batch(decoder, L_dense, syndromes_np):
    """
    Decode a batch of syndromes with an ldpc decoder (single-shot loop).
    Returns observable predictions as np.ndarray of shape (B,).
    """
    B = syndromes_np.shape[0]
    obs = np.zeros(B, dtype=np.uint8)
    for i in range(B):
        # Get the most-likely error configuration from the decoder for this syndrome.
        correction = decoder.decode(syndromes_np[i])
        # Project the correction onto the logical observable via L_dense (mod 2).
        # L_dense has shape (num_obs, num_errors); the first observable row is used.
        obs[i] = (
            int((L_dense @ correction).item() %
                2) if L_dense.shape[0] == 1 else int((L_dense @ correction)[0] % 2)
        )
    return obs


def _build_all_decoders(det_model, dist):
    """Build all decoders (PyMatching, LDPC, cudaq-qec) from the DEM"""
    import pymatching
    matcher_corr = pymatching.Matching.from_detector_error_model(
        det_model, enable_correlations=True
    )
    matcher_uncorr = pymatching.Matching.from_detector_error_model(
        det_model, enable_correlations=False
    )
    ldpc_decoders = _build_ldpc_decoders(det_model)
    cudaq_decoders = {}
    unavailable_decoders = []
    try:
        cudaq_decoders, unavailable_decoders = _build_cudaq_decoders(det_model)
        if dist.rank == 0:
            print(f"[Decoder Ablation] cudaq-qec decoders loaded: {list(cudaq_decoders.keys())}")
            if unavailable_decoders:
                print(f"[Decoder Ablation] cudaq-qec decoders unavailable: {unavailable_decoders}")
    except Exception as e:
        if dist.rank == 0:
            print(f"[Decoder Ablation] cudaq-qec decoders unavailable: {e}")
    return matcher_corr, matcher_uncorr, ldpc_decoders, cudaq_decoders, unavailable_decoders


def _build_logical_operators(D, code_rotation, device):
    """Build parity-check index tensors and logical operator masks for the surface code"""
    maps = _build_stab_maps(D, code_rotation)
    Hx_idx = maps["Hx_idx"].to(device=device, dtype=torch.long)
    Hz_idx = maps["Hz_idx"].to(device=device, dtype=torch.long)
    Hx_mask = maps["Hx_mask"].to(device=device, dtype=torch.bool)
    Hz_mask = maps["Hz_mask"].to(device=device, dtype=torch.bool)
    stab_indices_x = maps["stab_x"].to(device=device, dtype=torch.long)
    stab_indices_z = maps["stab_z"].to(device=device, dtype=torch.long)
    Kx, Kz = maps["Kx"], maps["Kz"]
    D2 = D * D
    if code_rotation.upper() in ("XV", "ZH"):
        Lx = torch.zeros((1, D2), dtype=torch.int32, device=device)
        Lx[0, :D] = 1
        Lz = torch.zeros((1, D2), dtype=torch.int32, device=device)
        Lz[0, ::D] = 1
    else:
        Lx = torch.zeros((1, D2), dtype=torch.int32, device=device)
        Lx[0, ::D] = 1
        Lz = torch.zeros((1, D2), dtype=torch.int32, device=device)
        Lz[0, :D] = 1
    return Hx_idx, Hz_idx, Hx_mask, Hz_mask, stab_indices_x, stab_indices_z, Kx, Kz, Lx, Lz


def _model_forward_and_residual(
    model,
    trainX,
    x_syn_diff,
    z_syn_diff,
    basis,
    B,
    D2,
    T,
    Hx_idx,
    Hz_idx,
    Hx_mask,
    Hz_mask,
    Kx,
    Kz,
    stab_indices_x,
    stab_indices_z,
    Lx,
    Lz,
    th_data,
    th_syn,
    sampling_mode,
    temperature_data,
    temperature_syn,
    cfg,
    device,
    num_boundary_dets,
    baseline_detectors_batch,
    det_model,
):
    """
    Run the pre-decoder model on one batch and build the residual syndrome.

    Returns:
        residual_np: (B, num_detectors) uint8 array - residual syndromes for global decoders.
        pre_L_np:    (B,) int64 array - logical frame contribution from data corrections.
    """
    with torch.amp.autocast(
        device_type=device.type if hasattr(device, "type") else "cuda",
        enabled=getattr(cfg, "enable_fp16", False),
    ):
        logits = model(trainX)
    z_data_corr = sample_predictions(logits[:, 0], th_data, sampling_mode, temperature_data)
    x_data_corr = sample_predictions(logits[:, 1], th_data, sampling_mode, temperature_data)
    syn_x_grid = sample_predictions(logits[:, 2], th_syn, sampling_mode, temperature_syn)
    syn_z_grid = sample_predictions(logits[:, 3], th_syn, sampling_mode, temperature_syn)

    z_flat = z_data_corr.permute(0, 2, 3, 1).contiguous().view(B, D2, T).to(torch.int32)
    x_flat = x_data_corr.permute(0, 2, 3, 1).contiguous().view(B, D2, T).to(torch.int32)
    z_exp = z_flat.unsqueeze(2).expand(B, D2, Kx, T)
    hx_idx_e = Hx_idx.clamp_min(0).view(1, -1, Kx, 1).expand(B, -1, -1, T)
    g_x = z_exp.gather(1, hx_idx_e)
    m_x = Hx_mask.view(1, -1, Kx, 1).expand_as(g_x)
    S_X = (g_x.masked_fill(~m_x, 0).sum(dim=2) & 1)
    x_exp = x_flat.unsqueeze(2).expand(B, D2, Kz, T)
    hz_idx_e = Hz_idx.clamp_min(0).view(1, -1, Kz, 1).expand(B, -1, -1, T)
    g_z = x_exp.gather(1, hz_idx_e)
    m_z = Hz_mask.view(1, -1, Kz, 1).expand_as(g_z)
    S_Z = (g_z.masked_fill(~m_z, 0).sum(dim=2) & 1)

    syn_x_flat = map_grid_to_stabilizer_tensor(syn_x_grid, stab_indices_x).to(torch.int32)
    syn_z_flat = map_grid_to_stabilizer_tensor(syn_z_grid, stab_indices_z).to(torch.int32)
    R_X = torch.empty_like(x_syn_diff, dtype=torch.uint8)
    R_X[:, :, 0] = (x_syn_diff[:, :, 0] + syn_x_flat[:, :, 0] + S_X[:, :, 0]) & 1
    if T > 1:
        R_X[:, :, 1:] = (
            x_syn_diff[:, :, 1:] + syn_x_flat[:, :, 1:] + syn_x_flat[:, :, :-1] + S_X[:, :, 1:]
        ) & 1
    R_Z = torch.empty_like(z_syn_diff, dtype=torch.uint8)
    R_Z[:, :, 0] = (z_syn_diff[:, :, 0] + syn_z_flat[:, :, 0] + S_Z[:, :, 0]) & 1
    if T > 1:
        R_Z[:, :, 1:] = (
            z_syn_diff[:, :, 1:] + syn_z_flat[:, :, 1:] + syn_z_flat[:, :, :-1] + S_Z[:, :, 1:]
        ) & 1

    # Logical frame from data corrections
    if basis == "X":
        pre_L_t = torch.einsum("ld,bdt->blt", Lx.to(torch.float32),
                               z_flat.to(torch.float32)).remainder_(2).to(torch.int32)
    else:
        pre_L_t = torch.einsum("ld,bdt->blt", Lz.to(torch.float32),
                               x_flat.to(torch.float32)).remainder_(2).to(torch.int32)
    pre_L = pre_L_t.sum(dim=2).remainder_(2).view(-1)

    # Build residual detectors (matching logical_error_rate.py exactly)
    if basis == "X":
        initial_detectors = R_X[:, :, 0].view(B, -1)
    else:
        initial_detectors = R_Z[:, :, 0].view(B, -1)
    R_X_rest = R_X[:, :, 1:]
    R_Z_rest = R_Z[:, :, 1:]
    R_cat_rest = torch.cat([R_X_rest, R_Z_rest], dim=1)
    rest_flat = R_cat_rest.permute(0, 2, 1).contiguous().view(B, -1)
    residual = torch.cat([initial_detectors, rest_flat], dim=1).to(torch.uint8)

    # Append boundary detectors from Stim (unchanged by pre-decoder)
    boundary_dets_batch = baseline_detectors_batch[:, -num_boundary_dets:]
    residual = torch.cat(
        [residual, torch.from_numpy(boundary_dets_batch).to(residual.device)], dim=1
    )

    if residual.shape[1] != det_model.num_detectors:
        raise ValueError(
            f"Residual shape {residual.shape} != DEM detectors {det_model.num_detectors}. "
            f"Check interleave order for basis '{basis}' and time slicing."
        )

    return residual.cpu().numpy(), pre_L.cpu().numpy()


def _run_decoders_on_batch(
    residual_np,
    pre_L_np,
    weights,
    ldpc_decoders,
    cudaq_decoders,
    matcher_uncorr,
    matcher_corr,
    cudaq_decoder_names,
    decoder_names,
    gt_obs_np,
    _timing,
    _cudaq_stats,
    weight_bucket_stats,
):
    """
    Run all configured decoders on one batch of residual syndromes.

    Mutates _timing, _cudaq_stats, and weight_bucket_stats in-place.
    Returns:
        all_finals: dict mapping decoder name -> (B,) int array of final observable predictions.
        n_agree:    number of samples where all decoders agreed.
    """
    import time as _t

    B = residual_np.shape[0]

    # 1. No-op: pred_obs = 0
    noop_final = pre_L_np % 2

    # 2. Union-Find (ldpc)
    _uf, _bp, _bplsd = LDPC_DECODER_NAMES
    _t0 = _t.perf_counter()
    uf_dec, uf_L = ldpc_decoders[_uf]
    uf_obs = _decode_ldpc_batch(uf_dec, uf_L, residual_np)
    uf_final = (pre_L_np + uf_obs) % 2
    _timing["uf_decode"] += _t.perf_counter() - _t0

    # 3. BP-only (no LSD fallback)
    _t0 = _t.perf_counter()
    bp_dec, bp_L = ldpc_decoders[_bp]
    bp_obs = _decode_ldpc_batch(bp_dec, bp_L, residual_np)
    bp_final = (pre_L_np + bp_obs) % 2
    _timing["bp_only_decode"] += _t.perf_counter() - _t0

    # 4. BP+LSD-0 (ldpc)
    _t0 = _t.perf_counter()
    bplsd_dec, bplsd_L = ldpc_decoders[_bplsd]
    bplsd_obs = _decode_ldpc_batch(bplsd_dec, bplsd_L, residual_np)
    bplsd_final = (pre_L_np + bplsd_obs) % 2
    _timing["bplsd_decode"] += _t.perf_counter() - _t0

    # 5. Uncorrelated PyMatching
    _t0 = _t.perf_counter()
    uncorr_pred = _decode_batch(matcher_uncorr, residual_np, False)
    uncorr_pred = np.asarray(uncorr_pred, dtype=np.int64).reshape(-1)
    uncorr_final = (pre_L_np + uncorr_pred) % 2
    _timing["uncorr_pm"] += _t.perf_counter() - _t0

    # 6. Correlated PyMatching
    _t0 = _t.perf_counter()
    corr_pred = _decode_batch(matcher_corr, residual_np, True)
    corr_pred = np.asarray(corr_pred, dtype=np.int64).reshape(-1)
    corr_final = (pre_L_np + corr_pred) % 2
    _timing["corr_pm"] += _t.perf_counter() - _t0

    # 7. cudaq-qec GPU-accelerated decoders
    cudaq_finals = {}
    for cn in cudaq_decoder_names:
        _t0 = _t.perf_counter()
        cdec, cL = cudaq_decoders[cn]
        c_obs, c_stats = _decode_cudaq_batch(cdec, cL, residual_np)
        c_final = (pre_L_np + c_obs) % 2
        cudaq_finals[cn] = c_final
        _timing[f"{cn}_decode"] += _t.perf_counter() - _t0
        # Accumulate per-sample convergence, iteration, and error stats
        conv_flags = c_stats["converged_flags"]
        iters = c_stats["iter_counts"]
        fails = (c_final != gt_obs_np)
        _cudaq_stats[cn]["converged_flags"].append(conv_flags)
        _cudaq_stats[cn]["iter_counts"].append(iters)
        _cudaq_stats[cn]["error_flags"].append(fails)

    _t0 = _t.perf_counter()
    all_finals = {
        DECODER_NAMES[0]: noop_final,
        _uf: uf_final,
        _bp: bp_final,
        _bplsd: bplsd_final,
        DECODER_NAMES[4]: uncorr_final,
        DECODER_NAMES[5]: corr_final,
    }
    all_finals.update(cudaq_finals)

    stacked = np.stack([all_finals[n] for n in decoder_names], axis=0)  # (n_decoders, B)
    agree = np.all(stacked == stacked[0:1], axis=0)  # (B,)

    for i in range(B):
        w = int(weights[i])
        bucket = w if w <= 6 else 7  # 0-6, 7+
        if bucket not in weight_bucket_stats:
            weight_bucket_stats[bucket] = {n: [0, 0] for n in decoder_names}
        weight_bucket_stats[bucket]["_total"] = weight_bucket_stats[bucket].get("_total", 0) + 1
        for name in decoder_names:
            if name not in weight_bucket_stats[bucket]:
                weight_bucket_stats[bucket][name] = [0, 0]
            weight_bucket_stats[bucket][name][1] += 1
            if all_finals[name][i] != gt_obs_np[i]:
                weight_bucket_stats[bucket][name][0] += 1

    _timing["bookkeeping"] += _t.perf_counter() - _t0

    return all_finals, int(agree.sum())


def _print_ablation_results(
    basis,
    D,
    cfg,
    total_scanned,
    baseline_errors,
    decoder_errors,
    decoder_names,
    cudaq_decoder_names,
    unavailable_decoders,
    _cudaq_stats,
    n_all_agree,
    all_residual_weights,
    weight_bucket_stats,
    _timing,
):
    """Print timing breakdown, LER summary, convergence stats, and generate plots."""
    _total_time = sum(_timing.values())
    print(f"\n{'='*60}")
    print(f"TIMING BREAKDOWN  (total loop = {_total_time:.2f}s)")
    print(f"{'='*60}")
    for k, v in sorted(_timing.items(), key=lambda x: -x[1]):
        pct = v / max(_total_time, 1e-9) * 100
        print(f"  {k:<20s}  {v:8.2f}s  ({pct:5.1f}%)")
    print(f"{'='*60}")

    print(f"\n{'='*70}")
    print(
        f"DECODER ABLATION STUDY  |  basis={basis}  d={D}  r={cfg.n_rounds}"
        f"  p={getattr(cfg.test, 'p_error', 0.003)}"
    )
    print(f"{'='*70}")
    print(f"Total samples: {total_scanned}")

    baseline_ler = baseline_errors / max(1, total_scanned)
    print(f"\n--- Logical Error Rates ---")
    print(
        f"  {'Baseline (no pre-dec)':<25s}  LER = {baseline_ler:.6f}"
        f"  ({baseline_errors} errors)"
    )
    for name in decoder_names:
        ler = decoder_errors[name] / max(1, total_scanned)
        print(f"  {name:<25s}  LER = {ler:.6f}  ({decoder_errors[name]} errors)")
    if unavailable_decoders:
        for name in unavailable_decoders:
            print(f"  {name:<25s}  LER = {'N/A':>13s}  (unavailable)")

    # cudaq decoder convergence and iteration stats
    if _cudaq_stats:
        print(f"\n--- cudaq-qec BP Convergence & Iteration Breakdown ---")
        print(
            f"  {'Decoder':<20s} {'Conv%':>7s} {'AvgIt':>6s} "
            f"{'Conv.It':>8s} {'Conv.LER':>9s} {'Conv.Err':>9s} "
            f"{'!Conv.It':>8s} {'!Conv.LER':>10s} {'!Conv.Err':>10s}"
        )
        for cn in cudaq_decoder_names:
            st = _cudaq_stats[cn]
            conv_all = np.concatenate(st["converged_flags"])
            iters_all = np.concatenate(st["iter_counts"])
            errs_all = np.concatenate(st["error_flags"])
            N = len(conv_all)
            n_conv = int(conv_all.sum())
            n_noconv = N - n_conv
            conv_pct = n_conv / max(1, N) * 100
            has_iters = iters_all.sum() > 0

            # Converged subset
            if n_conv > 0 and has_iters:
                conv_avg_it = iters_all[conv_all].mean()
                conv_ler = errs_all[conv_all].mean()
                conv_errs = int(errs_all[conv_all].sum())
            else:
                conv_avg_it = conv_ler = 0.0
                conv_errs = 0

            # Non-converged subset
            if n_noconv > 0 and has_iters:
                noconv_avg_it = iters_all[~conv_all].mean()
                noconv_ler = errs_all[~conv_all].mean()
                noconv_errs = int(errs_all[~conv_all].sum())
            else:
                noconv_avg_it = noconv_ler = 0.0
                noconv_errs = 0

            if has_iters:
                avg_it_str = f"{iters_all.mean():5.1f}"
                conv_it_str = f"{conv_avg_it:7.1f}"
                noconv_it_str = f"{noconv_avg_it:7.1f}" if n_noconv > 0 else "    N/A"
            else:
                avg_it_str = "  N/A"
                conv_it_str = "    N/A"
                noconv_it_str = "    N/A"

            noconv_ler_str = f"{noconv_ler:9.6f}" if n_noconv > 0 else "      N/A"
            noconv_err_str = f"{noconv_errs:>9d}" if n_noconv > 0 else "      N/A"

            print(
                f"  {cn:<20s} {conv_pct:>6.1f}% {avg_it_str} "
                f"{conv_it_str} {conv_ler:>9.6f} {conv_errs:>9d} "
                f"{noconv_it_str} {noconv_ler_str} {noconv_err_str}"
            )

    agreement_rate = n_all_agree / max(1, total_scanned)
    print(f"\n--- Decoder Agreement ---")
    print(
        f"  All {len(decoder_names)} decoders agree:"
        f" {agreement_rate*100:.2f}% ({n_all_agree}/{total_scanned})"
    )

    weights_arr = np.array(all_residual_weights)
    print(f"\n--- Residual Weight Distribution ---")
    for w in sorted(weight_bucket_stats.keys()):
        label = f"{w}+" if w == 7 else str(w)
        count = weight_bucket_stats[w].get("_total", 0)
        pct = count / max(1, total_scanned) * 100
        print(f"  Weight {label:>3s}: {count:>7d} samples ({pct:6.2f}%)")
    print(f"  Mean weight: {weights_arr.mean():.3f},  Max: {int(weights_arr.max())}")

    print(f"\n--- Conditional LER by Residual Weight ---")
    header = f"  {'Weight':>7s}"
    for name in decoder_names:
        header += f"  {name:>12s}"
    print(header)
    for w in sorted(weight_bucket_stats.keys()):
        label = f"{w}+" if w == 7 else str(w)
        row = f"  {label:>7s}"
        for name in decoder_names:
            n_err, n_tot = weight_bucket_stats[w].get(name, [0, 0])
            if n_tot > 0:
                row += f"  {n_err/n_tot:>12.6f}"
            else:
                row += f"  {'N/A':>12s}"
        print(row)
    print(f"{'='*70}")

    # --- Plots ---
    _plot_residual_weight_histogram(all_residual_weights, basis, cfg)
    _plot_conditional_ler(weight_bucket_stats, decoder_names, basis, cfg)


@torch.inference_mode()
def decoder_ablation_study(model, device, dist, cfg):
    """
    Run the pre-decoder on the test set, then apply multiple global decoders
    of varying complexity to the same residual syndromes.
    Measures LER per decoder, residual weight distribution, and decoder agreement.

    Uses Stim datapipe (with boundary detectors) for baseline, ground truth, and
    DEM/matcher construction — matching the reference implementation in
    logical_error_rate.py for apples-to-apples comparison.
    """
    import time as _time
    from copy import deepcopy

    # --- Config ---
    th_data = float(getattr(cfg.test, "th_data", 0.0))
    th_syn = float(getattr(cfg.test, "th_syn", 0.0))
    sampling_mode = str(getattr(cfg.test, "sampling_mode", "threshold")).lower()
    temperature = float(getattr(cfg.test, "temperature", 1.0))
    temperature_data = getattr(cfg.test, "temperature_data", None)
    temperature_syn = getattr(cfg.test, "temperature_syn", None)
    temperature_data = float(temperature_data) if temperature_data is not None else temperature
    temperature_syn = float(temperature_syn) if temperature_syn is not None else temperature

    model.eval()
    basis = str(getattr(cfg.test, "meas_basis_test", "X")).upper()
    if basis not in ("X", "Z"):
        basis = "X"

    # --- Dataset ---
    total_samples = int(cfg.test.num_samples)
    samples_per_gpu = total_samples // max(1, dist.world_size)
    from data.factory import DatapipeFactory

    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        rank_seed = 12345 + dist.rank * 1000
        torch.manual_seed(rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rank_seed)
        np.random.seed(rank_seed)
        random.seed(rank_seed)
        cfg_copy = deepcopy(cfg)
        cfg_copy.test.num_samples = samples_per_gpu
        cfg_copy.test.meas_basis_test = basis
        test_dataset = DatapipeFactory.create_datapipe_inference(cfg_copy)
    finally:
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        np.random.set_state(np_state)
        random.setstate(py_state)

    circuit = test_dataset.circ.stim_circuit
    num_obs = circuit.num_observables
    assert num_obs == 1
    det_model = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )

    # --- Decoders ---
    matcher_corr, matcher_uncorr, ldpc_decoders, cudaq_decoders, unavailable_decoders = \
        _build_all_decoders(det_model, dist)
    cudaq_decoder_names = sorted(cudaq_decoders.keys())
    decoder_names = list(DECODER_NAMES) + cudaq_decoder_names

    # --- Baseline data ---
    stim_dets = np.asarray(test_dataset.dets_and_obs[:, :-num_obs], dtype=np.uint8)
    assert stim_dets.shape[1] == det_model.num_detectors, \
        f"Stim dets width {stim_dets.shape[1]} != DEM {det_model.num_detectors}"
    stim_obs = np.asarray(test_dataset.dets_and_obs[:, -num_obs:], dtype=np.uint8)

    surface_code = test_dataset.circ.code
    num_boundary_dets = surface_code.hx.shape[0] if basis == 'X' else surface_code.hz.shape[0]

    # --- Logical operators ---
    D = cfg.distance
    code_rotation = getattr(cfg.data, "code_rotation", "XV")
    Hx_idx, Hz_idx, Hx_mask, Hz_mask, stab_indices_x, stab_indices_z, Kx, Kz, Lx, Lz = \
        _build_logical_operators(D, code_rotation, device)
    D2 = D * D

    if dist.rank == 0:
        print(
            f"\n[Decoder Ablation] basis={basis}, d={D}, r={cfg.n_rounds},"
            f" p={getattr(cfg.test, 'p_error', 0.003)}"
        )
        print(
            f"[Decoder Ablation] Using Stim datapipe (with boundary detectors)"
            f" for apples-to-apples comparison"
        )
        print(
            f"[Decoder Ablation] DEM detectors: {det_model.num_detectors}"
            f" (incl. {num_boundary_dets} boundary)"
        )
        cudaq_names_str = ", ".join(cudaq_decoders.keys()) if cudaq_decoders else "(none)"
        print(
            f"[Decoder Ablation] Decoders: No-op, Union-Find, BP+LSD-0,"
            f" Uncorr PM, Corr PM, {cudaq_names_str}, + Baseline PM"
        )

    # --- Batch loop ---
    batch_size = int(getattr(cfg.test.dataloader, "batch_size", 2048))
    N = len(test_dataset)
    num_batches = (N + batch_size - 1) // batch_size

    total_scanned = 0
    baseline_errors = 0
    decoder_errors = {name: 0 for name in decoder_names}
    all_residual_weights = []
    all_baseline_weights = []
    weight_bucket_stats = {}
    n_all_agree = 0

    _timing = {
        k: 0.0 for k in (
            "collate",
            "baseline_pm",
            "model_fwd",
            "residual_build",
            "uf_decode",
            "bp_only_decode",
            "bplsd_decode",
            "uncorr_pm",
            "corr_pm",
            "bookkeeping",
        )
    }
    for cn in cudaq_decoder_names:
        _timing[f"{cn}_decode"] = 0.0
    _cudaq_stats = {
        cn: {
            "converged_flags": [],
            "iter_counts": [],
            "error_flags": []
        } for cn in cudaq_decoder_names
    }

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)
        B = end - start

        _t0 = _time.perf_counter()
        items = [test_dataset[i] for i in range(start, end)]
        x_syn_diff = torch.stack([it["x_syn_diff"] for it in items]
                                ).to(device=device, dtype=torch.int32)
        z_syn_diff = torch.stack([it["z_syn_diff"] for it in items]
                                ).to(device=device, dtype=torch.int32)
        trainX = torch.stack([it["trainX"] for it in items]).to(device=device)
        _timing["collate"] += _time.perf_counter() - _t0

        _, _, T = x_syn_diff.shape
        if T < 2:
            continue

        # Baseline: raw Stim syndromes + ground truth
        baseline_detectors_batch = stim_dets[start:end]
        gt_obs_batch = stim_obs[start:end]
        all_baseline_weights.extend(baseline_detectors_batch.sum(axis=1).tolist())

        _t0 = _time.perf_counter()
        baseline_pred_obs = _decode_batch(matcher_corr, baseline_detectors_batch, True)
        baseline_pred_obs = np.asarray(baseline_pred_obs, dtype=np.uint8).reshape(-1, num_obs)
        baseline_errors += int((baseline_pred_obs != gt_obs_batch).sum())
        _timing["baseline_pm"] += _time.perf_counter() - _t0

        gt_obs_np = gt_obs_batch.reshape(-1).astype(np.int64)

        # Pre-decoder forward pass + residual syndrome construction
        _t0 = _time.perf_counter()
        residual_np, pre_L_np = _model_forward_and_residual(
            model,
            trainX,
            x_syn_diff,
            z_syn_diff,
            basis,
            B,
            D2,
            T,
            Hx_idx,
            Hz_idx,
            Hx_mask,
            Hz_mask,
            Kx,
            Kz,
            stab_indices_x,
            stab_indices_z,
            Lx,
            Lz,
            th_data,
            th_syn,
            sampling_mode,
            temperature_data,
            temperature_syn,
            cfg,
            device,
            num_boundary_dets,
            baseline_detectors_batch,
            det_model,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        _timing["residual_build"] += _time.perf_counter() - _t0

        weights = residual_np.sum(axis=1)
        all_residual_weights.extend(weights.tolist())

        # All decoder runs
        all_finals, n_agree = _run_decoders_on_batch(
            residual_np,
            pre_L_np,
            weights,
            ldpc_decoders,
            cudaq_decoders,
            matcher_uncorr,
            matcher_corr,
            cudaq_decoder_names,
            decoder_names,
            gt_obs_np,
            _timing,
            _cudaq_stats,
            weight_bucket_stats,
        )
        for name in decoder_names:
            decoder_errors[name] += int((all_finals[name] != gt_obs_np).sum())
        n_all_agree += n_agree

        total_scanned += B
        if dist.rank == 0 and (batch_idx + 1) % 5 == 0:
            print(f"  [Ablation] Processed {total_scanned} samples...")

    if dist.rank == 0:
        _print_ablation_results(
            basis,
            D,
            cfg,
            total_scanned,
            baseline_errors,
            decoder_errors,
            decoder_names,
            cudaq_decoder_names,
            unavailable_decoders,
            _cudaq_stats,
            n_all_agree,
            all_residual_weights,
            weight_bucket_stats,
            _timing,
        )

    return (
        {
            "total_samples": total_scanned,
            "baseline_errors": baseline_errors,
            "decoder_errors": decoder_errors,
            "residual_weights": all_residual_weights,
            "baseline_weights": all_baseline_weights,
            "weight_bucket_stats": weight_bucket_stats,
            "agreement_count": n_all_agree,
            "unavailable_decoders": unavailable_decoders,
        } if dist.rank == 0 else {}
    )


def _plot_residual_weight_histogram(weights, basis, cfg):
    """Plot and save residual weight histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weights_arr = np.array(weights)
    max_w = min(int(weights_arr.max()) + 1, 20)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(-0.5, max_w + 1.5, 1)
    ax.hist(weights_arr, bins=bins, edgecolor="black", alpha=0.7, color="#4C72B0")
    ax.set_xlabel("Residual Weight (# non-zero detectors)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Residual Syndrome Weight Distribution\n"
        f"basis={basis}  d={cfg.distance}  r={cfg.n_rounds}"
        f"  p={getattr(cfg.test, 'p_error', 0.003)}  N={len(weights)}",
        fontsize=11,
    )
    ax.set_yscale("log")
    n_zero = int((weights_arr == 0).sum())
    pct_zero = n_zero / max(1, len(weights_arr)) * 100
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax.text(
        0.5,
        0.95,
        f"Weight-0: {pct_zero:.1f}%",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        color="red"
    )
    plt.tight_layout()
    output_dir = os.path.join(cfg.output, "plots")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"residual_weight_hist_{basis}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def _plot_conditional_ler(weight_bucket_stats, decoder_names, basis, cfg):
    """Plot conditional LER by residual weight for each decoder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buckets = sorted(weight_bucket_stats.keys())
    labels = [f"{w}+" if w == 7 else str(w) for w in buckets]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [
        "#999999", "#E24A33", "#348ABD", "#FBC15E", "#8EBA42", "#988ED5", "#777B7E", "#76B900",
        "#FF6F61", "#2CA02C", "#D62728", "#9467BD", "#17BECF"
    ]
    markers = ["x", "s", "D", "^", "o", "v", "P", "*", "h", "d", "<", ">", "X"]

    for idx, name in enumerate(decoder_names):
        lers = []
        x_pos = []
        for i, w in enumerate(buckets):
            n_err, n_tot = weight_bucket_stats[w].get(name, [0, 0])
            if n_tot >= 10:
                lers.append(n_err / n_tot)
                x_pos.append(i)
        if lers:
            ax.plot(
                x_pos,
                lers,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                label=name,
                linewidth=1.5,
                markersize=6,
            )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Residual Weight (# non-zero detectors)", fontsize=12)
    ax.set_ylabel("Logical Error Rate", fontsize=12)
    ax.set_title(
        f"Conditional LER by Residual Weight\n"
        f"basis={basis}  d={cfg.distance}  r={cfg.n_rounds}"
        f"  p={getattr(cfg.test, 'p_error', 0.003)}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=-0.02)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_dir = os.path.join(cfg.output, "plots")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"conditional_ler_{basis}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
