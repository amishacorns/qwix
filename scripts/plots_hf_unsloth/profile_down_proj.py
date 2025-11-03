#!/usr/bin/env python3
"""
Profile down_proj weights from unsloth/gpt-oss-20b-BF16:
- Download the repo and scan all .safetensors shards to find the target tensor.
- Locate the tensor for model.layers.0.mlp.experts.down_proj.* (weight/kernel).
# Quantize the weight with Qwix using ABSMAX calibration and subchannel tiling.
# Compute per-subchannel stats:
#     * unique quantized values per subchannel
# Emit a histogram of unique-counts across subchannels and a JSON with metrics.

Dependencies (install if missing):
  pip install huggingface_hub safetensors matplotlib

Example:
  python qwix/scripts/plots_hf_unsloth/profile_down_proj.py \
    --repo unsloth/gpt-oss-20b-BF16 \
    --pattern model.layers.0.mlp.experts.down_proj \
    --axis -1 --tile-size 128 --qtype float8_e4m3fn \
    --outdir /tmp/qwix_profile

This script always uses the Hugging Face repo to locate shards; no local path needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

# Optional deps
try:
  from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover - optional
  snapshot_download = None

try:
  from safetensors.numpy import load_file as sf_load
except Exception:  # pragma: no cover - optional
  sf_load = None

try:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional
  plt = None

# Ensure repository root is on path so we can import qwix.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import jax
from jax import numpy as jnp
from qwix._src.core import qarray


def _list_all_safetensors(repo_id: str, cache_dir: Optional[str]) -> List[Path]:
  if snapshot_download is None:
    raise RuntimeError("huggingface_hub not installed. Install with: pip install huggingface_hub")
  local_dir = snapshot_download(repo_id=repo_id, allow_patterns=["*.safetensors"], cache_dir=cache_dir)
  candidates: List[Path] = []
  for root, _dirs, files in os.walk(local_dir):
    for f in files:
      if f.endswith(".safetensors") and not f.endswith(".index.safetensors") and f != "model.safetensors.index.json":
        candidates.append(Path(root) / f)
  candidates.sort()
  return candidates


def _find_tensor_key_and_shard(repo_id: str, cache_dir: Optional[str], pattern: str) -> Tuple[Path, str]:
  shards = _list_all_safetensors(repo_id, cache_dir)
  if not shards:
    raise FileNotFoundError(f"No .safetensors found for repo {repo_id}")
  last_error: Exception | None = None
  for shard in shards:
    try:
      tensors = _load_safetensors(shard)
      key = _select_weight_key(tensors, pattern)
      return shard, key
    except Exception as e:  # keep searching other shards
      last_error = e
      continue
  raise RuntimeError(f"Failed to locate tensor matching pattern '{pattern}' in any shard. Last error: {last_error}")


def _load_safetensors(path: Path) -> dict:
  if sf_load is None:
    raise RuntimeError("safetensors not installed. Install with: pip install safetensors")
  return sf_load(str(path))


def _select_weight_key(tensors: dict, pattern: str) -> str:
  # Heuristic: match substring pattern and a typical leaf name
  preferred_suffixes = ["weight", "kernel", "w", "_weight"]
  matches = [k for k in tensors.keys() if pattern in k]
  if not matches:
    # Try replacing dots with slashes or vice versa if needed
    alt = pattern.replace("/", ".")
    matches = [k for k in tensors.keys() if alt in k]
  if not matches:
    raise KeyError(f"No tensors matched pattern '{pattern}'. Available example keys: {list(tensors.keys())[:10]}")
  # Prefer keys that end with common weight suffixes
  def score(k: str) -> Tuple[int, int]:
    suf_score = -1
    for i, suf in enumerate(preferred_suffixes):
      if k.endswith(suf) or k.endswith(suf + ":0"):
        suf_score = -i  # earlier suffix gets higher score
        break
    # shorter key gets slight preference
    return (suf_score, -len(k))
  matches.sort(key=score, reverse=True)
  return matches[0]


def _parse_qtype(qtype_str: str):
  """Restrict qtype to float8_e4m3fn or float4_e2m1fn (and common aliases)."""
  m = qtype_str.lower().replace("-", "_")
  if m in ("float8_e4m3fn", "f8", "fp8", "e4m3", "float8"):
    return jnp.float8_e4m3fn
  if m in ("float4_e2m1fn", "f4", "fp4", "e2m1", "float4"):
    return jnp.float4_e2m1fn
  raise ValueError(
      f"Unsupported qtype: {qtype_str}. Allowed: float8_e4m3fn | float4_e2m1fn"
  )


def _to_jnp(array: np.ndarray) -> jax.Array:
  # Some safetensors may contain bfloat16; cast to bf16 or fp32 for safety
  if array.dtype.name in ("bfloat16", "bf16"):
    return jnp.array(array, dtype=jnp.bfloat16)
  # Fallback to float32 to ensure quantization allowed
  if array.dtype.kind == "f" and array.dtype.itemsize >= 2:
    return jnp.array(array, dtype=jnp.bfloat16 if array.dtype.itemsize == 2 else jnp.float32)
  # If it's already a float of unsupported precision, cast to fp32
  return jnp.array(array, dtype=jnp.float32)


def _compute_unique_counts(qvals: np.ndarray, axis: int, tile_size: int) -> np.ndarray:
  """Compute unique-value counts per subchannel-tile, requiring exact tiling.

  Assumes Qwix pads the tiled axis so that tile_size divides that dimension.
  """
  ndim = qvals.ndim
  axis = axis if axis >= 0 else axis + ndim
  if qvals.shape[axis] % tile_size != 0:
    raise ValueError(f"Axis {axis} size {qvals.shape[axis]} not divisible by tile_size {tile_size}")
  tile_count = qvals.shape[axis] // tile_size
  # Move axis to last for convenience
  q = np.moveaxis(qvals, axis, -1)  # [..., K]
  new_shape = (-1, tile_count, tile_size)  # batch_of_subchannels, tile_count, tile_size
  q = q.reshape(*q.shape[:-1], tile_count, tile_size).reshape(new_shape)
  # Each subchannel is along the last dim per tile; compute unique per subchannel-tile
  # Result shape: (batch_of_subchannels, tile_count)
  unique_counts = np.zeros((q.shape[0], q.shape[1]), dtype=np.int32)
  for i in range(q.shape[0]):
    vals = q[i]
    for t in range(tile_count):
      unique_counts[i, t] = np.unique(vals[t]).size
  return unique_counts.reshape(-1)


# Removed MSE computation: weight may be padded; we keep only unique analysis.


def main():
  parser = argparse.ArgumentParser(description="Profile down_proj quantization stats")
  parser.add_argument("--repo", type=str, default="unsloth/gpt-oss-20b-BF16", help="HF repo id to download from if no --safetensors is given")
  parser.add_argument("--cache-dir", type=str, default=None, help="HF cache dir")
  parser.add_argument("--pattern", type=str, default="model.layers.1.mlp.experts.gate_up_proj", help="Substring to locate the target tensor key")
  parser.add_argument("--axis", type=int, default=-1, help="Axis to tile (subchannel axis)")
  parser.add_argument("--tile-size", type=int, default=128, help="Tile size for subchannel quantization")
  parser.add_argument(
      "--tile-sizes",
      type=str,
      default=None,
      help="Comma-separated list of tile sizes to evaluate, e.g. '64,128'. Overrides --tile-size if provided.",
  )
  parser.add_argument(
      "--qtype",
      type=str,
      default="float4_e2m1fn",
      help="Quantization type: float8_e4m3fn | float4_e2m1fn",
  )
  parser.add_argument("--outdir", type=str, default="/tmp/qwix_profile", help="Output directory for plots and metrics")
  parser.add_argument("--save-prefix", type=str, default="down_proj_layer0", help="Filename prefix for outputs")
  parser.add_argument(
      "--calib",
      type=str,
      default="absmax",
      help="Calibration method string, e.g. 'absmax' or 'mse,0.5,21'",
  )
  args = parser.parse_args()

  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)

  shard_path, key = _find_tensor_key_and_shard(args.repo, args.cache_dir, args.pattern)
  tensors = _load_safetensors(shard_path)
  print(f"Using safetensors shard: {shard_path}")
  print(f"Selected tensor key: {key}")

  weight_np = tensors[key]
  # Convert to jnp for Qwix
  w = _to_jnp(weight_np)
  print(f"Weight shape: {tuple(w.shape)}, dtype: {w.dtype}")

  # Common axis setup
  axis = args.axis if args.axis >= 0 else args.axis + w.ndim
  if axis < 0 or axis >= w.ndim:
    raise ValueError(f"Invalid axis {args.axis} for weight with ndim {w.ndim}")
  ch_axes = tuple(i for i in range(w.ndim) if i != axis)

  # Resolve tile sizes list
  if args.tile_sizes:
    tile_sizes = [int(s) for s in args.tile_sizes.split(',') if s.strip()]
  else:
    tile_sizes = [int(args.tile_size)]

  # Accumulate metrics per tile size
  metrics = {
    "repo": args.repo,
    "shard_path": str(shard_path),
    "tensor_key": key,
    "shape": list(map(int, w.shape)),
    "axis": int(axis),
    "qtype": str(args.qtype),
    "calib": str(args.calib),
    "by_tile_size": {},
  }

  for tsize in tile_sizes:
    print(f"\n=== Evaluating tile size {tsize} ===")
    
    # Run with both absmax and MSE calibration for comparison
    results = {}
    for calib_method in ['absmax', str(args.calib)]:
      print(f"  Calibration method: {calib_method}")
      how = qarray.HowToQuantize(
        qtype=_parse_qtype(args.qtype),
        channelwise_axes=ch_axes,
        tiled_axes={axis: int(tsize)},
        calibration_method=calib_method,
      )

      q = qarray.quantize(w, how)
      x_dq = qarray.dequantize(q)

      # If dequantized array is padded due to tiling, truncate to original dims for error computation.
      def _trunc(a, target_shape):
        if a.shape == target_shape:
          return a
        slices = tuple(slice(0, s) for s in target_shape)
        return a[slices]

      x_dq = _trunc(x_dq, w.shape)

      # Overall MSE vs original weights (compute in float32 for stability).
      mse_overall = float(jnp.square(jnp.asarray(w, jnp.float32) - jnp.asarray(x_dq, jnp.float32)).mean())

      # Unique counts
      qvals_np = np.array(jnp.asarray(q.qvalue, dtype=jnp.float32))
      unique_counts = _compute_unique_counts(qvals_np, axis=axis, tile_size=int(tsize))

      print(f"    Subchannels: {unique_counts.size}")
      print(f"    Unique qvalues per subchannel: mean={unique_counts.mean():.2f}, min={unique_counts.min()}, max={unique_counts.max()}")
      print(f"    MSE: {mse_overall:.3e}")

      # MSE multipliers (if available)
      mult = getattr(q, 'mse_multiplier', None)
      mult_1d = None
      if mult is not None:
        mult_arr = jnp.asarray(mult, dtype=jnp.float32)
        mult_arr = jnp.moveaxis(mult_arr, axis, -1)
        mult_1d = np.asarray(mult_arr, dtype=np.float32).reshape(-1)
        mult_path = outdir / f"{args.save_prefix}_tile_{tsize}_{calib_method}_mse_multipliers.npy"
        np.save(mult_path, mult_1d)

      results[calib_method] = {
        'mse': mse_overall,
        'unique_counts': unique_counts,
        'multipliers': mult_1d,
      }

    # Joint plot: unique histogram (left) and multipliers histogram (right), comparing absmax vs MSE
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: Unique counts comparison with side-by-side bars
    max_unique = max(results['absmax']['unique_counts'].max(), results[str(args.calib)]['unique_counts'].max())
    bins = np.arange(0, int(max_unique) + 2)
    
    # Compute histograms manually to place bars side-by-side
    hist_absmax, _ = np.histogram(results['absmax']['unique_counts'], bins=bins)
    hist_mse, _ = np.histogram(results[str(args.calib)]['unique_counts'], bins=bins)
    
    # Bar width and positions
    width = 0.35
    x = bins[:-1]
    # Define consistent colors
    color_absmax = '#1f77b4'  # Blue
    color_mse = '#ff7f0e'     # Orange
    axes[0].bar(x - width/2, hist_absmax, width, label='Absmax', color=color_absmax, edgecolor='black', alpha=0.8)
    axes[0].bar(x + width/2, hist_mse, width, label='MSE', color=color_mse, edgecolor='black', alpha=0.8)
    axes[0].set_title("Unique Quantized Values Per Subchannel")
    axes[0].set_xlabel("Unique Values in Subchannel")
    axes[0].set_ylabel("Count of Subchannels (Log)")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # Right panel: Multipliers comparison (only MSE has multipliers)
    if results[str(args.calib)]['multipliers'] is not None:
      mult_mse = results[str(args.calib)]['multipliers']
      axes[1].hist(mult_mse, bins=100, alpha=0.8, label='MSE', color=color_mse, edgecolor='black')
      axes[1].set_title("MSE-Chosen Multipliers Per Subchannel-Tile")
      axes[1].set_xlabel("Multiplier (Relative to Absmax)")
      axes[1].set_ylabel("Count (Log)")
      axes[1].set_yscale("log")
      axes[1].legend()
      axes[1].grid(True, linestyle='--', alpha=0.3)
    else:
      axes[1].text(0.5, 0.5, 'No Multipliers Available', ha='center', va='center')
      axes[1].axis('off')
    
    fig.suptitle(f"Tensor: {key}   |   Tile Size: {tsize}   |   Absmax MSE: {results['absmax']['mse']:.3e}   |   MSE Calib MSE: {results[str(args.calib)]['mse']:.3e}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    joint_path = outdir / f"{args.save_prefix}_tile_{tsize}_joint_hist.png"
    fig.savefig(joint_path)
    plt.close(fig)
    print(f"Saved joint histogram: {joint_path}")

    # Save metrics per tile size
    entry = {
      "tile_size": int(tsize),
      "absmax": {
        "mse_overall": results['absmax']['mse'],
        "unique_counts": {
          "num_subchannels": int(results['absmax']['unique_counts'].size),
          "mean": float(results['absmax']['unique_counts'].mean()),
          "min": int(results['absmax']['unique_counts'].min()),
          "max": int(results['absmax']['unique_counts'].max()),
        },
      },
      "mse_calib": {
        "mse_overall": results[str(args.calib)]['mse'],
        "unique_counts": {
          "num_subchannels": int(results[str(args.calib)]['unique_counts'].size),
          "mean": float(results[str(args.calib)]['unique_counts'].mean()),
          "min": int(results[str(args.calib)]['unique_counts'].min()),
          "max": int(results[str(args.calib)]['unique_counts'].max()),
        },
      },
      "joint_hist_path": str(joint_path),
    }
    
    if results[str(args.calib)]['multipliers'] is not None:
      mult_mse = results[str(args.calib)]['multipliers']
      entry["mse_calib"]["mse_multipliers_path"] = str(outdir / f"{args.save_prefix}_tile_{tsize}_{args.calib}_mse_multipliers.npy")
      entry["mse_calib"]["mse_multipliers_stats"] = {
        "count": int(mult_mse.size),
        "mean": float(mult_mse.mean()),
        "min": float(mult_mse.min()),
        "max": float(mult_mse.max()),
        "std": float(mult_mse.std()),
      }
    
    metrics["by_tile_size"][str(tsize)] = entry

  # After loop: plot MSE across tile sizes for both calibration methods
  tile_sizes_sorted = sorted(metrics["by_tile_size"].keys(), key=int)
  mse_absmax = [metrics["by_tile_size"][ts]["absmax"]["mse_overall"] for ts in tile_sizes_sorted]
  mse_calib = [metrics["by_tile_size"][ts]["mse_calib"]["mse_overall"] for ts in tile_sizes_sorted]
  
  fig_mse, ax_mse = plt.subplots(figsize=(8, 5))
  ax_mse.plot([int(t) for t in tile_sizes_sorted], mse_absmax, marker='o', linewidth=2, markersize=8, label='Absmax')
  ax_mse.plot([int(t) for t in tile_sizes_sorted], mse_calib, marker='s', linewidth=2, markersize=8, label='MSE Calib')
  ax_mse.set_title(f"MSE vs Tile Size\nTensor: {key}")
  ax_mse.set_xlabel("Tile Size")
  ax_mse.set_ylabel("Overall MSE")
  ax_mse.legend()
  ax_mse.grid(True, linestyle='--', alpha=0.3)
  fig_mse.tight_layout()
  mse_plot_path = outdir / f"{args.save_prefix}_mse_vs_tile_size.png"
  fig_mse.savefig(mse_plot_path)
  plt.close(fig_mse)
  print(f"Saved MSE vs tile size plot: {mse_plot_path}")

  # Save metrics to JSON
  with open(outdir / f"{args.save_prefix}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
  print(f"Saved metrics JSON: {outdir / f'{args.save_prefix}_metrics.json'}")


if __name__ == "__main__":
  main()
