# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from qwix._src.core import qarray

jax.config.update('jax_threefry_partitionable', False)


class MSECalibrationTest(parameterized.TestCase):
    @parameterized.named_parameters(
        # Per-tensor simple
        dict(testcase_name='int8_per_tensor', qtype=jnp.int8, tiled_axes={}, channelwise_axes=[], shape=(128, 1024)),
        dict(testcase_name='f8_e4m3_per_tensor', qtype=jnp.float8_e4m3fn, tiled_axes={}, channelwise_axes=[], shape=(64, 256)),
        dict(testcase_name='f4_e2m1_per_tensor', qtype=jnp.float4_e2m1fn, tiled_axes={}, channelwise_axes=[], shape=(64, 256)),
        dict(testcase_name='f4_e2m1_per_tensor_big', qtype=jnp.float4_e2m1fn, tiled_axes={}, channelwise_axes=[], shape=(192, 1024)),
        # Subchannel along last axis
        dict(testcase_name='int8_subchannel_128', qtype=jnp.int8, tiled_axes={1: 128}, channelwise_axes=[0], shape=(64, 1024)),
        dict(testcase_name='int8_subchannel_32', qtype=jnp.int8, tiled_axes={1: 32}, channelwise_axes=[], shape=(64, 1024)),
        dict(testcase_name='int8_subchannel_256', qtype=jnp.int8, tiled_axes={1: 256}, channelwise_axes=[], shape=(64, 1024)),
        dict(testcase_name='f4_subchannel_64', qtype=jnp.float4_e2m1fn, tiled_axes={1: 64}, channelwise_axes=[], shape=(64, 1024)),
        dict(testcase_name='f4_subchannel_256', qtype=jnp.float4_e2m1fn, tiled_axes={1: 256}, channelwise_axes=[], shape=(64, 1024)),
        # Padding case: last axis not divisible by tile -> pad_to_tile kicks in
        dict(testcase_name='int8_pad_to_tile', qtype=jnp.int8, tiled_axes={1: 128}, channelwise_axes=[], shape=(31, 1000)),
        dict(testcase_name='f4_pad_to_tile', qtype=jnp.float4_e2m1fn, tiled_axes={1: 128}, channelwise_axes=[], shape=(45, 1005)),
        # Tile along first axis
        dict(testcase_name='int8_tile_axis0', qtype=jnp.int8, tiled_axes={0: 7}, channelwise_axes=[], shape=(55, 16)),
        dict(testcase_name='f4_tile_axis0_8', qtype=jnp.float4_e2m1fn, tiled_axes={0: 8}, channelwise_axes=[], shape=(60, 64)),
        # Channelwise only on last axis
        dict(testcase_name='f8_channelwise_last', qtype=jnp.float8_e4m3fn, tiled_axes={}, channelwise_axes=[1], shape=(17, 129)),
        # Fractional tile spec: 1.0 => shared along that axis (single tile)
        dict(testcase_name='int8_fractional_tile_shared', qtype=jnp.int8, tiled_axes={1: 1.0}, channelwise_axes=[], shape=(64, 256)),
        # Fractional tile spec: 0.5 => 2 tiles along that axis when divisible
        dict(testcase_name='int8_fractional_tile_half', qtype=jnp.int8, tiled_axes={1: 0.5}, channelwise_axes=[], shape=(64, 256)),
        # Odd sizes per-tensor
        dict(testcase_name='int8_per_tensor_odd', qtype=jnp.int8, tiled_axes={}, channelwise_axes=[], shape=(33, 101))
    )
    def test_mse_sweep_not_worse_than_absmax(self, qtype, tiled_axes, channelwise_axes, shape):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, shape, dtype=jnp.bfloat16)

        how_abs = qarray.HowToQuantize(
            qtype=qtype,
            channelwise_axes=channelwise_axes,
            tiled_axes=tiled_axes,
            calibration_method='absmax',
        )
        q_abs = qarray.quantize(x, how_abs)
        x_abs = qarray.dequantize(q_abs)

        how_mse = qarray.HowToQuantize(
            qtype=qtype,
            channelwise_axes=channelwise_axes,
            tiled_axes=tiled_axes,
            calibration_method='mse',  # default sweep
        )
        q_mse = qarray.quantize(x, how_mse)
        x_mse = qarray.dequantize(q_mse)

        # Dequantized arrays may be padded for tiling; truncate to original dims.
        def trunc(a, target_shape):
            if a.shape == target_shape:
                return a
            slices = tuple(slice(0, s) for s in target_shape)
            return a[slices]
        x_abs = trunc(x_abs, x.shape)
        x_mse = trunc(x_mse, x.shape)

        # Expect MSE calibration to be no worse (often strictly better) than absmax.
        mse_abs = jnp.square(x - x_abs).mean()
        mse_mse = jnp.square(x - x_mse).mean()
        self.assertLessEqual(mse_mse, mse_abs)

    def test_mse_sweep_custom_range(self):
        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (64, 256), dtype=jnp.bfloat16)

        # Narrow sweep range to encourage similar result to absmax.
        how_mse_narrow = qarray.HowToQuantize(
            qtype=jnp.int8,
            channelwise_axes=[0],
            tiled_axes={},
            calibration_method='mse,0.9,1.1,7',
        )
        q_mse = qarray.quantize(x, how_mse_narrow)
        # The chosen scale should be close to absmax scale on average.
        how_abs = qarray.HowToQuantize(
            qtype=jnp.int8,
            channelwise_axes=[0],
            tiled_axes={},
            calibration_method='absmax',
        )
        q_abs = qarray.quantize(x, how_abs)
        ratio = (q_mse.scale / q_abs.scale).astype(jnp.float32)
        self.assertTrue(jnp.all((ratio > 0.9) & (ratio < 1.1)))

    def test_mse_two_arg_form_includes_one_and_not_worse(self):
        key = jax.random.PRNGKey(2)
        x = jax.random.normal(key, (48, 320), dtype=jnp.bfloat16)
        # Two-arg form: start,steps (end fixed at 1.0)
        how_mse_two = qarray.HowToQuantize(
                qtype=jnp.int8,
                channelwise_axes=[],
                tiled_axes={1: 64},  # will pad if needed
                calibration_method='mse,0.8,9',
        )
        how_abs = qarray.HowToQuantize(
                qtype=jnp.int8,
                channelwise_axes=[],
                tiled_axes={1: 64},
                calibration_method='absmax',
        )
        q_mse = qarray.quantize(x, how_mse_two)
        q_abs = qarray.quantize(x, how_abs)
        x_mse = qarray.dequantize(q_mse)
        x_abs = qarray.dequantize(q_abs)
        # Truncate possible tiling padding.
        def trunc(a, target_shape):
            if a.shape == target_shape:
                return a
            slices = tuple(slice(0, s) for s in target_shape)
            return a[slices]
        x_mse = trunc(x_mse, x.shape)
        x_abs = trunc(x_abs, x.shape)
        mse_abs = jnp.square(x - x_abs).mean()
        mse_mse = jnp.square(x - x_mse).mean()
        self.assertLessEqual(mse_mse, mse_abs)

        def test_mse_large_matrix_per_tensor_fast(self):
            # Bigger matrix, keep sweep short for test runtime.
            key = jax.random.PRNGKey(5)
            x = jax.random.normal(key, (256, 2048), dtype=jnp.bfloat16)
            how_abs = qarray.HowToQuantize(
                qtype=jnp.int8,
                channelwise_axes=[],
                tiled_axes={},
                calibration_method='absmax',
            )
            how_mse = qarray.HowToQuantize(
                qtype=jnp.int8,
                channelwise_axes=[],
                tiled_axes={},
                calibration_method='mse,0.8,1.0,5',  # short sweep to keep test fast
            )
            q_abs = qarray.quantize(x, how_abs)
            q_mse = qarray.quantize(x, how_mse)
            xa = qarray.dequantize(q_abs)
            xm = qarray.dequantize(q_mse)
            mse_abs = jnp.square(x - xa).mean()
            mse_mse = jnp.square(x - xm).mean()
            self.assertLessEqual(mse_mse, mse_abs)

    def test_mse_exact_one_matches_absmax(self):
        key = jax.random.PRNGKey(3)
        x = jax.random.normal(key, (40, 300), dtype=jnp.bfloat16)
        # Force the sweep to only include 1.0
        how_mse_one = qarray.HowToQuantize(
            qtype=jnp.int8,
            channelwise_axes=[],
            tiled_axes={},
            calibration_method='mse,1.0,1.0,1',
        )
        how_abs = qarray.HowToQuantize(
            qtype=jnp.int8,
            channelwise_axes=[],
            tiled_axes={},
            calibration_method='absmax',
        )
        q_mse = qarray.quantize(x, how_mse_one)
        q_abs = qarray.quantize(x, how_abs)
        # Scales should match exactly; MSE equal after dequant.
        self.assertTrue(jnp.all(q_mse.scale == q_abs.scale))
        xm = qarray.dequantize(q_mse)
        xa = qarray.dequantize(q_abs)
        self.assertEqual(float(jnp.square(x - xm).mean()), float(jnp.square(x - xa).mean()))

    @parameterized.named_parameters(
        dict(testcase_name='fp8_subchannel_pad', qtype=jnp.float8_e4m3fn, tiled_axes={1: 64}, channelwise_axes=[], shape=(31, 1001)),
        dict(testcase_name='int8_two_axis_tiling_pad', qtype=jnp.int8, tiled_axes={0: 8, 1: 96}, channelwise_axes=[], shape=(63, 1000)),
    )
    def test_mse_not_worse_multi_axis_and_fp8(self, qtype, tiled_axes, channelwise_axes, shape):
        key = jax.random.PRNGKey(4)
        x = jax.random.normal(key, shape, dtype=jnp.bfloat16)
        how_abs = qarray.HowToQuantize(qtype=qtype, channelwise_axes=channelwise_axes, tiled_axes=tiled_axes, calibration_method='absmax')
        how_mse = qarray.HowToQuantize(qtype=qtype, channelwise_axes=channelwise_axes, tiled_axes=tiled_axes, calibration_method='mse,0.7,1.0,13')
        q_abs = qarray.quantize(x, how_abs)
        q_mse = qarray.quantize(x, how_mse)
        xa = qarray.dequantize(q_abs)
        xm = qarray.dequantize(q_mse)
        # Truncate for potential padding on either axis.
        def trunc(a, target):
            if a.shape == target:
                return a
            return a[tuple(slice(0, s) for s in target)]
        xa = trunc(xa, x.shape)
        xm = trunc(xm, x.shape)
        mse_abs = jnp.square(x - xa).mean()
        mse_mse = jnp.square(x - xm).mean()
        self.assertLessEqual(mse_mse, mse_abs + 1e-7)


if __name__ == '__main__':
  absltest.main()
