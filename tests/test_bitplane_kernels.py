"""Real Metal parity for the vendored FP16/BF16 and affine q4/q8 kernels."""
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from omlx.custom_kernels.bitplane import runtime as bp
from omlx.custom_kernels.bitplane.codec import pack_affine_codes, pack_float_words


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("bits", [None, 4, 8])
def test_metal_linear_and_native_reconstruction(dtype_name, bits):
    mx.random.seed(93)
    dtype = bp.DTYPES[dtype_name]
    original = nn.Linear(64, 32, bias=False)
    original.set_dtype(dtype)
    if bits is None:
        arrays = pack_float_words(np.array(original.weight.view(mx.uint16)), dtype_name)
        kind, width, group = dtype_name, arrays["bp_planes"].shape[1], 0
    else:
        original = nn.QuantizedLinear.from_linear(original, group_size=32, bits=bits)
        arrays = pack_affine_codes(np.array(original.weight), bits, 64)
        arrays.update(bp_scale=original.scales, bp_offset=original.biases)
        kind, width, group = "affine", bits, 32
    layer = bp.BitplaneLinear(64, 32, width, kind, activation_dtype=dtype_name, group_size=group)
    layer.load_weights([(k, mx.array(v)) for k, v in arrays.items()])
    for rows in (1, 8):
        x = mx.random.normal((rows, 64)).astype(dtype)
        actual, expected = layer(x), original(x)
        mx.eval(actual, expected)
        np.testing.assert_allclose(np.array(actual.astype(mx.float32)),
                                   np.array(expected.astype(mx.float32)), atol=.025, rtol=.025)
    restored = layer._source_floats() if bits is None else layer._source_codes()
    expected = original.weight
    if bits is None:
        restored, expected = restored.view(mx.uint16), expected.view(mx.uint16)
    np.testing.assert_array_equal(np.array(restored), np.array(expected))


def test_installed_native_loader_preserves_manifest_and_generates(tmp_path):
    """Requires staged dependencies in a fresh Python process (see README)."""
    from mlx_lm.models.qwen3 import Model, ModelArgs
    from mlx_lm.utils import load_model
    import inspect

    if "omlx bitplane loader hook" not in inspect.getsource(load_model):
        pytest.skip("Run in the staged MLX environment to exercise installed hooks")
    cfg = dict(model_type="qwen3", hidden_size=32, num_hidden_layers=1,
               intermediate_size=64, num_attention_heads=2, num_key_value_heads=1,
               head_dim=16, rms_norm_eps=1e-6, vocab_size=64, rope_theta=10000.,
               max_position_embeddings=128, tie_word_embeddings=True)
    source = Model(ModelArgs.from_dict(cfg))
    source.set_dtype(mx.bfloat16)
    weights = dict(tree_flatten(source.parameters()))
    projections = {}
    prefix = "model.layers.0.mlp"
    for name in ("gate_proj", "up_proj", "down_proj"):
        weight = weights.pop(f"{prefix}.{name}.weight")
        arrays = pack_float_words(np.array(weight.view(mx.uint16)), "bfloat16")
        projections[name] = dict(in_features=weight.shape[1], out_features=weight.shape[0],
                                 weight_planes=arrays["bp_planes"].shape[1], source_kind="bfloat16")
        weights.update({f"{prefix}.{name}.{k}": mx.array(v) for k, v in arrays.items()})
    weights[f"{prefix}.silu_table"] = bp.make_silu_table(mx.bfloat16)
    cfg["bitffn_bitplane"] = dict(format=bp.FORMAT, activation_dtype="bfloat16",
        layers={"layers.0.mlp": {"projections": projections}},
        prefill_backend="native", native_prefill_min_rows=8)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights, metadata={"format": "mlx"})
    loaded, _ = load_model(tmp_path)
    assert isinstance(loaded.layers[0].mlp, bp.BitplaneMLP)
    assert loaded.layers[0].mlp.native_prefill_min_rows == 8
    for tokens in ([[1]], [[1, 2, 3, 4, 5, 6, 7, 8]]):
        actual, expected = loaded(mx.array(tokens)), source(mx.array(tokens))
        mx.eval(actual, expected)
        np.testing.assert_allclose(np.array(actual.astype(mx.float32)),
                                   np.array(expected.astype(mx.float32)), atol=.03, rtol=.03)
