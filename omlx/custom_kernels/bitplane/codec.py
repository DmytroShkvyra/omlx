"""Lossless numeric bit planes. No fitting, calibration, or model training.

Each 32-element tile is sign/magnitude fixed point with a shared power of two.
The shared exponent is chosen without discarding ANY nonzero source bit.
FP16 needs at most 40 magnitude planes; BF16 needs at most 261.
This is a representation change, not one bit per original parameter.
"""
import numpy as np


LAYOUTS = {"float16": (10, 5, 15, 40), "bfloat16": (7, 8, 127, 261)}


def _pack_masks(bits):
    return np.sum(bits.astype(np.uint32) * (np.uint32(1) << np.arange(32, dtype=np.uint32)),
                  axis=-1, dtype=np.uint32)


def pack_float_words(raw, dtype):
    """Pack a [out,in] UINT16 array of IEEE words; preserve signed zero too."""
    if dtype not in LAYOUTS:
        raise ValueError("Only IEEE float16 and bfloat16 sources are supported")
    raw = np.asarray(raw)
    if raw.dtype != np.uint16 or raw.ndim != 2 or min(raw.shape) < 1:
        raise ValueError("Expected nonempty [out,in] uint16 IEEE words")
    fraction, exponent_bits, bias, _ = LAYOUTS[dtype]
    n, k = raw.shape
    a = np.pad(raw, ((0, 0), (0, (-k) % 32))).reshape(n, -1, 32)
    e = (a >> fraction) & ((1 << exponent_bits) - 1)
    if np.any(e == (1 << exponent_bits) - 1):
        raise ValueError("Nonfinite source weights are unsupported")
    m = (a & ((1 << fraction) - 1)).astype(np.uint32)
    m |= np.where(e > 0, 1 << fraction, 0).astype(np.uint32)
    low = np.where(e > 0, e.astype(np.int32) - bias - fraction,
                   1 - bias - fraction)
    # m fits in 11 bits, so this float64 operation is exact.
    high = low + np.frexp(m.astype(np.float64))[1] - 1
    base = np.min(np.where(m != 0, low, 10000), axis=-1)
    top = np.max(np.where(m != 0, high, -10000), axis=-1)
    count = np.maximum(top - base + 1, 0).astype(np.int32)
    base = np.where(count > 0, base, 0).astype(np.int32)
    width = max(1, int(count.max()))
    planes = np.zeros((n, width, a.shape[1]), dtype=np.uint32)
    for p in range(width):
        offset = base[..., None] + p - low
        active = (offset >= 0) & (offset <= fraction) & (p < count[..., None])
        bits = (m >> np.clip(offset, 0, fraction).astype(np.uint32)) & 1
        planes[:, p, :] = _pack_masks(bits * active)
    return {"bp_planes": planes, "bp_sign": _pack_masks(a >> 15),
            "bp_exponent": base, "bp_count": count}


def unpack_float_words(arrays, in_features, dtype):
    """Independent reconstruction for export validation, not an inference path."""
    fraction, exponent_bits, bias, _ = LAYOUTS[dtype]
    planes = np.asarray(arrays["bp_planes"])
    n, width, words = planes.shape
    base = np.asarray(arrays["bp_exponent"])
    sign = np.asarray(arrays["bp_sign"])
    count = np.asarray(arrays["bp_count"])
    value = np.zeros((n, words, 32), dtype=np.float64)
    lanes = np.arange(32, dtype=np.uint32)
    for p in range(width):
        b = ((planes[:, p, :, None] >> lanes) & 1) * (p < count[..., None])
        value += np.ldexp(b.astype(np.float64), base[..., None] + p)
    value = np.copysign(value, np.where(((sign[..., None] >> lanes) & 1) != 0, -1., 1.))
    value = value.reshape(n, words * 32)[:, :in_features]
    if dtype == "float16":
        return value.astype(np.float16).view(np.uint16)
    # Every reconstructed value is exactly representable in BF16 (hence FP32).
    return (value.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def pack_affine_codes(packed, bits, in_features):
    """Reorder existing MLX integer codes; NEVER quantize/dequantize weights."""
    packed = np.asarray(packed)
    if bits not in (4, 8) or packed.dtype != np.uint32 or packed.ndim != 2:
        raise ValueError("Expected existing MLX 4/8-bit affine uint32 weights")
    if in_features <= 0 or packed.shape[1] * (32 // bits) != in_features:
        raise ValueError("Source code shape does not match in_features")
    shifts = np.arange(32 // bits, dtype=np.uint32) * bits
    codes = ((packed[..., None] >> shifts) & ((1 << bits) - 1)).astype(np.uint8)
    tiles = codes.reshape(len(packed), in_features // 32, 32)
    return {"bp_planes": np.stack([_pack_masks((tiles >> p) & 1) for p in range(bits)], axis=1),
            "bp_sign": np.zeros((1,), np.uint32), "bp_exponent": np.zeros((1,), np.int32),
            "bp_count": np.full((1,), bits, np.int32)}


def unpack_affine_codes(arrays, bits, in_features):
    planes = np.asarray(arrays["bp_planes"])
    if planes.shape[1] != bits:
        raise ValueError("Unexpected code plane count")
    codes = np.zeros((len(planes), planes.shape[2], 32), np.uint32)
    for p in range(bits):
        codes |= ((planes[:, p, :, None] >> np.arange(32, dtype=np.uint32)) & 1) << p
    codes = codes.reshape(len(planes), in_features)
    grouped = codes.reshape(len(planes), -1, 32 // bits)
    return np.sum(grouped << (np.arange(32 // bits, dtype=np.uint32) * bits), axis=-1, dtype=np.uint32)


