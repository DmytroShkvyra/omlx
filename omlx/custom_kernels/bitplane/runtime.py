"""Apple Metal exact-weight bit-plane FFNs, with optional lossy storage factors.

Weights retain all source information. Activations retain all FP16/BF16 bits.
The affine path uses XOR signs, masked popcount and bounded UINT32 partial
sums, followed by FP32 scaling and reduction. No activation bits are dropped.
Native prefill restores original IEEE words or affine codes losslessly;
single-token decode keeps the tiled XOR/popcount implementation for all sources.
FP32 reduction order is different from MLX: bitwise model parity is NOT promised.
Lossy storage variants are explicitly marked in config and expand once at load.
"""
import re
from pathlib import Path
import json
import logging

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

FORMAT = "bitffn.bitplane.lossless-weights.v1"
KERNEL_IMPLEMENTATION = "xor-hybrid-v7-unified"
DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16}
MAX_PLANES = {"float16": 40, "bfloat16": 261}
COMPACT_PLANES = 32
# Grouping two words is available for controlled experiments. On M4 Max
# it was slower than word32, so the production kernels keep one word.
AFFINE_GROUP_WORDS = 1


def _tile_shape(rows):
    # Bounded scalar partial sums permit a wider down/linear tile than v4.
    return (1, 4)

# A SIMD group packs 32 input elements together. Only the prefix `counts`
# of each output tile is read; unused output planes are deliberately unwritten.
_PACK_SOURCE = r'''
uint word = thread_position_in_grid.x / 32;
uint lane = thread_index_in_simdgroup;
uint row = thread_position_in_grid.y;
if (word >= WORDS || row >= ROWS) return;
uint col = word * 32 + lane;
ushort raw = col < K ? as_type<ushort>(x[row * K + col]) : ushort(0);
uint e = (uint(raw) >> FRAC) & EMASK;
uint m = (uint(raw) & ((1u << FRAC) - 1u)) | (e ? (1u << FRAC) : 0u);
bool nonfinite = e == EMASK;
int low = e ? int(e) - BIAS - FRAC : 1 - BIAS - FRAC;
int bottom = simd_min(m && !nonfinite ? low : 10000);
int top = simd_max(m && !nonfinite ? low + 31 - int(clz(m)) : -10000);
int count = max(top - bottom + 1, 0);
uint invalid = simd_max(uint(nonfinite));
uint sign = simd_sum(((uint(raw) >> 15) & 1u) << lane);
if (lane == 0) {
 signs[row * WORDS + word] = sign;
 exponents[row * WORDS + word] = count ? bottom : 0;
 counts[row * WORDS + word] = invalid ? -1 : count;
 // Preserve the v1 sequential FP32 sum exactly, but compute it ONCE per
 // input tile instead of repeating it for every output neuron.
 float sum = 0.0f;
 for (uint j = 0; j < 32 && word * 32 + j < K; ++j)
  sum += float(x[row * K + word * 32 + j]);
 sums[row * WORDS + word] = sum;
}
if (invalid) return;
constexpr int EXTRA_BITS = FRAC == 10 ? 40 - ABITS : 261 - ABITS;
for (int p = 0; p < count; ++p) {
 int offset = bottom + p - low;
 uint bit = (offset >= 0 && offset <= FRAC) ? ((m >> uint(offset)) & 1u) : 0u;
 uint mask = simd_sum(bit << lane);
 if (lane == 0) {
  if (p < ABITS) planes[(row * ABITS + p) * WORDS + word] = mask;
  else overflow[(row * EXTRA_BITS + p - ABITS) * WORDS + word] = mask;
 }
}
'''
_PACK = mx.fast.metal_kernel(
    name="bp_pack_float16_exact_v7_overflow", input_names=["x"],
    output_names=["planes", "signs", "exponents", "counts", "sums", "overflow"],
    source=_PACK_SOURCE, compile_options={"math_mode": "safe"})

# Each SIMD group owns an MT x NT output tile. The 8/4 source code planes
# stay in registers across MT tokens; input planes are reused across NT
# neurons. Each unsigned partial sum covers 27-WBITS activation planes:
# 32 * (2**WBITS-1) * (2**(27-WBITS)-1) < 2**32. Its sign correction is
# subtracted exactly ONCE in integers, before conversion to FP32. Wider
# activations use more partials; every bit is retained without INT64.
_AFFINE_DOT_SOURCE = r'''
uint n0 = (thread_position_in_grid.x / 32) * NT;
uint r0 = thread_position_in_grid.y * MT;
uint lane = thread_index_in_simdgroup;
if (n0 >= N || r0 >= ROWS) return;
constexpr int CHANNELS = /* CHANNELS */;
float accum[MT][CHANNELS];
#pragma unroll
for (int m = 0; m < MT; ++m)
 #pragma unroll
 for (int n = 0; n < CHANNELS; ++n) accum[m][n] = 0.0f;
constexpr int WORDS_PER_STEP = /* WORDS_PER_STEP */;
for (uint first_word = lane * WORDS_PER_STEP; first_word < WORDS; first_word += 32 * WORDS_PER_STEP) {
 float scales[CHANNELS], offsets[CHANNELS];
 uint group = (first_word * 32) / GROUP_SIZE;
 #pragma unroll
 for (int n = 0; n < CHANNELS; ++n) {
  uint neuron = n0 + n % NT;
  scales[n] = neuron < N ? float(/* SCALE */) : 0.0f;
  offsets[n] = neuron < N ? float(/* OFFSET */) : 0.0f;
 }
 float step_dot[MT][CHANNELS], step_sum[MT];
 #pragma unroll
 for (int m = 0; m < MT; ++m) {
  step_sum[m] = 0.0f;
  #pragma unroll
  for (int n = 0; n < CHANNELS; ++n) step_dot[m][n] = 0.0f;
 }
 // For GROUP_SIZE=64 both words share their original scale and offset.
 // Keep the exact per-word packing; combine only at the FP32 affine stage.
 for (int part_word = 0; part_word < WORDS_PER_STEP; ++part_word) {
  uint word = first_word + part_word;
  uint wb[CHANNELS][WBITS];
  #pragma unroll
  for (int n = 0; n < CHANNELS; ++n) {
   uint neuron = n0 + n % NT;
   #pragma unroll
   for (int b = 0; b < WBITS; ++b)
    wb[n][b] = neuron < N ? /* WEIGHT */ : 0u;
  }
  #pragma unroll
  for (int m = 0; m < MT; ++m) {
  uint row = r0 + m;
  if (row >= ROWS) continue;
  int count = xc[row * WORDS + word];
  if (count < 0) {
   #pragma unroll
   for (int n = 0; n < CHANNELS; ++n) step_dot[m][n] = NAN;
   continue;
  }
  if (count == 0) continue;
  uint negative = xs[row * WORDS + word];
  int exponent = xe[row * WORDS + word];
  float dot[CHANNELS];
  uint sign_correction[CHANNELS];
  #pragma unroll
  for (int n = 0; n < CHANNELS; ++n) {
   dot[n] = 0.0f;
   sign_correction[n] = 0;
   #pragma unroll
   for (int b = 0; b < WBITS; ++b)
    sign_correction[n] += popcount(wb[n][b] & negative) << b;
  }
  constexpr int PARTIAL_BITS = 27 - WBITS;
  for (int first = 0; first < count; first += PARTIAL_BITS) {
   uint partial[CHANNELS];
   #pragma unroll
   for (int n = 0; n < CHANNELS; ++n) partial[n] = 0;
   int width = min(PARTIAL_BITS, count - first);
   // Horner order replaces a variable shift/add with multiply-add by two.
   // The unsigned integer result is identical; only plane read order changes.
   for (int digit = width - 1; digit >= 0; --digit) {
    int a = first + digit;
    /* INPUT_PLANE */
    plane ^= negative;
    #pragma unroll
    for (int n = 0; n < CHANNELS; ++n) {
     uint coefficient = 0u;
     #pragma unroll
     for (int b = 0; b < WBITS; ++b)
      coefficient += popcount(plane & wb[n][b]) << b;
     partial[n] = partial[n] * 2u + coefficient;
    }
   }
   #pragma unroll
   for (int n = 0; n < CHANNELS; ++n) {
    uint correction = sign_correction[n] * ((1u << width) - 1u);
    // Neither operand nor the absolute difference exceeds UINT32_MAX.
    // Subtract BEFORE the float conversion to preserve cancellation bits.
    float value = partial[n] >= correction ? float(partial[n] - correction)
                                          : -float(correction - partial[n]);
    dot[n] += ldexp(value, exponent + first);
   }
  }
  step_sum[m] += sums[row * WORDS + word];
  #pragma unroll
  for (int n = 0; n < CHANNELS; ++n)
   step_dot[m][n] += dot[n];
  }
 }
 #pragma unroll
 for (int m = 0; m < MT; ++m) {
  if (r0 + m >= ROWS) continue;
  #pragma unroll
  for (int n = 0; n < CHANNELS; ++n)
   accum[m][n] += step_dot[m][n] * scales[n] + step_sum[m] * offsets[n];
 }
}
/* OUTPUT */
'''


_DENSE_DOT_SOURCE = r'''
uint n0 = (thread_position_in_grid.x / 32) * NT;
uint r0 = thread_position_in_grid.y * MT;
uint lane = thread_index_in_simdgroup;
if (n0 >= N || r0 >= ROWS) return;
constexpr int CHANNELS = /* CHANNELS */;
// Bound BOTH operands: 32*(2**8-1)*(2**19-1) < UINT32_MAX.
// Every weight/activation plane is visited, including wide exponent tiles.
constexpr int BLOCK_BITS = 8;
constexpr int PARTIAL_BITS = 27 - BLOCK_BITS;
float accum[MT][CHANNELS];
#pragma unroll
for (int m = 0; m < MT; ++m)
 #pragma unroll
 for (int n = 0; n < CHANNELS; ++n) accum[m][n] = 0.0f;
for (uint word = lane; word < WORDS; word += 32) {
 int counts[CHANNELS], bases[CHANNELS];
 uint signs[CHANNELS];
 int maximum = 0;
 #pragma unroll
 for (int n = 0; n < CHANNELS; ++n) {
  uint neuron = n0 + n % NT;
  counts[n] = neuron < N ? /* WCOUNT */ : 0;
  bases[n] = neuron < N ? /* WEXPONENT */ : 0;
  signs[n] = neuron < N ? /* WSIGN */ : 0u;
  maximum = max(maximum, counts[n]);
 }
 for (int first_b = 0; first_b < maximum; first_b += BLOCK_BITS) {
  uint wb[CHANNELS][BLOCK_BITS];
  #pragma unroll
  for (int n = 0; n < CHANNELS; ++n) {
   uint neuron = n0 + n % NT;
   #pragma unroll
   for (int b = 0; b < BLOCK_BITS; ++b)
    wb[n][b] = first_b + b < counts[n] ? /* WEIGHT */ : 0u;
  }
  #pragma unroll
  for (int m = 0; m < MT; ++m) {
   uint row = r0 + m;
   if (row >= ROWS) continue;
   int count = xc[row * WORDS + word];
   if (count < 0) {
    #pragma unroll
    for (int n = 0; n < CHANNELS; ++n) accum[m][n] = NAN;
    continue;
   }
   if (count == 0) continue;
   int exponent = xe[row * WORDS + word];
   uint negative[CHANNELS], correction[CHANNELS];
   #pragma unroll
   for (int n = 0; n < CHANNELS; ++n) {
    negative[n] = xs[row * WORDS + word] ^ signs[n];
    correction[n] = 0u;
    #pragma unroll
    for (int b = 0; b < BLOCK_BITS; ++b)
     correction[n] += popcount(wb[n][b] & negative[n]) << b;
   }
   for (int first = 0; first < count; first += PARTIAL_BITS) {
    uint partial[CHANNELS];
    #pragma unroll
    for (int n = 0; n < CHANNELS; ++n) partial[n] = 0u;
    int width = min(PARTIAL_BITS, count - first);
    for (int digit = width - 1; digit >= 0; --digit) {
     int a = first + digit;
     /* INPUT_PLANE */
     #pragma unroll
     for (int n = 0; n < CHANNELS; ++n) {
      uint coefficient = 0u;
      uint signed_plane = plane ^ negative[n];
      #pragma unroll
      for (int b = 0; b < BLOCK_BITS; ++b)
       coefficient += popcount(signed_plane & wb[n][b]) << b;
      partial[n] = partial[n] * 2u + coefficient;
     }
    }
    #pragma unroll
    for (int n = 0; n < CHANNELS; ++n) {
     uint subtract = correction[n] * ((1u << width) - 1u);
     float value = partial[n] >= subtract ? float(partial[n] - subtract)
                                         : -float(subtract - partial[n]);
     accum[m][n] += ldexp(value, exponent + bases[n] + first_b + first);
    }
   }
  }
 }
 // Nonfinite activations must propagate even for an all-zero weight tile.
 #pragma unroll
 for (int m = 0; m < MT; ++m)
  if (r0 + m < ROWS && xc[(r0 + m) * WORDS + word] < 0)
   #pragma unroll
   for (int n = 0; n < CHANNELS; ++n) accum[m][n] = NAN;
}
/* OUTPUT */
'''


def _affine_dot_kernel(fused, compact=False, pack_output=False, pair_scale=(AFFINE_GROUP_WORDS == 2), dense=False):
    """Shared tiling, exact packing and epilogues for IEEE and affine weights."""
    def load(name, index):
        plain = f"{name}[{index}]"
        return f"(n < NT ? {plain} : up_{plain})" if fused else plain

    epilogue = r'''
#pragma unroll
for (int m = 0; m < MT; ++m) {
 #pragma unroll
 for (int n = 0; n < NT; ++n) {
  float value = simd_sum(accum[m][n]);
  if (lane == 0 && r0 + m < ROWS && n0 + n < N)
   out[(r0 + m) * N + n0 + n] = T(value);
 }
}
'''
    if fused:
        epilogue = r'''
#pragma unroll
for (int m = 0; m < MT; ++m) {
 #pragma unroll
 for (int n = 0; n < NT; ++n) {
  float gate_value = simd_sum(accum[m][n]);
  float up_value = simd_sum(accum[m][NT + n]);
  if (lane == 0 && r0 + m < ROWS && n0 + n < N) {
   // Keep BOTH projection roundings and the final product rounding from v2.
   T gate = T(gate_value), up = T(up_value);
   uint code = uint(as_type<ushort>(gate));
   out[(r0 + m) * N + n0 + n] = T(float(table[code]) * float(up));
  }
 }
}
'''
    if pack_output:
        # One threadgroup owns 32 adjacent output neurons. SIMD groups do
        # the dot products; their rounded SwiGLU values meet in shared memory.
        # The first SIMD group then runs the SAME exact packer as _PACK.
        epilogue = r'''
threadgroup T hidden[32];
#pragma unroll
for (int n = 0; n < NT; ++n) {
 float gate_value = simd_sum(accum[0][n]);
 float up_value = simd_sum(accum[0][NT + n]);
 if (lane == 0) {
  T gate = T(gate_value), up = T(up_value);
  uint code = uint(as_type<ushort>(gate));
  T value = n0 + n < N ? T(float(table[code]) * float(up)) : T(0);
  hidden[(thread_index_in_threadgroup / 32) * NT + n] = value;
  // Keep raw words for rare >32-plane reconstruction in the down kernel.
  if (n0 + n < N) out[r0 * N + n0 + n] = value;
 }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (thread_index_in_threadgroup < 32) {
'''
        pack_source = (_PACK_SOURCE
            .replace("uint word = thread_position_in_grid.x / 32;", "uint word = threadgroup_position_in_grid.x;")
            .replace("uint row = thread_position_in_grid.y;", "uint row = r0;")
            .replace("x[row * K + col]", "hidden[lane]")
            .replace("x[row * K + word * 32 + j]", "hidden[j]"))
        for before, after in (("WORDS", "OUT_WORDS"), ("ABITS", "OUT_ABITS"), ("K", "N"), ("sums", "packed_sums")):
            pack_source = re.sub(r"\b" + before + r"\b", after, pack_source)
        epilogue += pack_source + "\n}\n"
    input_plane = "uint plane = xp[(row * ABITS + a) * WORDS + word];"
    if compact:
        input_plane = r'''
   // Compact hot prefix plus overflow masks packed ONCE per input word.
   // Wide BF16 SiLU tails must never reconstruct 32 IEEE values per neuron.
   constexpr int EXTRA_BITS = FRAC == 10 ? 40 - ABITS : 261 - ABITS;
   uint plane = a < ABITS ? xp[(row * ABITS + a) * WORDS + word]
                         : xo[(row * EXTRA_BITS + a - ABITS) * WORDS + word];
'''
    weight_width = "(n < NT ? WBITS : UP_WBITS)" if fused and dense else "WBITS"
    weight_bit = "first_b + b" if dense else "b"
    source = ((_DENSE_DOT_SOURCE if dense else _AFFINE_DOT_SOURCE).replace("/* CHANNELS */", "2 * NT" if fused else "NT")
              .replace("/* WORDS_PER_STEP */", "(GROUP_SIZE == 64 ? 2 : 1)" if pair_scale else "1")
              .replace("/* WEIGHT */", load("wp", f"(neuron * {weight_width} + {weight_bit}) * WORDS + word"))
              .replace("/* WCOUNT */", load("wc", "neuron * WORDS + word"))
              .replace("/* WEXPONENT */", load("we", "neuron * WORDS + word"))
              .replace("/* WSIGN */", load("ws", "neuron * WORDS + word"))
              .replace("/* SCALE */", load("scale", "neuron * GROUPS + group"))
              .replace("/* OFFSET */", load("offset", "neuron * GROUPS + group"))
              .replace("/* INPUT_PLANE */", input_plane)
              .replace("/* OUTPUT */", epilogue))
    if pack_output:
        # Even a partial final threadgroup must reach the barrier together.
        source = source.replace("if (n0 >= N || r0 >= ROWS) return;", "if (r0 >= ROWS) return;")
    metadata = ["ws", "we", "wc"] if dense else ["scale", "offset"]
    inputs = ["xp", "xs", "xe", "xc", "sums", "xo", "wp", *metadata]
    if fused:
        inputs += ["up_wp", *("up_" + name for name in metadata), "table"]
    if compact:
        inputs += ["x"]
    name = ("bp_dense_gate_up_v7" if fused else "bp_dense_dot_v7") if dense else (
        "bp_affine_gate_up_swiglu_v6" if fused else "bp_affine_popcount_xor_v6")
    name += ("_compact" if compact else "") + ("_packed" if pack_output else "") + "_overflow"
    if not dense:
        name += "_pair64" if pair_scale else "_word32"
    return mx.fast.metal_kernel(
        name=name,
        input_names=inputs, output_names=(["out", "planes", "signs", "exponents", "counts", "packed_sums", "overflow"] if pack_output else ["out"]), source=source,
        compile_options={"math_mode": "safe"})


_DOT_TILED = _affine_dot_kernel(False)
_DOT_COMPACT = _affine_dot_kernel(False, compact=True)
_GATE_UP = _affine_dot_kernel(True, compact=True)
_GATE_UP_PACKED = _affine_dot_kernel(True, compact=True, pack_output=True)
_DENSE_TILED = _affine_dot_kernel(False, dense=True)
_DENSE_COMPACT = _affine_dot_kernel(False, compact=True, dense=True)
_DENSE_GATE_UP = _affine_dot_kernel(True, compact=True, dense=True)
_DENSE_GATE_UP_PACKED = _affine_dot_kernel(True, compact=True, pack_output=True, dense=True)

# Transpose bit planes back into the ORIGINAL packed integer source codes.
# This is a lossless bit permutation, without floating-point weight conversion.
_SOURCE_CODES = mx.fast.metal_kernel(
    name="bp_restore_affine_source_codes_v4", input_names=["planes"],
    output_names=["codes"], source=r'''
uint i = thread_position_in_grid.x;
if (i >= N * PACKED_COLUMNS) return;
uint neuron = i / PACKED_COLUMNS;
uint column = (i % PACKED_COLUMNS) * VALUES;
uint word = column / 32, first = column % 32;
uint packed = 0u;
#pragma unroll
for (uint b = 0; b < WBITS; ++b) {
 uint plane = planes[(neuron * WBITS + b) * WORDS + word] >> first;
 if (WBITS == 8) {
  // Spread four bits to byte positions 0,8,16,24 with one multiply.
  plane = ((plane & 15u) * 0x00204081u) & 0x01010101u;
 } else {
  // Spread eight bits to nibble positions 0,4,...,28 (SWAR).
  plane &= 255u;
  plane = (plane | (plane << 12)) & 0x000f000fu;
  plane = (plane | (plane << 6)) & 0x03030303u;
  plane = (plane | (plane << 3)) & 0x11111111u;
 }
 packed |= plane << b;
}
codes[i] = packed;
''', compile_options={"math_mode": "safe"})

# Reconstruct IEEE WORDS, not a floating sum of planes: subnormals, signed
# zero and the full BF16 exponent span therefore round-trip bit for bit.
_SOURCE_FLOATS = mx.fast.metal_kernel(
    name="bp_restore_ieee_words_v7", input_names=["wp", "ws", "we", "wc"],
    output_names=["weight"], source=r'''
uint i = thread_position_in_grid.x;
if (i >= N * K) return;
uint neuron = i / K, col = i % K;
uint word = col / 32, lane = col % 32;
uint meta = neuron * WORDS + word;
int count = wc[meta], base = we[meta];
uint raw = ((ws[meta] >> lane) & 1u) << 15;
if (!IEEE_WORDS) {
 // Lossy plane predictors can produce mantissas wider than the original
 // IEEE format. Keep their numeric values in FP32, never truncate to BF16.
 float value = 0.0f;
 for (int p = count - 1; p >= 0; --p) {
  uint bit = (wp[(neuron * WBITS + p) * WORDS + word] >> lane) & 1u;
  value += ldexp(float(bit), base + p);
 }
 weight[i] = W(raw ? -value : value);
 return;
}
int top = -1;
for (int p = count - 1; p >= 0; --p) {
 if ((wp[(neuron * WBITS + p) * WORDS + word] >> lane) & 1u) {
  top = p;
  break;
 }
}
if (top >= 0) {
 int e = base + top;
 bool normal = e >= 1 - WBIAS;
 if (normal) raw |= uint(e + WBIAS) << WFRAC;
 int low = normal ? e - WFRAC : 1 - WBIAS - WFRAC;
 for (int b = 0; b < WFRAC; ++b) {
  int p = low + b - base;
  if (p >= 0 && p < count)
   raw |= ((wp[(neuron * WBITS + p) * WORDS + word] >> lane) & 1u) << b;
 }
}
weight[i] = W(as_type<SOURCE_T>(ushort(raw)));
''', compile_options={"math_mode": "safe"})


def _gate_up_tile_shape(rows):
    # Each output neuron keeps two projections in registers.
    return (1, 1)


def _packed_gate_up_tile_shape(rows):
    # Sixteen SIMD groups, two neurons each: one exact 32-element pack word.
    return (1, 2)


def _prefer_fused_down_pack(rows):
    # On the measured M4 Max decode path, the separate pack plus smaller
    # gate/up threadgroups beats collective packing. Preserve the existing
    # multi-row policy, for which this ablation has not established a win.
    return rows > 1


def _can_fuse_gate_up(gate, up):
    same_shape = all(getattr(gate, key) == getattr(up, key) for key in
                     ("in_features", "out_features", "activation_dtype"))
    if gate.source_kind == "affine" or up.source_kind == "affine":
        return (same_shape and gate.source_kind == up.source_kind == "affine" and
                gate.weight_planes == up.weight_planes and gate.group_size == up.group_size)
    return same_shape


def _can_pack_gate_up(gate, up, down):
    return (_can_fuse_gate_up(gate, up) and
            gate.out_features == down.in_features and
            gate.activation_dtype == down.activation_dtype)

_DOT = mx.fast.metal_kernel(
    name="bp_xor_popcount_dot", input_names=["xp", "xs", "xe", "xc", "wp", "ws", "we", "wc", "scale", "offset", "x"],
    output_names=["out"], source=r'''
uint neuron = thread_position_in_grid.x / 32;
uint lane = thread_index_in_simdgroup;
uint row = thread_position_in_grid.y;
if (neuron >= N || row >= ROWS) return;
float total = 0.0f;
for (uint word = lane; word < WORDS; word += 32) {
 int a_count = xc[row * WORDS + word];
 int w_count = AFFINE ? WBITS : wc[neuron * WORDS + word];
 if (a_count < 0) { total = NAN; continue; }
 if (a_count == 0) continue;
 uint negative = xs[row * WORDS + word] ^ (AFFINE ? 0u : ws[neuron * WORDS + word]);
 int exponent = xe[row * WORDS + word] + (AFFINE ? 0 : we[neuron * WORDS + word]);
 float dot = 0.0f;
 // Sum each power-of-two diagonal in integers before FP32 accumulation.
 for (int d = 0; d < a_count + w_count - 1; ++d) {
  int coefficient = 0;
  for (int a = max(0, d - w_count + 1); a <= min(a_count - 1, d); ++a) {
   uint active = xp[(row * ABITS + a) * WORDS + word] & wp[(neuron * WBITS + d - a) * WORDS + word];
   coefficient += int(popcount(active)) - 2 * int(popcount(active & negative));
  }
  dot += ldexp(float(coefficient), exponent + d);
 }
 if (AFFINE) {
  uint group = (word * 32) / GROUP_SIZE;
  float sum_x = 0.0f;
  for (uint j = 0; j < 32 && word * 32 + j < K; ++j)
   sum_x += float(x[row * K + word * 32 + j]);
  dot = dot * float(scale[neuron * GROUPS + group]) + sum_x * float(offset[neuron * GROUPS + group]);
 }
 total += dot;
}
float result = simd_sum(total);
if (lane == 0) out[row * N + neuron] = T(result);
''', compile_options={"math_mode": "safe"})

_SWIGLU = mx.fast.metal_kernel(
    name="bp_swiglu_lut", input_names=["gate", "up", "table"], output_names=["out"], source=r'''
uint i = thread_position_in_grid.x;
if (i >= SIZE) return;
uint code = uint(as_type<ushort>(gate[i]));
out[i] = T(float(table[code]) * float(up[i]));
''', compile_options={"math_mode": "safe"})


def make_silu_table(dtype):
    """Full truth table of the installed MLX unary SiLU, including special words."""
    if dtype not in DTYPES.values():
        raise TypeError("SiLU LUT requires FP16 or BF16")
    values = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16).view(dtype)
    table = nn.silu(values)
    mx.eval(table)
    return table


def lut_swiglu(gate, up, table):
    if gate.shape != up.shape or gate.dtype != up.dtype or gate.dtype != table.dtype:
        raise ValueError("SwiGLU ports and LUT must have the same shape/dtype")
    if table.shape != (65536,) or gate.dtype not in DTYPES.values():
        raise ValueError("Expected a complete 16-bit SiLU truth table")
    if gate.size == 0:
        raise ValueError("Empty activation")
    return _SWIGLU(inputs=[mx.contiguous(gate), mx.contiguous(up), table],
                   template=[("SIZE", gate.size), ("T", gate.dtype)],
                   grid=(gate.size, 1, 1), threadgroup=(256, 1, 1),
                   output_shapes=[gate.shape], output_dtypes=[gate.dtype])[0]


class BitplaneLinear(nn.Module):
    def __init__(self, in_features, out_features, weight_planes, source_kind,
                 activation_dtype="float16", chunk_rows=32, group_size=0):
        super().__init__()
        if activation_dtype not in DTYPES or source_kind not in (*DTYPES, "affine"):
            raise ValueError("Unsupported bit-plane precision")
        if any(type(v) is not int for v in (in_features, out_features, weight_planes, chunk_rows)):
            raise ValueError("Dimensions must be integers")
        if min(in_features, out_features, weight_planes, chunk_rows) < 1:
            raise ValueError("Invalid dimensions")
        if source_kind == "affine" and (weight_planes not in (4, 8) or type(group_size) is not int or group_size < 32 or group_size % 32 or in_features % group_size):
            raise ValueError("Existing affine codes require 4/8 planes and groups divisible by 32")
        if source_kind != "affine" and weight_planes > MAX_PLANES[source_kind]:
            raise ValueError("Too many weight planes for the source dtype")
        self.in_features, self.out_features = in_features, out_features
        self.weight_planes, self.source_kind = weight_planes, source_kind
        self.activation_dtype, self.chunk_rows = activation_dtype, chunk_rows
        self.group_size = group_size
        self._native_cache = None
        self._native_numeric = False
        words = (in_features + 31) // 32
        self.bp_planes = mx.zeros((out_features, weight_planes, words), mx.uint32)
        meta_shape = (1,) if source_kind == "affine" else (out_features, words)
        self.bp_sign = mx.zeros(meta_shape, mx.uint32)
        self.bp_exponent = mx.zeros(meta_shape, mx.int32)
        self.bp_count = mx.full(meta_shape, weight_planes if source_kind == "affine" else 0, mx.int32)
        if source_kind == "affine":
            self.bp_scale = mx.ones((out_features, in_features // group_size), DTYPES[activation_dtype])
            self.bp_offset = mx.zeros((out_features, in_features // group_size), DTYPES[activation_dtype])

    def _check_input(self, x):
        if x.ndim < 1 or x.shape[-1] != self.in_features or x.size == 0:
            raise ValueError("Invalid bit-plane projection input shape")
        dtype = DTYPES[self.activation_dtype]
        if x.dtype != dtype:
            raise TypeError(f"Expected {self.activation_dtype} activations; implicit lossy casts are disabled")
        return dtype

    def _source_codes(self, cache=True):
        """Restore source codes, optionally caching outside model parameters.

        Keep the source array identity so load_weights()/parameter replacement
        cannot accidentally reuse codes from an earlier weight matrix.
        """
        if self.source_kind != "affine":
            raise ValueError("Native prefill requires existing affine source codes")
        if not cache:
            self._native_cache = None
        if self._native_cache is None or self._native_cache[0] is not self.bp_planes:
            values = 32 // self.weight_planes
            columns = self.in_features // values
            codes = _SOURCE_CODES(inputs=[self.bp_planes],
                template=[("N", self.out_features), ("PACKED_COLUMNS", columns),
                          ("VALUES", values), ("WBITS", self.weight_planes),
                          ("WORDS", self.in_features // 32)],
                grid=(self.out_features * columns, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(self.out_features, columns)], output_dtypes=[mx.uint32])[0]
            if not cache:
                return codes
            self._native_cache = (self.bp_planes, codes)
        return self._native_cache[1]

    def _native_project(self, x, cache=True):
        self._check_input(x)
        if self.source_kind != "affine":
            source = x.astype(mx.float32) if self._native_numeric else x
            return (source @ self._source_floats(cache=cache).T).astype(x.dtype)
        return mx.quantized_matmul(x, self._source_codes(cache=cache), self.bp_scale, self.bp_offset,
                                  transpose=True, group_size=self.group_size,
                                  bits=self.weight_planes, mode="affine")

    def _source_floats(self, cache=False):
        """Exact IEEE reconstruction, streamed by default; no second full model."""
        if self.source_kind == "affine":
            raise ValueError("Affine weights restore source codes instead")
        identity = (self.bp_planes, self.bp_sign, self.bp_exponent, self.bp_count, self._native_numeric)
        if not cache:
            self._native_cache = None
        if self._native_cache is not None and all(a is b for a, b in zip(identity, self._native_cache[0])):
            return self._native_cache[1]
        weight_dtype = mx.float32 if self._native_numeric else DTYPES[self.source_kind]
        weight = _SOURCE_FLOATS(inputs=list(identity[:4]),
            template=[("N", self.out_features), ("K", self.in_features),
                      ("WORDS", (self.in_features + 31) // 32), ("WBITS", self.weight_planes),
                      ("WFRAC", 10 if self.source_kind == "float16" else 7),
                      ("WBIAS", 15 if self.source_kind == "float16" else 127),
                      ("IEEE_WORDS", not self._native_numeric), ("SOURCE_T", DTYPES[self.source_kind]), ("W", weight_dtype)],
            grid=(self.out_features * self.in_features, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(self.out_features, self.in_features)],
            output_dtypes=[weight_dtype])[0]
        if cache:
            self._native_cache = (identity, weight)
        return weight

    def _pack_chunks(self, x, compact=False):
        self._check_input(x)
        rows, k = x.size // self.in_features, self.in_features
        words = (k + 31) // 32
        abits = COMPACT_PLANES if compact else MAX_PLANES[self.activation_dtype]
        fraction, bias = (10, 15) if self.activation_dtype == "float16" else (7, 127)
        flat = x.reshape(rows, k)
        for start in range(0, rows, self.chunk_rows):
            chunk = mx.contiguous(flat[start:start + self.chunk_rows])
            count = chunk.shape[0]
            packed = _PACK(inputs=[chunk],
                template=[("K", k), ("WORDS", words), ("ROWS", count), ("ABITS", abits),
                          ("FRAC", fraction), ("BIAS", bias), ("EMASK", 31 if fraction == 10 else 255)],
                grid=(words * 32, count, 1), threadgroup=(128, 1, 1),
                output_shapes=[(count, abits, words), (count, words), (count, words), (count, words), (count, words),
                               (count, max(1, MAX_PLANES[self.activation_dtype] - abits), words)],
                output_dtypes=[mx.uint32, mx.uint32, mx.int32, mx.int32, mx.float32, mx.uint32])
            yield chunk, packed

    def _project_chunk(self, chunk, packed):
        k, n, count = self.in_features, self.out_features, chunk.shape[0]
        words, abits = (k + 31) // 32, packed[0].shape[1]
        xp, xs, xe, xc, sums, overflow = packed
        dtype = DTYPES[self.activation_dtype]
        if self.source_kind == "affine":
            # Register tiles, not mandatory threadgroup-memory staging. Keep
            # decode's single row separate from multi-token prefill.
            mt, nt = _tile_shape(count)
            compact = abits < MAX_PLANES[self.activation_dtype]
            kernel = _DOT_COMPACT if compact else _DOT_TILED
            inputs = [xp, xs, xe, xc, sums, overflow, self.bp_planes, self.bp_scale, self.bp_offset]
            template = [("K", k), ("N", n), ("WORDS", words), ("ROWS", count),
                        ("ABITS", abits), ("WBITS", self.weight_planes), ("MT", mt), ("NT", nt),
                        ("GROUP_SIZE", self.group_size), ("GROUPS", k // self.group_size), ("T", dtype)]
            if compact:
                inputs.append(chunk)
                template += [("FRAC", 10 if self.activation_dtype == "float16" else 7),
                             ("BIAS", 15 if self.activation_dtype == "float16" else 127)]
            output = kernel(inputs=inputs, template=template,
                grid=(((n + nt - 1) // nt) * 32, (count + mt - 1) // mt, 1), threadgroup=(128, 1, 1),
                output_shapes=[(count, n)], output_dtypes=[dtype])[0]
        else:
            mt, nt = _tile_shape(count)
            kernel = _DENSE_COMPACT if abits < MAX_PLANES[self.activation_dtype] else _DENSE_TILED
            output = kernel(inputs=[*packed, self.bp_planes, self.bp_sign,
                                  self.bp_exponent, self.bp_count, *([chunk] if kernel is _DENSE_COMPACT else [])],
                template=[("K", k), ("N", n), ("WORDS", words), ("ROWS", count),
                          ("ABITS", abits), ("WBITS", self.weight_planes),
                          ("MT", mt), ("NT", nt), ("T", dtype),
                          ("FRAC", 10 if self.activation_dtype == "float16" else 7),
                          ("BIAS", 15 if self.activation_dtype == "float16" else 127)],
                grid=(((n + nt - 1) // nt) * 32, (count + mt - 1) // mt, 1), threadgroup=(128, 1, 1),
                output_shapes=[(count, n)], output_dtypes=[dtype])[0]
        return output

    def __call__(self, x):
        outputs = [self._project_chunk(chunk, packed) for chunk, packed in
                   self._pack_chunks(x, compact=True)]
        n = self.out_features
        return mx.concatenate(outputs, axis=0).reshape(*x.shape[:-1], n)


class BitplaneMLP(nn.Module):
    def __init__(self, projections, activation_dtype, chunk_rows=32,
                 prefill_backend="bitplane", native_prefill_min_rows=32, native_prefill_cache=False,
                 numeric_weights=False):
        super().__init__()
        if prefill_backend not in ("bitplane", "native"):
            raise ValueError("prefill_backend must be bitplane or native")
        if type(native_prefill_min_rows) is not int or native_prefill_min_rows < 2:
            raise ValueError("native_prefill_min_rows must be an integer >= 2")
        if type(native_prefill_cache) is not bool:
            raise ValueError("native_prefill_cache must be boolean")
        self.prefill_backend = prefill_backend
        self.native_prefill_min_rows = native_prefill_min_rows
        self.native_prefill_cache = native_prefill_cache
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, BitplaneLinear(**projections[name], activation_dtype=activation_dtype, chunk_rows=chunk_rows))
            getattr(self, name)._native_numeric = bool(numeric_weights)
        self.silu_table = mx.zeros((65536,), DTYPES[activation_dtype])

    def _gate_up_chunk(self, chunk, packed, pack_output=False):
        gate, up = self.gate_proj, self.up_proj
        if not _can_fuse_gate_up(gate, up):
            if pack_output:
                raise ValueError("Packed gate/up requires compatible projection dimensions")
            return lut_swiglu(gate._project_chunk(chunk, packed), up._project_chunk(chunk, packed), self.silu_table)
        k, n, rows = gate.in_features, gate.out_features, chunk.shape[0]
        mt, nt = _packed_gate_up_tile_shape(rows) if pack_output else _gate_up_tile_shape(rows)
        threads = 32 * (32 // nt) if pack_output else 128
        if pack_output and (mt != 1 or nt not in (1, 2, 4, 8)):
            raise ValueError("Packed gate/up requires MT=1 and NT in 1,2,4,8")
        words = (n + 31) // 32
        shapes = [(rows, n)]
        dtypes = [DTYPES[gate.activation_dtype]]
        if pack_output:
            shapes += [(rows, COMPACT_PLANES, words)] + [(rows, words)] * 4
            dtypes += [mx.uint32, mx.uint32, mx.int32, mx.int32, mx.float32]
            shapes += [(rows, MAX_PLANES[gate.activation_dtype] - COMPACT_PLANES, words)]
            dtypes += [mx.uint32]
        dense = gate.source_kind != "affine"
        metadata = lambda mod: ([mod.bp_sign, mod.bp_exponent, mod.bp_count] if dense else [mod.bp_scale, mod.bp_offset])
        kernel = (_DENSE_GATE_UP_PACKED if pack_output else _DENSE_GATE_UP) if dense else (_GATE_UP_PACKED if pack_output else _GATE_UP)
        outputs = kernel(
            inputs=[*packed, gate.bp_planes, *metadata(gate),
                    up.bp_planes, *metadata(up), self.silu_table, chunk],
            template=[("K", k), ("N", n), ("WORDS", (k + 31) // 32), ("ROWS", rows),
                      ("ABITS", packed[0].shape[1]), ("WBITS", gate.weight_planes),
                      ("UP_WBITS", up.weight_planes),
                      ("MT", mt), ("NT", nt), ("GROUP_SIZE", gate.group_size),
                      ("GROUPS", k // max(1, gate.group_size)), ("T", DTYPES[gate.activation_dtype]),
                      ("FRAC", 10 if gate.activation_dtype == "float16" else 7),
                      ("BIAS", 15 if gate.activation_dtype == "float16" else 127)] +
                     ([("OUT_WORDS", words), ("OUT_ABITS", COMPACT_PLANES),
                       ("EMASK", 31 if gate.activation_dtype == "float16" else 255)] if pack_output else []),
            grid=(words * threads if pack_output else ((n + nt - 1) // nt) * 32,
                  (rows + mt - 1) // mt, 1), threadgroup=(threads, 1, 1),
            output_shapes=shapes, output_dtypes=dtypes)
        return (outputs[0], outputs[1:]) if pack_output else outputs[0]

    def __call__(self, x):
        self.up_proj._check_input(x)
        rows = x.size // self.up_proj.in_features
        if self.prefill_backend == "native" and rows >= self.native_prefill_min_rows:
            gate = self.gate_proj._native_project(x, cache=self.native_prefill_cache)
            up = self.up_proj._native_project(x, cache=self.native_prefill_cache)
            return self.down_proj._native_project(lut_swiglu(gate, up, self.silu_table), cache=self.native_prefill_cache)
        outputs = []
        for chunk, packed in self.gate_proj._pack_chunks(x, compact=True):
            if _prefer_fused_down_pack(chunk.shape[0]) and _can_pack_gate_up(self.gate_proj, self.up_proj, self.down_proj):
                hidden, down_packed = self._gate_up_chunk(chunk, packed, pack_output=True)
                outputs.append(self.down_proj._project_chunk(hidden, down_packed))
            else:
                outputs.append(self.down_proj(self._gate_up_chunk(chunk, packed)))
        return mx.concatenate(outputs, axis=0).reshape(*x.shape[:-1], self.down_proj.out_features)


def expand_plane_factors(weights):
    """Expand storage-only lossy predictors once, before normal strict loading.

    Integer sign-factor sums are exact in FP32 at the supported ranks. Original
    sign/exponent/count/affine metadata stays untouched. Inference uses the
    ordinary v6 expanded plane kernels, with no factorized-DOT speed claim.
    """
    for shape_key in list(weights):
        if not shape_key.endswith('.bp_storage_shape'):
            continue
        prefix = shape_key[:-len('.bp_storage_shape')]
        rows, planes, words, columns = map(int, weights.pop(shape_key).tolist())
        if min(rows, planes, words, columns) < 1 or words != (columns + 31) // 32:
            raise ValueError('Invalid compressed plane dimensions')
        if prefix + '.bp_planes' in weights:
            raise ValueError('Ambiguous compressed and expanded planes')
        def signs(a, width):
            bits = ((a[..., None] >> mx.arange(32, dtype=mx.uint32)) & 1).reshape(a.shape[0], -1)
            return bits[:, :width].astype(mx.float32) * 2 - 1
        output = []
        for plane in range(planes):
            base = prefix + f'.bp_storage_{plane}'
            if base + '_raw' in weights:
                packed = weights.pop(base + '_raw')
            else:
                q = weights.pop(base + '_q')
                rank = q.size
                if rank * 255 >= 1 << 24:
                    raise ValueError('Compressed predictor exceeds exact integer range')
                first = signs(weights.pop(base + '_w1'), columns)
                second = signs(weights.pop(base + '_w2'), rank)
                pieces = []
                for start in range(0, rows, 128):
                    bits = (second[start:start + 128] * q.astype(mx.float32)[None, :]) @ first >= 0
                    bits = mx.pad(bits, [(0, 0), (0, words * 32 - columns)]).reshape(-1, words, 32)
                    value = mx.sum(bits.astype(mx.uint32) << mx.arange(32, dtype=mx.uint32), -1)
                    mx.eval(value)
                    pieces.append(value)
                packed = mx.concatenate(pieces, 0)
            if packed.shape != (rows, words) or packed.dtype != mx.uint32:
                raise ValueError('Malformed compressed plane')
            output.append(packed)
        weights[prefix + '.bp_planes'] = mx.stack(output, axis=1)
        mx.eval(weights[prefix + '.bp_planes'])


def materialize_weights(weights, batch_bytes=512 << 20):
    """Bound loading/evaluation batches so large cold bundles do not time out.

    A single stored tensor can exceed the budget and is evaluated on its own.
    Array values and inference arithmetic are unchanged; lazy loaders skip this.
    """
    if type(batch_bytes) is not int or batch_bytes < 1:
        raise ValueError('Positive materialization budget required')
    pending, size = [], 0
    total = sum(v.nbytes for v in weights.values())
    logger = logging.getLogger(__name__)
    if total >= 8 << 30:
        logger.info('Loading %.1f GiB of bit-plane model parameters in bounded batches', total / 2**30)
    for value in weights.values():
        if pending and size + value.nbytes > batch_bytes:
            mx.eval(*pending)
            pending, size = [], 0
        pending.append(value)
        size += value.nbytes
        if size >= batch_bytes:
            mx.eval(*pending)
            pending, size = [], 0
    if pending:
        mx.eval(*pending)


def install(model, config, weights, *, materialize=False):
    """Explicit config-driven installation; standard model loading remains strict."""
    spec = config.get("bitffn_bitplane")
    if spec is None:
        if any(".bp_planes" in k for k in weights):
            raise ValueError("Bit-plane weights require a manifest")
        return
    expand_plane_factors(weights)
    if spec.get("format") != FORMAT or spec.get("activation_dtype") not in DTYPES:
        raise ValueError("Unsupported bit-plane manifest")
    if (config.get("quantization") or config.get("quantization_config")) and not spec.get("preserve_source_codes"):
        raise ValueError("Quantized models are unsupported by the no-quantization backend")
    entries = spec.get("layers")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Empty bit-plane layer manifest")
    modules, replacements, expected = dict(model.named_modules()), [], set()
    # Early bundles retain HF tensor names/layouts while their plane shards
    # carry format=mlx. mlx-vlm then skips its normal sanitization and invokes
    # this hook before its MTP load_weights override can repair those names.
    # Normalize only the unmistakable raw Qwen layout, once, in memory.
    raw_tables = [k[:-len('.silu_table')] for k in weights
                  if k.startswith('model.language_model.') and k.endswith('.silu_table')]
    if config.get('model_type') == 'qwen3_5' and raw_tables and any(p not in modules for p in raw_tables):
        if not hasattr(model, 'sanitize'):
            raise ValueError('Raw Qwen bit-plane bundle requires architecture sanitization')
        normalized = model.sanitize(weights)
        vision = getattr(model, 'vision_tower', None)
        if hasattr(vision, 'sanitize'):
            normalized = vision.sanitize(normalized)
        weights.clear()
        weights.update(normalized)
    for suffix, desc in entries.items():
        if not re.fullmatch(r"layers\.\d+\.mlp", suffix):
            raise ValueError(f"Unsupported MLP path: {suffix}")
        # Native MTP adds e.g. language_model.mtp.layers.0.mlp alongside
        # language_model.model.layers.0.mlp. The converted SiLU table binds the
        # exact target; an unqualified suffix alone cannot identify the FFN.
        candidates = [p for p in modules if p == suffix or p.endswith("." + suffix)]
        matches = [p for p in candidates if p + '.silu_table' in weights]
        if len(matches) != 1:
            raise ValueError(f"MLP not uniquely bound to bit-plane tensors: {suffix}; candidates={candidates}, bound={matches}")
        path = matches[0]
        projections = desc.get("projections", {})
        if set(projections) != {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError("All three SwiGLU projections are required")
        original = modules[path]
        for name, projection in projections.items():
            source = getattr(original, name, None)
            dims = (projection.get("out_features"), projection.get("in_features"))
            if not isinstance(original, BitplaneMLP) and (
                not isinstance(source, (nn.Linear, nn.QuantizedLinear)) or
                (source.weight.shape[0], source.weight.shape[1] * (32 // source.bits) if isinstance(source, nn.QuantizedLinear) else source.weight.shape[1]) != dims or "bias" in source
            ):
                raise ValueError(f"Expected original bias-free Linear at {path}.{name}")
        replacement = BitplaneMLP(projections, spec["activation_dtype"], spec.get("chunk_rows", 32),
                                 spec.get("prefill_backend", "bitplane"), spec.get("native_prefill_min_rows", 32),
                                 spec.get("native_prefill_cache", False),
                                 numeric_weights=int(suffix.split('.')[1]) in spec.get('lossy_storage', {}).get('layers', []))
        from mlx.utils import tree_flatten
        for key, value in tree_flatten(replacement.parameters()):
            full = path + "." + key
            expected.add(full)
            if full not in weights or weights[full].shape != value.shape or weights[full].dtype != value.dtype:
                raise ValueError(f"Missing/malformed bit-plane parameter: {full}")
        for name, projection in projections.items():
            base = path + "." + name
            counts, exponents = weights[base + ".bp_count"], weights[base + ".bp_exponent"]
            if projection["source_kind"] == "affine":
                if not bool((mx.all(counts == projection["weight_planes"]) & mx.all(exponents == 0) &
                             mx.all(weights[base + ".bp_sign"] == 0) & mx.all(mx.isfinite(weights[base + ".bp_scale"])) &
                             mx.all(mx.isfinite(weights[base + ".bp_offset"]))).item()):
                    raise ValueError(f"Invalid source-code metadata: {base}")
                continue
            lower, upper = (-24, 5) if projection["source_kind"] == "float16" else (-133, 120)
            valid = mx.all((counts >= 0) & (counts <= projection["weight_planes"]))
            valid = valid & mx.all((exponents >= lower) & (exponents <= upper))
            max_exponent = 15 if projection["source_kind"] == "float16" else 127
            valid = valid & mx.all((counts == 0) | (exponents + counts - 1 <= max_exponent))
            if not bool(valid.item()):
                raise ValueError(f"Invalid bit-plane tile metadata: {base}")
        replacements.append((path, replacement))
    actual = {k for k in weights if ".bp_" in k or k.endswith(".silu_table")}
    if actual != expected:
        raise ValueError("Bit-plane weights do not match manifest")
    model.update_modules(tree_unflatten(replacements))
    if materialize:
        materialize_weights(weights)


def load_model(model_path, lazy=False):
    """Load a standalone bundle without patching any installed MLX/vMLX packages."""
    from mlx_lm.utils import _get_classes
    model_path = Path(model_path)
    config = json.loads((model_path / "config.json").read_text())
    Model, ModelArgs = _get_classes(config)
    model = Model(ModelArgs.from_dict(config))
    index = json.loads((model_path / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for shard in sorted(set(index.values())):
        if Path(shard).name != shard:
            raise ValueError("Unsafe shard path")
        weights.update(mx.load(str(model_path / shard)))
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    install(model, config, weights, materialize=not lazy)
    quantization = config.get("quantization", config.get("quantization_config"))
    if quantization:
        def predicate(path, module):
            if isinstance(module, BitplaneLinear):
                return False
            if path in quantization:
                return quantization[path]
            return hasattr(module, "to_quantized") and path + ".scales" in weights
        # Instantiate unchanged quantized module types, then load their ORIGINAL
        # source tensors below. This does not re-quantize any source weights.
        nn.quantize(model, group_size=quantization["group_size"], bits=quantization["bits"],
                    mode=quantization.get("mode", "affine"), class_predicate=predicate)
    model.load_weights(list(weights.items()), strict=True)
    if not lazy:
        mx.eval(model.parameters())
    model.eval()
    return model, config


def auto_install_from_weights(model, weights):
    """Support vMLX loaders that call Module.load_weights without a config."""
    original_weights = weights
    weights = weights if isinstance(weights, dict) else dict(weights)
    expand_plane_factors(weights)
    if isinstance(original_weights, list):
        original_weights[:] = list(weights.items())
    paths = {k.rsplit(".silu_table", 1)[0] for k in weights if k.endswith(".silu_table")}
    if not paths:
        return
    modules, layers = dict(model.named_modules()), {}
    if all(isinstance(modules.get(path), BitplaneMLP) for path in paths):
        # A config-aware loader already installed and validated these modules.
        # Preserve its chunk_rows setting and avoid replacing it a second time.
        return
    dtype = None
    preserve = False
    for path in paths:
        matches = [p for p in modules if p == path or p.endswith("." + path)]
        if len(matches) != 1:
            raise ValueError(f"Cannot identify bit-plane MLP: {path}")
        original = modules[matches[0]]
        current_dtype = next((k for k, v in DTYPES.items() if v == weights[path + ".silu_table"].dtype), None)
        if current_dtype is None or dtype not in (None, current_dtype):
            raise ValueError("Mixed/unsupported bit-plane activation precision")
        dtype = current_dtype
        desc = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            prefix = path + "." + name
            source = getattr(original, name)
            if isinstance(source, BitplaneLinear):
                i, o = source.in_features, source.out_features
            elif isinstance(source, nn.QuantizedLinear):
                o, i = source.weight.shape[0], source.weight.shape[1] * (32 // source.bits)
            else:
                o, i = source.weight.shape
            affine = prefix + ".bp_scale" in weights
            preserve |= affine
            desc[name] = {"in_features": i, "out_features": o,
                          "weight_planes": weights[prefix + ".bp_planes"].shape[1],
                          "source_kind": "affine" if affine else dtype}
            if affine:
                desc[name]["group_size"] = i // weights[prefix + ".bp_scale"].shape[1]
        suffix = re.search(r"layers\.\d+\.mlp$", path)
        if suffix is None:
            raise ValueError(f"Unsupported bit-plane path: {path}")
        layers[suffix.group()] = {"projections": desc}
    install(model, {"bitffn_bitplane": {"format": FORMAT, "activation_dtype": dtype,
                                        "preserve_source_codes": preserve, "layers": layers}}, weights)
