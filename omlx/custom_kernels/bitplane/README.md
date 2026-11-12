# Bitplane FFN backend

Vendored from our `bit-ffn-lab` project on 2026-11-10. `provenance.json`
records the original filenames and SHA256 of the unchanged runtime and codec.
Implementation: `xor-hybrid-v7-unified`; format:
`bitffn.bitplane.lossless-weights.v1`.

`runtime.py` contains the complete Metal sources and MLX dispatch code for
activation packing, XOR/masked-popcount projection, fused gate/up/SiLU,
and reconstruction of original weights/codes for native prefill.
`codec.py` contains the NumPy packing and reconstruction routines.
Supported sources are FP16, BF16 and existing affine q4/q8 codes. This is
an experimental FFN backend; it does not promise native speed or bitwise
model-output parity. Explicitly marked lossy storage variants also exist.

## Compiled kernels

This backend uses `mx.fast.metal_kernel`. MLX specializes and compiles each
kernel at runtime for the requested dimensions, types and dispatch flags.
There is no standalone precompiled Bitplane `.so` or `.metallib` to copy.
The Metal sources are embedded in `runtime.py` and travel with the wheel/app.
Do not ship a machine's opaque Metal cache as a portable kernel library.
An ahead-of-time library would require a separate C++/MLX extension and
dispatch implementation; it is not part of this import.

## App builds

Both `apps/omlx-mac/Scripts/build.sh` and `build_meeting_app.py` install the
loader hooks into the **staged** dependency tree before signing the app.
The hooks import the bundled runtime from this package. Rebuilding the fork
therefore restores Bitplane support even after replacing the donor runtime.
No external `bit-ffn-lab` checkout or weight reconversion is needed.

The installer validates every target before writing, keeps original-file
backups, and refuses unknown loader layouts. MLX dependency updates that
change the anchors fail the build rather than silently dropping support.
Only the hook blocks are removed on restore, preserving later edits.

For a source/venv install, or to check a staged application:

```sh
python -m omlx.custom_kernels.bitplane.install --site-packages /path/to/site-packages --check
python -m omlx.custom_kernels.bitplane.install --site-packages /path/to/site-packages
# Undo our loader hooks:
python -m omlx.custom_kernels.bitplane.install --site-packages /path/to/site-packages --restore
```

Restart the server after installing hooks into an existing environment.
Stock upstream app updates do not include these fork changes.

## Verification

```sh
python -m pytest tests/test_bitplane_bundle.py tests/test_bitplane_kernels.py
```

The GPU tests compile and run FP16/BF16 and q4/q8 kernels, compare projection
outputs with MLX and check bit-exact source reconstruction. With hooks installed
in a fresh process, they also load a tiny converted Qwen3 through native
MLX-LM and compare its one-token and eight-token forward outputs.
