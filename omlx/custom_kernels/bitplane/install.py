"""Install reversible loader hooks into staged MLX dependencies before signing.

Run directly with --site-packages, or use python -m with the installed package.
No MLX import or GPU is needed to stage these hooks.
"""
import argparse
import ast
import json
from pathlib import Path
import shutil

MARKER = "# omlx bitplane loader hook v1"
CONFIG_HOOK = '''    # omlx bitplane loader hook v1
    if "bitffn_bitplane" in config:
        from omlx.custom_kernels.bitplane import runtime as _bp_runtime
        _bp_runtime.install(model, config, weights)
        if not lazy:
            _bp_runtime.materialize_weights(weights)

'''
BASE_HOOK = '''        # omlx bitplane loader hook v1
        if any(".bp_" in name or name.endswith(".silu_table") for name, _ in weights):
            from omlx.custom_kernels.bitplane.runtime import auto_install_from_weights
            auto_install_from_weights(self, weights)

'''
TARGETS = [
    ("mlx/nn/layers/base.py", "        if strict:\n            new_weights = dict(weights)\n", BASE_HOOK),
    ("mlx_lm/utils.py", "    def _quantize(quantization):\n", CONFIG_HOOK),
    ("mlx_vlm/utils.py", "    if not has_quantization:\n", CONFIG_HOOK),
]


def install_into_site_packages(root, *, check=False, restore=False):
    root = Path(root).resolve()
    backup = root / "omlx_bitplane_backup_v1"
    planned, existing = {}, []
    for name, anchor, hook in TARGETS:
        path = root / name
        if not path.is_file():
            if name.startswith("mlx_vlm/"):
                continue
            raise ValueError(f"Missing MLX dependency: {path}")
        source = path.read_text()
        # Do not stack two independent runtimes on an old vMLX-patched donor.
        if "# bitffn_bitplane loader hook v1" in source:
            raise ValueError(f"Restore the legacy Bit FFN Lab hooks first: {path}")
        if MARKER in source:
            if source.count(hook) != 1:
                raise ValueError(f"Unsupported modified Bitplane hook: {path}")
            existing.append(name)
            if restore:
                planned[name] = source.replace(hook, "", 1)
        elif not restore:
            if source.count(anchor) != 1:
                raise ValueError(f"Unsupported loader anchor: {path}")
            planned[name] = source.replace(anchor, hook + anchor, 1)
        if name in planned:
            ast.parse(planned[name])
    result = {"site_packages": str(root), "existing_hooks": existing,
              "pending_hooks": list(planned)}
    if check:
        return result
    # Validate every target before touching any dependency. Remove only our
    # exact block on restore, retaining unrelated upstream changes.
    for name, source in planned.items():
        path = root / name
        saved = backup / name
        if not restore and not saved.exists():
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, saved)
        temporary = path.with_suffix(".bp-tmp")
        temporary.write_text(source)
        temporary.replace(path)
        for pyc in (path.parent / "__pycache__").glob(path.stem + ".*.pyc"):
            pyc.unlink()
    return {**result, "restored_hooks" if restore else "installed_hooks": list(planned)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    print(json.dumps(install_into_site_packages(args.site_packages,
                     check=args.check, restore=args.restore), indent=2))


if __name__ == "__main__":
    main()
