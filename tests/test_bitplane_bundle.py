"""Staging contracts for the vendored Bitplane loader, without a GPU."""
import ast
from pathlib import Path

import pytest

from omlx.custom_kernels.bitplane.install import (
    MARKER, TARGETS, install_into_site_packages,
)


def _dependencies(root):
    originals = {}
    for name, anchor, _ in TARGETS:
        source = ("class M:\n    def load(self, weights, strict):\n"
                  if name.startswith("mlx/") else "def load(config, weights, lazy=False):\n")
        source += anchor + "        pass\n"
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        originals[name] = source
    return originals


def test_staging_idempotency_and_restore(tmp_path):
    originals = _dependencies(tmp_path)
    assert len(install_into_site_packages(tmp_path, check=True)["pending_hooks"]) == 3
    install_into_site_packages(tmp_path)
    assert install_into_site_packages(tmp_path)["installed_hooks"] == []
    for name in originals:
        path = tmp_path / name
        assert path.read_text().count(MARKER) == 1
        ast.parse(path.read_text())
        path.write_text(path.read_text() + "\n# upstream change\n")
    install_into_site_packages(tmp_path, restore=True)
    for name, source in originals.items():
        assert (tmp_path / name).read_text() == source + "\n# upstream change\n"


def test_unsupported_dependency_never_partially_patched(tmp_path):
    originals = _dependencies(tmp_path)
    path = tmp_path / "mlx_vlm/utils.py"
    path.write_text("# changed upstream\n")
    with pytest.raises(ValueError, match="Unsupported loader anchor"):
        install_into_site_packages(tmp_path)
    assert (tmp_path / "mlx/nn/layers/base.py").read_text() == originals["mlx/nn/layers/base.py"]
    assert not (tmp_path / "omlx_bitplane_backup_v1").exists()


def test_hooks_use_bundled_runtime_and_builds_install_them():
    root = Path(__file__).resolve().parents[1]
    for _, _, hook in TARGETS:
        assert "omlx.custom_kernels.bitplane" in hook
    for build in ("build.sh", "build_meeting_app.py"):
        source = (root / "apps/omlx-mac/Scripts" / build).read_text()
        assert "omlx/custom_kernels/bitplane/install.py" in source
        assert "--site-packages" in source
