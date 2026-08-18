import os
from pathlib import Path

import pytest

from analysis.json_log_viewer import json_log_viewer
from analysis.sft_data_viewer import sft_data_viewer

VIEWERS = [
    pytest.param(json_log_viewer, ".json", id="json"),
    pytest.param(sft_data_viewer, ".jsonl", id="sft"),
]


def configure_viewer(module, safe_root):
    module.app.config["SAFE_ROOT"] = safe_root
    if hasattr(module, "configure_safe_root"):
        module.configure_safe_root(safe_root)


@pytest.mark.parametrize(("module", "suffix"), VIEWERS)
def test_confined_open_rejects_hard_links(module, suffix, tmp_path):
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    configure_viewer(module, safe_root)
    outside = tmp_path / f"outside{suffix}"
    outside.write_text("outside", encoding="utf-8")
    linked = safe_root / f"linked{suffix}"
    try:
        linked.hardlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Hard links are unavailable: {exc}")

    with pytest.raises(module.SafePathError) as exc_info:
        module.open_confined_server_file(linked.name)

    assert exc_info.value.status_code == 403


@pytest.mark.parametrize(("module", "suffix"), VIEWERS)
def test_confined_open_rejects_symlink_ancestors(module, suffix, tmp_path):
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    configure_viewer(module, safe_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / f"records{suffix}").write_text("outside", encoding="utf-8")
    linked = safe_root / "linked"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable: {exc}")

    with pytest.raises(module.SafePathError) as exc_info:
        module.open_confined_server_file(f"linked/records{suffix}")

    assert exc_info.value.status_code in {403, 404}


@pytest.mark.skipif(
    os.name == "nt" or os.open not in os.supports_dir_fd,
    reason="Descriptor-relative ancestor-swap test requires POSIX openat support",
)
@pytest.mark.parametrize(("module", "suffix"), VIEWERS)
def test_confined_open_is_bound_to_pinned_ancestor(
    module, suffix, tmp_path, monkeypatch
):
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    configure_viewer(module, safe_root)
    current = safe_root / "current"
    current.mkdir()
    inside_file = current / f"records{suffix}"
    inside_file.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / f"records{suffix}").write_text("outside", encoding="utf-8")
    pinned = safe_root / "pinned"
    real_open = module.os.open
    swapped = False

    def swap_before_final_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == f"records{suffix}" and dir_fd is not None and not swapped:
            current.rename(pinned)
            current.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "open", swap_before_final_open)
    monkeypatch.setattr(
        module.os,
        "supports_dir_fd",
        module.os.supports_dir_fd | {swap_before_final_open},
    )
    descriptor = module.open_confined_server_file(f"current/records{suffix}")
    try:
        with os.fdopen(os.dup(descriptor), "r", encoding="utf-8") as source:
            assert source.read() == "inside"
    finally:
        os.close(descriptor)
