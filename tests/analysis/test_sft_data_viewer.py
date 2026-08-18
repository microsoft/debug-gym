import sys
import threading
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest

from analysis.sft_data_viewer import sft_data_viewer


@pytest.fixture
def viewer(tmp_path):
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    sft_data_viewer.app.config.update(
        TESTING=True,
        SAFE_ROOT=safe_root,
        UPLOAD_FOLDER=str(upload_root),
        MAX_CONTENT_LENGTH=64 * 1024 * 1024,
    )
    if hasattr(sft_data_viewer, "configure_safe_root"):
        sft_data_viewer.configure_safe_root(safe_root)
    sft_data_viewer.clear_current_file()
    yield sft_data_viewer.app.test_client(), safe_root
    sft_data_viewer.clear_current_file()


def _write_jsonl(path: Path):
    path.write_text('{"messages": [], "problem": "test"}\n', encoding="utf-8")


def test_load_file_accepts_in_root_jsonl(viewer):
    client, safe_root = viewer
    path = safe_root / "records.jsonl"
    _write_jsonl(path)

    response = client.post(
        "/load_file",
        data={"filepath": path.relative_to(safe_root).as_posix()},
    )

    assert response.status_code == 302


def test_upload_does_not_follow_preexisting_filename_symlink(viewer, tmp_path):
    client, _ = viewer
    outside = tmp_path / "outside.jsonl"
    outside.write_text("unchanged", encoding="utf-8")
    link = Path(sft_data_viewer.app.config["UPLOAD_FOLDER"]) / "records.jsonl"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable: {exc}")

    response = client.post(
        "/upload",
        data={
            "file": (
                BytesIO(b'{"problem": "inside", "messages": []}\n'),
                "records.jsonl",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 302
    assert outside.read_text(encoding="utf-8") == "unchanged"
    upload_files = [
        path
        for path in Path(sft_data_viewer.app.config["UPLOAD_FOLDER"]).iterdir()
        if path != link
    ]
    assert len(upload_files) == 1

    response = client.post(
        "/upload",
        data={
            "file": (
                BytesIO(b'{"problem": "replacement", "messages": []}\n'),
                "replacement.jsonl",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 302
    replacement_files = [
        path
        for path in Path(sft_data_viewer.app.config["UPLOAD_FOLDER"]).iterdir()
        if path != link
    ]
    assert len(replacement_files) == 1
    assert replacement_files[0] != upload_files[0]

    assert client.post("/change_file").status_code == 302
    assert list(Path(sft_data_viewer.app.config["UPLOAD_FOLDER"]).iterdir()) == [link]


def test_upload_size_limit_does_not_retain_partial_file(viewer):
    client, _ = viewer
    sft_data_viewer.app.config["MAX_CONTENT_LENGTH"] = 128

    response = client.post(
        "/upload",
        data={"file": (BytesIO(b"x" * 1024), "oversized.jsonl")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 413
    assert not list(Path(sft_data_viewer.app.config["UPLOAD_FOLDER"]).iterdir())


def test_load_file_rejects_traversal_and_absolute_outside_paths(viewer, tmp_path):
    client, safe_root = viewer
    outside = tmp_path / "outside.jsonl"
    _write_jsonl(outside)

    for path in ("../outside.jsonl", str(outside)):
        response = client.post("/load_file", data={"filepath": path})

        assert response.status_code in {400, 403, 404}
        assert str(outside) not in response.get_data(as_text=True)


def test_load_file_rejects_sibling_prefix_path(viewer):
    client, safe_root = viewer
    sibling = safe_root.parent / f"{safe_root.name}-sibling"
    sibling.mkdir()
    path = sibling / "records.jsonl"
    _write_jsonl(path)

    response = client.post("/load_file", data={"filepath": str(path)})

    assert response.status_code in {400, 403, 404}


def test_load_file_rejects_wrong_suffix(viewer):
    client, safe_root = viewer
    path = safe_root / "records.json"
    path.write_text("{}", encoding="utf-8")

    response = client.post(
        "/load_file",
        data={"filepath": path.relative_to(safe_root).as_posix()},
    )

    assert response.status_code == 400


def test_load_file_rejects_symlink_escape(viewer, tmp_path):
    client, safe_root = viewer
    outside = tmp_path / "outside.jsonl"
    _write_jsonl(outside)
    link = safe_root / "linked.jsonl"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable: {exc}")

    response = client.post("/load_file", data={"filepath": "linked.jsonl"})

    assert response.status_code in {400, 403, 404}


def test_loaded_file_cannot_be_swapped_for_outside_symlink(viewer, tmp_path):
    client, safe_root = viewer
    path = safe_root / "records.jsonl"
    path.write_text('{"problem": "inside", "messages": []}\n', encoding="utf-8")
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"problem": "outside", "messages": []}\n', encoding="utf-8")
    assert (
        client.post("/load_file", data={"filepath": "records.jsonl"}).status_code == 302
    )

    try:
        path.unlink()
        path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Open-file replacement is unavailable: {exc}")

    response = client.get("/api/record/0")

    assert response.status_code == 200
    assert response.get_json()["problem"] == "inside"


def test_load_file_rejects_hard_link_to_outside_file(viewer, tmp_path):
    client, safe_root = viewer
    outside = tmp_path / "outside.jsonl"
    _write_jsonl(outside)
    hard_link = safe_root / "hard-linked.jsonl"
    try:
        hard_link.hardlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Hard links are unavailable: {exc}")

    response = client.post("/load_file", data={"filepath": "hard-linked.jsonl"})

    assert response.status_code == 403


def test_change_file_never_deletes_server_owned_source(viewer):
    client, safe_root = viewer
    path = safe_root / "records.jsonl"
    _write_jsonl(path)
    assert (
        client.post("/load_file", data={"filepath": "records.jsonl"}).status_code == 302
    )

    assert client.post("/change_file").status_code == 302

    assert path.exists()


def test_concurrent_loads_publish_one_complete_snapshot(viewer, monkeypatch):
    _, safe_root = viewer
    first = safe_root / "first.jsonl"
    first.write_text('{"problem": "first", "messages": []}\n', encoding="utf-8")
    second = safe_root / "second.jsonl"
    second.write_text(
        "".join(
            f'{{"problem": "second-{index}", "messages": []}}\n' for index in range(20)
        ),
        encoding="utf-8",
    )
    first_waiting = threading.Event()
    release_first = threading.Event()
    original_replace = sft_data_viewer.replace_current_file

    def blocking_replace(*args, **kwargs):
        filename = args[2]
        if filename == "first.jsonl":
            first_waiting.set()
            assert release_first.wait(timeout=5)
        return original_replace(*args, **kwargs)

    monkeypatch.setattr(sft_data_viewer, "replace_current_file", blocking_replace)
    results = {}

    def load_first():
        with sft_data_viewer.app.test_client() as client:
            results["first"] = client.post(
                "/load_file", data={"filepath": "first.jsonl"}
            ).status_code

    thread = threading.Thread(target=load_first)
    thread.start()
    assert first_waiting.wait(timeout=5)
    with sft_data_viewer.app.test_client() as client:
        results["second"] = client.post(
            "/load_file", data={"filepath": "second.jsonl"}
        ).status_code
    release_first.set()
    thread.join(timeout=5)
    assert not thread.is_alive()

    with sft_data_viewer.app.test_client() as client:
        response = client.get("/")

    assert results == {"first": 302, "second": 302}
    html = response.get_data(as_text=True)
    assert "first.jsonl" in html
    assert "1 trajectories" in html


def test_tool_call_json_uses_text_nodes_for_untrusted_strings():
    template = (
        Path(sft_data_viewer.__file__).parent / "templates" / "record_detail.html"
    ).read_text(encoding="utf-8")

    assert "valueSpan.textContent" in template
    assert "valueContainer.innerHTML = formatJSONValue" not in template
    assert 'return `<span class="json-string">"${value}"</span>`' not in template


def test_statistics_serializes_untrusted_labels_as_json(viewer):
    client, safe_root = viewer
    path = safe_root / "malicious-labels.jsonl"
    path.write_text(
        (
            '{"messages":[{"role":"x\\\\","content":""}],'
            '"satisfied_criteria":["\\\\\\u003c/script\\u003e"]}\n'
        ),
        encoding="utf-8",
    )
    assert (
        client.post(
            "/load_file",
            data={"filepath": "malicious-labels.jsonl"},
        ).status_code
        == 302
    )

    response = client.get("/statistics")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'labels: ["X\\\\"]' in html
    assert "\\u003c/script\\u003e" in html
    assert "'X\\'" not in html


def test_main_ignores_non_loopback_host_environment(monkeypatch, tmp_path):
    run = Mock()
    monkeypatch.setenv("SFT_DATA_VIEWER_HOST", "0.0.0.0")
    monkeypatch.setattr(sft_data_viewer.app, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["sft_data_viewer.py", "--safe-root", str(tmp_path)],
    )

    sft_data_viewer.main()

    run.assert_called_once_with(host="127.0.0.1", port=sft_data_viewer.DEFAULT_PORT)


def test_cross_origin_mutation_is_rejected(viewer):
    client, _ = viewer

    response = client.post(
        "/load_file",
        data={"filepath": "records.jsonl"},
        headers={"Origin": "https://evil.example"},
    )

    assert response.status_code == 403


def test_change_file_is_post_only(viewer):
    client, _ = viewer

    assert client.get("/change_file").status_code == 405


def test_untrusted_host_is_rejected(viewer):
    client, safe_root = viewer
    path = safe_root / "records.jsonl"
    _write_jsonl(path)

    response = client.post(
        "/load_file",
        data={"filepath": str(path)},
        headers={"Host": "evil.example"},
    )

    assert response.status_code == 400


def test_server_defaults_to_loopback_and_current_directory():
    assert sft_data_viewer.DEFAULT_HOST == "127.0.0.1"
    assert sft_data_viewer.DEFAULT_SAFE_ROOT == Path.cwd().resolve()
