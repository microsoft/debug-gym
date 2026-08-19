import json
import sys
import threading
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest

from analysis.json_log_viewer import json_log_viewer


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    json_log_viewer.app.config.update(
        TESTING=True,
        SAFE_ROOT=safe_root,
        UPLOAD_FOLDER=str(upload_root),
        ALLOWED_ORIGINS=(),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    )
    if hasattr(json_log_viewer, "configure_safe_root"):
        json_log_viewer.configure_safe_root(safe_root)
    if hasattr(json_log_viewer, "clear_loaded_state"):
        json_log_viewer.clear_loaded_state()
    else:
        json_log_viewer.data = None
        json_log_viewer.current_file = None
    yield json_log_viewer.app.test_client(), safe_root
    if hasattr(json_log_viewer, "clear_loaded_state"):
        json_log_viewer.clear_loaded_state()


def _write_log(path: Path):
    path.write_text(
        json.dumps(
            {
                "problem": "test",
                "config": {},
                "uuid": "test",
                "success": True,
                "log": [{"action": None}],
            }
        ),
        encoding="utf-8",
    )


def test_load_file_from_path_accepts_in_root_json(viewer):
    client, safe_root = viewer
    path = safe_root / "trajectory.json"
    _write_log(path)

    response = client.post(
        "/load_file_from_path",
        json={"path": path.relative_to(safe_root).as_posix()},
    )

    assert response.status_code == 200
    assert response.get_json()["success"] is True


def test_upload_does_not_follow_preexisting_filename_symlink(viewer, tmp_path):
    client, _ = viewer
    outside = tmp_path / "outside.json"
    outside.write_text("unchanged", encoding="utf-8")
    link = Path(json_log_viewer.app.config["UPLOAD_FOLDER"]) / "trajectory.json"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable: {exc}")
    payload = json.dumps(
        {
            "problem": "test",
            "config": {},
            "uuid": "test",
            "success": True,
            "log": [],
        }
    ).encode()

    response = client.post(
        "/upload",
        data={"file": (BytesIO(payload), "trajectory.json")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 302
    assert outside.read_text(encoding="utf-8") == "unchanged"
    assert list(Path(json_log_viewer.app.config["UPLOAD_FOLDER"]).iterdir()) == [link]


@pytest.mark.parametrize(
    ("route", "method"),
    [("/load_file_from_path", "post"), ("/browse_directory", "get")],
)
def test_routes_reject_traversal_and_absolute_outside_paths(
    viewer, tmp_path, route, method
):
    client, safe_root = viewer
    outside = tmp_path / "outside.json"
    _write_log(outside)

    for path in ("../outside.json", str(outside)):
        if method == "post":
            response = client.post(route, json={"path": path})
        else:
            response = client.get(route, query_string={"path": path})

        assert response.status_code in {400, 403, 404}
        assert str(outside) not in response.get_data(as_text=True)


@pytest.mark.parametrize(
    ("route", "method"),
    [("/load_file_from_path", "post"), ("/browse_directory", "get")],
)
def test_routes_reject_sibling_prefix_path(viewer, tmp_path, route, method):
    client, safe_root = viewer
    sibling = safe_root.parent / f"{safe_root.name}-sibling"
    sibling.mkdir()
    path = sibling / "trajectory.json"
    _write_log(path)

    if method == "post":
        response = client.post(route, json={"path": str(path)})
    else:
        response = client.get(route, query_string={"path": str(path)})

    assert response.status_code in {400, 403, 404}


def test_load_file_from_path_rejects_wrong_suffix(viewer):
    client, safe_root = viewer
    path = safe_root / "trajectory.txt"
    path.write_text("{}", encoding="utf-8")

    response = client.post(
        "/load_file_from_path",
        json={"path": path.relative_to(safe_root).as_posix()},
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    ("route", "method"),
    [("/load_file_from_path", "post"), ("/browse_directory", "get")],
)
def test_routes_reject_symlink_escape(viewer, tmp_path, route, method):
    client, safe_root = viewer
    outside = tmp_path / "outside.json"
    _write_log(outside)
    link = safe_root / "linked.json"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable: {exc}")

    if method == "post":
        response = client.post(route, json={"path": "linked.json"})
    else:
        response = client.get(route, query_string={"path": "linked.json"})

    assert response.status_code in {400, 403, 404}


def test_browse_directory_accepts_safe_root(viewer):
    client, safe_root = viewer
    _write_log(safe_root / "trajectory.json")

    response = client.get("/browse_directory")

    assert response.status_code == 200
    assert response.get_json()["current_path"] == "."


def test_load_file_from_path_rejects_hard_link_to_outside_file(viewer, tmp_path):
    client, safe_root = viewer
    outside = tmp_path / "outside.json"
    _write_log(outside)
    hard_link = safe_root / "hard-linked.json"
    try:
        hard_link.hardlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Hard links are unavailable: {exc}")

    response = client.post(
        "/load_file_from_path",
        json={"path": "hard-linked.json"},
    )

    assert response.status_code == 403


def test_cors_is_limited_to_explicit_allowed_origins(viewer):
    client, safe_root = viewer
    path = safe_root / "trajectory.json"
    _write_log(path)
    json_log_viewer.app.config["ALLOWED_ORIGINS"] = ("https://gray.example",)

    denied = client.post(
        "/load_file_from_path",
        json={"path": path.relative_to(safe_root).as_posix()},
        headers={"Origin": "https://evil.example"},
    )
    allowed = client.post(
        "/load_file_from_path",
        json={"path": path.relative_to(safe_root).as_posix()},
        headers={"Origin": "https://gray.example"},
    )

    assert "Access-Control-Allow-Origin" not in denied.headers
    assert allowed.headers["Access-Control-Allow-Origin"] == "https://gray.example"


def test_same_origin_post_is_allowed_without_cors_configuration(viewer):
    client, safe_root = viewer
    path = safe_root / "trajectory.json"
    _write_log(path)

    response = client.post(
        "/load_file_from_path",
        json={"path": "trajectory.json"},
        headers={"Origin": "http://localhost"},
    )

    assert response.status_code == 200
    assert "Access-Control-Allow-Origin" not in response.headers


def test_untrusted_host_cannot_bypass_origin_allowlist(viewer):
    client, safe_root = viewer
    path = safe_root / "trajectory.json"
    _write_log(path)

    response = client.post(
        "/load_file_from_path",
        json={"path": path.relative_to(safe_root).as_posix()},
        headers={
            "Host": "evil.example",
            "Origin": "http://evil.example",
        },
    )

    assert response.status_code == 400


def test_state_changing_file_load_is_post_only(viewer):
    client, _ = viewer

    assert (
        client.get("/load_file_from_path", query_string={"path": "x.json"}).status_code
        == 405
    )
    assert client.get("/change_file").status_code == 405


@pytest.mark.parametrize("route", ["/browse_directory", "/load_file_from_path"])
def test_safe_path_failures_never_expose_exception_text(viewer, monkeypatch, route):
    client, _ = viewer
    target = (
        "open_confined_directory"
        if route == "/browse_directory"
        else "open_confined_server_file"
    )
    monkeypatch.setattr(
        json_log_viewer,
        target,
        Mock(side_effect=json_log_viewer.SafePathError("private detail", 403)),
    )

    if route == "/browse_directory":
        response = client.get(route)
    else:
        response = client.post(route, json={"path": "trajectory.json"})

    assert response.status_code == 403
    assert "private detail" not in response.get_data(as_text=True)


def test_loaded_state_snapshot_never_mixes_data_and_filename():
    assert hasattr(json_log_viewer, "replace_loaded_state")
    assert hasattr(json_log_viewer, "get_loaded_state")
    failures = []
    start = threading.Barrier(3)

    def writer(marker):
        start.wait()
        for _ in range(500):
            json_log_viewer.replace_loaded_state(
                {
                    "problem": marker,
                    "config": {},
                    "uuid": marker,
                    "success": True,
                    "log": [],
                },
                f"{marker}.json",
            )

    threads = [
        threading.Thread(target=writer, args=("a",)),
        threading.Thread(target=writer, args=("b",)),
    ]
    for thread in threads:
        thread.start()
    start.wait()
    while any(thread.is_alive() for thread in threads):
        state = json_log_viewer.get_loaded_state()
        if state is not None and state.filename != f"{state.data['problem']}.json":
            failures.append((state.filename, state.data["problem"]))
    for thread in threads:
        thread.join()

    assert failures == []


def test_file_browser_uses_text_nodes_for_untrusted_names():
    template = (
        Path(json_log_viewer.__file__).parent / "templates" / "upload.html"
    ).read_text(encoding="utf-8")

    assert "filenameSpan.textContent = item.name" in template
    assert "browserItem.addEventListener('click'" in template
    assert 'onclick="handleItemClick' not in template
    assert "contentDiv.innerHTML = html" not in template


def test_step_identifier_is_escaped_before_dynamic_rendering():
    template = (
        Path(json_log_viewer.__file__).parent / "templates" / "index.html"
    ).read_text(encoding="utf-8")

    assert "const safeDisplayStepId = escapeHTML(displayStepId)" in template
    assert "Step ${safeDisplayStepId}" in template
    assert "Step ${displayStepId}" not in template


def test_main_ignores_non_loopback_host_environment(monkeypatch, tmp_path):
    run = Mock()
    monkeypatch.setenv("JSON_LOG_VIEWER_HOST", "0.0.0.0")
    monkeypatch.setattr(json_log_viewer.app, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["json_log_viewer.py", "--safe-root", str(tmp_path)],
    )

    json_log_viewer.main()

    run.assert_called_once_with(host="127.0.0.1", port=json_log_viewer.DEFAULT_PORT)


def test_server_defaults_to_loopback_and_current_directory():
    assert json_log_viewer.DEFAULT_HOST == "127.0.0.1"
    assert json_log_viewer.DEFAULT_SAFE_ROOT == Path.cwd().resolve()
