from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_arena import _resolve_mlflow_tracking_uri


def test_resolve_mlflow_tracking_uri_prefers_cli_value(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    uri = _resolve_mlflow_tracking_uri(tmp_path, "http://localhost:5000")

    assert uri == "http://localhost:5000"


@pytest.mark.parametrize("env_uri", [None, "http://mlflow:5000", "http://mlflow:5000/"])
def test_resolve_mlflow_tracking_uri_defaults_to_local_sqlite(
    tmp_path: Path,
    monkeypatch,
    env_uri: str | None,
) -> None:
    if env_uri is None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    else:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", env_uri)

    uri = _resolve_mlflow_tracking_uri(tmp_path, None)

    assert uri == f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"


def test_resolve_mlflow_tracking_uri_honors_non_docker_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    uri = _resolve_mlflow_tracking_uri(tmp_path, None)

    assert uri == "http://localhost:5000"
