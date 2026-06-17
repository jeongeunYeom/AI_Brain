from pathlib import Path

from app.core.config import PROJECT_ROOT, resolve_data_dir


def test_default_data_dir_resolves_to_project_root_data(monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)

    assert resolve_data_dir() == (PROJECT_ROOT / "data").resolve()


def test_relative_data_dir_resolves_from_project_root():
    assert resolve_data_dir("custom_data") == (PROJECT_ROOT / "custom_data").resolve()


def test_absolute_data_dir_is_preserved(tmp_path: Path):
    assert resolve_data_dir(str(tmp_path)) == tmp_path.resolve()
