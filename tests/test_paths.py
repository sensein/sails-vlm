"""Path resolution: env var wins, config file second, clear error last."""
import pytest

from sails_vlm import paths


def test_env_var_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("SAILS_DATA_ROOT", str(tmp_path))
    assert paths.data_root() == tmp_path


def test_config_file_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("SAILS_DATA_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sails-vlm.yaml").write_text(f"data_root: {tmp_path}\n")
    assert paths.data_root() == tmp_path


def test_missing_both_raises_with_instructions(tmp_path, monkeypatch):
    monkeypatch.delenv("SAILS_DATA_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        paths.data_root()
    assert "SAILS_DATA_ROOT" in str(exc.value)
    assert "sails-vlm.yaml" in str(exc.value)


def test_interpolate_nested(monkeypatch, tmp_path):
    monkeypatch.setenv("SAILS_DATA_ROOT", str(tmp_path))
    cfg = {
        "data": {"video_dir": "${SAILS_DATA_ROOT}/rmm/clips"},
        "task": {"labels": ["rocking", "jumping"]},
        "output": {"save_dir": "${SAILS_DATA_ROOT}/out"},
    }
    out = paths.interpolate(cfg)
    assert out["data"]["video_dir"] == f"{tmp_path}/rmm/clips"
    assert out["task"]["labels"] == ["rocking", "jumping"]  # non-path strings untouched
