"""
Regression tests for DatasetManager.

These tests are fully offline and do not require the real 5G-NIDD
or CLAIR-5G dataset.

Run:

    pytest tests/test_dataset_manager.py -v -s

The tests verify:
    - Dataset path configuration
    - Dataset existence validation
    - File validation
    - Readability validation
    - Dataset metadata generation
    - Missing dataset handling
    - Invalid dataset path handling
"""


from pathlib import Path

import pytest

from app.data.dataset_manager import DatasetManager


# ============================================================
# Initialization
# ============================================================


def test_dataset_manager_uses_default_dataset_name():
    manager = DatasetManager()

    assert manager.dataset_name == "5G-NIDD"
    assert manager.dataset_path is None


def test_dataset_manager_accepts_custom_dataset_name_and_path(
    tmp_path,
):
    dataset_path = tmp_path / "clair_5g.csv"

    manager = DatasetManager(
        dataset_name="CLAIR-5G",
        dataset_path=str(dataset_path),
    )

    assert manager.dataset_name == "CLAIR-5G"
    assert manager.dataset_path == str(dataset_path)


# ============================================================
# get_latest_dataset()
# ============================================================


def test_get_latest_dataset_returns_absolute_path(tmp_path):
    dataset_path = tmp_path / "5g_nidd.csv"
    dataset_path.write_text(
        "feature,label\n1,0\n",
        encoding="utf-8",
    )

    manager = DatasetManager(
        dataset_name="5G-NIDD",
        dataset_path=str(dataset_path),
    )

    result = manager.get_latest_dataset()

    assert result == str(dataset_path.resolve())
    assert Path(result).is_absolute()
    assert Path(result).is_file()


def test_get_latest_dataset_raises_when_no_path_configured():
    manager = DatasetManager(
        dataset_name="5G-NIDD",
    )

    with pytest.raises(
        ValueError,
        match="No dataset path configured for 5G-NIDD",
    ):
        manager.get_latest_dataset()


def test_get_latest_dataset_raises_when_dataset_does_not_exist(
    tmp_path,
):
    dataset_path = tmp_path / "missing.csv"

    manager = DatasetManager(
        dataset_name="5G-NIDD",
        dataset_path=str(dataset_path),
    )

    with pytest.raises(
        FileNotFoundError,
        match="Dataset not found",
    ):
        manager.get_latest_dataset()


def test_get_latest_dataset_raises_when_path_is_directory(
    tmp_path,
):
    dataset_directory = tmp_path / "dataset"
    dataset_directory.mkdir()

    manager = DatasetManager(
        dataset_name="5G-NIDD",
        dataset_path=str(dataset_directory),
    )

    with pytest.raises(
        ValueError,
        match="Dataset path is not a file",
    ):
        manager.get_latest_dataset()


# ============================================================
# validate_dataset()
# ============================================================


def test_validate_dataset_uses_configured_path(tmp_path):
    dataset_path = tmp_path / "5g_nidd.csv"
    dataset_path.write_text(
        "feature,label\n1,0\n",
        encoding="utf-8",
    )

    manager = DatasetManager(
        dataset_name="5G-NIDD",
        dataset_path=str(dataset_path),
    )

    assert manager.validate_dataset() is True


def test_validate_dataset_accepts_explicit_path(tmp_path):
    dataset_path = tmp_path / "clair_5g.csv"
    dataset_path.write_text(
        "feature,label\n1,0\n",
        encoding="utf-8",
    )

    manager = DatasetManager(
        dataset_name="CLAIR-5G",
    )

    assert manager.validate_dataset(
        str(dataset_path)
    ) is True


def test_validate_dataset_raises_when_no_path_is_available():
    manager = DatasetManager(
        dataset_name="CLAIR-5G",
    )

    with pytest.raises(
        ValueError,
        match="No dataset path provided for CLAIR-5G",
    ):
        manager.validate_dataset()


def test_validate_dataset_raises_when_dataset_does_not_exist(
    tmp_path,
):
    dataset_path = tmp_path / "missing.csv"

    manager = DatasetManager(
        dataset_name="5G-NIDD",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Dataset not found",
    ):
        manager.validate_dataset(str(dataset_path))


def test_validate_dataset_raises_when_path_is_directory(
    tmp_path,
):
    dataset_directory = tmp_path / "dataset"
    dataset_directory.mkdir()

    manager = DatasetManager(
        dataset_name="5G-NIDD",
    )

    with pytest.raises(
        ValueError,
        match="Dataset path is not a file",
    ):
        manager.validate_dataset(str(dataset_directory))


# ============================================================
# get_dataset_info()
# ============================================================


def test_get_dataset_info_returns_expected_metadata(tmp_path):
    dataset_path = tmp_path / "5g_nidd.csv"
    dataset_content = (
        "feature,label\n"
        "1,0\n"
        "2,1\n"
    )

    dataset_path.write_text(
        dataset_content,
        encoding="utf-8",
    )

    manager = DatasetManager(
        dataset_name="5G-NIDD",
        dataset_path=str(dataset_path),
    )

    info = manager.get_dataset_info()

    assert info["dataset_name"] == "5G-NIDD"
    assert info["dataset_path"] == str(dataset_path.resolve())
    assert info["filename"] == "5g_nidd.csv"
    assert info["file_size_bytes"] == dataset_path.stat().st_size
    assert info["file_size_mb"] == round(
        dataset_path.stat().st_size / (1024 * 1024),
        2,
    )
    assert info["last_modified"] == dataset_path.stat().st_mtime


def test_get_dataset_info_accepts_explicit_path(tmp_path):
    dataset_path = tmp_path / "clair_5g.csv"

    dataset_path.write_text(
        "feature,label\n1,0\n",
        encoding="utf-8",
    )

    manager = DatasetManager(
        dataset_name="CLAIR-5G",
    )

    info = manager.get_dataset_info(
        str(dataset_path)
    )

    assert info["dataset_name"] == "CLAIR-5G"
    assert info["dataset_path"] == str(dataset_path.resolve())
    assert info["filename"] == "clair_5g.csv"


def test_get_dataset_info_raises_when_no_dataset_is_configured():
    manager = DatasetManager(
        dataset_name="5G-NIDD",
    )

    with pytest.raises(
        ValueError,
        match="No dataset path configured for 5G-NIDD",
    ):
        manager.get_dataset_info()


def test_get_dataset_info_raises_for_missing_dataset(
    tmp_path,
):
    dataset_path = tmp_path / "missing.csv"

    manager = DatasetManager(
        dataset_name="CLAIR-5G",
        dataset_path=str(dataset_path),
    )

    with pytest.raises(
        FileNotFoundError,
        match="Dataset not found",
    ):
        manager.get_dataset_info()
