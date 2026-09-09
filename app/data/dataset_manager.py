"""
Dataset Manager

Provides a single interface for resolving the dataset used by
the automated experiment pipeline.

Current implementation:
    - Supports the local/offline 5G-NIDD dataset.
    - Validates that the dataset exists and is readable.

Future implementation:
    - Can be extended to retrieve the latest validated dataset
      from the CLAIR-5G online data source.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class DatasetManager:
    """
    Manage dataset selection for experiments.

    The experiment pipeline should use this class instead of
    directly hardcoding a dataset path.
    """

    def __init__(
        self,
        dataset_name: str = "5G-NIDD",
        dataset_path: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

    # ========================================================
    # Dataset Resolution
    # ========================================================

    def get_latest_dataset(self) -> str:
        """
        Return the dataset that should be used for the experiment.

        Current behaviour:
            Uses the configured local dataset path.

        Future behaviour:
            This method can be extended to query the CLAIR-5G
            data source and return the latest valid dataset.

        Returns:
            Absolute path to the dataset.

        Raises:
            FileNotFoundError:
                If the configured dataset does not exist.

            ValueError:
                If no dataset path has been configured.
        """

        if not self.dataset_path:
            raise ValueError(
                f"No dataset path configured for {self.dataset_name}."
            )

        dataset_path = Path(self.dataset_path).expanduser().resolve()

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}"
            )

        if not dataset_path.is_file():
            raise ValueError(
                f"Dataset path is not a file: {dataset_path}"
            )

        return str(dataset_path)

    # ========================================================
    # Dataset Validation
    # ========================================================

    def validate_dataset(self, dataset_path: Optional[str] = None) -> bool:
        """
        Validate that a dataset exists and is a readable file.

        Args:
            dataset_path:
                Optional dataset path. If omitted, the configured
                dataset path is used.

        Returns:
            True if the dataset is valid.

        Raises:
            FileNotFoundError:
                If the dataset does not exist.

            ValueError:
                If the path does not point to a file.
        """

        path = dataset_path or self.dataset_path

        if not path:
            raise ValueError(
                f"No dataset path provided for {self.dataset_name}."
            )

        dataset_file = Path(path).expanduser().resolve()

        if not dataset_file.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_file}"
            )

        if not dataset_file.is_file():
            raise ValueError(
                f"Dataset path is not a file: {dataset_file}"
            )

        if not os.access(dataset_file, os.R_OK):
            raise PermissionError(
                f"Dataset is not readable: {dataset_file}"
            )

        return True

    # ========================================================
    # Dataset Information
    # ========================================================

    def get_dataset_info(
        self,
        dataset_path: Optional[str] = None,
    ) -> dict:
        """
        Return basic metadata about the selected dataset.

        This does not load the entire dataset into memory.

        Returns:
            Dictionary containing dataset metadata.
        """

        path = dataset_path or self.get_latest_dataset()
        dataset_file = Path(path).expanduser().resolve()

        self.validate_dataset(str(dataset_file))

        return {
            "dataset_name": self.dataset_name,
            "dataset_path": str(dataset_file),
            "filename": dataset_file.name,
            "file_size_bytes": dataset_file.stat().st_size,
            "file_size_mb": round(
                dataset_file.stat().st_size / (1024 * 1024),
                2,
            ),
            "last_modified": dataset_file.stat().st_mtime,
        }