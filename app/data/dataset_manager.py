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

import io
import os
import re
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd


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
        dataset_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_url = dataset_url
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir
            else Path("data").expanduser().resolve()
        )

    # ========================================================
    # Dataset Resolution
    # ========================================================

    @staticmethod
    def _is_remote_dataset(value: Optional[str]) -> bool:
        if not value:
            return False

        parsed = urlparse(str(value).strip())
        return parsed.scheme in {"http", "https"}

    @staticmethod
    def _normalise_remote_filename(
        dataset_name: str,
        dataset_url: str,
    ) -> str:
        parsed = urlparse(dataset_url)
        candidate = Path(parsed.path).name

        if not candidate:
            candidate = f"{dataset_name}.csv"

        safe_name = re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            candidate,
        ).strip("._")

        if not safe_name:
            safe_name = f"{dataset_name}.csv"

        if not safe_name.lower().endswith((".csv", ".parquet", ".tsv", ".json")):
            safe_name = f"{safe_name}.csv"

        return safe_name

    def _download_remote_dataset(
        self,
        dataset_url: str,
        destination: Optional[str] = None,
    ) -> str:
        url = str(dataset_url).strip()

        if not url:
            raise ValueError(
                "A dataset URL is required to download a remote dataset."
            )

        if not self._is_remote_dataset(url):
            raise ValueError(
                f"Dataset source is not a remote URL: {url}"
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        file_name = self._normalise_remote_filename(
            self.dataset_name,
            url,
        )
        download_path = (
            Path(destination).expanduser().resolve()
            if destination is not None
            else (self.cache_dir / file_name).resolve()
        )

        download_path.parent.mkdir(parents=True, exist_ok=True)

        request = Request(
            url,
            headers={
                "User-Agent": "AI-Co-Scientist/1.0",
            },
        )

        with urlopen(request, timeout=60) as response:
            payload = response.read()

        if file_name.lower().endswith(".zip") or url.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                csv_candidates = [
                    member for member in archive.namelist()
                    if member.lower().endswith((".csv", ".parquet", ".tsv", ".json"))
                ]
                if not csv_candidates:
                    raise ValueError(
                        "Remote dataset archive does not contain a CSV-like dataset."
                    )

                extracted = archive.read(csv_candidates[0])
                download_path = download_path.with_suffix(".csv")
                download_path.write_bytes(extracted)
        else:
            download_path.write_bytes(payload)

        self.dataset_path = str(download_path)
        return str(download_path)

    def get_latest_dataset(
        self,
        dataset_path: Optional[str] = None,
        dataset_url: Optional[str] = None,
    ) -> str:
        """
        Return the dataset that should be used for the experiment.

        The source may be a local file path, a remote HTTP(S) URL, or a
        configured default path.
        """
        candidate_path = dataset_path if dataset_path is not None else self.dataset_path
        candidate_url = dataset_url if dataset_url is not None else self.dataset_url

        if candidate_path is not None and self._is_remote_dataset(candidate_path):
            return self._download_remote_dataset(str(candidate_path))

        if candidate_url is not None and self._is_remote_dataset(candidate_url):
            return self._download_remote_dataset(str(candidate_url))

        if candidate_path is None:
            if candidate_url is None:
                raise ValueError(
                    f"No dataset path configured for {self.dataset_name}."
                )
            return self._download_remote_dataset(str(candidate_url))

        dataset_file = Path(candidate_path).expanduser().resolve()

        if not dataset_file.exists():
            if candidate_url is not None and self._is_remote_dataset(candidate_url):
                return self._download_remote_dataset(
                    str(candidate_url),
                    str(dataset_file),
                )
            raise FileNotFoundError(
                f"Dataset not found: {dataset_file}"
            )

        if not dataset_file.is_file():
            raise ValueError(
                f"Dataset path is not a file: {dataset_file}"
            )

        return str(dataset_file)

    # ========================================================
    # Dataset Validation
    # ========================================================

    def validate_dataset(self, dataset_path: Optional[str] = None) -> bool:
        """
        Validate that a dataset exists and is a readable file.

        Args:
            dataset_path:
                Optional dataset path or remote dataset URL. If omitted, the
                configured dataset source is used.

        Returns:
            True if the dataset is valid.

        Raises:
            FileNotFoundError:
                If the dataset does not exist.

            ValueError:
                If the path does not point to a file.
        """

        path = dataset_path or self.dataset_path or self.dataset_url

        if not path:
            raise ValueError(
                f"No dataset path provided for {self.dataset_name}."
            )

        if self._is_remote_dataset(path):
            resolved_path = self.get_latest_dataset(
                dataset_path=None,
                dataset_url=str(path),
            )
            dataset_file = Path(resolved_path).expanduser().resolve()
        else:
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


    # ========================================================
    # Dataset Schema Inspection
    # ========================================================

    def inspect_schema(
        self,
        dataset_path: Optional[str] = None,
        sample_rows: int = 1000,
    ) -> dict:
        """
        Inspect the dataset schema without loading the entire
        dataset into memory.
        The method is dataset-agnostic and does not assume a
        fixed target-column name. It provides schema information
        that can be passed to the AI/code-generation agent.
        """

        path = dataset_path or self.get_latest_dataset()

        self.validate_dataset(path)

        dataset_file = Path(path).expanduser().resolve()

        df = pd.read_csv(
            dataset_file,
            nrows=sample_rows,
        )

        columns = df.columns.tolist()

        column_types = {
            column: str(df[column].dtype)
            for column in columns
        }

        numeric_columns = [
            column
            for column in columns
            if pd.api.types.is_numeric_dtype(df[column])
        ]

        categorical_columns = [
            column
            for column in columns
            if not pd.api.types.is_numeric_dtype(df[column])
        ]

        target_candidates = []

        for column in columns:
            series = df[column]

            unique_count = series.nunique(dropna=True)
            candidate_reasons = []

            if not pd.api.types.is_numeric_dtype(series):
                candidate_reasons.append("categorical")

            if unique_count <= 50:
                candidate_reasons.append(
                    f"low_cardinality_{unique_count}"
                )

            normalized_name = (
                str(column)
                .strip()
                .lower()
            )

            label_keywords = (
                "label",
                "target",
                "class",
                "category",
                "attack",
            )

            if any(
                keyword in normalized_name
                for keyword in label_keywords
            ):
                candidate_reasons.append(
                    "label_like_column_name"
                )

            if candidate_reasons:
                value_counts = series.value_counts(dropna=True)

                sample_values = [
                    value.item() if hasattr(value, "item") else value
                    for value in value_counts.index.tolist()[:10]
                ]

                candidate = {
                    "column": column,
                    "dtype": str(series.dtype),
                    "unique_values": unique_count,
                    "sample_values": sample_values,
                    "sample_value_counts": {
                        str(key): int(value)
                        for key, value in value_counts.head(10).items()
                    },
                    "missing_count": int(series.isna().sum()),
                    "reasons": candidate_reasons,
                }

                target_candidates.append(candidate)

        return {
            "dataset_name": self.dataset_name,
            "dataset_path": str(dataset_file),
            "row_count_sampled": len(df),
            "columns": columns,
            "column_types": column_types,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "target_candidates": target_candidates,
        }