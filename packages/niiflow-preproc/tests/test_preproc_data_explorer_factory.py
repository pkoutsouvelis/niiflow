"""Tests for :func:`get_data_explorer` and nifti-finder filter construction.

These tests cover the data-exploration factory directly. Workflow integration
with explorer configuration remains in ``test_preproc_workflows_dynamic.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nifti_finder.explorers import AllPurposeFileExplorer

from niiflow.preproc.data.explorer_factory import get_data_explorer


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Small BIDS-like dataset used to exercise pattern and filter discovery."""
    root = tmp_path / "dataset"
    layout = [
        "sub-01/anat/sub-01_T1w.nii.gz",
        "sub-01/anat/sub-01_T2w.nii.gz",
        "sub-01/anat/sub-01_T1w_seg.nii.gz",
        "sub-02/anat/sub-02_T1w.nii.gz",
        "sub-02/anat/sub-02_T2w.nii",
        "sub-02/anat/sub-02_T1w_seg.nii.gz",
        "sub-03/anat/sub-03_T1w.nii.gz",
        "README.txt",
    ]
    for rel in layout:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return root


def _list_files(explorer: AllPurposeFileExplorer, root: Path) -> list[str]:
    return sorted(path.name for path in explorer.list(root, sort=True, unique=True))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestGetDataExplorerConstruction:
    def test_returns_all_purpose_file_explorer(self) -> None:
        explorer = get_data_explorer(pattern="*.nii*")

        assert isinstance(explorer, AllPurposeFileExplorer)

    def test_none_pattern_defaults_to_glob_star(self, dataset_root: Path) -> None:
        explorer = get_data_explorer(pattern=None)

        names = _list_files(explorer, dataset_root)
        assert "README.txt" in names
        assert "sub-01_T1w.nii.gz" in names

    def test_omitted_pattern_defaults_to_glob_star(self, dataset_root: Path) -> None:
        explorer = get_data_explorer()

        names = _list_files(explorer, dataset_root)
        assert "README.txt" in names


# ---------------------------------------------------------------------------
# Pattern validation
# ---------------------------------------------------------------------------


class TestPatternValidation:
    def test_rejects_non_string_pattern(self) -> None:
        with pytest.raises(ValueError, match="`pattern` must be a string or list"):
            get_data_explorer(pattern=123)  # type: ignore[arg-type]

    def test_rejects_list_with_non_string_entries(self) -> None:
        with pytest.raises(ValueError, match="list of strings"):
            get_data_explorer(pattern=["*.nii*", 42])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Discovery behaviour
# ---------------------------------------------------------------------------


class TestFileDiscovery:
    def test_discovers_all_nifti_files_with_simple_pattern(
        self, dataset_root: Path
    ) -> None:
        explorer = get_data_explorer(pattern="*.nii*")

        assert _list_files(explorer, dataset_root) == sorted(
            [
                "sub-01_T1w.nii.gz",
                "sub-01_T2w.nii.gz",
                "sub-01_T1w_seg.nii.gz",
                "sub-02_T1w.nii.gz",
                "sub-02_T2w.nii",
                "sub-02_T1w_seg.nii.gz",
                "sub-03_T1w.nii.gz",
            ]
        )

    def test_discovers_with_bids_like_pattern(self, dataset_root: Path) -> None:
        explorer = get_data_explorer(pattern="sub-*/anat/*T1w.nii*")

        assert _list_files(explorer, dataset_root) == sorted(
            [
                "sub-01_T1w.nii.gz",
                "sub-02_T1w.nii.gz",
                "sub-03_T1w.nii.gz",
            ]
        )

    def test_multiple_patterns_are_deduplicated(self, dataset_root: Path) -> None:
        explorer = get_data_explorer(
            pattern=["*.nii*", "sub-*/anat/*T1w.nii*"],
        )

        paths = explorer.list(dataset_root, sort=True, unique=True)
        names = [path.name for path in paths]

        assert len(names) == len(set(names))
        assert names == sorted(names)
        assert names.count("sub-01_T1w.nii.gz") == 1

    def test_single_filter_excludes_segmentation_files(
        self, dataset_root: Path
    ) -> None:
        explorer = get_data_explorer(
            pattern="sub-*/anat/*T1w*.nii*",
            filter_kwargs={
                "name": "ExcludeFileRegex",
                "kwargs": {"regex": r".*_seg\.nii.*"},
            },
        )

        assert _list_files(explorer, dataset_root) == sorted(
            [
                "sub-01_T1w.nii.gz",
                "sub-02_T1w.nii.gz",
                "sub-03_T1w.nii.gz",
            ]
        )

    def test_compose_filter_combines_multiple_criteria(
        self, dataset_root: Path
    ) -> None:
        explorer = get_data_explorer(
            pattern="*.nii*",
            filter_kwargs={
                "name": "ComposeFilter",
                "kwargs": {
                    "filters": [
                        {
                            "name": "ExcludeFileRegex",
                            "kwargs": {"regex": r".*T2w.*"},
                        },
                        {
                            "name": "IncludeFileRegex",
                            "kwargs": {"regex": r".*_seg.*"},
                        },
                    ],
                    "logic": "AND",
                },
            },
        )

        assert _list_files(explorer, dataset_root) == sorted(
            [
                "sub-01_T1w_seg.nii.gz",
                "sub-02_T1w_seg.nii.gz",
            ]
        )

    def test_compose_filter_accepts_single_nested_filter_dict(
        self, dataset_root: Path
    ) -> None:
        explorer = get_data_explorer(
            pattern="*.nii*",
            filter_kwargs={
                "name": "ComposeFilter",
                "kwargs": {
                    "filters": {
                        "name": "ExcludeFileRegex",
                        "kwargs": {"regex": r".*T2w.*"},
                    },
                    "logic": "AND",
                },
            },
        )

        names = _list_files(explorer, dataset_root)
        assert all("T2w" not in name for name in names)
        assert "sub-01_T1w.nii.gz" in names

    def test_compose_filter_skips_none_entries(self, dataset_root: Path) -> None:
        explorer = get_data_explorer(
            pattern="*.nii*",
            filter_kwargs={
                "name": "ComposeFilter",
                "kwargs": {
                    "filters": [
                        None,
                        {
                            "name": "ExcludeFileRegex",
                            "kwargs": {"regex": r".*T2w.*"},
                        },
                    ],
                    "logic": "AND",
                },
            },
        )

        names = _list_files(explorer, dataset_root)
        assert all("T2w" not in name for name in names)


# ---------------------------------------------------------------------------
# Filter config validation
# ---------------------------------------------------------------------------


class TestFilterConfigValidation:
    def test_filter_entry_requires_name(self) -> None:
        with pytest.raises(ValueError, match="`name` key is required"):
            get_data_explorer(
                filter_kwargs={"kwargs": {"regex": ".*"}},  # type: ignore[typeddict-item]
            )

    def test_filter_entry_requires_kwargs(self) -> None:
        with pytest.raises(ValueError, match="`kwargs` key is required"):
            get_data_explorer(
                filter_kwargs={"name": "ExcludeFileRegex"},  # type: ignore[typeddict-item]
            )

    def test_rejects_non_mapping_filter_config(self) -> None:
        with pytest.raises(ValueError, match="Invalid type for `filters`"):
            get_data_explorer(filter_kwargs="not-a-dict")  # type: ignore[arg-type]

    def test_rejects_unknown_filter_class(self) -> None:
        with pytest.raises(ImportError, match="Filter 'NoSuchFilter' not found"):
            get_data_explorer(
                filter_kwargs={"name": "NoSuchFilter", "kwargs": {}},
            )

    def test_rejects_invalid_compose_filter_inner_type(self) -> None:
        with pytest.raises(
            ValueError, match="Invalid type for `filters` in `ComposeFilter`"
        ):
            get_data_explorer(
                filter_kwargs={
                    "name": "ComposeFilter",
                    "kwargs": {"filters": 123, "logic": "AND"},
                },
            )
