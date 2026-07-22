"""Tests for :func:`get_data_explorer` and nifti-finder filter construction.

These tests cover the data-exploration factory directly. Workflow integration
with explorer configuration remains in ``test_preproc_workflows_dynamic.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nifti_finder.explorers import FileFinder

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


def _list_files(explorer: FileFinder, root: Path) -> list[str]:
    return sorted(path.name for path in explorer.list(root, sort=True, unique=True))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestGetDataExplorerConstruction:
    def test_returns_file_finder(self) -> None:
        explorer = get_data_explorer(patterns="*.nii*")

        assert isinstance(explorer, FileFinder)

    def test_omitted_patterns_defaults_to_nifti_glob(self, dataset_root: Path) -> None:
        explorer = get_data_explorer()

        names = _list_files(explorer, dataset_root)
        assert "README.txt" not in names
        assert "sub-01_T1w.nii.gz" in names

    def test_levels_none_enables_flat_recursive_search(
        self, dataset_root: Path
    ) -> None:
        explorer = get_data_explorer(patterns="*.nii*", levels=None)

        assert "sub-01_T1w.nii.gz" in _list_files(explorer, dataset_root)

    def test_levels_restricts_traversal(self, dataset_root: Path) -> None:
        explorer = get_data_explorer(
            patterns="*.nii*",
            levels={"subjects": "sub-*", "anat": "anat"},
        )

        names = _list_files(explorer, dataset_root)
        assert "sub-01_T1w.nii.gz" in names
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestArgumentValidation:
    def test_rejects_non_string_patterns(self) -> None:
        with pytest.raises(ValueError, match="`patterns` must be a string or sequence"):
            get_data_explorer(patterns=123)  # type: ignore[arg-type]

    def test_rejects_list_with_non_string_entries(self) -> None:
        with pytest.raises(ValueError, match="sequence of strings"):
            get_data_explorer(patterns=["*.nii*", 42])  # type: ignore[list-item]

    def test_rejects_empty_levels_mapping(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            get_data_explorer(levels={})

    def test_rejects_non_mapping_levels(self) -> None:
        with pytest.raises(ValueError, match="`levels` must be a dictionary or None"):
            get_data_explorer(levels=["sub-*"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Discovery behaviour
# ---------------------------------------------------------------------------


class TestFileDiscovery:
    def test_discovers_all_nifti_files_with_simple_patterns(
        self, dataset_root: Path
    ) -> None:
        explorer = get_data_explorer(patterns="*.nii*")

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

    def test_discovers_with_bids_like_patterns(self, dataset_root: Path) -> None:
        explorer = get_data_explorer(patterns="sub-*/anat/*T1w.nii*")

        assert _list_files(explorer, dataset_root) == sorted(
            [
                "sub-01_T1w.nii.gz",
                "sub-02_T1w.nii.gz",
                "sub-03_T1w.nii.gz",
            ]
        )

    def test_multiple_patterns_are_deduplicated(self, dataset_root: Path) -> None:
        explorer = get_data_explorer(
            patterns=["*.nii*", "sub-*/anat/*T1w.nii*"],
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
            patterns="sub-*/anat/*T1w*.nii*",
            filters={
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
            patterns="*.nii*",
            filters={
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
            patterns="*.nii*",
            filters={
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
            patterns="*.nii*",
            filters={
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
                filters={"kwargs": {"regex": ".*"}},  # type: ignore[typeddict-item]
            )

    def test_filter_entry_requires_kwargs(self) -> None:
        with pytest.raises(ValueError, match="`kwargs` key is required"):
            get_data_explorer(
                filters={"name": "ExcludeFileRegex"},  # type: ignore[typeddict-item]
            )

    def test_rejects_non_mapping_filter_config(self) -> None:
        with pytest.raises(ValueError, match="Invalid type for `filters`"):
            get_data_explorer(filters="not-a-dict")  # type: ignore[arg-type]

    def test_rejects_unknown_filter_class(self) -> None:
        with pytest.raises(ImportError, match="Filter 'NoSuchFilter' not found"):
            get_data_explorer(
                filters={"name": "NoSuchFilter", "kwargs": {}},
            )

    def test_rejects_invalid_compose_filter_inner_type(self) -> None:
        with pytest.raises(
            ValueError, match="Invalid type for `filters` in `ComposeFilter`"
        ):
            get_data_explorer(
                filters={
                    "name": "ComposeFilter",
                    "kwargs": {"filters": 123, "logic": "AND"},
                },
            )
