"""Tests for :func:`read_paths_from_file`."""

from __future__ import annotations

from pathlib import Path

import pytest

from niiflow.preproc.data import read_paths_from_file


def _write_list(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestReadPathsFromFile:
    def test_reads_existing_files_in_order(self, tmp_path: Path) -> None:
        a = tmp_path / "a.nii.gz"
        b = tmp_path / "b.nii.gz"
        a.write_bytes(b"")
        b.write_bytes(b"")
        list_file = _write_list(tmp_path / "files.txt", [str(b), str(a)])

        paths = read_paths_from_file(list_file)

        assert paths == [b.resolve(), a.resolve()]

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        a = tmp_path / "a.nii.gz"
        a.write_bytes(b"")
        list_file = _write_list(tmp_path / "files.txt", ["", str(a), "  ", ""])

        assert read_paths_from_file(list_file) == [a.resolve()]

    def test_must_exist_raises_on_missing_path(self, tmp_path: Path) -> None:
        list_file = _write_list(
            tmp_path / "files.txt",
            [str(tmp_path / "missing.nii.gz")],
        )

        with pytest.raises(ValueError, match="not an existing file"):
            read_paths_from_file(list_file, must_exist=True)

    def test_must_exist_raises_on_directory(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        list_file = _write_list(tmp_path / "files.txt", [str(sub)])

        with pytest.raises(ValueError, match="directory"):
            read_paths_from_file(list_file, must_exist=True)

    def test_must_exist_false_keeps_missing_and_directories(
        self, tmp_path: Path
    ) -> None:
        a = tmp_path / "a.nii.gz"
        a.write_bytes(b"")
        missing = tmp_path / "missing.nii.gz"
        sub = tmp_path / "subdir"
        sub.mkdir()
        list_file = _write_list(
            tmp_path / "files.txt",
            [str(missing), str(sub), str(a)],
        )

        assert read_paths_from_file(list_file, must_exist=False) == [
            missing.resolve(),
            sub.resolve(),
            a.resolve(),
        ]

    def test_resolve_false_keeps_missing_without_stat(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.nii.gz"
        list_file = _write_list(tmp_path / "files.txt", [str(missing)])

        paths = read_paths_from_file(list_file, must_exist=False, resolve=False)

        assert paths == [Path(str(missing))]

    def test_rejects_non_txt_list_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "files.csv"
        bad.write_text("x\n", encoding="utf-8")

        with pytest.raises(ValueError, match=r"\.txt"):
            read_paths_from_file(bad)

    def test_rejects_missing_list_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_paths_from_file(tmp_path / "missing.txt")

    def test_rejects_non_bool_must_exist(self, tmp_path: Path) -> None:
        list_file = _write_list(tmp_path / "files.txt", [])
        with pytest.raises(TypeError, match="must_exist"):
            read_paths_from_file(list_file, must_exist="yes")  # type: ignore[arg-type]

    def test_resolve_false_keeps_listed_form(self, tmp_path: Path) -> None:
        a = tmp_path / "a.nii.gz"
        a.write_bytes(b"")
        listed = str(a)  # absolute under tmp_path, but not .resolve()'d
        list_file = _write_list(tmp_path / "files.txt", [listed])

        paths = read_paths_from_file(list_file, resolve=False)

        assert paths == [Path(listed)]
        assert paths[0] == a

    def test_deprecated_aliases_warn_until_removed(self, tmp_path: Path) -> None:
        a = tmp_path / "a.nii.gz"
        a.write_bytes(b"")
        list_file = _write_list(tmp_path / "files.txt", [str(a)])

        with pytest.warns(DeprecationWarning, match="`strict` is deprecated"):
            assert read_paths_from_file(list_file, strict=False) == [a.resolve()]
        with pytest.warns(DeprecationWarning, match="skip_resolve_filepaths"):
            assert read_paths_from_file(list_file, skip_resolve_filepaths=True) == [
                Path(str(a))
            ]

    def test_default_resolves_listed_paths(self, tmp_path: Path) -> None:
        a = tmp_path / "a.nii.gz"
        a.write_bytes(b"")
        list_file = _write_list(tmp_path / "files.txt", [str(a)])

        paths = read_paths_from_file(list_file)

        assert paths == [a.resolve()]

    def test_rejects_non_bool_resolve(self, tmp_path: Path) -> None:
        list_file = _write_list(tmp_path / "files.txt", [])
        with pytest.raises(TypeError, match="`resolve` must be a boolean"):
            read_paths_from_file(list_file, resolve="yes")  # type: ignore[arg-type]
