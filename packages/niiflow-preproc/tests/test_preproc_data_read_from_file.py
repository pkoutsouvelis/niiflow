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

    def test_strict_raises_on_missing_path(self, tmp_path: Path) -> None:
        list_file = _write_list(
            tmp_path / "files.txt",
            [str(tmp_path / "missing.nii.gz")],
        )

        with pytest.raises(ValueError, match="not an existing file"):
            read_paths_from_file(list_file, strict=True)

    def test_strict_raises_on_directory(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        list_file = _write_list(tmp_path / "files.txt", [str(sub)])

        with pytest.raises(ValueError, match="directory"):
            read_paths_from_file(list_file, strict=True)

    def test_non_strict_skips_missing_and_directories(self, tmp_path: Path) -> None:
        a = tmp_path / "a.nii.gz"
        a.write_bytes(b"")
        sub = tmp_path / "subdir"
        sub.mkdir()
        list_file = _write_list(
            tmp_path / "files.txt",
            [str(tmp_path / "missing.nii.gz"), str(sub), str(a)],
        )

        assert read_paths_from_file(list_file, strict=False) == [a.resolve()]

    def test_rejects_non_txt_list_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "files.csv"
        bad.write_text("x\n", encoding="utf-8")

        with pytest.raises(ValueError, match=r"\.txt"):
            read_paths_from_file(bad)

    def test_rejects_missing_list_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_paths_from_file(tmp_path / "missing.txt")

    def test_rejects_non_bool_strict(self, tmp_path: Path) -> None:
        list_file = _write_list(tmp_path / "files.txt", [])
        with pytest.raises(TypeError, match="strict"):
            read_paths_from_file(list_file, strict="yes")  # type: ignore[arg-type]

    def test_skip_resolve_filepaths_keeps_listed_form(self, tmp_path: Path) -> None:
        a = tmp_path / "a.nii.gz"
        a.write_bytes(b"")
        listed = str(a)  # absolute under tmp_path, but not .resolve()'d
        list_file = _write_list(tmp_path / "files.txt", [listed])

        paths = read_paths_from_file(list_file, skip_resolve_filepaths=True)

        assert paths == [Path(listed)]
        assert paths[0] == a

    def test_default_resolves_listed_paths(self, tmp_path: Path) -> None:
        a = tmp_path / "a.nii.gz"
        a.write_bytes(b"")
        list_file = _write_list(tmp_path / "files.txt", [str(a)])

        paths = read_paths_from_file(list_file)

        assert paths == [a.resolve()]

    def test_rejects_non_bool_skip_resolve_filepaths(self, tmp_path: Path) -> None:
        list_file = _write_list(tmp_path / "files.txt", [])
        with pytest.raises(TypeError, match="skip_resolve_filepaths"):
            read_paths_from_file(
                list_file, skip_resolve_filepaths="yes"  # type: ignore[arg-type]
            )
