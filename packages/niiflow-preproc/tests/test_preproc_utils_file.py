"""Tests for :mod:`niiflow.preproc.utils.file` JSON helpers."""

from __future__ import annotations

from pathlib import Path

from niiflow.preproc.utils.file import read_json, write_json


def test_write_json_accepts_list_root(tmp_path: Path) -> None:
    path = tmp_path / "ranges.json"
    ranges = ((3, 7), (3, 7), (3, 7))

    write_json(ranges, path)

    assert read_json(path) == [[3, 7], [3, 7], [3, 7]]


def test_write_json_dict_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "doc.json"
    payload = {"steps": [], "nested": {"count": 2}}

    write_json(payload, path)

    assert read_json(path) == payload
