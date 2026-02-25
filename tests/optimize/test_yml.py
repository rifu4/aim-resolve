"""Tests for aim_resolve.optimize.yml — YAML load/save utilities."""

import os
import pytest

from aim_resolve.optimize.yml import yaml_load, yaml_save, get_vals


class TestGetVals:
    def test_replaces_spaces(self):
        result = get_vals({"hello world": 1, "foo bar": {"baz qux": 2}})
        assert result == {"hello_world": 1, "foo_bar": {"baz_qux": 2}}

    def test_no_spaces(self):
        result = get_vals({"key": "value"})
        assert result == {"key": "value"}


class TestYamlSaveLoad:
    def test_roundtrip(self, tmp_path):
        data = {"alpha": 1, "beta": {"gamma": [1, 2, 3]}}
        fpath = str(tmp_path / "test.yml")
        yaml_save(data, fpath)
        loaded = yaml_load(fpath)
        assert loaded["alpha"] == 1
        assert loaded["beta"]["gamma"] == [1, 2, 3]

    def test_list_of_dicts(self, tmp_path):
        data = [{"a": 1}, {"b": 2}]
        fpath = str(tmp_path / "multi.yml")
        yaml_save(data, fpath)
        loaded = yaml_load(fpath)
        assert "a" in loaded
        assert "b" in loaded

    def test_load_missing_raises(self):
        with pytest.raises(RuntimeError):
            yaml_load("/nonexistent/file.yml")

    def test_save_invalid_raises(self, tmp_path):
        fpath = str(tmp_path / "bad.yml")
        with pytest.raises(TypeError):
            yaml_save(["not_a_dict"], fpath)
