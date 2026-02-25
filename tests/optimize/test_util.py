"""Tests for aim_resolve.optimize.util — utility functions."""

import pytest

from aim_resolve.optimize.util import (
    add_dicts,
    check_dict,
    eval_list,
    eval_string,
    extend_reps,
    fun2mode,
    get_it,
    has_key,
    has_val,
    is_or_contains_type,
    merge_dicts,
    pop_key,
    pop_val,
)

# ---------- merge_dicts ----------


class TestMergeDicts:
    def test_simple(self):
        result = merge_dicts([{"a": 1}, {"b": 2}])
        assert result == {"a": 1, "b": 2}

    def test_override(self):
        result = merge_dicts([{"a": 1}, {"a": 2}])
        assert result == {"a": 2}

    def test_nested_merge(self):
        result = merge_dicts([{"x": {"a": 1}}, {"x": {"b": 2}}])
        assert result == {"x": {"a": 1, "b": 2}}

    def test_empty(self):
        result = merge_dicts([{}, {}])
        assert result == {}


# ---------- has_key / has_val ----------


class TestHasKey:
    def test_top_level(self):
        assert has_key({"a": 1}, "a") is True

    def test_nested(self):
        assert has_key({"x": {"a": 1}}, "a") is True

    def test_missing(self):
        assert has_key({"a": 1}, "b") is False


class TestHasVal:
    def test_top_level(self):
        assert has_val({"a": 1}, 1) is True

    def test_nested(self):
        assert has_val({"x": {"a": 42}}, 42) is True

    def test_missing(self):
        assert has_val({"a": 1}, 99) is False


# ---------- pop_key / pop_val ----------


class TestPopKey:
    def test_removes_key(self):
        result = pop_key({"a": 1, "b": 2}, "a")
        assert result == {"b": 2}

    def test_nested(self):
        result = pop_key({"x": {"a": 1, "b": 2}}, "a")
        assert result == {"x": {"b": 2}}


class TestPopVal:
    def test_removes_value(self):
        result = pop_val({"a": "x", "b": "y"}, "x")
        assert result == {"b": "y"}

    def test_nested(self):
        result = pop_val({"top": {"a": "x", "b": "y"}}, "x")
        assert result == {"top": {"b": "y"}}


# ---------- add_dicts ----------


class TestAddDicts:
    def test_add_scalars(self):
        result = add_dicts({"a": 1}, {"a": 2})
        assert result == {"a": 3}

    def test_add_lists(self):
        result = add_dicts({"a": [1, 2]}, {"a": [3]})
        assert result == {"a": [1, 2, 3]}

    def test_nested(self):
        result = add_dicts({"x": {"a": 1}}, {"x": {"a": 2}})
        assert result == {"x": {"a": 3}}


# ---------- is_or_contains_type ----------


class TestIsOrContainsType:
    def test_direct(self):
        assert is_or_contains_type([1, 2], list) is True

    def test_nested_dict(self):
        assert is_or_contains_type({"a": [1]}, list) is True

    def test_not_found(self):
        assert is_or_contains_type({"a": 42}, list) is False


# ---------- get_it ----------


class TestGetIt:
    def test_list(self):
        assert get_it([10, 20, 30], 1) == 20

    def test_scalar(self):
        assert get_it(42, 0) == 42

    def test_dict(self):
        result = get_it({"a": [1, 2], "b": 3}, 1)
        assert result == {"a": 2, "b": 3}


# ---------- extend_reps ----------


class TestExtendReps:
    def test_extend(self):
        result = extend_reps([1, 2], 5)
        assert result == [1, 2, 2, 2, 2]

    def test_truncate(self):
        result = extend_reps([1, 2, 3, 4], 2)
        assert result == [1, 2]

    def test_exact(self):
        result = extend_reps([1, 2, 3], 3)
        assert result == [1, 2, 3]

    def test_scalar_input(self):
        result = extend_reps(5, 3)
        assert result == [5, 5, 5]

    def test_custom_add_val(self):
        result = extend_reps([1], 4, add_val=0)
        assert result == [1, 0, 0, 0]


# ---------- eval_string ----------


class TestEvalString:
    def test_simple_list(self):
        result = eval_string("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_multiplication(self):
        result = eval_string("3*[1]")
        assert result == [1, 1, 1]

    def test_addition(self):
        result = eval_string("[1] + [2]")
        assert result == [1, 2]

    def test_string_in_list(self):
        result = eval_string("[hello, world]")
        assert result == ["hello", "world"]


class TestEvalList:
    def test_nested(self):
        result = eval_list(["3*[1]", 5])
        assert result == [[1, 1, 1], 5]


# ---------- check_dict ----------


class TestCheckDict:
    def test_valid(self):
        result = check_dict({"a": 1, "b": 2}, {"a"}, {"b"})
        assert result == {"a": 1, "b": 2}

    def test_missing_needed_raises(self):
        with pytest.raises(ValueError, match="missing"):
            check_dict({"b": 2}, {"a"}, {"b"})

    def test_removes_extra(self):
        result = check_dict({"a": 1, "c": 3}, {"a"}, set())
        assert result == {"a": 1}

    def test_none_dict(self):
        result = check_dict(None, {"a"})
        assert result is None


# ---------- fun2mode ----------


class TestFun2Mode:
    def test_lh_fast(self):
        dct = {"lh.0": {"fun": "fast_lh"}}
        result = fun2mode(dct)
        assert result["lh.0"]["mode"] == "fast"
        assert "fun" not in result["lh.0"]

    def test_lh_radio(self):
        dct = {"lh.0": {"fun": "radio_fn"}}
        result = fun2mode(dct)
        assert result["lh.0"]["mode"] == "radio"

    def test_lh_image(self):
        dct = {"lh.0": {"fun": "image_fn"}}
        result = fun2mode(dct)
        assert result["lh.0"]["mode"] == "image"

    def test_data_radio(self):
        dct = {"data.0": {"fun": "radio_data"}}
        result = fun2mode(dct)
        assert result["data.0"]["mode"] == "radio"

    def test_no_fun_key(self):
        dct = {"lh.0": {"mode": "fast"}}
        result = fun2mode(dct)
        assert result["lh.0"]["mode"] == "fast"
