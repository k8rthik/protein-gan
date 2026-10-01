"""Tests for the deterministic, structure-level train/validation split."""

from __future__ import annotations

import pytest

from folduzz.errors import InvalidInputError
from folduzz.splits import assign_split, split_counts


class TestAssignSplit:
    def test_is_deterministic(self):
        assert assign_split("1ABC") == assign_split("1ABC")

    def test_is_case_insensitive(self):
        assert assign_split("1abc") == assign_split("1ABC")

    def test_zero_fraction_is_all_train(self):
        assert all(assign_split(f"{i}ABC", val_fraction=0.0) == "train" for i in range(10))

    def test_one_fraction_is_all_val(self):
        assert all(assign_split(f"{i}ABC", val_fraction=1.0) == "val" for i in range(10))

    def test_fraction_is_roughly_honoured(self):
        ids = [f"{i:04x}" for i in range(4000)]
        val = sum(assign_split(i, val_fraction=0.15) == "val" for i in ids)
        assert 0.12 < val / len(ids) < 0.18

    def test_salt_changes_assignment(self):
        ids = [f"{i:04x}" for i in range(400)]
        a = [assign_split(i, salt="a") for i in ids]
        b = [assign_split(i, salt="b") for i in ids]
        assert a != b

    @pytest.mark.parametrize("fraction", [-0.1, 1.5])
    def test_rejects_bad_fraction(self, fraction):
        with pytest.raises(InvalidInputError):
            assign_split("1ABC", val_fraction=fraction)

    def test_rejects_empty_id(self):
        with pytest.raises(InvalidInputError):
            assign_split("")


class TestSplitCounts:
    def test_counts_both_splits(self):
        counts = split_counts(["1ABC", "2DEF", "3GHI"], val_fraction=1.0)
        assert counts == {"train": 0, "val": 3}
