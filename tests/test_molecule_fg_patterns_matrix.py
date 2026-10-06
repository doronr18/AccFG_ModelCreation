import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / 'molecule-fg-data'
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from bernoulli_mixture_clustering import patterns_to_matrix  # noqa: E402

EXPECTED = np.array([[0, 1, 0, 1], [1, 1, 1, 0], [0, 0, 0, 0]], dtype=np.int8)


def test_string_list():
    out = patterns_to_matrix(['0101', '1110', '0000'])
    assert out.dtype == np.int8
    assert np.array_equal(out, EXPECTED)


def test_numpy_array_of_strings_keeps_every_row():
    out = patterns_to_matrix(np.array(['0101', '1110', '0000']))
    assert np.array_equal(out, EXPECTED)


def test_numeric_matrix_passes_through():
    assert np.array_equal(patterns_to_matrix(EXPECTED.astype(np.uint8)), EXPECTED)


def test_ragged_or_non_binary_patterns_are_rejected():
    with pytest.raises(ValueError):
        patterns_to_matrix(['010', '01'])
    with pytest.raises(ValueError):
        patterns_to_matrix(['012'])
