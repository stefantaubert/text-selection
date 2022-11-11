import numpy as np

from text_selection_core.types import Dataset
from text_selection_core.validation import ErrorType, ValidationErr
from text_selection_core.weights.calculation import get_from_lines


def test_parse_as_float():
  ds = Dataset(3, "Basis")
  result = get_from_lines(ds, ["1", "2.5", "0"])
  np.testing.assert_array_equal(result, np.array([1.0, 2.5, 0.0], dtype=np.float16))
  assert result.dtype == np.float16


def test_parse_as_int():
  ds = Dataset(3, "Basis")
  result = get_from_lines(ds, ["1", "2", "0"])
  np.testing.assert_array_equal(result, np.array([1.0, 2.5, 0.0], dtype=np.uint8))
  assert result.dtype == np.uint8


def test_negative_number_returns_validation_error():
  ds = Dataset(1, "Basis")
  result = get_from_lines(ds, ["-1"])
  assert isinstance(result, ValidationErr)
  assert result.error_type == ErrorType.WEIGHTS_INVALID


def test_text_returns_validation_error():
  ds = Dataset(1, "Basis")
  result = get_from_lines(ds, ["abc"])
  assert isinstance(result, ValidationErr)
  assert result.error_type == ErrorType.WEIGHTS_INVALID
