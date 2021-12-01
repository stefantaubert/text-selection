import numpy as np
from text_selection.kld.kld_iterator import remove_nan_rows


def test_axis_0_empty__returns_empty():
  qk = np.array([], dtype=np.float64)
  result = remove_nan_rows(qk, axis=0)
  np.testing.assert_array_equal(result, np.empty(shape=(0), dtype=np.float64))


def test_axis_1_empty_returns_empty():
  qk = np.array([[]], dtype=np.float64)
  result = remove_nan_rows(qk, axis=1)
  np.testing.assert_array_equal(result, np.empty(shape=(0, 0), dtype=np.float64))


def test_axis_0_one_nan__returns_full_array():
  qk = np.array([1, np.nan], dtype=np.float64)
  result = remove_nan_rows(qk, axis=0)
  np.testing.assert_array_equal(result, np.array([1, np.nan], dtype=np.float64))


def test_axis_0_full_nan__returns_empty():
  qk = np.array([np.nan, np.nan], dtype=np.float64)
  result = remove_nan_rows(qk, axis=0)

  np.testing.assert_array_equal(result, np.empty(shape=(0), dtype=np.float64))


def test_axis_1_full_nan_rows__returns_full_nan_rows_removed():
  qk = np.array([[1, 2, np.nan], [np.nan, np.nan, np.nan], [1, 0, 0]], dtype=np.float64)
  result = remove_nan_rows(qk, axis=1)
  np.testing.assert_array_equal(result, np.array([[1, 2, np.nan], [1, 0, 0]], dtype=np.float64))
