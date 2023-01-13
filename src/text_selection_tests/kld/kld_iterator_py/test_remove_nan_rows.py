import numpy as np

from text_selection.kld.kld_iterator import remove_nan_rows


# region empty
def test_s0_empty__returns_empty():
  qk = np.array([], dtype=np.float64)
  result = remove_nan_rows(qk, axis=0)
  np.testing.assert_array_equal(result, np.empty(shape=(0,)))


def test_s0x0_empty_returns_empty():
  qk = np.empty(shape=(0, 0), dtype=np.float64)
  result = remove_nan_rows(qk, axis=1)
  np.testing.assert_array_equal(result, np.empty(shape=(0, 0)))


def test_s1x0_empty_returns_empty():
  qk = np.array([[]], dtype=np.float64)
  result = remove_nan_rows(qk, axis=1)
  np.testing.assert_array_equal(result, np.empty(shape=(1, 0)))


def test_s2x0_empty_returns_empty():
  qk = np.array([[], []], dtype=np.float64)
  result = remove_nan_rows(qk, axis=1)
  np.testing.assert_array_equal(result, np.empty(shape=(2, 0)))
# endregion


# region normal
def test_s1_no_nan__returns_full_array():
  qk = np.array([1], dtype=np.float64)
  result = remove_nan_rows(qk, axis=0)
  np.testing.assert_array_equal(result, np.array([1]))


def test_s2_no_nan__returns_full_array():
  qk = np.array([1, 2], dtype=np.float64)
  result = remove_nan_rows(qk, axis=0)
  np.testing.assert_array_equal(result, np.array([1, 2]))


def test_s1x1_no_nan__returns_full_array():
  qk = np.array([[1]], dtype=np.float64)
  result = remove_nan_rows(qk, axis=1)
  np.testing.assert_array_equal(result, np.array([[1]]))


def test_s1x2_no_nan__returns_full_array():
  qk = np.array([[1, 2]], dtype=np.float64)
  result = remove_nan_rows(qk, axis=1)
  np.testing.assert_array_equal(result, np.array([[1, 2]]))


def test_s2x1_no_nan__returns_full_array():
  qk = np.array([[1], [2]], dtype=np.float64)
  result = remove_nan_rows(qk, axis=1)
  np.testing.assert_array_equal(result, np.array([[1], [2]]))


def test_s2x2_no_nan__returns_full_array():
  qk = np.array([[1, 2], [2, 1]], dtype=np.float64)
  result = remove_nan_rows(qk, axis=1)
  np.testing.assert_array_equal(result, np.array([[1, 2], [2, 1]]))
# endregion


# region all nan
def test_s1_all_nan__returns_empty():
  qk = np.array([np.nan], dtype=np.float64)
  result = remove_nan_rows(qk, axis=0)
  np.testing.assert_array_equal(result, np.empty(shape=(0,)))


def test_s2_all_nan__returns_all_entries():
  qk = np.array([np.nan, np.nan], dtype=np.float64)
  result = remove_nan_rows(qk, axis=0)
  np.testing.assert_array_equal(result, np.empty(shape=(0,)))


def test_s1x1_all_nan__returns_all_entries():
  qk = np.array([[np.nan]], dtype=np.float64)
  result = remove_nan_rows(qk, axis=1)
  np.testing.assert_array_equal(result, np.empty(shape=(0, 1)))


def test_s1x2_all_nan__returns_all_entries():
  qk = np.array([[np.nan, np.nan]], dtype=np.float64)
  result = remove_nan_rows(qk, axis=1)
  np.testing.assert_array_equal(result, np.empty(shape=(0, 2)))


def test_s2x1_all_nan__returns_all_entries():
  qk = np.array([[np.nan], [np.nan]], dtype=np.float64)
  result = remove_nan_rows(qk, axis=1)
  np.testing.assert_array_equal(result, np.empty(shape=(0, 1)))


def test_s2x2_all_nan__returns_all_entries():
  qk = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64)
  result = remove_nan_rows(qk, axis=1)
  np.testing.assert_array_equal(result, np.empty(shape=(0, 2)))
# endregion


# region one nan
def test_s2_one_nan__returns_all_entries():
  qk = np.array([1.0, np.nan], dtype=np.float64)
  result = remove_nan_rows(qk, axis=0)
  np.testing.assert_array_equal(result, np.array([1.0, np.nan]))


def test_s1x2_one_nan__returns_all_entries():
  qk = np.array([[1.0, np.nan]], dtype=np.float64)
  result = remove_nan_rows(qk, axis=1)
  np.testing.assert_array_equal(result, np.array([[1.0, np.nan]]))


def test_s2x2_one_nan__returns_all_entries():
  qk = np.array([[np.nan, 1], [1, np.nan]], dtype=np.float64)
  result = remove_nan_rows(qk, axis=1)
  np.testing.assert_array_equal(result, np.array([[np.nan, 1], [1, np.nan]]))
# endregion


def test_s3x3_component_test():
  qk = np.array([[1, 2, np.nan], [np.nan, np.nan, np.nan], [1, 0, 0]], dtype=np.float64)
  result = remove_nan_rows(qk, axis=1)
  np.testing.assert_array_equal(result, np.array([[1, 2, np.nan], [1, 0, 0]]))
