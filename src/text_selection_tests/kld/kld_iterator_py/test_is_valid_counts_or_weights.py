import numpy as np

from text_selection.kld.kld_iterator import is_valid_counts_or_weights


# region empty
def test_s0_empty__returns_true():
  counts = np.array([], dtype=np.int32)
  result = is_valid_counts_or_weights(counts, axis=0)
  assert result


def test_s0x0_empty__returns_true():
  counts = np.empty(shape=(0, 0), dtype=np.int32)
  result = is_valid_counts_or_weights(counts, axis=1)
  assert result


def test_s1x0_empty__returns_true():
  counts = np.array([[]], dtype=np.int32)
  result = is_valid_counts_or_weights(counts, axis=1)
  assert result


def test_s2x0_empty__returns_true():
  counts = np.array([[], []], dtype=np.int32)
  result = is_valid_counts_or_weights(counts, axis=1)
  assert result
# endregion


# region normal
def test_s1_normal__returns_true():
  counts = np.array([1], dtype=np.int32)
  result = is_valid_counts_or_weights(counts, axis=0)
  assert result


def test_s2_normal__returns_true():
  counts = np.array([1, ], dtype=np.int32)
  result = is_valid_counts_or_weights(counts, axis=0)
  assert result


def test_s1x1_normal__returns_true():
  counts = np.array([[1]], dtype=np.int32)
  result = is_valid_counts_or_weights(counts, axis=1)
  assert result


def test_s1x2_normal__returns_true():
  counts = np.array([[1, 2]], dtype=np.int32)
  result = is_valid_counts_or_weights(counts, axis=1)
  assert result


def test_s2x1_normal__returns_true():
  counts = np.array([[2], [1]], dtype=np.int32)
  result = is_valid_counts_or_weights(counts, axis=1)
  assert result


def test_s2x2_normal__returns_true():
  counts = np.array([[2, 1], [3, 0]], dtype=np.int32)
  result = is_valid_counts_or_weights(counts, axis=1)
  assert result
# endregion


# region negative
def test_s1_negative__returns_false():
  counts = np.array([-1], dtype=np.int32)
  result = is_valid_counts_or_weights(counts, axis=0)
  assert not result


def test_s2_negative__returns_false():
  counts = np.array([-1, 0], dtype=np.int32)
  result = is_valid_counts_or_weights(counts, axis=0)
  assert not result


def test_s1x1_negative__returns_false():
  counts = np.array([[-1]], dtype=np.int32)
  result = is_valid_counts_or_weights(counts, axis=1)
  assert not result


def test_s1x2_negative__returns_false():
  counts = np.array([[-1, 3]], dtype=np.int32)
  result = is_valid_counts_or_weights(counts, axis=1)
  assert not result


def test_s2x1_negative__returns_false():
  counts = np.array([[-1], [5]], dtype=np.int32)
  result = is_valid_counts_or_weights(counts, axis=1)
  assert not result


def test_s2x2_negative__returns_false():
  counts = np.array([[-1, 3], [4, 5]], dtype=np.int32)
  result = is_valid_counts_or_weights(counts, axis=1)
  assert not result
# endregion


# region nan
def test_s1_nan__returns_false():
  counts = np.array([np.nan], dtype=np.float64)
  result = is_valid_counts_or_weights(counts, axis=0)
  assert not result


def test_s2_nan__returns_false():
  counts = np.array([2, np.nan], dtype=np.float64)
  result = is_valid_counts_or_weights(counts, axis=0)
  assert not result


def test_s1x1_nan__returns_false():
  counts = np.array([[np.nan]], dtype=np.float64)
  result = is_valid_counts_or_weights(counts, axis=1)
  assert not result


def test_s1x2_nan__returns_false():
  counts = np.array([[2, np.nan]], dtype=np.float64)
  result = is_valid_counts_or_weights(counts, axis=1)
  assert not result


def test_s2x1_one_nan__returns_false():
  counts = np.array([[1], [np.nan]], dtype=np.float64)
  result = is_valid_counts_or_weights(counts, axis=1)
  assert not result


def test_s2x2_one_nan__returns_false():
  counts = np.array([[1, 2], [2, np.nan]], dtype=np.float64)
  result = is_valid_counts_or_weights(counts, axis=1)
  assert not result
# endregion
