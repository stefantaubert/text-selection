import numpy as np
import pytest
from text_selection.kld.kld_iterator import get_distribution


def test_twodim_counts_empty__returns_empty_array():
  counts = np.array([[]], dtype=np.uint32)

  result = get_distribution(counts, axis=1)

  assert_result = np.array([[]], np.float64)
  assert np.array_equal(result, assert_result)
  assert result.dtype == assert_result.dtype


def test_twodim_counts_one_one__returns_len_one_array_with_one():
  counts = np.array([[1]], dtype=np.uint32)

  result = get_distribution(counts, axis=1)

  assert_result = np.array([[1.0]], np.float64)
  assert np.array_equal(result, assert_result)
  assert result.dtype == assert_result.dtype


def test_twodim_counts_one_zero__returns_len_one_array_with_nan():
  counts = np.array([[0]], dtype=np.uint32)

  result = get_distribution(counts, axis=1)

  assert len(result) == 1
  assert np.isnan(result[0, 0])
  assert result.dtype == np.float64


def test_twodim_multiple_entries__returns_correct_distributions():
  counts = np.array([[1, 2], [3, 4], [0, 0], [np.nan, np.nan], [0.4, 0.6]], dtype=np.float64)

  result = get_distribution(counts, axis=1)

  assert_result = np.array(
    [
      [1 / 3, 2 / 3],
      [3 / 7, 4 / 7],
      [np.nan, np.nan],
      [np.nan, np.nan],
      [0.4, 0.6],
    ],
    dtype=np.float64,
  )

  assert np.array_equal(result[0], assert_result[0])
  assert np.array_equal(result[1], assert_result[1])
  assert np.all(np.isnan(result[2]))
  assert np.all(np.isnan(result[3]))
  assert np.array_equal(result[4], assert_result[4])
  assert result.dtype == assert_result.dtype


def test_counts_empty__returns_empty_array():
  counts = np.array([], dtype=np.uint32)

  result = get_distribution(counts, axis=0)

  assert_result = np.array([], np.float64)
  assert np.array_equal(result, assert_result)
  assert result.dtype == assert_result.dtype


def test_counts_one_one__returns_len_one_array_with_one():
  counts = np.array([1], dtype=np.uint32)

  result = get_distribution(counts, axis=0)

  assert_result = np.array([1.0], np.float64)
  assert np.array_equal(result, assert_result)
  assert result.dtype == assert_result.dtype


def test_counts_one_zero__returns_len_one_array_with_nan():
  counts = np.array([0], dtype=np.uint32)

  result = get_distribution(counts, axis=0)

  assert len(result) == 1
  assert np.isnan(result[0])
  assert result.dtype == np.float64


def test_counts_two_zeros__returns_len_two_array_with_nan():
  counts = np.array([0, 0], dtype=np.uint32)

  result = get_distribution(counts, axis=0)

  assert len(result) == 2
  assert np.all(np.isnan(result))
  assert result.dtype == np.float64


def test_counts_one_one_and_one_zero__returns_len_two_array_with_one_and_zero():
  counts = np.array([1, 0], dtype=np.uint32)

  result = get_distribution(counts, axis=0)

  assert_result = np.array([1.0, 0], np.float64)
  assert np.array_equal(result, assert_result)
  assert result.dtype == assert_result.dtype


def test_counts_two_ones__returns_len_two_array_with_point_five_for_both():
  counts = np.array([1, 1], dtype=np.uint32)

  result = get_distribution(counts, axis=0)

  assert_result = np.array([0.5, 0.5], np.float64)
  assert np.array_equal(result, assert_result)
  assert result.dtype == assert_result.dtype
  assert np.sum(result, axis=0) == 1.0


def test_counts_one_two__returns_len_two_array_with_one_third_two_third():
  counts = np.array([1, 2], dtype=np.uint32)

  result = get_distribution(counts, axis=0)

  assert_result = np.array([1 / 3, 2 / 3], np.float64)
  assert np.array_equal(result, assert_result)
  assert result.dtype == assert_result.dtype
  assert np.sum(result, axis=0) == 1.0


def test_counts_float__returns_len_two_array_with_float():
  counts = np.array([0.4, 0.6], dtype=np.float64)

  result = get_distribution(counts, axis=0)

  assert_result = np.array([0.4, 0.6], np.float64)
  assert np.array_equal(result, assert_result)
  assert result.dtype == assert_result.dtype
  assert np.sum(result, axis=0) == 1.0


def test_counts_negative__raises_assert():
  counts = np.array([-1], dtype=np.int32)
  with pytest.raises(AssertionError):
    get_distribution(counts, axis=0)
