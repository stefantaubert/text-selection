import numpy as np
import pytest
from text_selection.kld.kld_iterator import get_distribution


def test_axis_1_empty__returns_empty_array():
  counts = np.array([[]], dtype=np.uint32)

  result = get_distribution(counts, axis=1)

  assert_result = np.array([[]], np.float64)
  assert np.array_equal(result, assert_result)
  assert result.dtype == assert_result.dtype


def test_axis_1_one_one__returns_len_one_array_with_one():
  counts = np.array([[1]], dtype=np.uint32)

  result = get_distribution(counts, axis=1)

  assert_result = np.array([[1.0]], np.float64)
  assert np.array_equal(result, assert_result)
  assert result.dtype == assert_result.dtype


def test_axis_1_one_zero__returns_len_one_array_with_nan():
  counts = np.array([[0]], dtype=np.uint32)

  result = get_distribution(counts, axis=1)

  assert len(result) == 1
  assert np.isnan(result[0, 0])
  assert result.dtype == np.float64


def test_axis_1_multiple_entries__returns_correct_distributions():
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


def test_axis_0_empty__returns_empty():
  counts = np.array([], dtype=np.uint32)
  result = get_distribution(counts, axis=0)
  np.testing.assert_array_equal(result, np.empty(shape=(0)))


def test_axis_0_one__returns_one():
  counts = np.array([1], dtype=np.uint32)
  result = get_distribution(counts, axis=0)
  np.testing.assert_array_equal(result, np.array([1.0]))


def test_axis_0_zero__returns_nan():
  counts = np.array([0], dtype=np.uint32)
  result = get_distribution(counts, axis=0)
  np.testing.assert_array_equal(result, np.array([np.nan]))


def test_axis_0_two_zeros__returns_two_nans():
  counts = np.array([0, 0], dtype=np.uint32)
  result = get_distribution(counts, axis=0)
  np.testing.assert_array_equal(result, np.array([np.nan, np.nan]))


def test_axis_0_two_ones__returns_point_five_for_both():
  counts = np.array([1, 1], dtype=np.uint32)
  result = get_distribution(counts, axis=0)
  np.testing.assert_array_equal(result, np.array([0.5, 0.5]))


def test_axis_0_one_and_two__returns_len_two_array_with_one_third_two_third():
  counts = np.array([1, 2], dtype=np.uint32)
  result = get_distribution(counts, axis=0)
  np.testing.assert_array_equal(result, np.array([1 / 3, 2 / 3]))


def test_axis_0_weights__returns_normalized_weights():
  weights = np.array([0.8, 1.2], dtype=np.float64)
  result = get_distribution(weights, axis=0)
  np.testing.assert_array_equal(result, np.array([0.4, 0.6]))
