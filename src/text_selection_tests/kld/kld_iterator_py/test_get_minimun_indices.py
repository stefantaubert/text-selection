
import numpy as np

from text_selection.kld.kld_iterator import get_minimum_indices


def test_one_entry__returns_zero():
  array = np.array([1.2], dtype=np.float64)
  min_value, min_indices = get_minimum_indices(array)
  assert min_value == 1.2
  np.testing.assert_array_equal(min_indices, np.array([0]))


def test_two_same_entries__returns_zero_and_one():
  array = np.array([1.2, 1.2], dtype=np.float64)
  min_value, min_indices = get_minimum_indices(array)
  assert min_value == 1.2
  np.testing.assert_array_equal(min_indices, np.array([0, 1]))


def test_two_different_entries__returns_min():
  array = np.array([1, 0.2], dtype=np.float64)
  min_value, min_indices = get_minimum_indices(array)
  assert min_value == 0.2
  np.testing.assert_array_equal(min_indices, np.array([1]))


def test_two_same_entries_with_one_different_entry__returns_min():
  array = np.array([0.2, 1, 0.2], dtype=np.float64)
  min_value, min_indices = get_minimum_indices(array)
  assert min_value == 0.2
  np.testing.assert_array_equal(min_indices, np.array([0, 2]))


def test_inf_and_one__returns_min():
  array = np.array([np.inf, 1.2], dtype=np.float64)
  min_value, min_indices = get_minimum_indices(array)
  assert min_value == 1.2
  np.testing.assert_array_equal(min_indices, np.array([1]))


def test_only_inf__returns_min():
  array = np.array([np.inf, np.inf], dtype=np.float64)
  min_value, min_indices = get_minimum_indices(array)
  assert np.isinf(min_value)
  np.testing.assert_array_equal(min_indices, np.array([0, 1]))
