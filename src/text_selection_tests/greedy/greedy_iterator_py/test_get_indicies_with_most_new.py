import numpy as np

from text_selection.greedy.greedy_iterator import get_indices_with_most_new


def test_4x3_componenttest():
  covered_counts = np.array(
    [0, 1, 1]
  )
  data = np.array([
    [0, 1, 1],
    [1, 1, 0],
    [0, 1, 0],
    [1, 1, 1],
  ])

  result = get_indices_with_most_new(data, covered_counts)

  np.testing.assert_array_equal(result, np.array([1, 3]))


def test_1x0__returns_zero():
  covered_counts = np.array([])
  data = np.array([[]])

  result = get_indices_with_most_new(data, covered_counts)

  np.testing.assert_array_equal(result, np.array([0]))


def test_1x1__returns_zero():
  covered_counts = np.array([0])
  data = np.array([[1]])

  result = get_indices_with_most_new(data, covered_counts)

  np.testing.assert_array_equal(result, np.array([0]))


def test_1x2__returns_zero():
  covered_counts = np.array([0, 1])
  data = np.array([[1, 1]])

  result = get_indices_with_most_new(data, covered_counts)

  np.testing.assert_array_equal(result, np.array([0]))


def test_2x1_same_two__returns_zero_and_one():
  covered_counts = np.array([1])
  data = np.array([[1], [1]])

  result = get_indices_with_most_new(data, covered_counts)

  np.testing.assert_array_equal(result, np.array([0, 1]))


def test_2x1_last_most__returns_one():
  covered_counts = np.array([0])
  data = np.array([[0], [1]])

  result = get_indices_with_most_new(data, covered_counts)

  np.testing.assert_array_equal(result, np.array([1]))


def test_2x1_last_most_but_all_covered__returns_zero_and_one():
  covered_counts = np.array([1])
  data = np.array([[0], [1]])

  result = get_indices_with_most_new(data, covered_counts)

  np.testing.assert_array_equal(result, np.array([0, 1]))
