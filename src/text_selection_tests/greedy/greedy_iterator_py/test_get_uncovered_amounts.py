import numpy as np

from text_selection.greedy.greedy_iterator import get_uncovered_amounts


def test_0x0__returns_empty():
  data = np.empty(shape=(0, 0))

  result = get_uncovered_amounts(data)

  np.testing.assert_array_equal(result, np.array([]))


def test_1x0__counts_zero():
  data = np.array([
    [],
  ])

  result = get_uncovered_amounts(data)

  np.testing.assert_array_equal(result, np.array([0]))


def test_1x1_zero__counts_zero():
  data = np.array([
    [0],
  ])

  result = get_uncovered_amounts(data)

  np.testing.assert_array_equal(result, np.array([0]))


def test_1x1_non_zero__counts_one():
  data = np.array([
    [5],
  ])

  result = get_uncovered_amounts(data)

  np.testing.assert_array_equal(result, np.array([1]))


def test_1x2_two_non_zero__counts_two():
  data = np.array([
    [5, 3],
  ])

  result = get_uncovered_amounts(data)

  np.testing.assert_array_equal(result, np.array([2]))


def test_4x3_component():
  data = np.array([
    [0, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [1, 1, 1],
  ])

  result = get_uncovered_amounts(data)

  np.testing.assert_array_equal(result, np.array([0, 2, 1, 3]))
