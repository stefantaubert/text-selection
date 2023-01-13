import numpy as np

from text_selection.greedy.greedy_iterator import get_maximum_indices


def test_one_entry():
  data = np.array([1])

  res_value, res_indices = get_maximum_indices(data)

  assert res_value == 1
  np.testing.assert_array_equal(res_indices, np.array([0]))


def test_component():
  data = np.array([0, 1, 1, 0])

  res_value, res_indices = get_maximum_indices(data)

  assert res_value == 1
  np.testing.assert_array_equal(res_indices, np.array([1, 2]))
