import numpy as np
from text_selection.greedy.greedy_iterator import get_max_new_counts


def test_a():
  data = np.array([
    [0, 1, 2],
    [0, 2, 0],
    [1, 2, 0],
  ])

  result = get_max_new_counts(data)

  np.testing.assert_array_equal(result, np.array([0, 2]))
