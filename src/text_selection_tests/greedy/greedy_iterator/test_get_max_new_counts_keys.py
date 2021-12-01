import numpy as np
from ordered_set import OrderedSet
from text_selection.greedy.greedy_iterator import get_max_new_counts_keys


def test_a():
  data = np.array([
    [0, 1, 2],
    [0, 2, 0],
    [1, 2, 0],
    [1, 2, 0],
  ])
  keys = OrderedSet((0, 1, 3))
  covered = np.array([0, 6, 0])

  result = get_max_new_counts_keys(data, keys, covered)

  assert result == OrderedSet((0, 3))
