import numpy as np
from ordered_set import OrderedSet

from text_selection.greedy.greedy_N_iterator import GreedyNIterator
from text_selection.selection import FirstKeySelector


def test_8x3():
  data = np.array([
    [0, 0, 0],  # 0
    [0, 0, 1],  # 1
    [2, 0, 1],  # 2
    [0, 3, 1],  # 3
    [1, 1, 1],  # 4
  ])
  data_indices = OrderedSet(range(len(data)))
  preselection = np.array(
    [0, 0, 0]
  )
  iterator = GreedyNIterator(
    data=data,
    data_indices=data_indices,
    key_selector=FirstKeySelector(),
    preselection=preselection,
    cover_per_epoch=2,
  )

  result = OrderedSet(iterator)

  assert result == OrderedSet((2, 3, 4, 1, 0))
