import numpy as np
from ordered_set import OrderedSet

from text_selection.greedy.optimized_greedy_iterator import OptimizedGreedyIterator
from text_selection.selection import FirstKeySelector


def test_8x3_all_combinations__return_correct_order():
  data = np.array([
    [0, 0, 0, 0],  # 0
    [0, 0, 0, 1],  # 1
    [0, 1, 0, 0],  # 2
    [0, 1, 0, 1],  # 3
    [1, 0, 0, 0],  # 4
    [1, 0, 0, 1],  # 5
    [1, 1, 0, 0],  # 6
    [1, 1, 0, 1],  # 7
  ])
  data_indices = OrderedSet(range(len(data)))
  preselection = np.array(
    [0, 0, 0, 0]
  )
  iterator = OptimizedGreedyIterator(
    data=data,
    data_indices=data_indices,
    key_selector=FirstKeySelector(),
    preselection=preselection,
  )

  result = list(iterator)

  assert result == [7, 3, 4, 5, 2, 6, 1, 0]
