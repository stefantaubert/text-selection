import numpy as np
from ordered_set import OrderedSet

from text_selection.greedy.greedy_iterator import GreedyIterator
from text_selection.selection import FirstKeySelector


def test_8x3_all_combinations__return_correct_order():
  data = np.array([
    [0, 0, 0],  # 0
    [0, 0, 1],  # 1
    [0, 1, 0],  # 2
    [0, 1, 1],  # 3
    [1, 0, 0],  # 4
    [1, 0, 1],  # 5
    [1, 1, 0],  # 6
    [1, 1, 1],  # 7
  ])
  data_indices = OrderedSet(range(len(data)))
  preselection = np.array(
    [0, 0, 0]
  )
  iterator = GreedyIterator(
    data=data,
    data_indices=data_indices,
    key_selector=FirstKeySelector(),
    preselection=preselection,
  )

  result = OrderedSet(iterator)

  assert result == OrderedSet((7, 3, 4, 5, 2, 6, 1, 0))


def test_4x3_all_ones__keys_0_1_3_2__returns_0_1_3_2():
  data = np.ones(shape=(4, 3), dtype=np.uint16)
  data_indices = OrderedSet((0, 1, 3, 2))
  preselection = np.zeros(data.shape[1], data.dtype)
  iterator = GreedyIterator(
    data=data,
    data_indices=data_indices,
    key_selector=FirstKeySelector(),
    preselection=preselection,
  )

  result = OrderedSet(iterator)

  assert result == OrderedSet((0, 1, 3, 2))
