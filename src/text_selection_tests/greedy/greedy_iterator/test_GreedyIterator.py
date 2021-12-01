
import numpy as np
from ordered_set import OrderedSet
from text_selection.greedy.greedy_iterator import GreedyIterator
from text_selection.selection import FirstKeySelector


def test_empty_indicies__return_empty_set():
  data = np.ones(shape=(4, 3), dtype=np.uint32)
  data_indicies = OrderedSet((0, 1, 3, 2,))
  preselection = np.zeros(data.shape[1], data.dtype)
  iterator = GreedyIterator(
    data=data,
    data_indicies=data_indicies,
    key_selector=FirstKeySelector(),
    preselection=preselection,
  )

  result = OrderedSet(iterator)

  assert result == OrderedSet((0, 1, 3, 2,))
