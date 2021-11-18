import numpy as np
from ordered_set import OrderedSet
from text_selection.kld.kld_iterator import KldIterator
from text_selection.selection import FirstKeySelector


def test__empty_indicies():
  data = np.ones(shape=(4, 3), dtype=np.uint32)
  data_indicies = OrderedSet()
  preselection = np.zeros(data.shape[1], data.dtype)
  iterator = KldIterator(
    data=data,
    data_indicies=data_indicies,
    key_selector=FirstKeySelector(),
    preselection=preselection,
    weights=np.array([1, 1, 1], dtype=np.uint32),
  )

  result = OrderedSet(iterator)

  assert result == OrderedSet()


def test__all_equal():
  data = np.ones(shape=(4, 3), dtype=np.uint32)
  data_indicies = OrderedSet([0, 2, 1, 3])
  preselection = np.zeros(data.shape[1], data.dtype)
  iterator = KldIterator(
    data=data,
    data_indicies=data_indicies,
    key_selector=FirstKeySelector(),
    preselection=preselection,
    weights=np.array([1, 1, 1], dtype=np.uint32),
  )

  result = OrderedSet(iterator)

  assert result == OrderedSet([0, 2, 1, 3])


def test_init__selects_zeros_first():
  data = np.array(
    [
      [1, 0, 1],  # 3
      [0, 0, 1],  # 5
      [0, 1, 0],  # 4
      [0, 0, 0],  # 1
      [0, 0, 0],  # 2
    ],
    dtype=np.uint32,
  )
  data_indicies = OrderedSet([0, 1, 2, 3, 4])
  preselection = np.zeros(data.shape[1], data.dtype)
  iterator = KldIterator(
    data=data,
    data_indicies=data_indicies,
    key_selector=FirstKeySelector(),
    preselection=preselection,
    weights=np.array([1, 1, 1], dtype=np.uint32),
  )

  result = OrderedSet(iterator)

  assert result == OrderedSet([3, 4, 0, 2, 1])
