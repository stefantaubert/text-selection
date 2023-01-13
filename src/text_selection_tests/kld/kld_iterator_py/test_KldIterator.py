import numpy as np
from ordered_set import OrderedSet

from text_selection.kld.kld_iterator import KldIterator
from text_selection.selection import FirstKeySelector


def test_empty_indicies__return_empty_set():
  data = np.ones(shape=(4, 3), dtype=np.uint16)
  data_indicies = OrderedSet()
  preselection = np.zeros(data.shape[1], data.dtype)
  iterator = KldIterator(
    data=data,
    data_indices=data_indicies,
    key_selector=FirstKeySelector(),
    preselection=preselection,
    weights=np.array([1, 1, 1], dtype=np.uint16),
  )

  result = OrderedSet(iterator)

  assert result == OrderedSet()


def test_all_equal_returns_all_in_same_key_order():
  data = np.ones(shape=(4, 3), dtype=np.uint16)
  data_indicies = OrderedSet((0, 2, 1, 3))
  preselection = np.zeros(data.shape[1], data.dtype)
  iterator = KldIterator(
    data=data,
    data_indices=data_indicies,
    key_selector=FirstKeySelector(),
    preselection=preselection,
    weights=np.array([1, 1, 1], dtype=np.uint16),
  )

  result = OrderedSet(iterator)

  assert result == OrderedSet((0, 2, 1, 3))


def test_select_zeros_not_first():
  data = np.array(
    [
      [1, 0, 1],  # 0
      [0, 0, 1],  # 4
      [0, 1, 0],  # 1
      [0, 0, 0],  # 2
      [0, 0, 0],  # 3
    ],
    dtype=np.uint16,
  )
  data_indicies = OrderedSet((0, 1, 2, 3, 4))
  preselection = np.zeros(data.shape[1], data.dtype)
  iterator = KldIterator(
    data=data,
    data_indices=data_indicies,
    key_selector=FirstKeySelector(),
    preselection=preselection,
    weights=np.array([1, 1, 1], dtype=np.uint16),
  )

  result = OrderedSet(iterator)

  assert result == OrderedSet((0, 2, 3, 4, 1))


def test_empty_ngrams__returns_same_input_order():
  data = np.empty(shape=(5, 0), dtype=np.uint16)
  data_indicies = OrderedSet((0, 1, 2, 3, 4))
  preselection = np.empty(shape=(0))
  iterator = KldIterator(
    data=data,
    data_indices=data_indicies,
    key_selector=FirstKeySelector(),
    preselection=preselection,
    weights=np.empty(shape=(0,), dtype=np.uint16),
  )

  result = OrderedSet(iterator)

  assert result == OrderedSet((0, 1, 2, 3, 4))
