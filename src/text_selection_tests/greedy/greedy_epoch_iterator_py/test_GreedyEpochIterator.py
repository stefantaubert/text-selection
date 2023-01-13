import numpy as np
from ordered_set import OrderedSet

from text_selection.greedy.greedy_epoch_iterator import EpochProxyIterator
from text_selection.greedy.greedy_iterator import GreedyIterator
from text_selection.selection import FirstKeySelector


def test_update_is_correct():
  it = GreedyIterator(
    data=np.array([[0, 1], [1, 1], [1, 0]]),
    data_indices=OrderedSet([0, 1, 2]),
    key_selector=FirstKeySelector(),
    preselection=np.array([0, 0]),
  )

  it = EpochProxyIterator(
    iterator=it,
    epochs=2,
  )

  assert_updates = (1, 0, 1)

  for _, assert_update in zip(it, assert_updates):
    assert it.tqdm_update == assert_update


def test_returns_greedy_indices():
  it = GreedyIterator(
    data=np.array([[0, 1], [1, 1], [1, 0]]),
    data_indices=OrderedSet([0, 1, 2]),
    key_selector=FirstKeySelector(),
    preselection=np.array([0, 0]),
  )

  it = EpochProxyIterator(
    iterator=it,
    epochs=2,
  )

  result = list(it)

  assert result == [1, 0, 2]
