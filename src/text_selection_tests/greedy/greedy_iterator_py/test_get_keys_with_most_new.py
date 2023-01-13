import numpy as np
from ordered_set import OrderedSet

from text_selection.greedy.greedy_iterator import get_keys_with_most_new


def test_4x3_component():
  covered = np.array(
    [0, 6, 0]
  )
  data = np.array([
    [0, 1, 2],
    [0, 2, 0],
    [1, 2, 0],
    [1, 2, 0],
  ])
  keys = OrderedSet((0, 1, 3))

  result = get_keys_with_most_new(data, keys, covered)

  assert list(result) == [0, 3]


def test_1x1__returns_0():
  covered = np.array(
    [0]
  )
  data = np.array([
    [1],
  ])
  keys = OrderedSet((0,))

  result = get_keys_with_most_new(data, keys, covered)

  assert list(result) == [0]


def test_2x1_keys_1__returns_1():
  covered = np.array(
    [0]
  )
  data = np.array([
    [1],
    [1],
  ])
  keys = OrderedSet((1,))

  result = get_keys_with_most_new(data, keys, covered)

  assert list(result) == [1]
