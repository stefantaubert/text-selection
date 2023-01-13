import numpy as np
from ordered_set import OrderedSet

from text_selection.kld.kld_iterator import get_minimum_kld_keys


def test_componenttest_without_preselection():
  data = np.array([[1, 2], [1, 1], [1, 1], [1, 1], [0, 0]], dtype=np.uint16)
  keys = OrderedSet((0, 1, 3, 4))
  covered = np.array([0, 0], dtype=np.uint16)
  target_dist = np.full(shape=(5, 2), fill_value=0.5, dtype=np.float64)
  div, result = get_minimum_kld_keys(
    data=data,
    covered_counts=covered,
    keys=keys,
    target_distributions=target_dist,
  )

  assert div == 0.0
  assert result == OrderedSet((1, 3))


def test_componenttest_with_preselection():
  data = np.array([[1, 2], [2, 1], [2, 1], [2, 1], [0, 0]], dtype=np.uint16)
  keys = OrderedSet((0, 1, 3, 4))
  covered = np.array([1, 2], dtype=np.uint16)
  target_dist = np.full(shape=(5, 2), fill_value=0.5, dtype=np.float64)
  div, result = get_minimum_kld_keys(
    data=data,
    covered_counts=covered,
    keys=keys,
    target_distributions=target_dist,
  )

  assert div == 0.0
  assert result == OrderedSet((1, 3))


def test_componenttest_with_preselection_zero_is_selected():
  data = np.array([[1, 2], [2, 1], [2, 1], [2, 1], [0, 0]], dtype=np.uint16)
  keys = OrderedSet((0, 1, 3, 4))
  covered = np.array([1, 1], dtype=np.uint16)
  target_dist = np.full(shape=(5, 2), fill_value=0.5, dtype=np.float64)
  div, result = get_minimum_kld_keys(
    data=data,
    covered_counts=covered,
    keys=keys,
    target_distributions=target_dist,
  )

  assert div == 0.0
  assert result == OrderedSet((4,))
