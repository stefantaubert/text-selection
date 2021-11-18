import numpy as np
from ordered_set import OrderedSet
from text_selection.kld.kld_iterator import get_minimum_kld_keys


def test_multitest_with_preselection():
  data = np.array([[1, 2], [2, 1], [2, 1], [2, 1], [0, 0]], dtype=np.uint32)
  keys = OrderedSet((0, 1, 3, 4))
  covered_counts = np.array([1, 2], dtype=np.uint32)
  target_dist = np.array([0.5, 0.5], dtype=np.float64)
  div, result = get_minimum_kld_keys(
    data=data,
    covered_counts=covered_counts,
    keys=keys,
    target_dist=target_dist,
  )

  assert div == 0.0
  assert result == OrderedSet((1, 3))

def test_multitest_without_preselection():
  data = np.array([[1, 2], [2, 1], [2, 1], [2, 1], [0, 0]], dtype=np.uint32)
  keys = OrderedSet((0, 1, 3, 4))
  covered_counts = np.array([0, 0], dtype=np.uint32)
  target_dist = np.array([0.5, 0.5], dtype=np.float64)
  div, result = get_minimum_kld_keys(
    data=data,
    covered_counts=covered_counts,
    keys=keys,
    target_dist=target_dist,
  )

  assert div == 0.0
  assert result == OrderedSet((1, 3))
