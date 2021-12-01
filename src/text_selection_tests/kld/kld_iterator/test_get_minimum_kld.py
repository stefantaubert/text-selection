import numpy as np
from text_selection.kld.kld_iterator import get_minimum_kld

# region empty

def test_s1x0_empty__returns_div_zero_index_zero():
  counts = np.array([[]])
  distribution = np.array([[]])
  divergence, indices = get_minimum_kld(counts, distribution)
  assert divergence == 0.0
  np.testing.assert_array_equal(indices, np.array([0]))


def test_s2x0_empty__returns_div_zero_indices_zero_and_one():
  counts = np.array([[], []])
  distribution = np.array([[], []])
  divergence, indices = get_minimum_kld(counts, distribution)
  assert divergence == 0.0
  np.testing.assert_array_equal(indices, np.array([0, 1]))

# endregion


def test_aeia():
  data = np.array([], dtype=np.uint32)
  target_dist = np.array([0.5, 0.5], dtype=np.float64)
  divergence, indices = get_minimum_kld(data, target_dist)

  assert divergence == 0
  np.testing.assert_array_equal(indices, np.array([0, 1]))


def test_s2x2__returns_one_min():
  counts = np.array([[1, 2], [1, 1]])
  distribution = np.array([[0.5, 0.5], [0.5, 0.5]])
  divergence, indices = get_minimum_kld(counts, distribution)
  assert divergence == 0.0
  np.testing.assert_array_equal(indices, np.array([1]))


def test_multitest_without_preselection():
  data = np.array([[1, 2], [2, 1], [2, 1], [2, 1], [0, 0]], dtype=np.uint32)
  target_dist = np.array([0.5, 0.5], dtype=np.float64)
  div, result = get_minimum_kld_keys(
    data=data,
    covered_counts=covered_counts,
    keys=keys,
    target_distribution=target_dist,
  )

  assert div == 0.0
  assert result == OrderedSet((1, 3))
