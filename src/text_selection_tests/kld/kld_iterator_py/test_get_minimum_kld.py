import numpy as np

from text_selection.kld.kld_iterator import get_minimum_kld


# region empty
def test_s1x0_empty__returns_div_zero_index_zero():
  counts = np.empty(shape=(1, 0), dtype=np.float64)
  distribution = np.empty(shape=(1, 0), dtype=np.float64)
  divergence, indices = get_minimum_kld(counts, distribution)
  assert divergence == 0.0
  np.testing.assert_array_equal(indices, np.array([0]))


def test_s2x0_empty__returns_div_zero_indices_zero_and_one():
  counts = np.empty(shape=(2, 0), dtype=np.float64)
  distribution = np.empty(shape=(2, 0), dtype=np.float64)
  divergence, indices = get_minimum_kld(counts, distribution)
  assert divergence == 0.0
  np.testing.assert_array_equal(indices, np.array([0, 1]))
# endregion


# region one entry
def test_s1x1__returns_one_min():
  counts = np.array([[1]])
  distribution = np.array([[1.0]])
  divergence, indices = get_minimum_kld(counts, distribution)
  assert divergence == 0.0
  np.testing.assert_array_equal(indices, np.array([0]))


def test_s1x2__returns_one_min():
  counts = np.array([[1, 1]])
  distribution = np.array([[0.5, 0.5]])
  divergence, indices = get_minimum_kld(counts, distribution)
  assert divergence == 0.0
  np.testing.assert_array_equal(indices, np.array([0]))


def test_s1x3__returns_one_min():
  counts = np.array([[2, 1, 1]])
  distribution = np.array([[0.5, 0.25, 0.25]])
  divergence, indices = get_minimum_kld(counts, distribution)
  assert divergence == 0.0
  np.testing.assert_array_equal(indices, np.array([0]))
# endregion


# region two entries
def test_s2x2__returns_one_min():
  counts = np.array([[1, 2], [1, 1]])
  distribution = np.array([[0.5, 0.5], [0.5, 0.5]])
  divergence, indices = get_minimum_kld(counts, distribution)
  assert divergence == 0.0
  np.testing.assert_array_equal(indices, np.array([1]))


def test_s2x2_one_zeros__returns_one_min():
  counts = np.array([[0, 0], [1, 1]])
  distribution = np.array([[0.5, 0.5], [0.5, 0.5]])
  divergence, indices = get_minimum_kld(counts, distribution)
  assert divergence == 0.0
  np.testing.assert_array_equal(indices, np.array([1]))


def test_s2x2_only_zeros__returns_all():
  counts = np.array([[0, 0], [0, 0]])
  distribution = np.array([[0.5, 0.5], [0.5, 0.5]])
  divergence, indices = get_minimum_kld(counts, distribution)
  assert np.isinf(divergence)
  np.testing.assert_array_equal(indices, np.array([0, 1]))
# endregion
