import numpy as np

from text_selection.kld.kld_iterator import get_distribution


# region empty
def test_s0_empty__returns_empty():
  counts = np.array([], dtype=np.uint16)
  result = get_distribution(counts, axis=0)
  np.testing.assert_array_equal(result, np.empty(shape=(0)))


def test_s0x0_empty__returns_empty():
  counts = np.empty(shape=(0, 0), dtype=np.uint16)
  result = get_distribution(counts, axis=1)
  np.testing.assert_array_equal(result, np.empty(shape=(0, 0)))


def test_s1x0_empty__returns_empty():
  counts = np.array([[]], dtype=np.uint16)
  result = get_distribution(counts, axis=1)
  np.testing.assert_array_equal(result, np.empty(shape=(1, 0)))


def test_s2x0_empty__returns_empty():
  counts = np.array([[], []], dtype=np.uint16)
  result = get_distribution(counts, axis=1)
  np.testing.assert_array_equal(result, np.empty(shape=(2, 0)))
# endregion


# region normal
def test_s1_two__returns_one():
  counts = np.array([2], dtype=np.uint16)
  result = get_distribution(counts, axis=0)
  np.testing.assert_array_equal(result, np.array([1.0]))


def test_s2_one_and_two__returns_one_third_and_two_third():
  counts = np.array([1, 2], dtype=np.uint16)
  result = get_distribution(counts, axis=0)
  np.testing.assert_array_equal(result, np.array([1 / 3, 2 / 3]))


def test_s1x1_two__returns_one():
  counts = np.array([[2]], dtype=np.uint16)
  result = get_distribution(counts, axis=1)
  np.testing.assert_array_equal(result, np.array([[1.0]]))


def test_s1x2_one_and_two__returns_one_third_and_two_third():
  counts = np.array([[1, 2]], dtype=np.uint16)
  result = get_distribution(counts, axis=1)
  np.testing.assert_array_equal(result, np.array([[1 / 3, 2 / 3]]))


def test_s2x1_one_and_two__returns_one_and_one():
  counts = np.array([[1], [2]], dtype=np.uint16)
  result = get_distribution(counts, axis=1)
  np.testing.assert_array_equal(result, np.array([[1.0], [1.0]]))


def test_s2x2_one_and_two__returns_one_third_and_two_third():
  counts = np.array([[1, 2], [2, 1]], dtype=np.uint16)
  result = get_distribution(counts, axis=1)
  np.testing.assert_array_equal(result, np.array([[1 / 3, 2 / 3], [2 / 3, 1 / 3]]))


def test_s2x2_weights__returns_normalized_weights():
  weights = np.array([[0.8, 1.2], [0.2, 0.8]], dtype=np.float64)
  result = get_distribution(weights, axis=1)
  np.testing.assert_array_equal(result, np.array([[0.4, 0.6], [0.2, 0.8]]))
# endregion


# region zero
def test_s1_zero__returns_nan():
  counts = np.array([0], dtype=np.uint16)
  result = get_distribution(counts, axis=0)
  np.testing.assert_array_equal(result, np.array([np.nan]))


def test_s2_zero__returns_nan():
  np.seterr(invalid="print")
  counts = np.array([0, 0], dtype=np.uint16)
  result = get_distribution(counts, axis=0)
  assert np.geterr()["invalid"] == "print"
  np.testing.assert_array_equal(result, np.array([np.nan, np.nan]))


def test_s1x1_zero__returns_nan():
  counts = np.array([[0]], dtype=np.uint16)
  result = get_distribution(counts, axis=1)
  np.testing.assert_array_equal(result, np.array([[np.nan]]))


def test_s1x2_zero__returns_nan():
  counts = np.array([[0, 0]], dtype=np.uint16)
  result = get_distribution(counts, axis=1)
  np.testing.assert_array_equal(result, np.array([[np.nan, np.nan]]))


def test_s2x1_zero__returns_nan():
  counts = np.array([[0], [0]], dtype=np.uint16)
  result = get_distribution(counts, axis=1)
  np.testing.assert_array_equal(result, np.array([[np.nan], [np.nan]]))


def test_s2x2_zero__returns_nan():
  counts = np.array([[0, 0], [0, 0]], dtype=np.uint16)
  result = get_distribution(counts, axis=1)
  np.testing.assert_array_equal(result, np.array([[np.nan, np.nan], [np.nan, np.nan]]))
# endregion
