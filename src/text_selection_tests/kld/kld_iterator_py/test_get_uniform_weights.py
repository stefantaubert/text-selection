import numpy as np

from text_selection.kld.kld_iterator import get_uniform_weights


def test_0():
  result = get_uniform_weights(0)
  np.testing.assert_array_equal(result, np.empty(shape=(0,)))
  assert result.dtype == np.uint16


def test_1():
  result = get_uniform_weights(1)
  np.testing.assert_array_equal(result, np.array([1]))
  assert result.dtype == np.uint16


def test_2():
  result = get_uniform_weights(2)
  np.testing.assert_array_equal(result, np.array([1, 1]))
  assert result.dtype == np.uint16
