import numpy as np

from text_selection.kld.kld_iterator import get_distributions_from_weights


def test_0x0():
  data_len = 0
  weights_len = 0
  weights = np.ones(shape=(weights_len,), dtype=np.uint16)
  result = get_distributions_from_weights(weights, data_len=data_len)
  np.testing.assert_array_equal(result, np.empty(shape=(data_len, weights_len)))


def test_1x0():
  data_len = 1
  weights_len = 0
  weights = np.ones(shape=(weights_len,), dtype=np.uint16)
  result = get_distributions_from_weights(weights, data_len=data_len)
  np.testing.assert_array_equal(result, np.empty(shape=(data_len, weights_len)))


def test_0x1():
  data_len = 0
  weights_len = 1
  weights = np.ones(shape=(weights_len,), dtype=np.uint16)
  result = get_distributions_from_weights(weights, data_len=data_len)
  np.testing.assert_array_equal(result, np.empty(shape=(data_len, weights_len)))


def test_1x1():
  data_len = 1
  weights_len = 1
  weights = np.ones(shape=(weights_len,), dtype=np.uint16)
  result = get_distributions_from_weights(weights, data_len=data_len)
  np.testing.assert_array_equal(result, np.full(
    shape=(data_len, weights_len), fill_value=1 / weights_len))


def test_1x3():
  data_len = 1
  weights_len = 3
  weights = np.ones(shape=(weights_len,), dtype=np.uint16)
  result = get_distributions_from_weights(weights, data_len=data_len)
  np.testing.assert_array_equal(result, np.full(
    shape=(data_len, weights_len), fill_value=1 / weights_len))


def test_2x4():
  data_len = 2
  weights_len = 4
  weights = np.ones(shape=(weights_len,), dtype=np.uint16)
  result = get_distributions_from_weights(weights, data_len=data_len)
  np.testing.assert_array_equal(result, np.full(
    shape=(data_len, weights_len), fill_value=1 / weights_len))
