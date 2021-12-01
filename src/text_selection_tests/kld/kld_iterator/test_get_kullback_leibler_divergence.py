import numpy as np
import pytest
from text_selection.kld.kld_iterator import get_kullback_leibler_divergence


# region empty
def test_axis_0_empty_input__returns_zero():
  pk = np.array([], dtype=np.float64)
  qk = np.array([], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert result == 0.0


def test_axis_1_empty_input__returns_zero():
  pk = np.array([[]], dtype=np.float64)
  qk = np.array([[]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.0], dtype=np.float64))
# endregion


# region equal
def test_axis_0_equal_values__returns_zero():
  pk = np.array([0.5, 0.5], dtype=np.float64)
  qk = np.array([0.5, 0.5], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert result == 0.0


def test_axis_1_equal_values__returns_zero():
  pk = np.array([[0.5, 0.5]], dtype=np.float64)
  qk = np.array([[0.5, 0.5]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.0], dtype=np.float64))
# endregion


# region different
def test_axis_0_different_values__returns_correct_kld():
  pk = np.array([0.3, 0.7], dtype=np.float64)
  qk = np.array([0.5, 0.5], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert result == 0.08228287850505178


def test_axis_1_different_values__returns_correct_kld():
  pk = np.array([[0.3, 0.7]], dtype=np.float64)
  qk = np.array([[0.5, 0.5]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.08228287850505178], dtype=np.float64))
# endregion


# region nan
def test_axis_0_pk_nan__returns_inf():
  pk = np.array([np.nan, np.nan], dtype=np.float64)
  qk = np.array([0.5, 0.5], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert np.isinf(result)


def test_axis_1_pk_nan__returns_inf():
  pk = np.array([[np.nan, np.nan]], dtype=np.float64)
  qk = np.array([[0.5, 0.5]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([np.inf], dtype=np.float64))
# endregion


# region dtypes
def test_axis_0_dtype_32__returns_dtype_32():
  pk = np.array([0.3, 0.7], dtype=np.float32)
  qk = np.array([0.5, 0.5], dtype=np.float32)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  np.testing.assert_almost_equal(result, 0.08228287)
  assert result.dtype == np.float32


def test_axis_0_dtype_64__returns_dtype_64():
  pk = np.array([0.3, 0.7], dtype=np.float64)
  qk = np.array([0.5, 0.5], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert result == 0.08228287850505178
  assert result.dtype == np.float64


def test_axis_1_dtype_32__returns_dtype_32():
  pk = np.array([[0.3, 0.7]], dtype=np.float32)
  qk = np.array([[0.5, 0.5]], dtype=np.float32)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.08228287], dtype=np.float32))


def test_axis_1_dtype_64__returns_dtype_64():
  pk = np.array([[0.3, 0.7]], dtype=np.float64)
  qk = np.array([[0.5, 0.5]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  np.testing.assert_array_equal(result, np.array([0.08228287850505178], dtype=np.float64))
# endregion


def test_axis_1_component_test__returns_correct_klds():
  pk = np.array([[0.5, 0.5], [np.nan, np.nan], [0.3, 0.7]], dtype=np.float64)
  qk = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float64)

  result = get_kullback_leibler_divergence(pk, qk, axis=1)

  np.testing.assert_array_equal(result, np.array([
    0.0, np.inf, 0.08228287850505178
  ], dtype=np.float64))
