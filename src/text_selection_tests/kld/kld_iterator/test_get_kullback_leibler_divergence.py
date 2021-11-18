import numpy as np
import pytest
from text_selection.kld.kld_iterator import get_kullback_leibler_divergence


def test_twodim_empty_input__returns_zero():
  pk = np.array([[]], dtype=np.float64)
  qk = np.array([[]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  assert result == np.array([0.0], dtype=np.float64)


def test_twodim_equal_values__returns_zero():
  pk = np.array([[0.5, 0.5]], dtype=np.float64)
  qk = np.array([[0.5, 0.5]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  assert result == np.array([0.0], dtype=np.float64)


def test_twodim_multivalues__returns_correct_kld():
  pk = np.array([[0.5, 0.5], [np.nan, np.nan], [1.0, 0], [0.3, 0.7]], dtype=np.float64)
  qk = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=1)
  assert len(result) == 4
  assert result[0] == 0.0
  assert result[1] == 0.0
  assert result[2] == 0.6931471805599453
  assert result[3] == 0.08228287850505178
  assert result.dtype == np.float64


def test_empty_input__returns_zero():
  pk = np.array([], dtype=np.float64)
  qk = np.array([], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert result == 0.0


def test_equal_values__returns_zero():
  pk = np.array([0.5, 0.5], dtype=np.float64)
  qk = np.array([0.5, 0.5], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert result == 0.0


def test_pk_only_zeros__returns_zero():
  pk = np.array([0.0, 0.0], dtype=np.float64)
  qk = np.array([0.5, 0.5], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert result == 0.0


def test_pk_nan__returns_zero():
  pk = np.array([np.nan, np.nan], dtype=np.float64)
  qk = np.array([0.5, 0.5], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert result == 0.0


def test_one_in_pk_nan__returns_zero():
  pk = np.array([1.0, np.nan], dtype=np.float64)
  qk = np.array([0.5, 0.5], dtype=np.float64)
  result = get_kullback_leibler_divergence(pk, qk, axis=0)
  assert result == 0.0


def test_pk_negative__raises_assert():
  pk = np.array([-1.0], dtype=np.float64)
  qk = np.array([1.0], dtype=np.float64)
  with pytest.raises(AssertionError):
    get_kullback_leibler_divergence(pk, qk, axis=0)


def test_qk_zeros__raises_assert():
  pk = np.array([0.5, 0.5], dtype=np.float64)
  qk = np.array([0.0, 0.0], dtype=np.float64)
  with pytest.raises(AssertionError):
    # would return inf if not asserted
    get_kullback_leibler_divergence(pk, qk, axis=0)


def test_qk_nan__raises_assert():
  pk = np.array([0.5, 0.5])
  qk = np.array([np.nan, np.nan])
  with pytest.raises(AssertionError):
    get_kullback_leibler_divergence(pk, qk, axis=0)


def test_qk_negative__raises_assert():
  pk = np.array([1.0], dtype=np.float64)
  qk = np.array([-1.0], dtype=np.float64)
  with pytest.raises(AssertionError):
    get_kullback_leibler_divergence(pk, qk, axis=0)


def test_pk_greater_one__raises_assert():
  pk = np.array([1.1], dtype=np.float64)
  qk = np.array([1.0], dtype=np.float64)
  with pytest.raises(AssertionError):
    get_kullback_leibler_divergence(pk, qk, axis=0)


def test_pk_zero__raises_assert():
  pk = np.array([0], dtype=np.float64)
  qk = np.array([1.0], dtype=np.float64)
  with pytest.raises(AssertionError):
    get_kullback_leibler_divergence(pk, qk, axis=0)


def test_twodim_pk_zero__raises_assert():
  pk = np.array([[0, 0]], dtype=np.float64)
  qk = np.array([[1.0, 0]], dtype=np.float64)
  with pytest.raises(AssertionError):
    get_kullback_leibler_divergence(pk, qk, axis=1)


def test_qk_greater_one__raises_assert():
  pk = np.array([1.0], dtype=np.float64)
  qk = np.array([1.1], dtype=np.float64)
  with pytest.raises(AssertionError):
    get_kullback_leibler_divergence(pk, qk, axis=0)


def test_pk_multi_dim_array__raises_assert():
  pk = np.array([[1.0]], dtype=np.float64)
  qk = np.array([1.0], dtype=np.float64)
  with pytest.raises(AssertionError):
    get_kullback_leibler_divergence(pk, qk, axis=0)


def test_qk_multi_dim_array__raises_assert():
  pk = np.array([1.0], dtype=np.float64)
  qk = np.array([[1.0]], dtype=np.float64)
  with pytest.raises(AssertionError):
    get_kullback_leibler_divergence(pk, qk, axis=0)
