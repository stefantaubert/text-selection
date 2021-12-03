import numpy as np
from numpy.core.fromnumeric import shape
from ordered_set import OrderedSet
from text_selection.kld.optimized_kld_iterator import remove_empty_columns


def test_empty_arrays():
  res_data, res_preselection, res_weights = remove_empty_columns(
    data=np.empty(shape=(0, 0)),
    data_indices=set(),
    preselection=np.empty(shape=(0,)),
    weights=np.empty(shape=(0,)),
  )

  np.testing.assert_array_equal(res_data, np.empty(shape=(0, 0)))
  np.testing.assert_array_equal(res_preselection, np.empty(shape=(0,)))
  np.testing.assert_array_equal(res_weights, np.empty(shape=(0,)))


def test_empty_indices_full_preselection__changes_nothing():
  res_data, res_preselection, res_weights = remove_empty_columns(
    data=np.array([[1, 1], [0, 0]]),
    data_indices=set(),
    preselection=np.array([1, 1]),
    weights=np.array([1, 1]),
  )

  np.testing.assert_array_equal(res_data, np.array([[1, 1], [0, 0]]))
  np.testing.assert_array_equal(res_preselection, np.array([1, 1]))
  np.testing.assert_array_equal(res_weights, np.array([1, 1]))


def test_none_zero__removes_nothing():
  res_data, res_preselection, res_weights = remove_empty_columns(
    data=np.ones(shape=(1, 1)),
    data_indices={0},
    preselection=np.ones(shape=(1,)),
    weights=np.ones(shape=(1,)),
  )

  np.testing.assert_array_equal(res_data, np.ones(shape=(1, 1)))
  np.testing.assert_array_equal(res_preselection, np.ones(shape=(1,)))
  np.testing.assert_array_equal(res_weights, np.ones(shape=(1,)))


def test_all_zero__removes_everything():
  res_data, res_preselection, res_weights = remove_empty_columns(
    data=np.zeros(shape=(1, 1)),
    data_indices={0},
    preselection=np.zeros(shape=(1,)),
    weights=np.zeros(shape=(1,)),
  )

  np.testing.assert_array_equal(res_data, np.empty(shape=(1, 0)))
  np.testing.assert_array_equal(res_preselection, np.empty(shape=(0,)))
  np.testing.assert_array_equal(res_weights, np.empty(shape=(0,)))


def test_one_one_and_one_zero__removes_zero():
  res_data, res_preselection, res_weights = remove_empty_columns(
    data=np.array([[0, 1]]),
    data_indices={0},
    preselection=np.array([0, 1]),
    weights=np.ones(shape=(2,)),
  )

  np.testing.assert_array_equal(res_data, np.array([[1]]))
  np.testing.assert_array_equal(res_preselection, np.array([1]))
  np.testing.assert_array_equal(res_weights, np.array([1]))


def test_data_and_pre_together_non_zero__removes_nothing():
  res_data, res_preselection, res_weights = remove_empty_columns(
    data=np.array([[0, 1]]),
    data_indices={0},
    preselection=np.array([1, 1]),
    weights=np.ones(shape=(2,)),
  )

  np.testing.assert_array_equal(res_data, np.array([[0, 1]]))
  np.testing.assert_array_equal(res_preselection, np.array([1, 1]))
  np.testing.assert_array_equal(res_weights, np.array([1, 1]))


def test_zero_rows_are_not_removed():
  res_data, res_preselection, res_weights = remove_empty_columns(
    data=np.array([[1, 1], [0, 0]]),
    data_indices={0, 1},
    preselection=np.array([0, 0]),
    weights=np.array([1, 1]),
  )

  np.testing.assert_array_equal(res_data, np.array([[1, 1], [0, 0]]))
  np.testing.assert_array_equal(res_preselection, np.array([0, 0]))
  np.testing.assert_array_equal(res_weights, np.array([1, 1]))


def test_componenttest():
  res_data, res_preselection, res_weights = remove_empty_columns(
    data=np.array([[0, 0, 1], [0, 0, 0], [2, 0, 0]]),
    data_indices={0, 1, 2},
    preselection=np.array([3, 0, 4]),
    weights=np.array([1, 2, 3]),
  )

  np.testing.assert_array_equal(res_data, np.array([[0, 1], [0, 0], [2, 0]]))
  np.testing.assert_array_equal(res_preselection, np.array([3, 4]))
  np.testing.assert_array_equal(res_weights, np.array([1, 3]))
