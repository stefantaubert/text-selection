import numpy as np

from text_selection.common.helper import get_empty_columns


def test_empty_arrays():
  result = get_empty_columns(
    data=np.array([[]]),
    data_indices=set(),
    preselection=np.array([]),
  )

  np.testing.assert_array_equal(result, np.array([]))


def test_empty_indices_full_preselection__changes_nothing():
  result = get_empty_columns(
    data=np.array([[1, 1], [0, 0]]),
    data_indices=set(),
    preselection=np.array([1, 1]),
  )

  np.testing.assert_array_equal(result, np.array([]))


def test_none_zero__removes_nothing():
  result = get_empty_columns(
    data=np.ones(shape=(1, 1)),
    data_indices={0},
    preselection=np.ones(shape=(1,)),
  )

  np.testing.assert_array_equal(result, np.array([]))


def test_all_zero__removes_everything():
  result = get_empty_columns(
    data=np.array([[0]]),
    data_indices={0},
    preselection=np.zeros(shape=(1,)),
  )

  np.testing.assert_array_equal(result, np.array([0]))


def test_one_one_and_one_zero__removes_zero():
  result = get_empty_columns(
    data=np.array([[0, 1]]),
    data_indices={0},
    preselection=np.array([0, 1]),
  )

  np.testing.assert_array_equal(result, np.array([0]))


def test_data_and_pre_together_non_zero__removes_nothing():
  result = get_empty_columns(
    data=np.array([[0, 1]]),
    data_indices={0},
    preselection=np.array([1, 1]),
  )

  np.testing.assert_array_equal(result, np.array([]))


def test_zero_rows_are_summed_up():
  result = get_empty_columns(
    data=np.array([[1, 1], [0, 0]]),
    data_indices={0, 1},
    preselection=np.array([0, 0]),
  )

  np.testing.assert_array_equal(result, np.array([]))


def test_componenttest():
  result = get_empty_columns(
    data=np.array([[0, 0, 1], [0, 0, 0], [2, 0, 0]]),
    data_indices={0, 1, 2},
    preselection=np.array([3, 0, 4]),
  )

  np.testing.assert_array_equal(result, np.array([1]))
