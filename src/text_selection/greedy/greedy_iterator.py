from logging import getLogger
from typing import Iterator, Tuple

import numpy as np
from ordered_set import OrderedSet
from text_selection.selection import KeySelector


class GreedyIterator(Iterator[int]):

  def __init__(self, data: np.ndarray, data_indicies: OrderedSet[int], preselection: np.ndarray, key_selector: KeySelector) -> None:
    super().__init__()
    self._data = data
    self._key_selector = key_selector
    # defines the order for what the selection is based on
    self.__available_data_keys_ordered = data_indicies.copy()
    self.__covered_array = preselection.copy()
    self.__epochs = 0

  def __iter__(self) -> Iterator[int]:
    return self

  def __next__(self) -> int:
    if len(self.__available_data_keys_ordered) == 0:
      raise StopIteration()

    if np.all(self.__covered_array > 0, axis=0):
      # reset epoch
      self.__covered_array = np.zeros_like(self.__covered_array)
      # todo on init not + 1
      self.__epochs += 1

    potential_keys = get_max_new_counts_keys(
      data=self._data,
      keys=self.__available_data_keys_ordered,
      covered_counts=self.__covered_array,
    )

    if len(potential_keys) > 1:
      logger = getLogger(__name__)
      logger.info(f"Found {len(potential_keys)} candidates for the current iteration.")

    selected_key = self._key_selector.select_key(potential_keys)
    assert 0 <= selected_key < len(self._data)
    self.__covered_array += self._data[selected_key]
    self.__available_data_keys_ordered.remove(selected_key)

    return selected_key


def get_max_new_counts_keys(data: np.ndarray, keys: OrderedSet[int], covered_counts: np.ndarray) -> OrderedSet[int]:
  assert len(data.shape) == 2
  assert len(covered_counts.shape) == 1
  assert 0 < len(keys) <= len(data)
  data_subset: np.ndarray = data[keys]
  uncovered_indices = np.flatnonzero(covered_counts == 0)
  data_subset_uncovered_total_counts = data_subset[:, uncovered_indices]
  indices = get_max_new_counts(data_subset_uncovered_total_counts)

  mapped_indices = OrderedSet(keys[index] for index in indices)

  return mapped_indices


def get_max_new_counts(data: np.ndarray) -> np.ndarray:
  data_subset_amounts = data != 0
  data_subset_uncovered_counts = np.sum(data_subset_amounts, axis=1)
  _, max_indices = get_maximum_indices(data_subset_uncovered_counts)
  return max_indices


def get_maximum_indices(array: np.ndarray) -> Tuple[float, np.ndarray]:
  assert len(array) > 0
  max_value = array.max()
  max_indices = np.flatnonzero(array == max_value)
  return max_value, max_indices
