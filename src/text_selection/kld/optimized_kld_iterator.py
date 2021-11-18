from logging import getLogger
from typing import Iterable, Iterator, Optional

import numpy as np
from numpy.typing import NDArray
from ordered_set import OrderedSet
from text_selection.kld.distribution_factories import DistributionFactory
from text_selection.kld.kld_iterator import KldCoreIterator
from text_selection.selection import KeySelector

# TODO: split into two iterators -> empty rows at end & empty column remover


class OptimizedKldIterator(KldCoreIterator):
  def __init__(self, data: np.ndarray, data_indicies: OrderedSet[int], preselection: np.ndarray, distribution_factory: DistributionFactory, key_selector: KeySelector, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> None:
    logger = getLogger(__name__)
    logger.info("Copy data and preselection")
    super().__init__(
      data=data.copy(),
      preselection=preselection,
      data_indicies=data_indicies,
      distribution_factory=distribution_factory,
      batches=batches,
      chunksize=chunksize,
      key_selector=key_selector,
      maxtasksperchild=maxtasksperchild,
      n_jobs=n_jobs,
    )
    logger.info("Done")

    # remove empty rows
    empty_row_indicies = get_empty_row_indicies(self.data)
    remove_rows = len(empty_row_indicies) > 0
    if remove_rows:
      logger.info(
        f"Removing {len(empty_row_indicies)} empty row(s) out of {len(self.data)} rows...")
      remove_from_ordered_set_inplace(self.available_data_keys_ordered, empty_row_indicies)
      # self.data = np.delete(self.data, empty_row_indicies, axis=0)
      logger.info("Done.")
    self.available_empty_row_indicies = OrderedSet(empty_row_indicies)
    del empty_row_indicies
    # mapping = {index: key for index, key in enumerate(self.select_from_keys)}
    # self.mapping_iterator = MappingIterator(self, mapping=mapping)

    # remove empty columns, can only occur if len(symbols in utterance) = n_gram - 1
    data_counts: NDArray = np.sum(self.data, axis=0)
    all_counts: NDArray = data_counts + self.covered_array
    remove_ngram_indicies = np.where(all_counts == 0)[0]
    if len(remove_ngram_indicies) > 0:
      logger.info(
        f"Removing {len(remove_ngram_indicies)} out of {self.data.shape[1]} columns...")
      self.data: np.ndarray = np.delete(self.data, remove_ngram_indicies, axis=1)
      self.covered_array: np.ndarray = np.delete(self.covered_array, remove_ngram_indicies, axis=0)
      self.calculate_target_distribution_from_data_and_update_values()
      logger.info("Done.")

  def __iter__(self) -> Iterator[int]:
    return self

  def __next__(self) -> int:
    # index = super().__next__()
    # key = self.select_from_keys[index]
    # return key
    try:
      return super().__next__()
    except StopIteration:
      if len(self.available_empty_row_indicies) > 0:
        selected_key = self.key_selector.select_key(self.available_empty_row_indicies)
        assert 0 <= selected_key < len(self.data)
        assert np.sum(self.data[selected_key], axis=0) == 0
        self.available_empty_row_indicies.remove(selected_key)

        return selected_key
      else:
        raise StopIteration()


def get_empty_row_indicies(array: np.ndarray) -> np.ndarray:
  empty_entry_ids = np.where(~array.any(axis=1))[0]
  return empty_entry_ids


def remove_from_ordered_set_inplace(s: OrderedSet[int], indicies: Iterable[int]) -> None:
  for index in reversed(sorted(indicies)):
    assert 0 <= index < len(s)
    remove_entry = s[index]
    s.remove(remove_entry)
