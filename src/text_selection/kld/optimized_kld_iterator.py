from logging import getLogger
from typing import Set, Tuple

import numpy as np
from numpy import ndarray
from ordered_set import OrderedSet
from text_selection.kld.kld_iterator import KldIterator
from text_selection.selection import KeySelector


def remove_empty_columns(data: ndarray, data_indices: Set[int], preselection: ndarray, weights: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
  """ remove empty columns, can only occur if len(symbols in utterance) = n_gram - 1 """
  assert len(data.shape) == 2
  assert len(preselection.shape) == len(weights.shape) == 1
  assert data.shape[1] == preselection.shape[0] == weights.shape[0]

  if len(data_indices) == 0:
    data_counts = np.zeros_like(preselection)
  else:
    data_counts: ndarray = np.sum(data[list(data_indices)], axis=0)
  all_counts: ndarray = data_counts + preselection
  del data_counts
  remove_ngram_indices = np.where(all_counts == 0)[0]
  del all_counts

  if len(remove_ngram_indices) > 0:
    data: ndarray = np.delete(data, remove_ngram_indices, axis=1)
    preselection: ndarray = np.delete(preselection, remove_ngram_indices, axis=0)
    weights: ndarray = np.delete(weights, remove_ngram_indices, axis=0)

  del remove_ngram_indices

  return data, preselection, weights


class OptimizedKldIterator(KldIterator):
  def __init__(self, data: ndarray, data_indices: OrderedSet[int], preselection: ndarray, weights: ndarray, key_selector: KeySelector) -> None:
    logger = getLogger(__name__)

    old_count = data.shape[1]
    data, preselection, weights = remove_empty_columns(data, data_indices, preselection, weights)
    new_count = data.shape[1]

    if new_count < old_count:
      logger.info(
          f"Removed {old_count - new_count} out of {old_count} columns.")

    super().__init__(
      data=data,
      preselection=preselection,
      data_indices=data_indices,
      weights=weights,
      key_selector=key_selector,
    )
