from logging import getLogger

import numpy as np
from numpy import ndarray
from ordered_set import OrderedSet

from text_selection.common.helper import get_empty_columns
from text_selection.kld.kld_iterator import KldIterator
from text_selection.selection import KeySelector


class OptimizedKldIterator(KldIterator):
  def __init__(self, data: ndarray, data_indices: OrderedSet[int], preselection: ndarray, weights: ndarray, key_selector: KeySelector) -> None:
    logger = getLogger(__name__)

    empty_columns = get_empty_columns(data, data_indices, preselection)
    if len(empty_columns) > 0:
      old_count = data.shape[1]
      data: ndarray = np.delete(data, empty_columns, axis=1)
      preselection: ndarray = np.delete(preselection, empty_columns, axis=0)
      weights: ndarray = np.delete(weights, empty_columns, axis=0)
      logger.info(
          f"Removed {len(empty_columns)} out of {old_count} columns.")
      del empty_columns

    super().__init__(
      data=data,
      preselection=preselection,
      data_indices=data_indices,
      weights=weights,
      key_selector=key_selector,
    )
