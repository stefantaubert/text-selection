from logging import getLogger
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from ordered_set import OrderedSet
from text_selection.kld.kld_iterator import KldCoreIterator
from text_selection.selection import KeySelector


class OptimizedKldIterator(KldCoreIterator):
  def __init__(self, data: np.ndarray, data_indicies: OrderedSet[int], preselection: np.ndarray, weights: np.ndarray, key_selector: KeySelector, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> None:
    # remove empty columns, can only occur if len(symbols in utterance) = n_gram - 1
    data_counts: NDArray = np.sum(data[data_indicies], axis=0)
    all_counts: NDArray = data_counts + preselection
    del data_counts
    remove_ngram_indicies = np.where(all_counts == 0)[0]
    del all_counts

    if len(remove_ngram_indicies) > 0:
      logger = getLogger(__name__)
      logger.info(
        f"Removing {len(remove_ngram_indicies)} out of {self._data.shape[1]} columns...")
      data: np.ndarray = np.delete(data, remove_ngram_indicies, axis=1)
      preselection: np.ndarray = np.delete(preselection, remove_ngram_indicies, axis=0)
      weights: np.ndarray = np.delete(weights, remove_ngram_indicies, axis=0)
      logger.info("Done.")
    del remove_ngram_indicies

    super().__init__(
      data=data,
      preselection=preselection,
      data_indicies=data_indicies,
      weights=weights,
      batches=batches,
      chunksize=chunksize,
      key_selector=key_selector,
      maxtasksperchild=maxtasksperchild,
      n_jobs=n_jobs,
    )
