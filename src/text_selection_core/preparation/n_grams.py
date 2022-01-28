from logging import getLogger
from typing import Optional, Set

from ordered_set import OrderedSet
from text_selection.common.ngram_extractor2 import NGram, NGramExtractor2
from text_selection_core.types import (Dataset, DataSymbols, NGramSet,
                                       SubsetName, get_subsets_ids,
                                       item_to_symbols)
from text_utils import Symbol


def get_n_grams(dataset: Dataset, subset_names: OrderedSet[SubsetName], symbols: DataSymbols, n_gram: NGram, ignore_symbols: Set[Symbol], most_common: float, least_common: float, n_jobs: int, maxtasksperchild: Optional[int], chunksize: int) -> NGramSet:
  logger = getLogger(__name__)
  logger.debug("Getting ids...")
  keys = OrderedSet(get_subsets_ids(dataset, subset_names))
  logger.debug("Ok.")

  ngram_extractor = NGramExtractor2(n_jobs, maxtasksperchild, chunksize)

  logger.debug("Fitting...")
  data_symbols = (
    item_to_symbols(symbols[data_id])
    for data_id in keys
  )

  ngram_extractor.fit(data_symbols, n_gram, ignore_symbols)

  logger.debug("Predicting...")
  data_symbols = (
    item_to_symbols(symbols[data_id])
    for data_id in keys
  )

  data = ngram_extractor.predict(data_symbols, len(keys))

  result = NGramSet()
  result.data = data
  result.data_ids_to_indices = {i: data_id for i, data_id in enumerate(keys)}
  result.indices_to_data_ids = {data_id: i for i, data_id in enumerate(keys)}
  result.n_grams = ngram_extractor.ngram_nr_to_ngram
  del ngram_extractor
  return result
