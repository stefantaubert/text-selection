from typing import Dict, Optional, Set, Tuple

import numpy as np
from ordered_set import OrderedSet
from text_selection.common.ngram_extractor import NGramExtractor


def get_ngram_counts(data: Dict[int, Tuple[str, ...]], select_from_keys: OrderedSet[int], preselection_keys: Set[int], n_gram: int, ignore_symbols: Optional[Set[str]], n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]):
  ngram_extractor = NGramExtractor(data, n_jobs, maxtasksperchild, chunksize, batches)
  ngram_extractor.fit(select_from_keys | preselection_keys, n_gram, ignore_symbols)
  all_data_counts = ngram_extractor.predict(select_from_keys)
  all_preselected_counts = ngram_extractor.predict(preselection_keys)
  del ngram_extractor
  summed_preselection_counts: np.ndarray = np.sum(all_preselected_counts, axis=0)
  del all_preselected_counts
  return all_data_counts, summed_preselection_counts
