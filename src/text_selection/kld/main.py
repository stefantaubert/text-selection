from logging import getLogger
from typing import Dict, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from ordered_set import OrderedSet
from text_selection.kld.distribution_factories import \
    UniformDistributionFactory
from text_selection.kld.durations_iterator import iterate_durations_dict
from text_selection.kld.filter_durations import get_duration_keys
from text_selection.kld.mapping_iterator import MappingIterator
from text_selection.kld.ngram_extractor import NGramExtractor
from text_selection.kld.optimized_kld_iterator import OptimizedKldIterator
from text_selection.selection import FirstKeySelector
from text_selection.utils import DurationBoundary


def greedy_kld_uniform_ngrams_seconds_with_preselection_perf(data: Dict[int, Tuple[str, ...]], select_from_keys: OrderedSet[int], preselection_keys: Set[int], n_gram: int, ignore_symbols: Optional[Set[str]], select_from_durations_s: Dict[int, float], seconds: float, duration_boundary: DurationBoundary, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> None:
  logger = getLogger(__name__)

  select_from_keys = get_duration_keys(select_from_durations_s, select_from_keys, duration_boundary)

  ngram_extractor = NGramExtractor(data, n_jobs, maxtasksperchild, chunksize, batches)
  ngram_extractor.fit(select_from_keys | preselection_keys, n_gram, ignore_symbols)
  all_data_counts = ngram_extractor.predict(select_from_keys)
  all_preselected_counts = ngram_extractor.predict(preselection_keys)
  summed_preselection_counts: NDArray = np.sum(all_preselected_counts, axis=0)
  del all_preselected_counts
  del ngram_extractor

  with OptimizedKldIterator(
    data=all_data_counts,
    preselection=summed_preselection_counts,
    data_indicies=OrderedSet(range(len(all_data_counts))),
    key_selector=FirstKeySelector(),
    distribution_factory=UniformDistributionFactory(),
    n_jobs=n_jobs,
    maxtasksperchild=maxtasksperchild,
    chunksize=chunksize,
    batches=batches,
  ) as iterator:
    logger.info(f"Target (uniform) distribution: {iterator.get_target_distribution()[0]}")
    logger.info(f"Initial Kullback-Leibler divergence: {iterator.current_kld}")
    key_index_mapping = {index: key for index, key in enumerate(select_from_keys)}
    mapping_iterator = MappingIterator(iterator, key_index_mapping)
    greedy_selected, enough_data_was_available = iterate_durations_dict(
      mapping_iterator, select_from_durations_s, seconds)

    if not enough_data_was_available:
      logger.warning(
        f"Aborted since no further data had been available!")

    logger.info(f"Final Kullback-Leibler divergence: {iterator.previous_kld}")

  result = OrderedSet(greedy_selected)
  return result
