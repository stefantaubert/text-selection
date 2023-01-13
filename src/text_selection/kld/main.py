from logging import getLogger
from typing import Dict, Optional, Set, Tuple

import numpy as np
from ordered_set import OrderedSet
from tqdm import tqdm

from text_selection.common.durations_iterator import UntilProxyIterator
from text_selection.common.filter_durations import get_duration_keys
from text_selection.common.helper import get_ngram_counts
from text_selection.common.mapping_iterator import MappingIterator
from text_selection.kld.custom_kld_iterator import CustomKldIterator
from text_selection.kld.kld_iterator import get_uniform_weights
from text_selection.selection import FirstKeySelector
from text_selection.utils import DurationBoundary


def greedy_kld_uniform_ngrams_seconds_with_preselection_perf(data: Dict[int, Tuple[str, ...]], select_from_keys: OrderedSet[int], preselection_keys: Set[int], n_gram: int, ignore_symbols: Optional[Set[str]], select_from_durations_s: Dict[int, float], seconds: float, duration_boundary: DurationBoundary, n_jobs: int, maxtasksperchild: Optional[int], chunksize: Optional[int], batches: Optional[int]) -> OrderedSet[int]:
  logger = getLogger(__name__)

  select_from_keys = get_duration_keys(select_from_durations_s, select_from_keys, duration_boundary)

  all_data_counts, summed_preselection_counts = get_ngram_counts(
    data, select_from_keys, preselection_keys, n_gram, ignore_symbols, n_jobs, maxtasksperchild, chunksize, batches)

  selector = FirstKeySelector()
  weights = get_uniform_weights(all_data_counts.shape[1])
  kld_iterator = CustomKldIterator(
    data=all_data_counts,
    data_indices=OrderedSet(range(len(all_data_counts))),
    preselection=summed_preselection_counts,
    key_selector=selector,
    weights=weights,
  )

  until_values = np.array([select_from_durations_s[key] for key in select_from_keys])
  until_iterator = UntilProxyIterator(
    iterator=kld_iterator,
    until_values=until_values,
    until_value=seconds,
  )

  key_index_mapping = {index: key for index, key in enumerate(select_from_keys)}
  mapping_iterator = MappingIterator(until_iterator, key_index_mapping)

  logger.info(f"Initial Kullback-Leibler divergence: {kld_iterator.current_kld}")

  result = OrderedSet()

  with tqdm(total=round(seconds), desc="Selected duration", ncols=200, unit="s") as progress_bar_seconds:
    with tqdm(desc="Iterations", unit="it") as progress_bar_iterations:
      for item in mapping_iterator:
        result.add(item)
        progress_bar_seconds.update(round(until_iterator.tqdm_update))
        progress_bar_iterations.update()

  if not until_iterator.was_enough_data_available:
    logger.warning("Didn't had enough data!")

  logger.info(f"Final Kullback-Leibler divergence: {kld_iterator.previous_kld}")

  return result
