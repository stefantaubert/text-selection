from logging import getLogger
from typing import Dict

import numpy as np
from ordered_set import OrderedSet
from text_selection.common.durations_iterator import UntilIterator
from text_selection.common.filter_durations import get_duration_keys
from text_selection.common.mapping_iterator import MappingIterator
from text_selection.random.random_iterator import RandomIterator
from text_selection.utils import DurationBoundary


def random_seconds_perf(select_from_keys: OrderedSet[int], select_from_durations_s: Dict[int, float], seconds: float, duration_boundary: DurationBoundary, seed: int) -> None:
  logger = getLogger(__name__)

  select_from_keys = get_duration_keys(select_from_durations_s, select_from_keys, duration_boundary)

  select_from_keys_np = OrderedSet(range(len(select_from_keys)))

  kld_iterator = RandomIterator(
    data_indices=select_from_keys_np,
    seed=seed,
  )

  until_values = np.array([select_from_durations_s[key] for key in select_from_keys])
  until_iterator = UntilIterator(
    iterator=kld_iterator,
    until_values=until_values,
    until_value=seconds,
  )

  key_index_mapping = {index: key for index, key in enumerate(select_from_keys)}
  mapping_iterator = MappingIterator(until_iterator, key_index_mapping)

  result = OrderedSet(mapping_iterator)

  if not until_iterator.was_enough_data_available:
    logger.warning(
      f"Aborted since no further data had been available!")

  return result
