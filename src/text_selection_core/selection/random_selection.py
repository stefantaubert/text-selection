from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Optional

import numpy as np
from ordered_set import OrderedSet
from tqdm import tqdm

from text_selection.common.mapping_iterator import map_indices
from text_selection.kld.custom_kld_iterator import CustomKldIterator
from text_selection.kld.kld_iterator import get_uniform_weights
from text_selection.random.random_iterator import RandomIterator
from text_selection.selection import SelectionMode
from text_selection_core.common import (SelectionDefaultParameters, WeightSelectionParameters,
                                        get_selector, validate_selection_default_parameters,
                                        validate_weights_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.helper import get_initial_weights, get_target_weights_from_percent
from text_selection_core.selection.symbol_extractor import get_array_mp
from text_selection_core.types import (Lines, Subset, create_subset_if_it_not_exists,
                                       get_subsets_line_nrs_gen, move_lines_to_subset)
from text_selection_core.validation import ensure_lines_count_matches_dataset
from text_selection_core.weights.weights_iterator import WeightsIterator


def select_random(default_params: SelectionDefaultParameters, weight_params: WeightSelectionParameters, seed: int, logger: Logger) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error

  if error := validate_weights_parameters(weight_params, default_params.dataset):
    return error

  changed_anything = create_subset_if_it_not_exists(
    default_params.dataset, default_params.to_subset_name, logger)

  from_ids = OrderedSet(get_subsets_line_nrs_gen(default_params.dataset,
                        default_params.from_subset_names))
  to_subset = default_params.dataset.subsets[default_params.to_subset_name]
  assert len(from_ids.intersection(to_subset)) == 0

  initial_weights = get_initial_weights(
    to_subset, weight_params.weights, weight_params.target_incl_selection)

  if weight_params.target_percent:
    weight_params.target = get_target_weights_from_percent(
        from_ids, to_subset, weight_params.weights, weight_params.target, weight_params.target_incl_selection)
    logger.debug(f"Selected target from percent: {weight_params.target}")

  iterator = RandomIterator(from_ids, seed)

  weights_iterator = WeightsIterator(
    iterator, weight_params.weights, weight_params.target, initial_weights, logger)

  result: Subset = OrderedSet()
  with tqdm(desc="Selecting weight", unit="it", total=weights_iterator.target_weight, initial=weights_iterator.current_weight) as pbar:
    for line_nr in weights_iterator:
      result.add(line_nr)
      logger.info(f"Selected L-{line_nr+1}.")
      pbar.update(weights_iterator.tqdm_update)

  if len(result) > 0:
    logger = getLogger(__name__)
    logger.info(f"Selected {len(result)} lines.")

    # logger.info(
    #   f"Filtered {len(result)} out of {select_from_count} lines ({get_percent_str(len(result),select_from_count)}). {select_from_count-len(result)} lines remain.")

    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
    changed_anything = True

  return changed_anything
