from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Optional

import numpy as np
from ordered_set import OrderedSet

from text_selection.common.mapping_iterator import map_indices
from text_selection.greedy.greedy_epoch_iterator import EpochProxyIterator
from text_selection.greedy.optimized_greedy_iterator import OptimizedGreedyIterator
from text_selection.selection import SelectionMode
from text_selection_core.common import (SelectionDefaultParameters, WeightSelectionParameters,
                                        get_selector, validate_selection_default_parameters,
                                        validate_weights_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.helper import get_initial_weights, get_target_weights_from_percent
from text_selection_core.selection.symbol_extractor import get_array_mp
from text_selection_core.types import (Lines, Subset, get_subsets_line_nrs,
                                       get_subsets_line_nrs_gen, move_lines_to_subset)
from text_selection_core.weights.weights_iterator import WeightsIterator


@dataclass()
class GreedySelectionParameters():
  lines: Lines
  ssep: str
  consider_to_subset: bool
  id_selection: SelectionMode


def select_greedy(default_params: SelectionDefaultParameters, params: GreedySelectionParameters, weight_params: WeightSelectionParameters, logger: Logger) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error, False

  if error := validate_weights_parameters(weight_params, default_params.dataset):
    return error, False

  from_ids = get_subsets_line_nrs(default_params.dataset,
                                  default_params.from_subset_names)

  # if error := NGramsNotExistError.validate(params.lines, from_ids):
  #   return error, False

  select_from_indices = OrderedSet(map_indices(
    from_ids, params.lines.indices_to_data_ids))
  to_subset = default_params.dataset.subsets[default_params.to_subset_name]

  if params.consider_to_subset:
    # if error := NGramsNotExistError.validate(params.lines, from_ids):
    #   return error, False

    preselection_indices = map_indices(to_subset, params.lines.indices_to_data_ids)
    summed_preselection_counts = np.sum(params.lines.data[preselection_indices], axis=0)
  else:
    summed_preselection_counts = np.zeros(params.lines.data.shape[1])

  selector = get_selector(params.id_selection)

  iterator = OptimizedGreedyIterator(
    data=params.lines.data,
    data_indices=select_from_indices,
    preselection=summed_preselection_counts,
    key_selector=selector,
  )

  initial_weights = get_initial_weights(
    to_subset, weight_params.weights, weight_params.target_incl_selection)

  if weight_params.target_percent:
    weight_params.target = get_target_weights_from_percent(
        from_ids, to_subset, weight_params.weights, weight_params.target, weight_params.target_incl_selection)

  weights_iterator = WeightsIterator(
    iterator, weight_params.weights, weight_params.target, initial_weights)
  mapping_iterator = map_indices(weights_iterator, params.lines.data_ids_to_indices)

  result: Subset = OrderedSet(mapping_iterator)
  changed_anything = False

  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Selected {len(result)} lines.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
    changed_anything = True

  return None, changed_anything


def select_greedy_epochs(default_params: SelectionDefaultParameters, params: GreedySelectionParameters, epochs: int, chunksize: int, n_jobs: int, maxtasksperchild: Optional[int], logger: Logger) -> ExecutionResult:
  assert epochs > 0

  if error := validate_selection_default_parameters(default_params):
    return error, False

  from_line_nrs = OrderedSet(get_subsets_line_nrs_gen(default_params.dataset,
                                                      default_params.from_subset_names))

  # if error := NGramsNotExistError.validate(params.lines, from_ids):
  #   return error, False

  calc_line_nrs = from_line_nrs.copy()

  from_ids_mapping = dict(enumerate(from_line_nrs))

  if params.consider_to_subset:
    to_line_nrs = default_params.dataset.subsets[default_params.to_subset_name]
    to_ids_mapping = dict(enumerate(to_line_nrs, start=len(calc_line_nrs)))
    calc_line_nrs.update(to_line_nrs)

  data, symbols = get_array_mp(params.lines, calc_line_nrs,
                               params.ssep, logger, chunksize, n_jobs, maxtasksperchild)

  if params.consider_to_subset:
    # if error := NGramsNotExistError.validate(params.lines, from_ids):
    #   return error, False
    assert to_ids_mapping is not None
    preselection_data = data[list(to_ids_mapping.keys())]
    summed_preselection_counts = np.sum(preselection_data, axis=0)
  else:
    summed_preselection_counts = np.zeros(data.shape[1])

  data_indices = OrderedSet(from_ids_mapping.keys())
  selector = get_selector(params.id_selection)

  greedy_iterator = OptimizedGreedyIterator(
    data=data,
    data_indices=data_indices,
    preselection=summed_preselection_counts,
    key_selector=selector,
  )

  weights_iterator = EpochProxyIterator(greedy_iterator, epochs)
  mapping_iterator = map_indices(weights_iterator, from_ids_mapping)

  result: Subset = OrderedSet(mapping_iterator)

  if not weights_iterator.was_enough_data_available:
    warning = f"Not enough data was available! Stopped at epoch {greedy_iterator.current_epoch + 1}"
    if greedy_iterator.is_fresh_epoch:
      warning += " (not started yet)."
    else:
      warning += " (interrupted)."
    logger.warning(warning)

  changed_anything = False

  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Selected {len(result)} lines.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
    for line_nr in result:
      logger.debug(f"Selected L{line_nr}: \"{params.lines[line_nr]}\".")
    changed_anything = True

  return None, changed_anything
