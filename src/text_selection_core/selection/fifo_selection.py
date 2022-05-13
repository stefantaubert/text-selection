from logging import getLogger
from typing import Literal

from ordered_set import OrderedSet

from text_selection_core.algorithms.fifo import (get_fifo_original_positions_iterator,
                                                 get_fifo_subset_iterator)
from text_selection_core.common import (SelectionDefaultParameters, WeightSelectionParameters,
                                        validate_selection_default_parameters,
                                        validate_weights_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.helper import get_initial_weights, get_target_weights_from_percent
from text_selection_core.types import Subset, get_subsets_line_nrs, move_lines_to_subset
from text_selection_core.weights.weights_iterator import WeightsIterator

original_mode = "original"
subset_mode = "subset"


def select_fifo(default_params: SelectionDefaultParameters, weight_params: WeightSelectionParameters, mode: Literal["original", "subset"]) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error, False

  if error := validate_weights_parameters(weight_params, default_params.dataset):
    return error, False

  from_ids = OrderedSet(get_subsets_line_nrs(default_params.dataset,
                        default_params.from_subset_names))

  to_subset = default_params.dataset.subsets[default_params.to_subset_name]
  assert len(from_ids.intersection(to_subset)) == 0

  initial_weights = get_initial_weights(
    to_subset, weight_params.weights, weight_params.target_incl_selection)

  if weight_params.target_percent:
    weight_params.target = get_target_weights_from_percent(
        from_ids, to_subset, weight_params.weights, weight_params.target, weight_params.target_incl_selection)

  if mode == subset_mode:
    iterator = get_fifo_subset_iterator(from_ids)
  elif mode == original_mode:
    original_line_nrs = OrderedSet(default_params.dataset.get_line_nrs())
    iterator = get_fifo_original_positions_iterator(from_ids, original_line_nrs)
  else:
    assert False

  weights_iterator = WeightsIterator(
    iterator, weight_params.weights, weight_params.target, initial_weights)

  result: Subset = OrderedSet(weights_iterator)
  changed_anything = False

  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Selected {len(result)} lines.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
    changed_anything = True

  return None, changed_anything
