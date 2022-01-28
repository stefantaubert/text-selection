from logging import getLogger
from typing import Iterator, Literal, Union

from ordered_set import OrderedSet
from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters,
                                        validate_weights_parameters)
from text_selection_core.filtering.weights_filter import \
    WeightsFilterParameters
from text_selection_core.globals import ExecutionResult
from text_selection_core.helper import (get_initial_weights,
                                        get_target_weights_from_percent)
from text_selection_core.sorting.default import get_original_positions
from text_selection_core.types import (DataId, DataIds, Subset,
                                       get_subsets_ids, move_ids_to_subset)
from text_selection_core.weights.weights_iterator import WeightsIterator

original_mode = Literal["original"]
subset_mode = Literal["subset"]
id_mode = Literal["id"]


def select_fifo(default_params: SelectionDefaultParameters, weight_params: WeightsFilterParameters, mode: Union[original_mode, subset_mode, id_mode]) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error, False

  if error := validate_weights_parameters(weight_params, default_params.dataset):
    return error, False

  from_ids = OrderedSet(get_subsets_ids(default_params.dataset,
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
    iterator = get_original_positions(from_ids, default_params.dataset.ids)
  elif mode == id_mode:
    iterator = get_fifo_id_iterator(from_ids)
  else:
    assert False

  weights_iterator = WeightsIterator(
    iterator, weight_params.weights, weight_params.target, initial_weights)

  result: Subset = OrderedSet(weights_iterator)
  changed_anything = False

  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Selected {len(result)} Id's.")
    move_ids_to_subset(default_params.dataset, result, default_params.to_subset_name)
    changed_anything = True

  return None, changed_anything


def get_fifo_subset_iterator(ids: DataIds) -> Iterator[DataId]:
  return iter(ids)


def get_fifo_original_positions_iterator(subset: Subset, ids: DataIds) -> Iterator[DataId]:
  ordered_subset = ids.intersection(subset)
  return iter(ordered_subset)


def get_fifo_id_iterator(subset: Subset) -> Iterator[DataId]:
  ordered_subset = sorted(subset)
  return iter(ordered_subset)
