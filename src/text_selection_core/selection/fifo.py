from logging import getLogger
from typing import Generator, Iterator

from ordered_set import OrderedSet
from text_selection_core.globals import ExecutionResult
from text_selection_core.helper import (get_initial_weights,
                                        get_target_weights_from_percent)
from text_selection_core.types import (DataId, DataIds, Dataset, DataWeights,
                                       Subset, SubsetName, Weight, get_subsets_ids)
from text_selection_core.validation import (InvalidPercentualValueError,
                                            NonDivergentSubsetsError,
                                            SubsetNotExistsError,
                                            WeightsDoNotContainAllKeysError)
from text_selection_core.weights.weights_iterator import WeightsIterator


def select_fifo(dataset: Dataset, from_subset_names: OrderedSet[SubsetName], to_subset_name: SubsetName, weights: DataWeights, target: Weight, target_incl_selection: bool, target_percent: bool) -> ExecutionResult:
  assert target >= 0

  if error := SubsetNotExistsError.validate_names(dataset, from_subset_names):
    return error, False

  if error := SubsetNotExistsError.validate(dataset, to_subset_name):
    return error, False

  if error := NonDivergentSubsetsError.validate_names(from_subset_names, to_subset_name):
    return error, False

  if error := WeightsDoNotContainAllKeysError.validate(dataset, weights):
    return error, False

  if target_percent:
    if error := InvalidPercentualValueError.validate(target):
      return error, False

  from_ids = OrderedSet(get_subsets_ids(dataset, from_subset_names))
  to_subset = dataset.subsets[to_subset_name]
  assert len(from_ids.intersection(to_subset)) == 0

  initial_weights = get_initial_weights(to_subset, weights, target_incl_selection)

  if target_percent:
    target = get_target_weights_from_percent(
        from_ids, to_subset, weights, target, target_incl_selection)

  iterator = get_fifo_iterator(from_ids)
  weights_iterator = WeightsIterator(iterator, weights, target, initial_weights)
  result: Subset = OrderedSet(weights_iterator)
  changed_anything = False
  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Selected {len(result)} Id's.")
    dataset.move_ids_to_subset(result, to_subset_name)
    changed_anything = True
  return None, changed_anything


def get_fifo_iterator(ids: DataIds) -> Iterator[DataId]:
  return iter(ids)


def get_fifo_original_positions_iterator(subset: Subset, ids: DataIds) -> Iterator[DataId]:
  ordered_subset = ids.intersection(subset)
  return iter(ordered_subset)


def get_fifo_id_iterator(subset: Subset) -> Iterator[DataId]:
  ordered_subset = sorted(subset)
  return iter(ordered_subset)
