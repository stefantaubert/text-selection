from typing import Generator, Iterator

from ordered_set import OrderedSet
from text_selection_core.globals import ExecutionResult
from text_selection_core.types import (DataId, DataIds, Dataset, Subset,
                                       SubsetName)
from text_selection_core.validation import SubsetNotExistsError


def sort_fifo_after_id(dataset: Dataset, subset_names: OrderedSet[SubsetName]) -> ExecutionResult:
  if error := SubsetNotExistsError.validate_names(dataset, subset_names):
    return error, False

  for subset_name in subset_names:
    subset = dataset.subsets[subset_name]
    ordered_subset = OrderedSet(sorted(subset))
    dataset.subsets[subset_name] = ordered_subset


def sort_fifo_after_original_position(dataset: Dataset, subset_names: OrderedSet[SubsetName]) -> ExecutionResult:
  if error := SubsetNotExistsError.validate_names(dataset, subset_names):
    return error, False

  for subset_name in subset_names:
    subset = dataset.subsets[subset_name]
    result = OrderedSet(get_original_positions(subset, dataset.ids))
    dataset.subsets[subset_name] = result


def get_original_positions(subset: Subset, ids: DataIds) -> Iterator[DataId]:
  ordered_subset = ids.intersection(subset)
  return iter(ordered_subset)
