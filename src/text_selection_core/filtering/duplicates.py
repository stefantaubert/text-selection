from logging import getLogger
from typing import Generator, Iterable, Tuple, TypeVar

from ordered_set import OrderedSet
from text_selection_core.globals import ExecutionResult
from text_selection_core.types import Dataset, DataSymbols, Subset, SubsetName, get_subsets_ids
from text_selection_core.validation import (NonDivergentSubsetsError,
                                            SubsetNotExistsError)


def select_duplicates(dataset: Dataset, from_subset_names: OrderedSet[SubsetName], to_subset_name: SubsetName, data_symbols: DataSymbols) -> ExecutionResult:
  if error := SubsetNotExistsError.validate_names(dataset, from_subset_names):
    return error, False

  if error := SubsetNotExistsError.validate(dataset, to_subset_name):
    return error, False

  if error := NonDivergentSubsetsError.validate_names(from_subset_names, to_subset_name):
    return error, False

  select_from = ((data_id, data_symbols[data_id])
                 for data_id in get_subsets_ids(dataset, from_subset_names))
  duplicates = get_duplicates(select_from)
  result: Subset = OrderedSet(duplicates)
  changed_anything = False
  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Selected {len(result)} Id's.")
    dataset.move_ids_to_subset(result, to_subset_name)
    changed_anything = True
  return None, changed_anything


T = TypeVar("T")


def get_duplicates(items: Iterable[Tuple[T, str]]) -> Generator[T, None, None]:
  collected = set()
  for key, item in items:
    if item in collected:
      yield key
      continue
    collected.add(item)
