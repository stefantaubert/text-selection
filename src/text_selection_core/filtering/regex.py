import re
from logging import getLogger
from typing import Generator, Iterable, Tuple, TypeVar

from ordered_set import OrderedSet
from text_selection_core.globals import ExecutionResult
from text_selection_core.types import (Dataset, DataSymbols, Subset,
                                       SubsetName, item_to_text)
from text_selection_core.validation import (NonDivergentSubsetsError,
                                            SubsetNotExistsError)


def select_regex_match(dataset: Dataset, from_subset_names: OrderedSet[SubsetName], to_subset_name: SubsetName, data_symbols: DataSymbols, pattern: str) -> ExecutionResult:
  if error := SubsetNotExistsError.validate_names(dataset, from_subset_names):
    return error, False

  if error := SubsetNotExistsError.validate(dataset, to_subset_name):
    return error, False

  if error := NonDivergentSubsetsError.validate_names(from_subset_names, to_subset_name):
    return error, False

  from_subsets = (dataset[from_subset_name] for from_subset_name in from_subset_names)
  from_ids = (data_id for subset in from_subsets for data_id in subset)
  select_from = ((data_id, item_to_text(data_symbols[data_id])) for data_id in from_ids)
  re_pattern = re.compile(pattern)
  items = get_matching_items(select_from, re_pattern)
  result: Subset = OrderedSet(items)
  changed_anything = False
  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Selected {len(result)} Id's.")
    dataset.move_ids_to_subset(result, to_subset_name)
    changed_anything = True
  return None, changed_anything


T = TypeVar("T")


def get_matching_items(items: Iterable[Tuple[T, str]], pattern: re.Pattern) -> Generator[T, None, None]:
  for key, item in items:
    if pattern.match(item) is not None:
      yield key
