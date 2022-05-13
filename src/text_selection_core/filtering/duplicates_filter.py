from logging import getLogger
from typing import Generator, Iterable, Tuple, TypeVar

from ordered_set import OrderedSet

from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.types import Lines, Subset, get_subsets_line_nrs, move_lines_to_subset


def filter_duplicates(default_params: SelectionDefaultParameters, data_symbols: Lines) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error, False

  select_from = ((line_nr, data_symbols[line_nr])
                 for line_nr in get_subsets_line_nrs(default_params.dataset, default_params.from_subset_names))
  duplicates = get_duplicates(select_from)
  result: Subset = OrderedSet(duplicates)
  changed_anything = False
  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Filtered {len(result)} lines.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
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
