import re
from logging import getLogger
from typing import Generator, Iterable, Tuple, TypeVar

from ordered_set import OrderedSet
from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.types import (DataSymbols, Subset, get_subsets_ids,
                                       item_to_text, move_ids_to_subset)


def filter_regex_pattern(default_params: SelectionDefaultParameters, data_symbols: DataSymbols, pattern: str) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error, False

  select_from = ((data_id, item_to_text(data_symbols[data_id]))
                 for data_id in get_subsets_ids(default_params.dataset, default_params. from_subset_names))
  re_pattern = re.compile(pattern)
  items = get_matching_items(select_from, re_pattern)
  result: Subset = OrderedSet(items)
  changed_anything = False
  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Filtered {len(result)} Id's.")
    move_ids_to_subset(default_params.dataset, result, default_params.to_subset_name)
    changed_anything = True
  return None, changed_anything


T = TypeVar("T")


def get_matching_items(items: Iterable[Tuple[T, str]], pattern: re.Pattern) -> Generator[T, None, None]:
  for key, item in items:
    if pattern.match(item) is not None:
      yield key
