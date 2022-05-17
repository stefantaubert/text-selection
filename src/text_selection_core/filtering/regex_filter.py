import re
from logging import Logger, getLogger
from typing import Generator, Iterable, Tuple, TypeVar

from ordered_set import OrderedSet

from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.types import Lines, Subset, get_subsets_line_nrs_gen, move_lines_to_subset


def filter_regex_pattern(default_params: SelectionDefaultParameters, lines: Lines, pattern: str, logger: Logger) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error, False

  select_from = ((line_nr, lines[line_nr])
                 for line_nr in get_subsets_line_nrs_gen(default_params.dataset, default_params. from_subset_names))
  logger.debug(pattern)
  re_pattern = re.compile(pattern)
  items = get_matching_items(select_from, re_pattern)
  result: Subset = OrderedSet(items)
  changed_anything = False
  if len(result) > 0:
    logger.debug(f"Filtered {len(result)} lines.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
    for line_nr in result:
      logger.debug(f"Filtered L{line_nr+1}: \"{lines[line_nr]}\".")
    changed_anything = True
  return None, changed_anything


T = TypeVar("T")


def get_matching_items(items: Iterable[Tuple[T, str]], pattern: re.Pattern) -> Generator[T, None, None]:
  for key, item in items:
    if pattern.match(item) is not None:
      yield key
