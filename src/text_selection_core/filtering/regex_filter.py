import re
from logging import Logger, getLogger
from typing import Generator, Iterable, Iterator, Tuple, TypeVar

from ordered_set import OrderedSet

from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.helper import get_percent_str
from text_selection_core.types import (LineNr, LineNrs, Lines, Subset, get_subsets_line_nrs_count,
                                       get_subsets_line_nrs_gen, move_lines_to_subset)


def filter_regex_pattern(default_params: SelectionDefaultParameters, lines: Lines, pattern: str, logger: Logger) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error, False

  select_from_nrs = get_subsets_line_nrs_gen(
    default_params.dataset, default_params.from_subset_names)
  select_from_count = get_subsets_line_nrs_count(
    default_params.dataset, default_params.from_subset_names)
  logger.debug(f"Pattern: {pattern}")
  re_pattern = re.compile(pattern)
  items = get_matching_lines(lines, select_from_nrs, re_pattern)
  result: Subset = OrderedSet(items)
  changed_anything = False
  if len(result) > 0:
    logger.info(
      f"Filtered {len(result)} out of {select_from_count} lines ({get_percent_str(len(result),select_from_count)}). {select_from_count-len(result)} lines remain.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
    for line_nr in result:
      logger.debug(f"Filtered L{line_nr+1}: \"{lines[line_nr]}\".")
    changed_anything = True
  return None, changed_anything


def get_matching_lines(lines: Lines, line_nrs: Iterator[LineNr], pattern: re.Pattern) -> Generator[LineNr, None, None]:
  for line_nr in line_nrs:
    if pattern.match(lines[line_nr]) is not None:
      yield line_nr
