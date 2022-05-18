from logging import Logger, getLogger
from typing import Generator, Iterator, Optional, TypeVar

from ordered_set import OrderedSet

from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.helper import get_percent_str
from text_selection_core.types import (LineNr, Lines, Subset, get_subsets_line_nrs_count,
                                       get_subsets_line_nrs_gen, move_lines_to_subset)
from text_selection_core.validation import LinesCountNotMatchingError


def filter_duplicates(default_params: SelectionDefaultParameters, lines: Lines, logger: Optional[Logger]) -> ExecutionResult:
  if logger is None:
    logger = getLogger(__name__)

  if error := validate_selection_default_parameters(default_params):
    return error, False

  if error := LinesCountNotMatchingError.validate(default_params.dataset, lines):
    return error, False

  select_from_line_nrs = get_subsets_line_nrs_gen(
    default_params.dataset, default_params.from_subset_names)
  select_from_count = get_subsets_line_nrs_count(
    default_params.dataset, default_params. from_subset_names)

  duplicates = get_matching_lines(lines, select_from_line_nrs)
  result: Subset = OrderedSet(duplicates)

  if changed_anything := len(result) > 0:
    logger.info(
      f"Filtered {len(result)} out of {select_from_count} lines ({get_percent_str(len(result),select_from_count)}). {select_from_count-len(result)} lines remain.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
    for line_nr in result:
      logger.debug(f"Filtered L{line_nr+1}: \"{lines[line_nr]}\".")
  else:
    logger.info("No duplicate lines exist!")

  return None, changed_anything


def get_matching_lines(lines: Lines, line_nrs: Iterator[LineNr]) -> Generator[LineNr, None, None]:
  collected = set()
  for line_nr in line_nrs:
    item = lines[line_nr]
    if item in collected:
      yield line_nr
      continue
    collected.add(item)
