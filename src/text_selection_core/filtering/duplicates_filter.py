from logging import Logger, getLogger
from typing import Generator, Iterable, Iterator, Optional, Tuple, TypeVar

from ordered_set import OrderedSet

from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.types import (LineNr, Lines, Subset, get_subsets_line_nrs,
                                       move_lines_to_subset)
from text_selection_core.validation import LinesCountNotMatchingError


def filter_duplicates(default_params: SelectionDefaultParameters, lines: Lines, logger: Optional[Logger]) -> ExecutionResult:
  if logger is None:
    logger = getLogger(__name__)

  if error := validate_selection_default_parameters(default_params):
    return error, False

  if error := LinesCountNotMatchingError.validate(default_params.dataset, lines):
    return error, False

  line_nrs = get_subsets_line_nrs(default_params.dataset, default_params.from_subset_names)
  duplicates = get_duplicates_line_nrs(lines, line_nrs)
  result: Subset = OrderedSet(duplicates)

  if changed_anything := len(result) > 0:
    logger.debug(f"Filtered {len(result)} line(s).")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)

  return None, changed_anything


T = TypeVar("T")


def get_duplicates_line_nrs(lines: Lines, line_nrs: Iterator[LineNr]) -> Generator[T, None, None]:
  collected = set()
  for line_nr in line_nrs:
    item = lines[line_nr]
    if item in collected:
      yield line_nr
      continue
    collected.add(item)
