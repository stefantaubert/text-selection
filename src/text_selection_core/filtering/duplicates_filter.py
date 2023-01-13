from logging import Logger, getLogger
from typing import Iterable, Optional

from ordered_set import OrderedSet
from tqdm import tqdm

from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import TQDM_LINE_UNIT, ExecutionResult
from text_selection_core.helper import get_percent_str
from text_selection_core.types import (LineNr, Lines, Subset, get_subsets_line_nrs_count,
                                       get_subsets_line_nrs_gen, move_lines_to_subset)
from text_selection_core.validation import ensure_lines_count_matches_dataset


def filter_duplicates(default_params: SelectionDefaultParameters, lines: Lines, limit: Optional[int], logger: Optional[Logger]) -> ExecutionResult:
  if logger is None:
    logger = getLogger(__name__)

  if error := validate_selection_default_parameters(default_params):
    return error

  if error := ensure_lines_count_matches_dataset(default_params.dataset, lines):
    return error

  select_from_line_nrs = get_subsets_line_nrs_gen(
    default_params.dataset, default_params.from_subset_names)
  select_from_count = get_subsets_line_nrs_count(
    default_params.dataset, default_params. from_subset_names)

  if limit is None:
    limit = select_from_count
  tqdm_total = min(limit, select_from_count)

  if tqdm_total != select_from_count:
    assert limit is not None
    logger.info(f"Limit to first {limit} lines.")

  with tqdm(select_from_line_nrs, desc="Filtering duplicates",
            unit=TQDM_LINE_UNIT, total=tqdm_total) as line_nrs:
    result: Subset = filter_without_hash(line_nrs, lines, limit, logger)

  del lines

  if changed_anything := len(result) > 0:
    logger.info(
      f"Filtered {len(result)} out of {select_from_count} lines ({get_percent_str(len(result),select_from_count)}). {select_from_count-len(result)} lines remain.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
  else:
    logger.info("No duplicate lines exist!")

  del result

  return changed_anything


# is inefficient: @100mio lines ~20 GB ram usage
# def filter_with_hash(line_nrs: Iterable[LineNr], encoding: str, lines: Lines, logger: Logger) -> OrderedSet:
#   result: Subset = OrderedSet()
#   collected = set()
#   for line_nr in line_nrs:
#     line = lines[line_nr]
#     line_hash = hashlib.sha256(line.encode(encoding)).hexdigest()
#     if line_hash in collected:
#       result.add(line_nr)
#       logger.info(f"Filtered L-{line_nr+1}: \"{line}\".")
#     else:
#       collected.add(line_hash)
#     del line
#     del line_hash
#     del line_nr
#   del collected


def filter_without_hash(line_nrs: Iterable[LineNr], lines: Lines, limit: int, logger: Logger) -> Subset:
  # @100mio lines ~5 GB ram usage
  result: Subset = OrderedSet()
  collected = set()
  for i, line_nr in enumerate(line_nrs):
    if i >= limit:
      del i
      del line_nr
      break
    line = lines[line_nr]
    if line in collected:
      result.add(line_nr)
      logger.info(f"Filtered L-{line_nr+1}: \"{line}\".")
    else:
      collected.add(line)
    del line
    del line_nr
    del i
  del collected
  return result
