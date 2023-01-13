import re
from collections import Counter
from logging import Logger
from typing import Generator, Iterator, List, Tuple

from ordered_set import OrderedSet
from tqdm import tqdm

from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import TQDM_LINE_UNIT, ExecutionResult
from text_selection_core.helper import get_percent_str
from text_selection_core.types import (LineNr, Lines, Subset, get_subsets_line_nrs_count,
                                       get_subsets_line_nrs_gen, move_lines_to_subset)
from text_selection_core.validation import (ErrorType, ValidationErr,
                                            ensure_lines_count_matches_dataset)


def filter_regex_pattern(default_params: SelectionDefaultParameters, lines: Lines, pattern: str, mode: str, logger: Logger) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error

  if error := ensure_lines_count_matches_dataset(default_params.dataset, lines):
    return error

  select_from_nrs = get_subsets_line_nrs_gen(
    default_params.dataset, default_params.from_subset_names)
  select_from_count = get_subsets_line_nrs_count(
    default_params.dataset, default_params.from_subset_names)

  try:
    re_pattern = re.compile(pattern)
  except re.error as error:
    return ValidationErr(ErrorType.INVALID_PATTERN, error.args[0])

  logger.debug(f"Pattern: {re_pattern.pattern}")

  method = None
  if mode == "match":
    method = get_matching_lines_match
  elif mode == "find":
    method = get_matching_lines_find
  else:
    assert False
  assert method is not None

  result: Subset = OrderedSet()
  # unique_matches = set()
  matches_counter = Counter()
  with tqdm(select_from_nrs, desc="Filtering", unit=TQDM_LINE_UNIT, total=select_from_count) as select_from_nrs:
    for line_nr, matched_strs in method(lines, select_from_nrs, re_pattern):
      result.add(line_nr)
      logger.info(f"Filtered L-{line_nr+1}: \"{lines[line_nr]}\".")
      del line_nr
      for match in matched_strs:
        logger.info(f"Matched: \"{match}\"")
        del match
      matches_counter.update(matched_strs)
      # unique_matches |= matched_strs
      del matched_strs

  if len(matches_counter) > 0:
    logger.info(f"Matched the following texts (#{len(matches_counter)}):")
  for unique_match, match_count in sorted(matches_counter.items(), key=lambda kv: (-kv[1], kv[0])):
    logger.info(f"- \"{unique_match}\" (#{match_count})")
    del unique_match
  del matches_counter

  changed_anything = False
  if len(result) > 0:
    logger.info(
      f"Filtered {len(result)} out of {select_from_count} lines ({get_percent_str(len(result),select_from_count)}). {select_from_count-len(result)} lines remain.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
    changed_anything = True
  return changed_anything


def get_matching_lines_match(lines: Lines, line_nrs: Iterator[LineNr], pattern: re.Pattern) -> Generator[Tuple[LineNr, List[str]], None, None]:
  for line_nr in line_nrs:
    if (match := pattern.match(lines[line_nr])) is not None:
      matches = []
      for group in match.groups():
        if group is not None and group != "":
          matches.append(group)
        del group
      yield line_nr, matches


def get_matching_lines_find(lines: Lines, line_nrs: Iterator[LineNr], pattern: re.Pattern) -> Generator[Tuple[LineNr, List[str]], None, None]:
  for line_nr in line_nrs:
    if len(matches := pattern.findall(lines[line_nr])) > 0:
      all_matches = []
      for m in matches:
        if isinstance(m, tuple):
          all_matches.extend(m)
        else:
          assert isinstance(m, str)
          all_matches.append(m)
      all_matches = [m for m in all_matches if m != ""]
      yield line_nr, all_matches
