import math
from collections import Counter
from dataclasses import dataclass
from logging import Logger, getLogger
from math import inf
from typing import Dict, Generator, Iterable, List, Literal, Optional, Set, Tuple, Union

from ordered_set import OrderedSet
from tqdm import tqdm

from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import TQDM_LINE_UNIT, ExecutionResult
from text_selection_core.helper import get_percent_str, split_adv, xtqdm
from text_selection_core.types import (LineNr, Lines, Subset, get_subsets_line_nrs,
                                       get_subsets_line_nrs_count, get_subsets_line_nrs_gen,
                                       move_lines_to_subset)


@dataclass()
class LineUnitFrequencyFilterParameters():
  lines: Lines
  ssep: str
  from_count_incl: int
  to_count_excl: Union[int, float]
  mode: Literal["all", "any"]


def filter_lines_with_line_unit_frequencies(default_params: SelectionDefaultParameters, params: LineUnitFrequencyFilterParameters, logger: Logger) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error, False

  select_from_nrs = get_subsets_line_nrs_gen(
    default_params.dataset, default_params.from_subset_names)
  select_from_count = get_subsets_line_nrs_count(
    default_params.dataset, default_params.from_subset_names)

  if select_from_count == 0:
    logger.info("Nothing to select from.")
    return None, False

  select_from_nrs = tqdm(select_from_nrs, desc="Filtering",
                         unit=TQDM_LINE_UNIT, total=select_from_count)

  result: Subset = OrderedSet()
  if params.mode == "all":
    method = all
  elif params.mode == "any":
    method = any
  else:
    assert False

  result = OrderedSet()
  for line_nr in select_from_nrs:
    words = split_adv(params.lines[line_nr], params.ssep)
    word_counts = Counter(words)

    match = method(params.from_count_incl <= count < params.to_count_excl for count in word_counts.values())
    if match:
      result.add(line_nr)
      logger.debug(
        f"Filtered L{line_nr+1}: \"{params.lines[line_nr]}\".")
      result.add(line_nr)

  changed_anything = False
  if len(result) > 0:
    logger.info(
      f"Filtered {len(result)} out of {len(select_from_nrs)} lines ({get_percent_str(len(result),len(select_from_nrs))}). {len(select_from_nrs)-len(result)} lines remain.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
    changed_anything = True
  return None, changed_anything
