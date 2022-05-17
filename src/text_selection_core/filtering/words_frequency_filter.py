import math
from collections import Counter
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Dict, Generator, Iterable, List, Literal, Optional, Set, Tuple

from ordered_set import OrderedSet
from tqdm import tqdm

from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.helper import split_adv, xtqdm
from text_selection_core.types import (LineNr, Lines, Subset, get_subsets_line_nrs,
                                       get_subsets_line_nrs_gen, move_lines_to_subset)


@dataclass()
class CountFilterParameters():
  lines: Lines
  ssep: str
  from_count_incl: int
  to_count_excl: Optional[int]
  count_whole: bool
  mode: Literal["all", "any"]


def filter_lines_with_unit_frequencies(default_params: SelectionDefaultParameters, params: CountFilterParameters, logger: Logger) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error, False

  from_line_nrs = get_subsets_line_nrs(default_params.dataset, default_params.from_subset_names)

  if params.count_whole:
    line_nrs_to_count = default_params.dataset.get_line_nrs()
  else:
    line_nrs_to_count = from_line_nrs

  # units = (
  #   unit
  #   for line_nr in line_nrs_to_count
  #   for unit in split_adv(params.lines[line_nr], params.ssep)
  # )

  # counter = Counter(tqdm(units, desc="Calculating counts", unit=" unit(s)"))

  counters: Dict[LineNr, Counter] = {}
  for line_nr in tqdm(line_nrs_to_count, desc="Calculating counts", unit=" line(s)"):
    line_counts = Counter(split_adv(params.lines[line_nr], params.ssep))
    counters[line_nr] = line_counts

  total_counter = Counter()
  for counter in tqdm(counters.values(), desc="Summing counts", unit=" line(s)"):
    total_counter.update(counter)

  to_count = params.to_count_excl
  if to_count is None:
    to_count = math.inf

  result: Subset = OrderedSet()
  if params.mode == "all":
    method = all
  elif params.mode == "any":
    method = any
  else:
    assert False

  for line_nr in tqdm(from_line_nrs, desc="Filtering", unit=" line(s)"):
    counts = counters[line_nr]
    word_counts = list(total_counter[unit] for unit in counts.keys())
    match = method(params.from_count_incl <= count < to_count for count in word_counts)
    if match:
      result.add(line_nr)

  changed_anything = False
  if len(result) > 0:
    logger.info(f"Filtered {len(result)} out of {len(from_line_nrs)} lines.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
    for line_nr in result:
      logger.debug(f"Filtered L{line_nr+1}: \"{params.lines[line_nr]}\".")
    changed_anything = True
  return None, changed_anything
