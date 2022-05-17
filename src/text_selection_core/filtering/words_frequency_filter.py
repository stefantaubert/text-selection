from collections import Counter
from dataclasses import dataclass
from logging import getLogger
from typing import Generator, Iterable, List, Set, Tuple

from ordered_set import OrderedSet
from tqdm import tqdm

from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.helper import split_adv, xtqdm
from text_selection_core.types import (LineNr, Lines, Subset, get_subsets_line_nrs_gen,
                                       move_lines_to_subset)


@dataclass()
class CountFilterParameters():
  lines: Lines
  ssep: str
  from_count_incl: int
  to_count_excl: int


def filter_lines_with_frequencies(default_params: SelectionDefaultParameters, params: CountFilterParameters) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error, False

  counters: List[Counter] = []
  for line in tqdm(params.lines, desc="Calculating counts", unit=" line(s)"):
    line_counts = Counter(split_adv(line, params.ssep))
    counters.append(line_counts)

  select_from = ((line_nr, params.lines[line_nr])
                 for line_nr in get_subsets_line_nrs_gen(default_params.dataset, default_params. from_subset_names))
  counter = get_counter(select_from, params.ssep, params.trim_symbols, params.ignore_case)

  select_from = ((line_nr, params.lines[line_nr])
                 for line_nr in get_subsets_line_nrs_gen(default_params.dataset, default_params. from_subset_names))
  items = get_matching_items(select_from, params.ssep, params.trim_symbols,
                             params.ignore_case, params.from_count_incl, params.to_count_excl, counter)
  result: Subset = OrderedSet(items)
  changed_anything = False
  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Filtered {len(result)} lines.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
    changed_anything = True
  return None, changed_anything


def get_counter(items: Iterable[Tuple[LineNr, str]], word_sep: str, trim_symbols: Set[str], ignore_case: bool) -> Generator[LineNr, None, None]:
  assert len(word_sep) > 0

  words = (
      word
      for _, sentence in items
      for word in sentence.split(word_sep)
      if word != ""
  )

  if len(trim_symbols) > 0:
    trim_symbols = ''.join(trim_symbols)
    words = (word.strip(trim_symbols) for word in words)

  if ignore_case:
    words = (word.lower() for word in words)

  logger = getLogger(__name__)
  logger.debug("Counting words...")
  counter = Counter(tqdm(words))

  return counter


def get_matching_items(items: Iterable[Tuple[LineNr, str]], word_sep: str, trim_symbols: Set[str], ignore_case: bool, from_count_incl: int, to_count_excl: int, counter: Counter) -> Generator[LineNr, None, None]:
  assert len(word_sep) > 0

  items = (
      (data_id, (word for word in sentence.split(word_sep) if word != ""))
      for data_id, sentence in items
  )

  if len(trim_symbols) > 0:
    trim_symbols = ''.join(trim_symbols)
    items = (
      (data_id, (word.strip(trim_symbols) for word in words))
      for data_id, words in items
    )

  items = (
    (data_id, (''.join(word) for word in words))
    for data_id, words in items
  )

  if ignore_case:
    items = (
      (data_id, (word.lower() for word in words))
      for data_id, words in items
    )

  items_in_range = (
    (data_id, all(
      from_count_incl <= counter[word] < to_count_excl for word in words
    ))
    for data_id, words in items
  )

  for data_id, all_words_in_range in items_in_range:
    if all_words_in_range:
      yield data_id
