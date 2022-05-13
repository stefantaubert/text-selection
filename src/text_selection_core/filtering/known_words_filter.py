from dataclasses import dataclass
from logging import getLogger
from typing import Generator, Iterable, Literal, Set, Tuple

import enchant
from ordered_set import OrderedSet

from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.types import (Line, LineNr, Lines, Subset, get_subsets_line_nrs,
                                       move_lines_to_subset)

enchant_en_US = Literal["en_US"]


@dataclass()
class Parameters():
  symbols: Lines
  word_sep: str
  trim_symbols: Set[str]
  language: Literal[enchant_en_US]


def filter_sentences_containing_only_known_words(default_params: SelectionDefaultParameters, params: Parameters) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error, False

  select_from = ((data_id, params.data_symbols[data_id])
                 for data_id in get_subsets_line_nrs(default_params.dataset, default_params. from_subset_names))

  assert params.language == enchant_en_US
  lexicon = enchant.Dict(params.language)

  items = get_matching_items(select_from, params.word_sep, params.trim_symbols, lexicon)
  result: Subset = OrderedSet(items)
  changed_anything = False
  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Filtered {len(result)} lines.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
    changed_anything = True
  return None, changed_anything


def get_matching_items(items: Iterable[Tuple[LineNr, Line]], word_sep: str, trim_symbols: Set[str], lexicon: enchant.Dict) -> Generator[LineNr, None, None]:
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
    (data_id, (word for word in words))
    for data_id, words in items
  )

  # Note: all on first false occurrence: https://github.com/python/cpython/blob/b9d8980d89bfaa4bf16d60f0488adcc9d2cbf5ef/Doc/library/functions.rst#id7
  items = (
    (data_id, all(lexicon.check(word) for word in words))
    for data_id, words in items
  )

  for data_id, contains_only_known_words in items:
    if contains_only_known_words:
      yield data_id
