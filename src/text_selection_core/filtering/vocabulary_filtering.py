from dataclasses import dataclass
from functools import partial
from logging import Logger
from math import isinf
from typing import Callable, List, Literal, Set, Union

from ordered_set import OrderedSet
from tqdm import tqdm

from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import TQDM_LINE_UNIT, ExecutionResult
from text_selection_core.helper import get_percent_str, split_adv
from text_selection_core.types import (Lines, get_subsets_line_nrs_count, get_subsets_line_nrs_gen,
                                       move_lines_to_subset)
from text_selection_core.validation import ensure_lines_count_matches_dataset


@dataclass()
class VocabularyFilterParameters():
  lines: Lines
  ssep: str
  from_count_incl: int
  to_count_excl: Union[int, float]
  vocabulary: Set[str]
  mode: Literal["iv", "oov"]


def filter_lines_with_vocabulary_frequencies(default_params: SelectionDefaultParameters, params: VocabularyFilterParameters, logger: Logger) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error

  if error := ensure_lines_count_matches_dataset(default_params.dataset, params.lines):
    return error

  select_from_nrs = get_subsets_line_nrs_gen(
    default_params.dataset, default_params.from_subset_names)
  select_from_count = get_subsets_line_nrs_count(
    default_params.dataset, default_params.from_subset_names)

  if select_from_count == 0:
    logger.info("Nothing to select from.")
    return False

  if params.mode == "iv":
    method = get_match_iv_method(params.from_count_incl, params.to_count_excl)
  elif params.mode == "oov":
    method = get_match_oov_method(params.from_count_incl, params.to_count_excl)
  else:
    assert False

  select_from_nrs = tqdm(select_from_nrs, desc="Filtering",
                         unit=TQDM_LINE_UNIT, total=select_from_count)
  result = OrderedSet()
  for line_nr in select_from_nrs:
    words = split_adv(params.lines[line_nr], params.ssep)
    match = method(words, params.vocabulary)
    del words
    if match:
      logger.info(
        f"Filtered L-{line_nr+1}: \"{params.lines[line_nr]}\".")
      result.add(line_nr)

  changed_anything = False
  if len(result) > 0:
    logger.info(
      f"Filtered {len(result)} out of {len(select_from_nrs)} lines ({get_percent_str(len(result),len(select_from_nrs))}). {len(select_from_nrs)-len(result)} lines remain.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
    changed_anything = True
  return changed_anything


def get_match_iv_method(from_incl: int, to_excl: Union[int, float]) -> Callable[[List[str], Set[str]], bool]:
  if from_incl == 1 and isinf(to_excl):
    return matches_any_iv

  if from_incl > 1 and isinf(to_excl):
    res = partial(
      matches_iv_count,
      count=from_incl,
    )
    return res

  res = partial(
    matches_iv_boundary,
    from_incl=from_incl,
    to_excl=to_excl,
  )
  return res


def get_match_oov_method(from_incl: int, to_excl: Union[int, float]) -> Callable[[List[str], Set[str]], bool]:
  if from_incl == 1 and isinf(to_excl):
    return matches_any_oov

  if from_incl > 1 and isinf(to_excl):
    res = partial(
      matches_oov_count,
      count=from_incl,
    )
    return res

  res = partial(
    matches_oov_boundary,
    from_incl=from_incl,
    to_excl=to_excl,
  )
  return res


def matches_any_iv(words: List[str], vocabulary: Set[str]) -> bool:
  result = any(word in vocabulary for word in words)
  return result


def matches_any_oov(words: List[str], vocabulary: Set[str]) -> bool:
  result = any(word not in vocabulary for word in words)
  return result


def matches_oov_count(words: List[str], vocabulary: Set[str], count: int) -> bool:
  counter = 0
  for word in words:
    if word not in vocabulary:
      counter += 1
    if counter >= count:
      del counter
      return True
  if counter >= count:
    del counter
    return True
  del counter
  return False


def matches_iv_count(words: List[str], vocabulary: Set[str], count: int) -> bool:
  counter = 0
  for word in words:
    if word in vocabulary:
      counter += 1
    if counter >= count:
      del counter
      return True
  if counter >= count:
    del counter
    return True
  del counter
  return False


def matches_oov_boundary(words: List[str], vocabulary: Set[str], from_incl: int, to_excl: Union[int, float]) -> bool:
  counter = sum(1 for word in words if word not in vocabulary)
  result = from_incl <= counter < to_excl
  del counter
  return result


def matches_iv_boundary(words: List[str], vocabulary: Set[str], from_incl: int, to_excl: Union[int, float]) -> bool:
  counter = sum(1 for word in words if word in vocabulary)
  result = from_incl <= counter < to_excl
  del counter
  return result
