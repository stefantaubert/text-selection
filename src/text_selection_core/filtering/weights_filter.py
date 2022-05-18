from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Generator, Iterable, Iterator, Tuple

from ordered_set import OrderedSet

from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.helper import get_percent_str
from text_selection_core.types import (DataWeights, LineNr, Subset, Weight,
                                       get_subsets_line_nrs_count, get_subsets_line_nrs_gen,
                                       move_lines_to_subset)
from text_selection_core.validation import WeightsLinesCountNotMatchingError


@dataclass()
class WeightsFilterParameters():
  weights: DataWeights
  from_weight_incl: Weight
  to_weight_excl: Weight


def filter_weights(default_params: SelectionDefaultParameters, params: WeightsFilterParameters, logger: Logger) -> ExecutionResult:
  assert 0 <= params.from_weight_incl < params.to_weight_excl

  if error := validate_selection_default_parameters(default_params):
    return error, False

  if error := WeightsLinesCountNotMatchingError.validate(default_params.dataset, params.weights):
    return error, False

  select_from_nrs = get_subsets_line_nrs_gen(
    default_params.dataset, default_params. from_subset_names)
  select_from_count = get_subsets_line_nrs_count(
    default_params.dataset, default_params. from_subset_names)

  iterator = get_matching_lines(params.weights, select_from_nrs,
                                params.from_weight_incl, params.to_weight_excl)
  result: Subset = OrderedSet(iterator)
  changed_anything = False
  if len(result) > 0:
    logger.info(
      f"Filtered {len(result)} out of {select_from_count} lines ({get_percent_str(len(result),select_from_count)}). {select_from_count-len(result)} lines remain.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
    for line_nr in result:
      logger.debug(f"Filtered L{line_nr+1} with weight: {params.weights[line_nr]}.")
    changed_anything = True
  return None, changed_anything


def get_matching_lines(weights: Weight, line_nrs: Iterator[LineNr], from_weight_incl: Weight, to_weight_excl: Weight) -> Generator[LineNr, None, None]:
  for line_nr in line_nrs:
    if from_weight_incl <= weights[line_nr] < to_weight_excl:
      yield line_nr
