from dataclasses import dataclass
from logging import getLogger
from typing import Generator, Iterable, Tuple

from ordered_set import OrderedSet

from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.types import (DataWeights, LineNr, Subset, Weight, get_subsets_line_nrs_gen,
                                       move_lines_to_subset)
from text_selection_core.validation import WeightsDoNotContainAllKeysError


@dataclass()
class WeightsFilterParameters():
  weights: DataWeights
  from_weight_incl: Weight
  to_weight_excl: Weight


def filter_weights(default_params: SelectionDefaultParameters, params: WeightsFilterParameters) -> ExecutionResult:
  assert 0 <= params.from_weight_incl < params.to_weight_excl

  if error := validate_selection_default_parameters(default_params):
    return error, False

  if error := WeightsDoNotContainAllKeysError.validate(default_params.dataset, params.weights):
    return error, False

  select_from = ((data_id, params[data_id])
                 for data_id in get_subsets_line_nrs_gen(default_params.dataset, default_params. from_subset_names))

  iterator = get_weight_keys(select_from, params.from_weight_incl,
                             params.to_weight_excl)
  result: Subset = OrderedSet(iterator)
  changed_anything = False
  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Filtered {len(result)} lines.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
    changed_anything = True
  return None, changed_anything


def get_weight_keys(weights: Iterable[Tuple[LineNr, Weight]], from_weight_incl: Weight, to_weight_excl: Weight) -> Generator[LineNr, None, None]:
  for data_id, weight in weights:
    if from_weight_incl <= weight < to_weight_excl:
      yield data_id
