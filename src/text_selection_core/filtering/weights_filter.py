from dataclasses import dataclass
from logging import Logger

import numpy as np
from ordered_set import OrderedSet
from tqdm import tqdm

from text_selection_core.common import (SelectionDefaultParameters,
                                        validate_selection_default_parameters)
from text_selection_core.globals import TQDM_LINE_UNIT, ExecutionResult
from text_selection_core.helper import get_percent_str
from text_selection_core.types import (DataWeights, Subset, Weight, get_subsets_line_nrs_count,
                                       get_subsets_line_nrs_gen, move_lines_to_subset)
from text_selection_core.validation import ensure_weight_line_count_matches_dataset


@dataclass()
class WeightsFilterParameters():
  weights: DataWeights
  from_weight_incl: Weight
  to_weight_excl: Weight


def filter_weights(default_params: SelectionDefaultParameters, params: WeightsFilterParameters, logger: Logger) -> ExecutionResult:
  assert 0 <= params.from_weight_incl < params.to_weight_excl

  if error := validate_selection_default_parameters(default_params):
    return error

  if error := ensure_weight_line_count_matches_dataset(default_params.dataset, params.weights):
    return error

  logger.debug(params)
  select_from_nrs = get_subsets_line_nrs_gen(
    default_params.dataset, default_params.from_subset_names)
  select_from_count = get_subsets_line_nrs_count(
    default_params.dataset, default_params.from_subset_names)

  select_from_nrs = tqdm(select_from_nrs, desc="Preparing",
                         unit=TQDM_LINE_UNIT, total=select_from_count)
  select_from_nrs_list = list(select_from_nrs)
  del select_from_nrs
  logger.debug("Creating temporary numpy array...")
  select_from_nrs_arr = np.array(select_from_nrs_list)
  del select_from_nrs_list
  logger.debug("Selecting from array...")
  select_from_subarray = params.weights[select_from_nrs_arr]
  lines_np = np.flatnonzero((params.from_weight_incl <= select_from_subarray)
                            & (select_from_subarray < params.to_weight_excl))
  lines = select_from_nrs_arr[lines_np]
  # lines = get_matching_lines(params.weights, select_from_nrs,
  #                            params.from_weight_incl, params.to_weight_excl)
  result: Subset = OrderedSet(lines)

  for line_nr in result:
    logger.info(f"Filtered L-{line_nr+1} with weight: {params.weights[line_nr]}.")

  changed_anything = False
  if len(result) > 0:
    logger.info(
      f"Filtered {len(result)} out of {select_from_count} lines ({get_percent_str(len(result),select_from_count)}). {select_from_count-len(result)} lines remain.")
    move_lines_to_subset(default_params.dataset, result, default_params.to_subset_name, logger)
    changed_anything = True
  return changed_anything


# def get_matching_lines(weights: Weight, line_nrs: Iterator[LineNr], from_weight_incl: Weight, to_weight_excl: Weight) -> Generator[LineNr, None, None]:
#   for line_nr in line_nrs:
#     if from_weight_incl <= weights[line_nr] < to_weight_excl:
#       yield line_nr
