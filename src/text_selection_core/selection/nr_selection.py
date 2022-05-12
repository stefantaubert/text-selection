from logging import getLogger
from typing import Literal

from ordered_set import OrderedSet

from text_selection_core.algorithms.fifo import (get_fifo_original_positions_iterator,
                                                 get_fifo_subset_iterator)
from text_selection_core.common import (SelectionDefaultParameters, WeightSelectionParameters,
                                        validate_selection_default_parameters,
                                        validate_weights_parameters)
from text_selection_core.filtering.weights_filter import WeightsFilterParameters
from text_selection_core.globals import ExecutionResult
from text_selection_core.helper import get_initial_weights, get_target_weights_from_percent
from text_selection_core.types import (Dataset, LineNr, LineNrs, Subset, SubsetName,
                                       get_subsets_line_nrs, move_lines_to_subset)
from text_selection_core.validation import SubsetNotExistsError, ValidationError
from text_selection_core.weights.weights_iterator import WeightsIterator


class NrsNotExistError(ValidationError):
  def __init__(self, dataset: Dataset, ids: LineNrs) -> None:
    super().__init__()
    self.dataset = dataset
    self.ids = ids

  @classmethod
  def validate(cls, dataset: Dataset, ids: LineNrs):
    if not ids.issubset(dataset.nrs):
      return cls(dataset, ids)
    return None

  @property
  def default_message(self) -> str:
    return f"Ids {', '.join(self.ids.difference(self.dataset.nrs))} do not exist in the dataset!"


def select_ids(dataset: Dataset, to_subset_name: SubsetName, nrs: LineNrs) -> ExecutionResult:
  if error := SubsetNotExistsError.validate(dataset, to_subset_name):
    return error, False

  if error := NrsNotExistError.validate(dataset, nrs):
    return error, False

  to_subset = dataset.subsets[to_subset_name]

  new_ids = (entry_id for entry_id in nrs if entry_id not in to_subset)
  result: Subset = OrderedSet(new_ids)
  changed_anything = False

  logger = getLogger(__name__)

  if len(result) > 0:
    logger.debug(f"Selected {len(result)} Id's.")
    move_lines_to_subset(dataset, result, to_subset_name, logger)
    changed_anything = True

  if len(result) < len(nrs):
    logger.debug(f"{len(nrs) - len(result)} were already selected.")

  return None, changed_anything
