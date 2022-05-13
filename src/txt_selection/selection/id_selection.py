from logging import getLogger
from typing import Literal

from ordered_set import OrderedSet
from txt_selection.algorithms.fifo import get_fifo_id_iterator, get_fifo_original_positions_iterator, get_fifo_subset_iterator
from txt_selection.common import (SelectionDefaultParameters, WeightSelectionParameters,
                                  validate_selection_default_parameters,
                                  validate_weights_parameters)
from txt_selection.filtering.weights_filter import \
    WeightsFilterParameters
from txt_selection.globals import ExecutionResult
from txt_selection.helper import (get_initial_weights,
                                  get_target_weights_from_percent)
from txt_selection.types import (DataId, DataIds, Dataset, Subset, SubsetName,
                                 get_subsets_ids, move_ids_to_subset)
from txt_selection.validation import SubsetNotExistsError, ValidationError
from txt_selection.weights.weights_iterator import WeightsIterator


class IdsNotExistError(ValidationError):
  def __init__(self, dataset: Dataset, ids: DataIds) -> None:
    super().__init__()
    self.dataset = dataset
    self.ids = ids

  @classmethod
  def validate(cls, dataset: Dataset, ids: DataIds):
    if not ids.issubset(dataset.ids):
      return cls(dataset, ids)
    return None

  @property
  def default_message(self) -> str:
    return f"Ids {', '.join(self.ids.difference(self.dataset.ids))} do not exist in the dataset!"


def select_ids(dataset: Dataset, to_subset_name: SubsetName, ids: DataIds) -> ExecutionResult:
  if error := SubsetNotExistsError.validate(dataset, to_subset_name):
    return error, False

  if error := IdsNotExistError.validate(dataset, ids):
    return error, False

  to_subset = dataset.subsets[to_subset_name]

  new_ids = (entry_id for entry_id in ids if entry_id not in to_subset)
  result: Subset = OrderedSet(new_ids)
  changed_anything = False

  logger = getLogger(__name__)

  if len(result) > 0:
    logger.debug(f"Selected {len(result)} lines.")
    move_ids_to_subset(dataset, result, to_subset_name)
    changed_anything = True

  if len(result) < len(ids):
    logger.debug(f"{len(ids) - len(result)} were already selected.")

  return None, changed_anything
