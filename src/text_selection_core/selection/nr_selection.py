from logging import Logger, getLogger

from ordered_set import OrderedSet

from text_selection_core.globals import ExecutionResult
from text_selection_core.types import (Dataset, LineNrs, Subset, SubsetName, move_lines_to_subset)
from text_selection_core.validation import SubsetNotExistsError, ValidationError


class NrsNotExistError(ValidationError):
  def __init__(self, dataset: Dataset, ids: LineNrs) -> None:
    super().__init__()
    self.dataset = dataset
    self.ids = ids

  @classmethod
  def validate(cls, dataset: Dataset, ids: LineNrs):
    if not ids.issubset(dataset.get_line_nrs()):
      return cls(dataset, ids)
    return None

  @property
  def default_message(self) -> str:
    return f"Line number(s) {', '.join(self.ids.difference(self.dataset.get_line_nrs()))} do(es) not exist in the dataset!"


def select_ids(dataset: Dataset, to_subset_name: SubsetName, nrs: LineNrs, logger: Logger) -> ExecutionResult:
  if error := SubsetNotExistsError.validate(dataset, to_subset_name):
    return error, False

  if error := NrsNotExistError.validate(dataset, nrs):
    return error, False

  to_subset = dataset.subsets[to_subset_name]

  new_ids = (entry_id for entry_id in nrs if entry_id not in to_subset)
  result: Subset = OrderedSet(new_ids)
  changed_anything = False

  if len(result) > 0:
    logger.debug(f"Selected {len(result)} lines.")
    move_lines_to_subset(dataset, result, to_subset_name, logger)
    changed_anything = True

  if len(result) < len(nrs):
    logger.debug(f"{len(nrs) - len(result)} were already selected.")

  return None, changed_anything
