from logging import getLogger

from text_selection_core.globals import ExecutionResult
from text_selection_core.types import Dataset, LineNrs, Subset, SubsetName, move_lines_to_subset
from text_selection_core.validation import SubsetNotExistsError, ValidationError


class AnyLineAlreadySelectedError(ValidationError):
  def __init__(self, subset: Subset, nrs: LineNrs) -> None:
    super().__init__()
    self.subset = subset
    self.nrs = nrs

  @classmethod
  def validate(cls, subset: Subset, nrs: LineNrs):
    if not len(nrs.intersection(subset)) == 0:
      return cls(subset, nrs)
    return None

  @property
  def default_message(self) -> str:
    return f"Some of the lines ({', '.join(self.nrs.intersection(self.subset))}) were already filtered!"


def filter_ids(dataset: Dataset, to_subset_name: SubsetName, nrs: LineNrs) -> ExecutionResult:
  if error := SubsetNotExistsError.validate(dataset, to_subset_name):
    return error, False

  to_subset = dataset[to_subset_name]
  if error := AnyLineAlreadySelectedError.validate(to_subset, nrs):
    return error, False

  changed_anything = False
  if len(nrs) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Filtered {len(nrs)} numbers.")
    move_lines_to_subset(dataset, nrs, to_subset_name, logger)
    changed_anything = True
  return None, changed_anything
