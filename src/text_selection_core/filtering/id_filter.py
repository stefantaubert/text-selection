from logging import getLogger

from text_selection_core.globals import ExecutionResult
from text_selection_core.types import (DataIds, Dataset, Subset, SubsetName,
                                       move_ids_to_subset)
from text_selection_core.validation import (SubsetNotExistsError,
                                            ValidationError)


class AnyIdAlreadySelectedError(ValidationError):
  def __init__(self, subset: Subset, ids: DataIds) -> None:
    super().__init__()
    self.subset = subset
    self.ids = ids

  @classmethod
  def validate(cls, subset: Subset, ids: DataIds):
    if not len(ids.intersection(subset)) == 0:
      return cls(subset, ids)
    return None

  @property
  def default_message(self) -> str:
    return f"Some of the Id's ({', '.join(self.ids.intersection(self.subset))}) were already filtered!"


def filter_ids(dataset: Dataset, to_subset_name: SubsetName, ids: DataIds) -> ExecutionResult:
  if error := SubsetNotExistsError.validate(dataset, to_subset_name):
    return error, False

  to_subset = dataset[to_subset_name]
  if error := AnyIdAlreadySelectedError.validate(to_subset, ids):
    return error, False

  changed_anything = False
  if len(ids) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Filtered {len(ids)} Id's.")
    move_ids_to_subset(dataset, ids, to_subset_name)
    changed_anything = True
  return None, changed_anything
