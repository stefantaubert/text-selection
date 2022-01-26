from typing import Optional

from ordered_set import OrderedSet

from text_selection_core.globals import ExecutionResult
from text_selection_core.types import Dataset, SubsetName
from text_selection_core.validation import (SubsetAlreadyExistsError,
                                            SubsetNotExistsError,
                                            ValidationError)


class IsLastSubsetError(ValidationError):
  def __init__(self, dataset: Dataset) -> None:
    super().__init__()
    self.dataset = dataset

  @classmethod
  def validate(cls, dataset: Dataset):
    if len(dataset) == 1:
      return cls(dataset)
    return None

  @property
  def default_message(self) -> str:
    return f"The last subset could not be removed!"


def add_subsets(dataset: Dataset, names: OrderedSet[SubsetName]) -> ExecutionResult:
  if error := SubsetAlreadyExistsError.validate_names(dataset, names):
    return error, False

  for name in names:
    dataset.subsets[name] = OrderedSet()

  return None, True


def remove_subsets(dataset: Dataset, names: OrderedSet[SubsetName]) -> Optional[ValidationError]:
  if error := SubsetNotExistsError.validate_names(dataset, names):
    return error, False

  if error := IsLastSubsetError.validate(dataset):
    return error, False

  for name in names:
    dataset.pop(name)

  return None, True


def rename_subset(dataset: Dataset, name: SubsetName, new_name: SubsetName) -> Optional[ValidationError]:
  if error := SubsetNotExistsError.validate(dataset, name):
    return error, False

  if error := SubsetAlreadyExistsError.validate(dataset, new_name):
    return error, False

  dataset.subsets[new_name] = dataset.pop(name)

  return None, True
