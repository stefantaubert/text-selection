from logging import Logger

from ordered_set import OrderedSet

from text_selection_core.globals import ExecutionResult
from text_selection_core.types import Dataset, SubsetName
from text_selection_core.validation import (ErrorType, ValidationErr,
                                            ensure_not_only_n_subsets_exists, ensure_subset_exists,
                                            ensure_subset_not_already_exists, ensure_subsets_exist,
                                            ensure_subsets_not_already_exist)


def add_subsets(dataset: Dataset, names: OrderedSet[SubsetName], skip_existing: bool, logger: Logger) -> ExecutionResult:
  if not skip_existing and (error := ensure_subsets_not_already_exist(dataset, names)):
    return error

  changed_anything = False
  for name in names:
    if skip_existing and name in dataset.subsets:
      logger.info(f"Skipped existing set \"{name}\".")
      continue
    dataset.subsets[name] = OrderedSet()
    changed_anything = True

  return changed_anything


def remove_subsets(dataset: Dataset, names: OrderedSet[SubsetName], logger: Logger) -> ExecutionResult:
  if error := ensure_subsets_exist(dataset, names):
    return error

  if error := ensure_not_only_n_subsets_exists(dataset, n=len(names)):
    return error

  for name in names:
    if len(dataset.subsets[name]) > 0:
      return ValidationErr(ErrorType.SUBSET_NOT_EMPTY, name)
    dataset.subsets.pop(name)

  return True


def rename_subset(dataset: Dataset, name: SubsetName, new_name: SubsetName, logger: Logger) -> ExecutionResult:
  if error := ensure_subset_exists(dataset, name):
    return error

  if error := ensure_subset_not_already_exists(dataset, new_name):
    return error

  dataset.subsets[new_name] = dataset.subsets.pop(name)

  return True
