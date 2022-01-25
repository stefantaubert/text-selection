from ordered_set import OrderedSet
from text_selection_core.globals import ExecutionResult
from text_selection_core.types import Dataset, SubsetName
from text_selection_core.validation import SubsetNotExistsError



def sort_random(dataset: Dataset, subset_names: OrderedSet[SubsetName]) -> ExecutionResult:
  if error := SubsetNotExistsError.validate_names(dataset, subset_names):
    return error, False

  for subset_name in subset_names:
    subset = dataset[subset_name]
    ordered_subset = dataset.ids.intersection(subset)
    dataset[subset_name] = ordered_subset
