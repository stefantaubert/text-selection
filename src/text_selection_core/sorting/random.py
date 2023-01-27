
import random
from logging import Logger
from random import shuffle

from ordered_set import OrderedSet

from text_selection_core.common import SortingDefaultParameters, validate_sorting_default_parameters
from text_selection_core.globals import ExecutionResult


def sort_random(params: SortingDefaultParameters, seed: int, logger: Logger) -> ExecutionResult:
  if error := validate_sorting_default_parameters(params):
    return error

  changed_anything = False
  for subset_name in params.subset_names:
    subset = params.dataset.subsets[subset_name]
    subset_as_list = list(subset)
    random.seed(seed)
    shuffle(subset_as_list)
    subset_sorted = OrderedSet(subset_as_list)
    changed_anything |= subset != subset_sorted
    params.dataset.subsets[subset_name] = subset_sorted

  return changed_anything
