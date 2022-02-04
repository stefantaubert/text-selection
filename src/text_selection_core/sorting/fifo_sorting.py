from typing import Literal

from ordered_set import OrderedSet
from text_selection_core.algorithms.fifo import get_fifo_id_iterator, get_fifo_original_positions_iterator
from text_selection_core.common import SortingDefaultParameters, validate_sorting_default_parameters
from text_selection_core.globals import ExecutionResult

original_mode = "original"
id_mode = "id"


def sort_fifo(default_params: SortingDefaultParameters, mode: Literal["original", "id"]) -> ExecutionResult:
  if error := validate_sorting_default_parameters(default_params):
    return error, False

  changed_anything = False
  for subset_name in default_params.subset_names:
    subset = default_params.dataset.subsets[subset_name]

    if mode == original_mode:
      iterator = get_fifo_original_positions_iterator(subset, default_params.dataset.ids)
    elif mode == id_mode:
      iterator = get_fifo_id_iterator(subset)
    else:
      assert False

    ordered_subset = OrderedSet(iterator)
    if ordered_subset != subset:
      default_params.dataset.subsets[subset_name] = ordered_subset
      changed_anything = True
  return None, changed_anything
