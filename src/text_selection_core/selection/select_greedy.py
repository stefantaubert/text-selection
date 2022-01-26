from dataclasses import dataclass
from logging import getLogger
from typing import Iterable, Optional

import numpy as np
from ordered_set import OrderedSet
from text_selection.common.mapping_iterator import map_indices
from text_selection.greedy.optimized_greedy_iterator import \
    OptimizedGreedyIterator
from text_selection.selection import FirstKeySelector, SelectionMode
from text_selection_core.common import (SelectionDefaultParameters,
                                        WeightSelectionParameters,
                                        validate_selection_default_parameters,
                                        validate_weights_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.helper import (get_initial_weights,
                                        get_target_weights_from_percent)
from text_selection_core.types import (DataId, Dataset, DataWeights, NGramSet,
                                       Subset, SubsetName, Weight,
                                       get_subsets_ids, move_ids_to_subset)
from text_selection_core.validation import ValidationError
from text_selection_core.weights.weights_iterator import WeightsIterator


class NGramsNotExistError(ValidationError):
  def __init__(self, n_grams: NGramSet) -> None:
    super().__init__()
    self.n_grams = n_grams

  @classmethod
  def validate(cls, n_grams: NGramSet, ids: Iterable[DataId]):
    all_n_grams_exist = all(data_id in n_grams.indices_to_data_ids for data_id in ids)
    if not all_n_grams_exist:
      return cls(n_grams)
    return None

  @property
  def default_message(self) -> str:
    return f"Not all n-grams exist!"


@dataclass()
class GreedySelectionParameters():
  n_grams: NGramSet
  consider_to_subset: bool
  id_selection: SelectionMode


def select_greedy(default_params: SelectionDefaultParameters, params: GreedySelectionParameters, weight_params: WeightSelectionParameters) -> ExecutionResult:
  if error := validate_selection_default_parameters(default_params):
    return error, False

  if error := validate_weights_parameters(weight_params, default_params.dataset):
    return error, False

  from_ids = OrderedSet(get_subsets_ids(default_params.dataset,
                        default_params.from_subset_names))

  if error := NGramsNotExistError.validate(params.n_grams, from_ids):
    return error, False

  select_from_indices = OrderedSet(map_indices(
    from_ids, params.n_grams.indices_to_data_ids))
  to_subset = default_params.dataset.subsets[default_params.to_subset_name]

  if params.consider_to_subset:
    if error := NGramsNotExistError.validate(params.n_grams, from_ids):
      return error, False

    preselection_indices = map_indices(to_subset, params.n_grams.indices_to_data_ids)
    summed_preselection_counts = np.sum(params.n_grams.data[preselection_indices], axis=0)
  else:
    summed_preselection_counts = np.zeros(params.n_grams.data.shape[1])

  if params.id_selection == SelectionMode.FIRST:
    selector = FirstKeySelector()
  else:
    assert False

  iterator = OptimizedGreedyIterator(
    data=params.n_grams.data,
    data_indices=select_from_indices,
    preselection=summed_preselection_counts,
    key_selector=selector,
  )

  initial_weights = get_initial_weights(
    to_subset, weight_params.weights, weight_params.target_incl_selection)

  if weight_params.target_percent:
    target = get_target_weights_from_percent(
        from_ids, to_subset, weight_params.weights, weight_params.target, weight_params.target_incl_selection)

  weights_iterator = WeightsIterator(iterator, weight_params.weights, target, initial_weights)
  mapping_iterator = map_indices(weights_iterator, params.n_grams.data_ids_to_indices)

  result: Subset = OrderedSet(mapping_iterator)
  changed_anything = False

  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Selected {len(result)} Id's.")
    move_ids_to_subset(default_params.dataset, result, default_params.to_subset_name)
    changed_anything = True

  return None, changed_anything
