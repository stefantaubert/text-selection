from dataclasses import dataclass
from logging import getLogger

import numpy as np
from ordered_set import OrderedSet
from text_selection.common.mapping_iterator import map_indices
from text_selection.kld.custom_kld_iterator import CustomKldIterator
from text_selection.kld.kld_iterator import get_uniform_weights
from text_selection.selection import SelectionMode
from text_selection_core.common import (NGramsNotExistError, SelectionDefaultParameters,
                                        WeightSelectionParameters, get_selector,
                                        validate_selection_default_parameters,
                                        validate_weights_parameters)
from text_selection_core.globals import ExecutionResult
from text_selection_core.helper import (get_initial_weights,
                                        get_target_weights_from_percent)
from text_selection_core.types import (NGramSet, Subset,
                                       get_subsets_ids, move_ids_to_subset)
from text_selection_core.weights.weights_iterator import WeightsIterator


@dataclass()
class KldSelectionParameters():
  n_grams: NGramSet
  consider_to_subset: bool
  id_selection: SelectionMode


def select_kld(default_params: SelectionDefaultParameters, params: KldSelectionParameters, weight_params: WeightSelectionParameters) -> ExecutionResult:
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

  selector = get_selector(params.id_selection)

  kld_weights = get_uniform_weights(params.n_grams.data.shape[1])

  iterator = CustomKldIterator(
    data=params.n_grams.data,
    data_indices=select_from_indices,
    preselection=summed_preselection_counts,
    key_selector=selector,
    weights=kld_weights,
  )

  initial_weights = get_initial_weights(
    to_subset, weight_params.weights, weight_params.target_incl_selection)

  if weight_params.target_percent:
    weight_params.target = get_target_weights_from_percent(
        from_ids, to_subset, weight_params.weights, weight_params.target, weight_params.target_incl_selection)

  weights_iterator = WeightsIterator(
    iterator, weight_params.weights, weight_params.target, initial_weights)
  mapping_iterator = map_indices(weights_iterator, params.n_grams.data_ids_to_indices)

  result: Subset = OrderedSet(mapping_iterator)
  changed_anything = False

  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Selected {len(result)} Id's.")
    move_ids_to_subset(default_params.dataset, result, default_params.to_subset_name)
    changed_anything = True

  return None, changed_anything
