from logging import getLogger
from typing import Iterable

import numpy as np
from ordered_set import OrderedSet
from text_selection.common.mapping_iterator import MappingIterator, map_indices
from text_selection.greedy.optimized_greedy_iterator import \
    OptimizedGreedyIterator
from text_selection.selection import FirstKeySelector, SelectionMode
from text_selection_core.globals import ExecutionResult
from text_selection_core.helper import (get_initial_weights,
                                        get_target_weights_from_percent)
from text_selection_core.types import (DataId, Dataset, DataWeights, NGramSet,
                                       Subset, SubsetName, Weight)
from text_selection_core.validation import (InvalidPercentualValueError,
                                            NonDivergentSubsetsError,
                                            SubsetNotExistsError,
                                            ValidationError,
                                            WeightsDoNotContainAllKeysError)
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


def select_greedy(dataset: Dataset, from_subset_names: OrderedSet[SubsetName], to_subset_name: SubsetName, n_grams: NGramSet, weights: DataWeights, target: Weight, target_incl_selection: bool, target_percent: bool, consider_to_subset: bool, id_selection: SelectionMode) -> ExecutionResult:
  if error := SubsetNotExistsError.validate_names(dataset, from_subset_names):
    return error, False

  if error := SubsetNotExistsError.validate(dataset, to_subset_name):
    return error, False

  if error := NonDivergentSubsetsError.validate_names(from_subset_names, to_subset_name):
    return error, False

  if error := WeightsDoNotContainAllKeysError.validate(dataset, weights):
    return error, False

  if target_percent:
    if error := InvalidPercentualValueError.validate(target):
      return error, False

  from_ids = OrderedSet(dataset.get_subsets_ids(from_subset_names))

  if error := NGramsNotExistError.validate(n_grams, from_ids):
    return error, False

  select_from_indices = OrderedSet(map_indices(from_ids, n_grams.indices_to_data_ids))
  to_subset = dataset[to_subset_name]

  if consider_to_subset:
    if error := NGramsNotExistError.validate(n_grams, from_ids):
      return error, False

    preselection_indices = map_indices(to_subset, n_grams.indices_to_data_ids)
    summed_preselection_counts = np.sum(n_grams.data[preselection_indices], axis=0)
  else:
    summed_preselection_counts = np.zeros(n_grams.data.shape[1])

  if id_selection == SelectionMode.FIRST:
    selector = FirstKeySelector()
  else:
    assert False

  iterator = OptimizedGreedyIterator(
    data=n_grams.data,
    data_indices=select_from_indices,
    preselection=summed_preselection_counts,
    key_selector=selector,
  )

  initial_weights = get_initial_weights(to_subset, weights, target_incl_selection)

  if target_percent:
    target = get_target_weights_from_percent(
        from_ids, to_subset, weights, target, target_incl_selection)

  weights_iterator = WeightsIterator(iterator, weights, target, initial_weights)
  mapping_iterator = map_indices(weights_iterator, n_grams.data_ids_to_indices)

  result: Subset = OrderedSet(mapping_iterator)
  changed_anything = False
  if len(result) > 0:
    logger = getLogger(__name__)
    logger.debug(f"Selected {len(result)} Id's.")
    dataset.move_ids_to_subset(result, to_subset_name)
    changed_anything = True
  return None, changed_anything
