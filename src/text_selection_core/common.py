from dataclasses import dataclass
from typing import Optional

from ordered_set import OrderedSet

from text_selection_core.types import (DataId, Dataset, DataWeights, NGramSet,
                                       Subset, SubsetName, Weight,
                                       get_subsets_ids)
from text_selection_core.validation import InvalidPercentualValueError, NonDivergentSubsetsError, SubsetNotExistsError, ValidationError, WeightsDoNotContainAllKeysError


@dataclass()
class WeightSelectionParameters():
  weights: DataWeights
  target: Weight
  target_incl_selection: bool
  target_percent: bool


@dataclass()
class SelectionDefaultParameters():
  dataset: Dataset
  from_subset_names: OrderedSet[SubsetName]
  to_subset_name: SubsetName



def validate_selection_default_parameters(params: SelectionDefaultParameters) -> Optional[ValidationError]:
  if error := SubsetNotExistsError.validate_names(params.dataset, params.from_subset_names):
    return error, False

  if error := SubsetNotExistsError.validate(params.dataset, params.to_subset_name):
    return error, False

  if error := NonDivergentSubsetsError.validate_names(params.from_subset_names, params. to_subset_name):
    return error, False



def validate_weights_parameters(params: WeightSelectionParameters, dataset: Dataset) -> Optional[ValidationError]:
  if error := WeightsDoNotContainAllKeysError.validate(dataset, params.weights):
    return error, False

  if params.target_percent:
    if error := InvalidPercentualValueError.validate(params.target):
      return error, False
