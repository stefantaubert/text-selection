from dataclasses import dataclass
from typing import Iterable, Optional

from ordered_set import OrderedSet
from text_selection.selection import FirstKeySelector, SelectionMode

from text_selection_core.types import (DataId, Dataset, DataWeights, NGramSet,
                                       Subset, SubsetName, Weight,
                                       get_subsets_ids)
from text_selection_core.validation import (InvalidPercentualValueError,
                                            NonDivergentSubsetsError,
                                            SubsetNotExistsError,
                                            ValidationError,
                                            WeightsDoNotContainAllKeysError)


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

@dataclass()
class SortingDefaultParameters():
  dataset: Dataset
  subset_names: OrderedSet[SubsetName]


def validate_selection_default_parameters(params: SelectionDefaultParameters) -> Optional[ValidationError]:
  if error := SubsetNotExistsError.validate_names(params.dataset, params.from_subset_names):
    return error

  if error := SubsetNotExistsError.validate(params.dataset, params.to_subset_name):
    return error

  if error := NonDivergentSubsetsError.validate_names(params.from_subset_names, params. to_subset_name):
    return error


def validate_weights_parameters(params: WeightSelectionParameters, dataset: Dataset) -> Optional[ValidationError]:
  if error := WeightsDoNotContainAllKeysError.validate(dataset, params.weights):
    return error

  if params.target_percent:
    if error := InvalidPercentualValueError.validate(params.target):
      return error

def validate_sorting_default_parameters(params: SortingDefaultParameters) -> Optional[ValidationError]:
  if error := SubsetNotExistsError.validate_names(params.dataset, params.subset_names):
    return error


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

def get_selector(mode: SelectionMode):
  if mode == SelectionMode.FIRST:
    selector = FirstKeySelector()
    return selector
  else:
    assert False
