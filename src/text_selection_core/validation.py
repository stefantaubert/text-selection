from ordered_set import OrderedSet

from text_selection_core.types import DataSymbols, Dataset, DataWeights, SubsetName


class ValidationError():
  # pylint: disable=no-self-use
  @property
  def default_message(self) -> str:
    return ""


class InternalError(ValidationError):
  @property
  def default_message(self) -> str:
    return "Internal error!"


class SubsetAlreadyExistsError(ValidationError):
  def __init__(self, dataset: Dataset, name: SubsetName) -> None:
    super().__init__()
    self.dataset = dataset
    self.name = name

  @classmethod
  def validate(cls, dataset: Dataset, name: SubsetName):
    if name in dataset.subsets:
      return cls(dataset, name)
    return None

  @classmethod
  def validate_names(cls, dataset: Dataset, names: OrderedSet[SubsetName]):
    for name in names:
      if error := cls.validate(dataset, name):
        return error
    return None

  @property
  def default_message(self) -> str:
    return f"Subset \"{self.name}\" already exists!"


class SubsetNotExistsError(ValidationError):
  def __init__(self, dataset: Dataset, name: SubsetName) -> None:
    super().__init__()
    self.dataset = dataset
    self.name = name

  @classmethod
  def validate(cls, dataset: Dataset, name: SubsetName):
    if name not in dataset.subsets:
      return cls(dataset, name)
    return None

  @classmethod
  def validate_names(cls, dataset: Dataset, names: OrderedSet[SubsetName]):
    for name in names:
      if error := cls.validate(dataset, name):
        return error
    return None

  @property
  def default_message(self) -> str:
    return f"Subset \"{self.name}\" does not exist!"


class InvalidPercentualValueError(ValidationError):
  def __init__(self, percent: float) -> None:
    super().__init__()
    self.percent = percent

  @classmethod
  def validate(cls, percent: float):
    if not 0 <= percent <= 1:
      return cls(percent)
    return None

  @property
  def default_message(self) -> str:
    return f"Invalid percent!"


class NonDivergentSubsetsError(ValidationError):
  def __init__(self, name: SubsetName) -> None:
    super().__init__()
    self.name = name

  @classmethod
  def validate(cls, from_subset_name: SubsetName, to_subset_name: SubsetName):
    if from_subset_name == to_subset_name:
      return cls(from_subset_name)
    return None

  @classmethod
  def validate_names(cls, from_subset_names: OrderedSet[SubsetName], to_subset_name: SubsetName):
    for from_subset_name in from_subset_names:
      if error := cls.validate(from_subset_name, to_subset_name):
        return error
    return None

  @property
  def default_message(self) -> str:
    return f"Subsets need to be distinct!"


class WeightsDoNotContainAllKeysError(ValidationError):
  def __init__(self, dataset: Dataset, weights: DataWeights) -> None:
    super().__init__()
    self.dataset = dataset
    self.weights = weights

  @classmethod
  def validate(cls, dataset: Dataset, weights: DataWeights):
    if set(dataset.ids) != set(weights.keys()):
      return cls(dataset, weights)
    return None

  @property
  def default_message(self) -> str:
    return f"Weights Id's does not match with Id's from dataset!"


class SymbolsDoNotContainAllKeysError(ValidationError):
  def __init__(self, dataset: Dataset, symbols: DataSymbols) -> None:
    super().__init__()
    self.dataset = dataset
    self.symbols = symbols

  @classmethod
  def validate(cls, dataset: Dataset, symbols: DataSymbols):
    if set(dataset.ids) != set(symbols.keys()):
      return cls(dataset, symbols)
    return None

  @property
  def default_message(self) -> str:
    return f"Symbol Id's does not match with Id's from dataset!"
