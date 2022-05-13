from text_utils import StringFormat2
from pathlib import Path
from typing import Dict, Optional

from ordered_set import OrderedSet
from text_selection_cli.io_handling import (get_data_symbols_path,
                                            get_data_weights_path, get_dataset_path,
                                            try_load_dataset, save_data_symbols,
                                            try_save_data_weights, try_save_dataset)
from text_selection_core.types import (Dataset, Lines, DataWeights,
                                       Subset, SubsetName)


def check_dataset_is_valid(dataset: Dataset) -> bool:
  if not isinstance(dataset, Dataset):
    return False

  if not isinstance(dataset.create_line_nrs, OrderedSet):
    return False

  if not len(dataset.subsets) > 0:
    return False

  for key in dataset.subsets:
    if not isinstance(key, str):
      return False
    if not isinstance(dataset.subsets[key], OrderedSet):
      return False

    if not dataset.subsets[key].issubset(dataset.create_line_nrs):
      return False

  all_keys_from_all_subsets = list(
    key
    for subset in dataset.subsets.values()
    for key in subset
  )

  some_keys_duplicate = len(all_keys_from_all_subsets) != len(set(all_keys_from_all_subsets))
  if some_keys_duplicate:
    return False

  not_all_keys_in_subsets = len(all_keys_from_all_subsets) != len(dataset.create_line_nrs)
  if not_all_keys_in_subsets:
    return False

  return True


def check_symbols_are_valid(symbols: Lines) -> bool:
  if not isinstance(symbols, dict):
    return False

  for k, v in symbols.items():
    if not isinstance(k, int):
      return False
    if not isinstance(v, str):
      return False
    if not StringFormat2.SPACED.can_convert_string_to_symbols(v):
      return False
  return True


def check_weights_are_valid(weights: DataWeights) -> bool:
  if not isinstance(weights, dict):
    return False

  for k, v in weights.items():
    if not isinstance(k, int):
      return False
    if not (isinstance(v, int) or isinstance(v, float)):
      return False
    if v < 0:
      return False
  return True


def create(directory: Path, dataset: Dataset, weights: Dict[str, DataWeights], symbols: Optional[Lines], overwrite: bool) -> None:
  dataset_path = get_dataset_path(directory)
  if not overwrite and dataset_path.exists():
    raise ValueError("The dataset already exists!")

  if not check_dataset_is_valid(dataset):
    raise ValueError("The dataset is not valid!")

  for weight_name, data_weights in weights.items():
    if not check_weights_are_valid(data_weights):
      raise ValueError(f"Weights '{weight_name}' have the wrong format!")

    weights_path = get_data_weights_path(directory, weight_name)
    if weights_path.exists() and not overwrite:
      raise ValueError(f"Weights '{weight_name}' already exist!")
    # TODO extra function
    if data_weights.keys() != set(dataset.create_line_nrs):
      raise ValueError(f"Weights '{weight_name}' contain not all lines!")

  if symbols is not None:
    if not check_symbols_are_valid(symbols):
      raise ValueError("Symbols have the wrong format!")

    symbols_path = get_data_symbols_path(directory)
    if symbols_path.exists() and not overwrite:
      raise ValueError(f"Symbols already exist!")
    if symbols.keys() != set(dataset.create_line_nrs):
      raise ValueError(f"Symbols contain not all lines!")

  dataset_path = get_dataset_path(directory)
  try_save_dataset(dataset_path, dataset)

  for weight_name, data_weights in weights.items():
    weights_path = get_data_weights_path(directory, weight_name)
    try_save_data_weights(weights_path, data_weights)

  if symbols is not None:
    symbols_path = get_data_symbols_path(directory)
    save_data_symbols(symbols_path, symbols)


def get_selection(directory: Path, subset_name: SubsetName) -> Subset:
  dataset_path = get_dataset_path(directory)
  if not dataset_path.is_file():
    raise ValueError("Dataset does not exist!")
  dataset = try_load_dataset(dataset_path)
  if subset_name not in dataset.subsets:
    raise ValueError("Subset does not exist!")
  result = dataset.subsets[subset_name]
  return result
