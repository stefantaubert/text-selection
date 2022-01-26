from pathlib import Path
from typing import Dict, Iterable, Optional

from ordered_set import OrderedSet
from text_selection_core.subsets import add_subsets
from text_selection_core.types import (Dataset, DataSymbols, DataWeights,
                                       Subset, SubsetName)
from text_selection_core.weights.calculation import get_uniform_weights

from text_selection_app.io import (get_data_symbols_path,
                                   get_data_weights_path, get_dataset_path,
                                   load_data_n_grams, load_data_symbols,
                                   load_data_weights, load_dataset,
                                   save_data_n_grams, save_data_symbols,
                                   save_data_weights, save_dataset)


def check_dataset_is_valid(dataset: Dataset) -> bool:
  if not isinstance(dataset, Dataset):
    return False

  if not isinstance(dataset.ids, OrderedSet):
    return False

  if not len(dataset.subsets) > 0:
    return False

  for key in dataset.subsets:
    if not isinstance(key, str):
      return False
    if not isinstance(dataset.subsets[key], OrderedSet):
      return False

    if not dataset.subsets[key].issubset(dataset.ids):
      return False

  all_keys_from_all_subsets = list(
    key
    for subset in dataset.subsets.values()
    for key in subset
  )

  some_keys_duplicate = len(all_keys_from_all_subsets) != len(set(all_keys_from_all_subsets))
  if some_keys_duplicate:
    return False

  not_all_keys_in_subsets = len(all_keys_from_all_subsets) != len(dataset.ids)
  if not_all_keys_in_subsets:
    return False

  return True


def create(directory: Path, dataset: Dataset, weights: Dict[str, DataWeights], symbols: Optional[DataSymbols], overwrite: bool) -> None:
  dataset_path = get_dataset_path(directory)
  if not overwrite and dataset_path.exists():
    raise ValueError("The dataset already exists!")

  if not check_dataset_is_valid(dataset):
    raise ValueError("The dataset is not valid!")

  save_dataset(dataset_path, dataset)

  for weight_name, data_weights in weights.items():
    weights_path = get_data_weights_path(directory, weight_name)
    if weights_path.exists() and not overwrite:
      raise ValueError(f"Weights '{weight_name}' already exist!")
    # TODO extra function
    if data_weights.keys() != set(dataset.ids):
      raise ValueError(f"Weights '{weight_name}' contain not all Id's!")
    save_data_weights(weights_path, data_weights)

  if symbols is not None:
    symbols_path = get_data_symbols_path(directory)
    if symbols_path.exists() and not overwrite:
      raise ValueError(f"Symbols already exist!")
    if symbols.keys() != set(dataset.ids):
      raise ValueError(f"Symbols contain not all Id's!")
    save_data_symbols(symbols_path, symbols)


def get_selection(directory: Path, subset_name: SubsetName) -> Subset:
  dataset_path = get_dataset_path(directory)
  if not dataset_path.is_file():
    raise ValueError("Dataset does not exist!")
  dataset = load_dataset(dataset_path)
  if subset_name not in dataset.subsets:
    raise ValueError("Subset does not exist!")
  result = dataset.subsets[subset_name]
  return result
