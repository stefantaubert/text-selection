import pickle
from logging import getLogger
from pathlib import Path
from typing import Any, List, cast

from text_selection_core.types import Dataset, DataWeights, Lines

DATASET_NAME = "data"
# DATA_SYMBOLS_NAME = "symbols"
FILE_EXTENSION = ".pkl"
DATASET_FULL_NAME = f"{DATASET_NAME}{FILE_EXTENSION}"
# DATA_SYMBOLS_FULL_NAME = f"{DATA_SYMBOLS_NAME}{FILE_EXTENSION}"


def save_obj(obj: Any, path: Path) -> None:
  assert isinstance(path, Path)
  assert path.parent.exists() and path.parent.is_dir()
  with open(path, mode="wb") as file:
    pickle.dump(obj, file)


def load_obj(path: Path) -> Any:
  assert isinstance(path, Path)
  assert path.is_file()
  with open(path, mode="rb") as file:
    return pickle.load(file)


def get_dataset_path(directory: Path) -> Path:
  return directory / DATASET_FULL_NAME

def load_file(path: Path) -> Lines:
  pass

def load_dataset(path: Path) -> Dataset:
  logger = getLogger(__name__)
  logger.debug(f"Loading '{path}'...")
  result = cast(Dataset, load_obj(path))
  logger.debug("Ok.")
  return result


def save_dataset(path: Path, dataset: Dataset) -> None:
  logger = getLogger(__name__)
  logger.debug(f"Saving '{path}'...")
  path.parent.mkdir(parents=True, exist_ok=True)
  save_obj(dataset, path)
  logger.debug("Ok.")


# def get_data_symbols_path(directory: Path) -> Path:
#   return directory / DATA_SYMBOLS_FULL_NAME


# def load_data_symbols(path: Path) -> Lines:
#   logger = getLogger(__name__)
#   logger.debug(f"Loading '{path}'...")
#   result = cast(Lines, load_obj(path))
#   logger.debug(f"Ok.")
#   return result


# def save_data_symbols(path: Path, data_symbols: Lines) -> None:
#   logger = getLogger(__name__)
#   logger.debug(f"Saving '{path}'...")
#   path.parent.mkdir(parents=True, exist_ok=True)
#   save_obj(data_symbols, path)
#   logger.debug(f"Ok.")


def get_data_weights_path(directory: Path, name: str) -> Path:
  return directory / f"{name}{FILE_EXTENSION}"


def load_data_weights(path: Path) -> DataWeights:
  logger = getLogger(__name__)
  logger.debug(f"Loading '{path}'...")
  result = cast(DataWeights, load_obj(path))
  logger.debug("Ok.")
  return result


def save_data_weights(path: Path, data_weights: DataWeights) -> None:
  logger = getLogger(__name__)
  logger.debug(f"Saving '{path}'...")
  path.parent.mkdir(parents=True, exist_ok=True)
  save_obj(data_weights, path)
  logger.debug("Ok.")


# def get_data_n_grams_path(directory: Path, name: str) -> Path:
#   return directory / f"{name}{FILE_EXTENSION}"


# def load_data_n_grams(path: Path) -> NGramSet:
#   logger = getLogger(__name__)
#   logger.debug(f"Loading '{path}'...")
#   result = cast(NGramSet, load_obj(path))
#   logger.debug(f"Ok.")
#   return result


# def save_data_n_grams(path: Path, data_n_grams: NGramSet) -> None:
#   logger = getLogger(__name__)
#   logger.debug(f"Saving '{path}'...")
#   path.parent.mkdir(parents=True, exist_ok=True)
#   save_obj(data_n_grams, path)
#   logger.debug(f"Ok.")
