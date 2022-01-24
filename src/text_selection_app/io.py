from logging import getLogger
from pathlib import Path
from typing import List, cast

from general_utils import load_obj, save_obj
from text_selection_core.types import (DataIds, Dataset, DataSymbols,
                                       DataWeights, NGramSet)

DATASET_NAME = "data"
DATA_SYMBOLS_NAME = "symbols"
FILE_EXTENSION = ".pkl"
DATASET_FULL_NAME = f"{DATASET_NAME}{FILE_EXTENSION}"
DATA_SYMBOLS_FULL_NAME = f"{DATA_SYMBOLS_NAME}{FILE_EXTENSION}"


def get_dataset_path(directory: Path) -> Path:
  return directory / DATASET_FULL_NAME


def load_dataset(path: Path) -> Dataset:
  logger = getLogger(__name__)
  logger.debug(f"Loading '{path}'...")
  return cast(Dataset, load_obj(path))


def save_dataset(path: Path, dataset: Dataset) -> None:
  logger = getLogger(__name__)
  logger.debug(f"Saving '{path}'...")
  path.parent.mkdir(parents=True, exist_ok=True)
  save_obj(dataset, path)


def get_data_symbols_path(directory: Path) -> Path:
  return directory / DATA_SYMBOLS_FULL_NAME


def load_data_symbols(path: Path) -> DataSymbols:
  logger = getLogger(__name__)
  logger.debug(f"Loading '{path}'...")
  return cast(DataSymbols, load_obj(path))


def save_data_symbols(path: Path, data_symbols: DataSymbols) -> None:
  logger = getLogger(__name__)
  logger.debug(f"Saving '{path}'...")
  path.parent.mkdir(parents=True, exist_ok=True)
  save_obj(data_symbols, path)


def get_data_weights_path(directory: Path, name: str) -> Path:
  return directory / f"{name}{FILE_EXTENSION}"


def load_data_weights(path: Path) -> DataWeights:
  logger = getLogger(__name__)
  logger.debug(f"Loading '{path}'...")
  return cast(DataWeights, load_obj(path))


def save_data_weights(path: Path, data_weights: DataWeights) -> None:
  logger = getLogger(__name__)
  logger.debug(f"Saving '{path}'...")
  path.parent.mkdir(parents=True, exist_ok=True)
  save_obj(data_weights, path)


def get_data_n_grams_path(directory: Path, name: str) -> Path:
  return directory / f"{name}{FILE_EXTENSION}"


def load_data_n_grams(path: Path) -> NGramSet:
  logger = getLogger(__name__)
  logger.debug(f"Loading '{path}'...")
  return cast(NGramSet, load_obj(path))


def save_data_n_grams(path: Path, data_n_grams: NGramSet) -> None:
  logger = getLogger(__name__)
  logger.debug(f"Saving '{path}'...")
  path.parent.mkdir(parents=True, exist_ok=True)
  save_obj(data_n_grams, path)
