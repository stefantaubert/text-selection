from pathlib import Path
from typing import List, cast

from general_utils import load_obj, save_obj
from text_selection.core.types import (DataIds, Dataset, DataSymbols,
                                       DataWeights)

DATA_ID_NAME = "ids"
DATA_SYMBOLS_NAME = "symbols"
FILE_EXTENSION = ".pkl"


def load_data_ids(directory: Path) -> DataIds:
  path = directory / f"{DATA_ID_NAME}{FILE_EXTENSION}"
  result = cast(DataIds, load_obj(path))
  return result


def save_data_ids(directory: Path, data_ids: DataIds) -> None:
  path = directory / f"{DATA_ID_NAME}{FILE_EXTENSION}"
  path.parent.mkdir(parents=True, exist_ok=True)
  save_obj(data_ids, path)


def load_data_symbols(directory: Path) -> DataSymbols:
  path = directory / f"{DATA_SYMBOLS_NAME}{FILE_EXTENSION}"
  result = cast(DataSymbols, load_obj(path))
  return result


def save_data_symbols(directory: Path, data_symbols: DataSymbols) -> None:
  path = directory / f"{DATA_SYMBOLS_NAME}{FILE_EXTENSION}"
  path.parent.mkdir(parents=True, exist_ok=True)
  save_obj(data_symbols, path)


def load_data_weights(directory: Path, name: str) -> DataWeights:
  path = directory / f"{name}{FILE_EXTENSION}"
  result = cast(DataWeights, load_obj(path))
  return result


def save_data_weights(directory: Path, data_weights: DataWeights, name: str) -> None:
  path = directory / f"{name}{FILE_EXTENSION}"
  path.parent.mkdir(parents=True, exist_ok=True)
  save_obj(data_weights, path)


def load_dataset(directory: Path, name: str) -> Dataset:
  path = directory / f"{name}{FILE_EXTENSION}"
  result = cast(Dataset, load_obj(path))
  return result


def save_dataset(directory: Path, dataset: Dataset, name: str) -> None:
  path = directory / f"{name}{FILE_EXTENSION}"
  path.parent.mkdir(parents=True, exist_ok=True)
  save_obj(dataset, path)
