import codecs
from argparse import ArgumentTypeError
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Callable, Optional
from typing import OrderedDict as OrderedDictType
from typing import TypeVar

from general_utils import get_all_files_in_all_subfolders
from ordered_set import OrderedSet

from text_selection_app.io import DATASET_FULL_NAME


def get_datasets(folder: Path) -> OrderedSet[Path]:
  result = get_all_files_in_all_subfolders(folder)
  data_ids_paths = OrderedSet(sorted(
    path
    for path in result
    if path.name == DATASET_FULL_NAME
  ))

  logger = getLogger(__name__)
  logger.info(f"Found {len(data_ids_paths)} datasets.")
  return data_ids_paths
