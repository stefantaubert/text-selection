from logging import Logger
from pathlib import Path

from general_utils import get_all_files_in_all_subfolders
from ordered_set import OrderedSet

from text_selection_cli.io_handling import DATASET_FULL_NAME


def get_datasets(folder: Path, logger: Logger) -> OrderedSet[Path]:
  result = get_all_files_in_all_subfolders(folder)
  data_ids_paths = OrderedSet(sorted(
    path
    for path in result
    if path.name == DATASET_FULL_NAME
  ))

  logger.info(f"Found {len(data_ids_paths)} datasets.")
  return data_ids_paths
