from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from typing import cast

from text_selection_core.subsets import rename_subset

from text_selection_app.argparse_helper import (parse_existing_directory,
                                                parse_non_empty_or_whitespace)
from text_selection_app.helper import get_datasets
from text_selection_app.io_handling import get_dataset_path, load_dataset, save_dataset


def get_subset_renaming_parser(parser: ArgumentParser):
  parser.description = f"This command adds subsets."
  parser.add_argument("directory", type=parse_existing_directory, metavar="directory",
                      help="directory containing data")
  parser.add_argument("name", type=parse_non_empty_or_whitespace, metavar="name",
                      help="subset that should be renamed")

  parser.add_argument("new_name", type=parse_non_empty_or_whitespace, metavar="new-name",
                      help="new name")
  return rename_subsets_ns


def rename_subsets_ns(ns: Namespace):
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    dataset = load_dataset(dataset_path)

    error, changed_anything = rename_subset(dataset, ns.name, ns.new_name)

    success = error is None

    if not success:
      logger.error(f"{error.default_message}")
      logger.info("Skipped.")
      assert not changed_anything
    else:
      assert changed_anything
      save_dataset(get_dataset_path(data_folder), dataset)
