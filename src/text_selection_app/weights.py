from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from typing import cast

from text_selection_core.weights.calculation import get_uniform_weights

from text_selection_app.argparse_helper import (parse_existing_directory,
                                                parse_non_empty_or_whitespace)
from text_selection_app.helper import get_datasets
from text_selection_app.io import (get_data_weights_path, load_dataset,
                                   save_data_weights)


def get_uniform_weights_creation_parser(parser: ArgumentParser):
  parser.description = f"This command adds subsets."
  parser.add_argument("directory", type=parse_existing_directory, metavar="directory",
                      help="directory containing data")
  parser.add_argument("--name", type=parse_non_empty_or_whitespace, metavar="NAME",
                      help="name of the weights", default="weights")
  parser.add_argument("-o", "--overwrite", action="store_true",
                      help="overwrite weights")
  return create_uniform_weights_ns


def create_uniform_weights_ns(ns: Namespace) -> None:
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    weights_path = get_data_weights_path(data_folder, ns.name)

    if weights_path.is_file() and ns.overwrite:
      logger.error("Weights already exist! Skipped.")
      continue

    dataset = load_dataset(dataset_path)

    weights = get_uniform_weights(dataset.ids)

    save_data_weights(weights_path, weights)
