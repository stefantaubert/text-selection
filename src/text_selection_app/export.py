from argparse import ArgumentParser, Namespace
from logging import getLogger
from pathlib import Path
from typing import cast

from text_selection_app.argparse_helper import (get_optional, parse_existing_directory,
                                                parse_non_empty, parse_non_empty_or_whitespace,
                                                parse_path)
from text_selection_app.default_args import (add_directory_argument, add_encoding_argument,
                                             add_file_arguments)
from text_selection_app.helper import get_datasets
from text_selection_app.io_handling import DATASET_NAME, load_dataset
from text_selection_core.exporting.symbols_exporting import export_symbols


def get_export_txt_parser(parser: ArgumentParser):
  parser.description = f"This command."
  add_directory_argument(parser)
  parser.add_argument("subset", type=parse_non_empty_or_whitespace, metavar="subset",
                      help="subset which should be exported")
  add_file_arguments(parser)
  parser.add_argument("--name", type=get_optional(parse_non_empty_or_whitespace), metavar="NAME",
                      help="name of the exported text-file if not same as subset", default=None)
  parser.add_argument("-out", "--output-directory", type=get_optional(parse_path), metavar="PATH",
                      help="custom output directory if not same as input directory", default=None)

  return export_txt_ns


def export_txt_ns(ns: Namespace):
  logger = getLogger(__name__)
  logger.debug(ns)
  root_folder = cast(Path, ns.directory)
  datasets = get_datasets(root_folder)

  if ns.name in {DATASET_NAME}:
    logger.error("The given name is not valid.")
    return

  dataset_path: Path
  for i, dataset_path in enumerate(datasets, start=1):
    data_folder = dataset_path.parent
    data_name = str(data_folder.relative_to(root_folder)
                    ) if root_folder != data_folder else "root"
    logger.info(f"Processing {data_name} ({i}/{len(datasets)})")

    symbols_path = data_folder / cast(str, ns.file)
    if not symbols_path.exists():
      logger.error(
        "File was not found! Skipping...")
      continue

    output_directory = root_folder
    if ns.output_directory is not None:
      output_directory = ns.output_directory
    target_folder = output_directory / data_folder.relative_to(root_folder)
    name = ns.subset
    if ns.name is not None:
      name = ns.name
    target_file = target_folder / f"{name}.txt"

    if target_file.is_file() and not ns.overwrite:
      logger.info(f"File {str(target_file)} already exists. Skipping...")
      continue

    dataset = load_dataset(dataset_path)
    lines = symbols_path.read_text(ns.encoding).split(ns.lsep)

    logger.debug("Exporting...")
    error, text = export_symbols(dataset, ns.subset, lines)

    success = error is None

    if not success:
      logger.error(f"{error.default_message}")
      logger.info("Skipped.")
    else:
      assert text is not None
      target_file.write_text(text, ns.encoding)
      logger.debug(f"Written text to: '{str(target_file)}'.")
  return
