from argparse import ArgumentParser, Namespace
from logging import Logger
from pathlib import Path
from typing import cast

from text_selection_cli.argparse_helper import parse_non_empty_or_whitespace, parse_path
from text_selection_cli.default_args import add_dataset_argument, add_file_arguments
from text_selection_cli.globals import ExecutionResult
from text_selection_cli.io_handling import get_dataset_path, try_load_file, try_save_dataset
from text_selection_core.types import Dataset


def get_init_parser(parser: ArgumentParser):
  parser.description = "This command reads the lines of a textfile and initializes a dataset from it."
  add_file_arguments(parser)
  parser.add_argument("name", type=parse_non_empty_or_whitespace, metavar="SUBSET-NAME",
                      help="name of the initial subset containing all line numbers")
  parser.add_argument("dataset", type=parse_path, metavar="DATASET-PATH",
                      help="output dataset file")
  return create_dataset_from_text_ns


def create_dataset_from_text_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  # TODO maybe just count ns.lsep occurrences + 1
  lines = try_load_file(ns.file, ns.encoding, ns.lsep, logger)
  if lines is None:
    return False, None

  if len(lines) == 0:
    logger.error("File has no content!")
    return False, None

  logger.info("Creating dataset...")
  dataset = Dataset(len(lines), ns.name)

  success = try_save_dataset(ns.dataset, dataset, logger)

  return success, None
