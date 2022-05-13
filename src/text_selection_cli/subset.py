from argparse import ArgumentParser, Namespace
from logging import Logger

from text_selection_cli.argparse_helper import parse_non_empty_or_whitespace
from text_selection_cli.default_args import add_dataset_argument
from text_selection_cli.globals import ExecutionResult
from text_selection_cli.io_handling import try_load_dataset, try_save_dataset
from text_selection_core.subsets import rename_subset


def get_subset_renaming_parser(parser: ArgumentParser):
  parser.description = "This command rename a subset."
  add_dataset_argument(parser)
  parser.add_argument("name", type=parse_non_empty_or_whitespace, metavar="name",
                      help="subset that should be renamed")
  parser.add_argument("new_name", type=parse_non_empty_or_whitespace, metavar="new-name",
                      help="new name")
  return rename_subsets_ns


def rename_subsets_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.dataset, logger)
  if dataset is None:
    return False, False

  logger.info("Renaming subset...")
  error, changed_anything = rename_subset(dataset, ns.name, ns.new_name)

  success = error is None

  if not success:
    logger.error(error.default_message)
    return False, False

  if changed_anything:
    success = try_save_dataset(ns.dataset, dataset, logger)
    if not success:
      return False, False

  return True, changed_anything
