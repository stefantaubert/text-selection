from argparse import ArgumentParser, Namespace
from logging import Logger

from text_selection_cli.argparse_helper import (ConvertToOrderedSetAction, parse_non_empty_or_whitespace)
from text_selection_cli.default_args import add_project_argument
from text_selection_cli.globals import ExecutionResult
from text_selection_cli.io_handling import (try_load_dataset, try_save_dataset)
from text_selection_core.common import SortingDefaultParameters
from text_selection_core.sorting.fifo_sorting import sort_fifo
from text_selection_core.sorting.reverse_sorting import sort_reverse


def get_fifo_sorting_parser(parser: ArgumentParser):
  parser.description = "Sort lines by FIFO principle."
  add_project_argument(parser)
  parser.add_argument("subsets", type=parse_non_empty_or_whitespace, nargs="+", metavar="subsets",
                      help="subsets", action=ConvertToOrderedSetAction)
  return sort_fifo_from_ns


def sort_fifo_from_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.project, logger)
  if dataset is None:
    return False, False

  default_params = SortingDefaultParameters(dataset, ns.subsets)
  error, changed_anything = sort_fifo(default_params)

  success = error is None

  if not success:
    logger.error(f"{error.default_message}")
    return False, False

  if changed_anything:
    success = try_save_dataset(ns.project, dataset, logger)
    if not success:
      return False, False

  return True, changed_anything


def get_reverse_sorting_parser(parser: ArgumentParser):
  parser.description = "Reverse sorting."
  add_project_argument(parser)
  parser.add_argument("subsets", type=parse_non_empty_or_whitespace, nargs="+", metavar="subsets",
                      help="subsets", action=ConvertToOrderedSetAction)
  return sort_reverse_from_ns


def sort_reverse_from_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.project, logger)
  if dataset is None:
    return False, False

  default_params = SortingDefaultParameters(dataset, ns.subsets)
  error, changed_anything = sort_reverse(default_params)

  success = error is None

  if not success:
    logger.error(f"{error.default_message}")
    return False, False

  if changed_anything:
    success = try_save_dataset(ns.project, dataset, logger)
    if not success:
      return False, False

  return True, changed_anything
