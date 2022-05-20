from argparse import ArgumentParser, Namespace
from logging import Logger

from text_selection_cli.argparse_helper import (ConvertToOrderedSetAction, parse_existing_file,
                                                parse_non_empty_or_whitespace)
from text_selection_cli.default_args import add_dataset_argument
from text_selection_cli.globals import ExecutionResult
from text_selection_cli.io_handling import try_load_data_weights, try_load_dataset, try_save_dataset
from text_selection_core.common import SortingDefaultParameters
from text_selection_core.sorting.fifo_sorting import sort_fifo
from text_selection_core.sorting.reverse_sorting import sort_reverse
from text_selection_core.sorting.weights_sorting import sort_after_weights
from text_selection_core.validation import ValidationErrBase


def get_fifo_sorting_parser(parser: ArgumentParser):
  parser.description = "Sort lines by FIFO principle."
  add_dataset_argument(parser)
  parser.add_argument("subsets", type=parse_non_empty_or_whitespace, nargs="+", metavar="SUBSET",
                      help="subsets", action=ConvertToOrderedSetAction)
  return sort_fifo_from_ns


def sort_fifo_from_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.dataset, logger)
  if isinstance(dataset, ValidationErrBase):
    return dataset

  default_params = SortingDefaultParameters(dataset, ns.subsets)
  changed_anything = sort_fifo(default_params, flogger)
  if isinstance(changed_anything, ValidationErrBase):
    return changed_anything

  if changed_anything:
    if error := try_save_dataset(ns.dataset, dataset, logger):
      return error

  return changed_anything


def get_reverse_sorting_parser(parser: ArgumentParser):
  parser.description = "Reverse sorting."
  add_dataset_argument(parser)
  parser.add_argument("subsets", type=parse_non_empty_or_whitespace, nargs="+", metavar="SUBSET",
                      help="subsets", action=ConvertToOrderedSetAction)
  return sort_reverse_from_ns


def sort_reverse_from_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.dataset, logger)
  if isinstance(dataset, ValidationErrBase):
    return dataset

  default_params = SortingDefaultParameters(dataset, ns.subsets)
  changed_anything = sort_reverse(default_params, flogger)
  if isinstance(changed_anything, ValidationErrBase):
    return changed_anything

  if changed_anything:
    if error := try_save_dataset(ns.dataset, dataset, logger):
      return error

  return changed_anything


def get_weight_sorting_parser(parser: ArgumentParser):
  parser.description = "Reverse sorting."
  add_dataset_argument(parser)
  parser.add_argument("subsets", type=parse_non_empty_or_whitespace, nargs="+", metavar="SUBSET",
                      help="subsets", action=ConvertToOrderedSetAction)
  parser.add_argument("weights", type=parse_existing_file, metavar="WEIGHTS-PATH",
                      help="path to the weights")
  return sort_after_weights_ns


def sort_after_weights_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  dataset = try_load_dataset(ns.dataset, logger)
  if isinstance(dataset, ValidationErrBase):
    return dataset

  weights = try_load_data_weights(ns.weights, logger)
  if isinstance(weights, ValidationErrBase):
    return weights

  default_params = SortingDefaultParameters(dataset, ns.subsets)
  changed_anything = sort_after_weights(default_params, weights, flogger)
  if isinstance(changed_anything, ValidationErrBase):
    return changed_anything

  if changed_anything:
    if error := try_save_dataset(ns.dataset, dataset, logger):
      return error

  return changed_anything
