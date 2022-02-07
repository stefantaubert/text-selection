import re
import sys
import argparse
import logging
from argparse import ArgumentParser
from logging import getLogger
from typing import Callable, Dict, Generator, List, Tuple

from text_selection_app.datasets import (get_backup_parser,
                                         get_dataset_creation_from_text_parser,
                                         get_restore_parser)
from text_selection_app.export import get_export_txt_parser
from text_selection_app.filtering import get_duplicate_selection_parser
from text_selection_app.n_grams import get_n_grams_extraction_parser
from text_selection_app.selection import (get_fifo_selection_parser,
                                          get_greedy_selection_parser, get_kld_selection_parser)
from text_selection_app.sorting import get_fifo_sorting_parser, get_reverse_sorting_parser
from text_selection_app.statistics import get_statistics_generation_parser
from text_selection_app.subset import get_subset_renaming_parser
from text_selection_app.subsets import (get_subsets_creation_parser,
                                        get_subsets_removal_parser)
from text_selection_app.weights import (
    get_symbol_count_weights_creation_parser,
    get_uniform_weights_creation_parser, get_weights_division_parser,
    get_word_count_weights_creation_parser)

__version__ = "0.0.1"

INVOKE_HANDLER_VAR = "invoke_handler"


Parsers = Generator[Tuple[str, str, Callable], None, None]


def formatter(prog):
  return argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=40)


def get_dataset_parsers() -> Parsers:
  yield "create-from-text", "create dataset from text", get_dataset_creation_from_text_parser
  yield "backup", "backup dataset", get_backup_parser
  yield "restore", "restore dataset", get_restore_parser
  yield "get-stats", "getting statistics", get_statistics_generation_parser


def get_weights_parsers() -> Parsers:
  yield "create-uniform", "create uniform weights", get_uniform_weights_creation_parser
  yield "create-word-count", "create weights from word count", get_word_count_weights_creation_parser
  yield "create-symbol-count", "create weights from symbols count", get_symbol_count_weights_creation_parser
  yield "divide", "divide weights", get_weights_division_parser


def get_subset_parsers() -> Parsers:
  yield "rename", "rename subset", get_subset_renaming_parser
  yield "export", "export subset as text file", get_export_txt_parser


def get_subsets_parsers() -> Parsers:
  yield "add", "add subsets", get_subsets_creation_parser
  yield "remove", "remove subsets", get_subsets_removal_parser
  yield "select-fifo", "select entries FIFO-style", get_fifo_selection_parser
  yield "select-greedy", "select entries greedy-style", get_greedy_selection_parser
  yield "select-kld", "select entries kld-style", get_kld_selection_parser
  yield "filter-duplicates", "filter duplicates", get_duplicate_selection_parser
  yield "sort-fifo", "sort entries FIFO-style", get_fifo_sorting_parser
  yield "sort-reverse", "reverse entries", get_reverse_sorting_parser


def get_ngrams_parsers() -> Parsers:
  yield "create", "create n-grams", get_n_grams_extraction_parser


def _init_parser():
  main_parser = ArgumentParser(
    formatter_class=formatter,
    description="This program provides methods to select data.",
  )
  main_parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
  subparsers = main_parser.add_subparsers(help="description")

  parsers: Dict[str, Tuple[Parsers, str]] = {
    "dataset": (get_dataset_parsers(), "dataset commands"),
    "subsets": (get_subsets_parsers(), "subsets commands"),
    "subset": (get_subset_parsers(), "subset commands"),
    "weights": (get_weights_parsers(), "weights commands"),
    "n-grams": (get_ngrams_parsers(), "n-grams commands"),
  }

  for parser_name, (methods, help_str) in parsers.items():
    sub_parser = subparsers.add_parser(parser_name, help=help_str, formatter_class=formatter)
    subparsers_of_subparser = sub_parser.add_subparsers()
    for command, description, method in methods:
      method_parser = subparsers_of_subparser.add_parser(
        command, help=description, formatter_class=formatter)
      method_parser.set_defaults(**{
        INVOKE_HANDLER_VAR: method(method_parser),
      })

  return main_parser


def configure_logger() -> None:
  loglevel = logging.DEBUG if __debug__ else logging.INFO
  main_logger = getLogger()
  main_logger.setLevel(loglevel)
  main_logger.manager.disable = logging.NOTSET
  if len(main_logger.handlers) > 0:
    console = main_logger.handlers[0]
  else:
    console = logging.StreamHandler()
    main_logger.addHandler(console)

  logging_formatter = logging.Formatter(
    '[%(asctime)s.%(msecs)03d] (%(levelname)s) %(message)s',
    '%Y/%m/%d %H:%M:%S',
  )
  console.setFormatter(logging_formatter)
  console.setLevel(loglevel)


def parse_args(args: List[str]):
  configure_logger()
  parser = _init_parser()
  received_args = parser.parse_args(args)
  params = vars(received_args)

  if INVOKE_HANDLER_VAR in params:
    invoke_handler: Callable[[ArgumentParser], None] = params.pop(INVOKE_HANDLER_VAR)
    invoke_handler(received_args)
  else:
    parser.print_help()


if __name__ == "__main__":
  args = sys.argv[1:]
  parse_args(args)
