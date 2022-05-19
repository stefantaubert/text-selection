import argparse
import platform
import sys
from argparse import ArgumentParser
from importlib.metadata import version
from logging import getLogger
from pathlib import Path
from pkgutil import iter_modules
from tempfile import gettempdir
from time import perf_counter
from typing import Callable, Dict, Generator, List, Tuple

from text_selection_cli.argparse_helper import get_optional, parse_path
from text_selection_cli.datasets import get_init_parser
from text_selection_cli.export import get_export_txt_parser
from text_selection_cli.filtering import (get_duplicate_selection_parser,
                                          get_line_unit_frequency_parser,
                                          get_regex_match_selection_parser,
                                          get_string_filter_parser, get_unit_frequency_parser,
                                          get_vocabulary_filtering_parser,
                                          get_weight_filtering_parser)
from text_selection_cli.globals import ExecutionResult
from text_selection_cli.logging_configuration import (configure_root_logger, get_file_logger,
                                                      init_and_return_loggers, try_init_file_logger)
#from text_selection_app.n_grams import get_n_grams_extraction_parser
from text_selection_cli.selection import (get_fifo_selection_parser,
                                          get_greedy_selection_epoch_parser,
                                          get_greedy_selection_parser, get_id_selection_parser,
                                          get_kld_selection_parser)
from text_selection_cli.sorting import get_fifo_sorting_parser, get_reverse_sorting_parser
from text_selection_cli.statistics import get_statistics_generation_parser
from text_selection_cli.subset import get_subset_renaming_parser
from text_selection_cli.subsets import get_subsets_creation_parser, get_subsets_removal_parser
from text_selection_cli.weights import (get_uniform_weights_creation_parser,
                                        get_weights_division_parser,
                                        get_word_count_weights_creation_parser)

prog_name = "text-selection"

__version__ = version(prog_name)

INVOKE_HANDLER_VAR = "invoke_handler"

CONSOLE_PNT_GREEN = "\x1b[1;49;32m"
CONSOLE_PNT_RED = "\x1b[1;49;31m"
CONSOLE_PNT_RST = "\x1b[0m"


Parsers = Generator[Tuple[str, str, Callable], None, None]


Parsers = Generator[Tuple[str, str, Callable[[ArgumentParser],
                                             Callable[..., ExecutionResult]]], None, None]


def formatter(prog):
  return argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=40)


def get_dataset_parsers() -> Parsers:
  yield "create-from-text", "create dataset from text", get_init_parser
  yield "get-stats", "getting statistics", get_statistics_generation_parser


def get_weights_parsers() -> Parsers:
  yield "create-uniform", "create uniform weights", get_uniform_weights_creation_parser
  yield "create-count", "create weights from unit count", get_word_count_weights_creation_parser
  yield "divide", "divide weights", get_weights_division_parser


def get_subset_parsers() -> Parsers:
  yield "rename", "rename subset", get_subset_renaming_parser


def get_subsets_parsers() -> Parsers:
  yield "add", "add subsets", get_subsets_creation_parser
  yield "remove", "remove subsets", get_subsets_removal_parser
  yield "select-lines", "select entries by line number", get_id_selection_parser
  yield "select-fifo", "select entries FIFO-style", get_fifo_selection_parser
  yield "select-greedy", "select entries greedy-style", get_greedy_selection_parser
  yield "select-greedy-epochs", "select entries greedy-style epoch-based", get_greedy_selection_epoch_parser
  yield "select-kld", "select entries kld-style", get_kld_selection_parser
  yield "filter-duplicates", "filter duplicates", get_duplicate_selection_parser
  yield "filter-regex", "filter lines via regex", get_regex_match_selection_parser
  yield "filter-count", "filter lines via count of units", get_unit_frequency_parser
  yield "filter-by-weight", "filter lines by weight", get_weight_filtering_parser
  yield "filter-by-vocabulary", "filter lines by unit vocabulary", get_vocabulary_filtering_parser
  yield "filter-by-text", "filter lines by line text", get_string_filter_parser
  yield "filter-by-unit-freq", "filter lines by unit frequencies per line", get_line_unit_frequency_parser
  yield "sort-by-line-nr", "sort entries by line number", get_fifo_sorting_parser
  yield "sort-reverse", "reverse entries", get_reverse_sorting_parser
  yield "export", "export subsets lines", get_export_txt_parser


# def get_ngrams_parsers() -> Parsers:
#   # yield "create", "create n-grams", get_n_grams_extraction_parser
#   pass


def get_parsers() -> Dict[str, Tuple[Parsers, str]]:
  parsers: Dict[str, Tuple[Parsers, str]] = {
    "dataset": (get_dataset_parsers(), "dataset commands"),
    "subsets": (get_subsets_parsers(), "subsets commands"),
    "subset": (get_subset_parsers(), "subset commands"),
    "weights": (get_weights_parsers(), "weights commands"),
  }
  return parsers


def print_features():
  parsers = get_parsers()
  for parser_name, (methods, help_str) in parsers.items():
    print(f"- {parser_name}")
    for command, description, method in methods:
      print(f"  - {description}")


def _init_parser():
  main_parser = ArgumentParser(
    formatter_class=formatter,
    description="This program provides methods to modify TextGrids (.TextGrid) and their corresponding audios (.wav).",
  )
  main_parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
  subparsers = main_parser.add_subparsers(help="description")
  default_log_path = Path(gettempdir()) / f"{prog_name}.log"

  parsers = get_parsers()
  for parser_name, (methods, help_str) in parsers.items():
    sub_parser = subparsers.add_parser(parser_name, help=help_str, formatter_class=formatter)
    subparsers_of_subparser = sub_parser.add_subparsers()
    for command, description, method in methods:
      method_parser = subparsers_of_subparser.add_parser(
        command, help=description, formatter_class=formatter)
      # init parser
      invoke_method = method(method_parser)
      method_parser.set_defaults(**{
        INVOKE_HANDLER_VAR: invoke_method,
      })
      logging_group = method_parser.add_argument_group("logging arguments")
      logging_group.add_argument("--log", type=get_optional(parse_path), metavar="FILE",
                                 nargs="?", const=None, help="path to write the log", default=default_log_path)
      logging_group.add_argument("--debug", action="store_true",
                                 help="include debugging information in log")

  return main_parser


def parse_args(args: List[str]) -> None:
  configure_root_logger()
  root_logger = getLogger()

  local_debugging = debug_file_exists()
  if local_debugging:
    root_logger.debug(f"Received arguments: {str(args)}")

  parser = _init_parser()

  try:
    ns = parser.parse_args(args)
  except SystemExit:
    # invalid command supplied
    return

  if local_debugging:
    root_logger.debug(f"Parsed arguments: {str(ns)}")

  if hasattr(ns, INVOKE_HANDLER_VAR):
    invoke_handler: Callable[..., ExecutionResult] = getattr(ns, INVOKE_HANDLER_VAR)
    delattr(ns, INVOKE_HANDLER_VAR)
    log_to_file = ns.log is not None
    if log_to_file:
      log_to_file = try_init_file_logger(ns.log, local_debugging or ns.debug)
      if not log_to_file:
        root_logger.warning("Logging to file is not possible.")

    flogger = get_file_logger()
    if not local_debugging:
      sys_version = sys.version.replace('\n', '')
      flogger.debug(f"CLI version: {__version__}")
      flogger.debug(f"Python version: {sys_version}")
      flogger.debug("Modules: %s", ', '.join(sorted(p.name for p in iter_modules())))

      my_system = platform.uname()
      flogger.debug(f"System: {my_system.system}")
      flogger.debug(f"Node Name: {my_system.node}")
      flogger.debug(f"Release: {my_system.release}")
      flogger.debug(f"Version: {my_system.version}")
      flogger.debug(f"Machine: {my_system.machine}")
      flogger.debug(f"Processor: {my_system.processor}")

    flogger.debug(f"Received arguments: {str(args)}")
    flogger.debug(f"Parsed arguments: {str(ns)}")

    start = perf_counter()
    cmd_flogger, cmd_logger = init_and_return_loggers(__name__)
    success, changed_anything = invoke_handler(ns, cmd_logger, cmd_flogger)

    if success:
      root_logger.info(f"{CONSOLE_PNT_GREEN}Everything was successfull!{CONSOLE_PNT_RST}")
      flogger.info("Everything was successfull!")
    else:
      if log_to_file:
        root_logger.error(
          "Not everything was successfull! See log for details.")
      else:
        root_logger.error(
          "Not everything was successfull!")
      flogger.error("Not everything was successfull!")

    if changed_anything is not None and not changed_anything:
      root_logger.info("Didn't changed anything.")
      flogger.info("Didn't changed anything.")

    duration = perf_counter() - start
    flogger.debug(f"Total duration (s): {duration}")

    if log_to_file:
      # path not encapsulated in "" because it is only console out
      root_logger.info(f"Written log to: {ns.log.absolute()}")
  else:
    parser.print_help()


def run():
  arguments = sys.argv[1:]
  parse_args(arguments)


def run_prod():
  run()


def debug_file_exists():
  return (Path(gettempdir()) / f"{prog_name}-debug").is_file()


def create_debug_file():
  if not debug_file_exists():
    (Path(gettempdir()) / f"{prog_name}-debug").write_text("", "UTF-8")


if __name__ == "__main__":
  # print_features()
  create_debug_file()
  run()
