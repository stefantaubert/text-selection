import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Generator, List, Set, cast

from ordered_set import OrderedSet
from tqdm import tqdm

from text_selection_cli.argparse_helper import parse_non_negative_integer
from txt_selection_cli.globals import SEL_ENC, SEL_EXT, SEL_LSEP, ExecutionResult
from txt_selection_cli.helper import (ConvertToOrderedSetAction, add_encoding_argument,
                                      get_all_files_in_all_subfolders, parse_existing_directory,
                                      parse_non_empty, parse_non_empty_or_whitespace, parse_path,
                                      try_read_selection)
from txt_selection_cli.logging_configuration import get_file_logger, init_and_get_console_logger


def get_line_nr_moving_parser(parser: ArgumentParser):
  parser.description = "This command."
  parser.add_argument("directory", type=parse_existing_directory,
                      metavar="directory", help="directory containing files", action=ConvertToOrderedSetAction)
  parser.add_argument("from_set", type=parse_non_empty_or_whitespace,
                      metavar="from-set", help="from sets")
  parser.add_argument("to_set", type=parse_non_empty_or_whitespace, metavar="to-set",
                      help="to set")
  parser.add_argument("line_numbers", type=parse_non_negative_integer, nargs="+", metavar="line-numbers",
                      help="ids to select", action=ConvertToOrderedSetAction)
  return remove_subset_ns


def remove_subset_ns(ns: Namespace) -> ExecutionResult:
  logger = init_and_get_console_logger(__name__)
  flogger = get_file_logger()

  logger.info("Searching for selection files...")
  sel_files: List[Path] = list(
    file
    for file in get_all_files_in_all_subfolders(ns.directory)
    if file.suffix.lower() == SEL_EXT.lower()
  )

  if len(sel_files) == 0:
    logger.info("Did not find any selection files!")
    return True, None

  projects: Set[Path] = {sel_file.parent for sel_file in sel_files}

  from_sel_files = {
    proj_dir: proj_dir / f"{ns.from_set}{SEL_EXT}"
    for proj_dir in projects
  }

  all_successfull = True
  proj_path: Path
  for proj_path in tqdm(projects, desc="Processing projects(s)", unit=" project(s)"):
    flogger.info(f"Processing \"{proj_path}\"")

    from_sets: List[OrderedSet[int]] = []
    all_files_loaded = True
    sel_file: Path
    for sel_file in from_sel_files[proj_path]:
      selection = try_read_selection(sel_file, flogger)
      if selection is None:
        all_files_loaded = False
        break

      from_sets.append(selection)

    to_selection_path = proj_path / f"{ns.to_set}{SEL_EXT}"
    if to_selection_path.exists():
      to_selection = try_read_selection(to_selection_path, flogger)
      if to_selection is None:
        all_files_loaded = False
    else:
      to_selection = OrderedSet()

    if not all_files_loaded:
      logger.info("Skipped.")
      all_successfull = False
      continue

    flogger.info("Reading selection...")
    try:
      text = path.read_text(ns.encoding)
    except Exception as ex:
      flogger.error(f"File \"{path.absolute()}\" couldn't be loaded!")
      flogger.exception(ex)
      all_successfull = False
      continue
    if text != "":
      flogger.info("File is not empty and is therefore kept.")
      continue

    flogger.info(f"Removing \"{path}\"")
    try:
      os.remove(path)
    except Exception as ex:
      flogger.error("File couldn't be removed!")
      flogger.exception(ex)
      all_successfull = False
      continue
  return all_successfull, None
