
from argparse import ArgumentParser
from multiprocessing import cpu_count

from text_selection_app.argparse_helper import (get_optional,
                                                parse_positive_integer)

DEFAULT_N_JOBS = cpu_count()
DEFAULT_CHUNKSIZE = 500
DEFAULT_MAXTASKSPERCHILD = None


def add_n_jobs_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-j", "--n-jobs", metavar='N', type=int,
                      choices=range(1, cpu_count() + 1), default=DEFAULT_N_JOBS, help="amount of parallel cpu jobs")


def add_chunksize_argument(parser: ArgumentParser, target: str = "files", default: int = DEFAULT_CHUNKSIZE) -> None:
  parser.add_argument("-s", "--chunksize", type=parse_positive_integer, metavar="NUMBER",
                      help=f"amount of {target} to chunk into one job", default=default)


def add_maxtaskperchild_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-m", "--maxtasksperchild", type=get_optional(parse_positive_integer), metavar="NUMBER",
                      help="amount of tasks per child", default=DEFAULT_MAXTASKSPERCHILD)
