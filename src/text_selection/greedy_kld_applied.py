from logging import getLogger
from typing import Dict, List, OrderedDict, TypeVar

from ordered_set import OrderedSet

from text_selection.greedy_kld_methods import (
    get_uniform_distribution, sort_greedy_kld, sort_greedy_kld_iterations,
    sort_greedy_kld_until, sort_greedy_kld_until_with_preselection)
from text_selection.selection import SelectionMode

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


def greedy_kld_uniform_default(data: OrderedDict[_T1, List[_T2]]) -> OrderedSet[_T1]:
  uniform_distr = get_uniform_distribution(data)
  greedy_selected = sort_greedy_kld(
    data=data,
    target_dist=uniform_distr,
  )
  return greedy_selected


def greedy_kld_uniform_iterations(data: OrderedDict[_T1, List[_T2]], iterations: int) -> OrderedSet[_T1]:
  uniform_distr = get_uniform_distribution(data)
  greedy_selected = sort_greedy_kld_iterations(
    data=data,
    target_dist=uniform_distr,
    iterations=iterations
  )
  return greedy_selected


def greedy_kld_uniform_seconds(data: OrderedDict[_T1, List[_T2]], durations_s: Dict[int, float], seconds: float) -> OrderedSet[_T1]:
  logger = getLogger(__name__)
  uniform_distr = get_uniform_distribution(data)
  if len(uniform_distr) > 0:
    logger.info(f"Target uniform distribution: {list(uniform_distr.values())[0]}")
  greedy_selected = sort_greedy_kld_until(
    data=data,
    target_dist=uniform_distr,
    until_values=durations_s,
    until_value=seconds,
  )
  return greedy_selected


def greedy_kld_uniform_seconds_with_preselection(data: OrderedDict[_T1, List[_T2]], durations_s: Dict[int, float], seconds: float, preselection: OrderedDict[_T1, List[_T2]]) -> OrderedSet[_T1]:
  logger = getLogger(__name__)
  uniform_distr = get_uniform_distribution(data)
  if len(uniform_distr) > 0:
    logger.info(f"Target uniform distribution: {list(uniform_distr.values())[0]}")
  greedy_selected = sort_greedy_kld_until_with_preselection(
    data=data,
    target_dist=uniform_distr,
    until_values=durations_s,
    until_value=seconds,
    preselection=preselection,
  )

  return greedy_selected

def greedy_kld_uniform_count(data: OrderedDict[_T1, List[_T2]], chars: Dict[int, int], total_count: int) -> OrderedSet[_T1]:
  uniform_distr = get_uniform_distribution(data)
  greedy_selected = sort_greedy_kld_until(
    data=data,
    target_dist=uniform_distr,
    until_values=chars,
    until_value=total_count,
  )
  return greedy_selected
