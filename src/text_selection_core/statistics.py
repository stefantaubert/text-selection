from typing import Dict, Generator, Iterable, List, Tuple

import numpy as np
from ordered_set import OrderedSet
from pandas import DataFrame
from text_utils import Symbol

from text_selection_core.types import (Dataset, DataSymbols, DataWeights, Item,
                                       NGramSet, Subset, SubsetName,
                                       item_to_symbols)

SPACE_DISPL = "␣"


def generate_statistics(dataset: Dataset, symbols: DataSymbols, weights: List[Tuple[str, DataWeights]], n_grams: List[NGramSet]) -> Generator[Tuple[str, DataFrame], None, None]:
  yield "Selection", get_selection_statistics(dataset)
  yield "Symbols", get_symbols_statistics(dataset, symbols)
  yield "Weights", get_weights_statistics(weights)
  for subset in get_subsets_ordered(dataset):
    yield f"Weights {subset}", get_subset_weights_statistics(dataset[subset], weights)


def get_subsets_ordered(dataset: Dataset) -> OrderedSet[str]:
  return OrderedSet(sorted(dataset.keys()))


def get_all_symbols(items: Iterable[Item]) -> Generator[Symbol, None, None]:
  result = (
    symbol
    for item in items
    for symbols in item_to_symbols(item)
    for symbol in symbols
  )
  return result


def get_n_gram_statistics(dataset: Dataset, n_grams: List[NGramSet]):
  pass


def get_symbols_statistics(dataset: Dataset, symbols_strs: DataSymbols):
  data = []
  all_symbols = set(get_all_symbols(symbols_strs))
  all_symbols = {repr(symbol)[1:-1] if symbol != " " else SPACE_DISPL for symbol in all_symbols}
  all_symbols = sorted(all_symbols)
  # TODO this as n_gram stats
  columns = ["Symbols"]

  contains_symbol: Dict[SubsetName, Generator[str, None, None]] = {}

  for subset in get_subsets_ordered(dataset):
    columns.append(subset)
    subset_symbols_strs = (symbols_strs[data_id] for data_id in subset)
    subset_symbols = set(get_all_symbols(subset_symbols_strs))
    subset_matches = ("x" if symbol in subset_symbols else "-" for symbol in all_symbols)
    contains_symbol[subset] = subset_matches
    # TODO add percent un-/covered as line

  for symbol in all_symbols:
    line = [symbol] + list(next(contains_symbol[k]) for k in contains_symbol)
    data.append(line)

  df = DataFrame(data, columns)

  return df


def get_selection_statistics(dataset: Dataset):
  data = []
  for subset in get_subsets_ordered(dataset):
    data.append(
      subset,
      len(subset),
      len(dataset.ids),
      len(dataset.ids) - len(subset),
      len(subset) / len(dataset.ids) * 100,
      (len(dataset.ids) - len(subset)) / len(dataset.ids) * 100,
    )

  df = DataFrame(
    data,
    columns=(
      "Name",
      "Sel",
      "Rst",
      "Tot",
      "Sel %",
      "Rst %",
    ),
  )

  return df


def get_subset_weights_statistics(subset: Subset, weights: List[Tuple[str, DataWeights]]):
  weights_df_data = []
  for weights_name, data_weights in weights:
    data_weights_total = sum(data_weights.values())
    subset_weights = list(data_weights[k] for k in subset)
    weights_sum = sum(subset_weights)
    weights_df_data.append(
      weights_name,
      min(subset_weights),
      np.mean(subset_weights),
      np.median(subset_weights),
      max(subset_weights),
      weights_sum,
      data_weights_total - weights_sum,
      data_weights_total,
      weights_sum / data_weights_total * 100,
      (data_weights_total - weights_sum) / data_weights_total * 100,
    )

  weights_df = DataFrame(weights_df_data, columns=(
    "Name",
    "Min",
    "Avg",
    "Med",
    "Max",
    "Sum",
    "Rst",
    "Tot",
    "Sum %",
    "Rst %",
  ))

  return weights_df


def get_weights_statistics(weights: List[Tuple[str, DataWeights]]):
  data = []
  for weights_name, data_weights in weights:
    subset_weights = data_weights.values()
    data.append(
      weights_name,
      min(subset_weights),
      np.mean(subset_weights),
      np.median(subset_weights),
      max(subset_weights),
      sum(subset_weights),
    )

  df = DataFrame(
    data,
    columns=(
      "Name",
      "Min",
      "Avg",
      "Med",
      "Max",
      "Sum",
    ),
  )

  return df
