from collections import Counter
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np
from ordered_set import OrderedSet
from pandas import DataFrame
from text_utils import Symbol

from text_selection_core.types import (Dataset, DataSymbols, DataWeights, Item,
                                       NGramSet, Subset, SubsetName,
                                       item_to_symbols)

SPACE_DISPL = "␣"


def generate_statistics(dataset: Dataset, symbols: Optional[DataSymbols], weights: List[Tuple[str, DataWeights]], n_grams: List[Tuple[str, NGramSet]]) -> Generator[Tuple[str, DataFrame], None, None]:
  yield "Selection", get_selection_statistics(dataset)
  if len(weights) > 0:
    yield "Weights", get_weights_statistics(weights)
    for subset in get_subsets_ordered(dataset):
      yield f"Weights {subset}", get_subset_weights_statistics(dataset.subsets[subset], weights)
  if symbols is not None:
    yield "Symbols", get_symbols_statistics(dataset, symbols)


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
  # TODO this as n_gram stats
  data = []
  all_symbols = OrderedSet(sorted(get_all_symbols(symbols_strs.values())))

  subset_contains_symbol: Dict[SubsetName, Counter] = {}
  subset_names = get_subsets_ordered(dataset)
  for subset_name in subset_names:
    subset = dataset.subsets[subset_name]
    subset_symbols_strs = (symbols_strs[data_id] for data_id in subset)
    #subset_symbols = set(get_all_symbols(subset_symbols_strs))
    #subset_matches = {symbol: "x" if symbol in subset_symbols else "-" for symbol in all_symbols}
    subset_matches = Counter(get_all_symbols(subset_symbols_strs))
    subset_contains_symbol[subset_name] = subset_matches
    # TODO add percent un-/covered as line

  #total_symbol_count = sum(sum(counters.values()) for counters in subset_contains_symbol.values())
  total_symbols_count = {
    symbol: sum(counters[symbol] for counters in subset_contains_symbol.values())
    for symbol in all_symbols
  }
  total_symbol_count = sum(total_symbols_count.values())

  for symbol in all_symbols:
    symbol_repr = repr(symbol)[1:-1] if symbol != " " else SPACE_DISPL
    counts = list(subset_contains_symbol[subset][symbol]
                  for subset in subset_contains_symbol)
    if total_symbol_count == 0:
      counts_percent_total = []
    else:
      counts_percent_total = list(count / total_symbol_count * 100 for count in counts)

    if symbol not in total_symbols_count or total_symbols_count[symbol] == 0:
      counts_percent_subset = []
    else:
      counts_percent_subset = list(count / total_symbols_count[symbol] * 100 for count in counts)

    line = [symbol_repr, total_symbols_count[symbol]] + \
        counts + counts_percent_total + counts_percent_subset
    data.append(line)
  arr = np.array(data)

  df = DataFrame(arr, columns=["Symbol", "Tot"] + list(subset_names) +
                 list(subset_names) + list(subset_names))

  return df


def get_selection_statistics(dataset: Dataset):
  data = []
  for subset_name in get_subsets_ordered(dataset):
    subset = dataset.subsets[subset_name]
    data.append((
      subset_name,
      len(subset),
      len(dataset.ids) - len(subset),
      len(dataset.ids),
      len(subset) / len(dataset.ids) * 100,
      (len(dataset.ids) - len(subset)) / len(dataset.ids) * 100,
    ))

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


NOT_AVAIL_VAL = "-"


def get_subset_weights_statistics(subset: Subset, weights: List[Tuple[str, DataWeights]]):
  data = []
  for weights_name, data_weights in weights:
    data_weights_total = sum(data_weights.values())
    subset_weights = list(data_weights[k] for k in subset)
    weights_sum = sum(subset_weights)
    values = [weights_name]
    if len(subset_weights) == 0:
      values += [NOT_AVAIL_VAL] * 4
    else:
      values.extend((
        min(subset_weights),
        np.mean(subset_weights),
        np.median(subset_weights),
        max(subset_weights),
      ))
    values.extend((
      weights_sum,
      data_weights_total - weights_sum,
      data_weights_total,
    ))
    if data_weights_total == 0:
      values += [NOT_AVAIL_VAL] * 2
    else:
      values.extend((
        weights_sum / data_weights_total * 100,
        (data_weights_total - weights_sum) / data_weights_total * 100,
      ))
    data.append(values)

  df = DataFrame(data, columns=(
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

  return df


def get_weights_statistics(weights: List[Tuple[str, DataWeights]]):
  data = []
  for weights_name, data_weights in weights:
    subset_weights = list(data_weights.values())
    data.append((
      weights_name,
      min(subset_weights),
      np.mean(subset_weights),
      np.median(subset_weights),
      max(subset_weights),
      sum(subset_weights),
    ))

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
