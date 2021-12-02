from typing import Dict, Iterator, List, Tuple, Union

import numpy as np
from tqdm import tqdm


class UntilIterator(Iterator[int]):
  def __init__(self, iterator: Iterator[int], until_values: np.ndarray, until_value: Union[int, float]) -> None:
    super().__init__()
    self.__iterator = iterator
    self.__until_values = until_values
    self.__until_value = until_value
    self.__tqdm: tqdm = None
    max_until = np.sum(until_values, axis=0)
    self.__current_total = 0.0
    self.__adjusted_until = round(min(until_value, max_until))
    self.__enough_data_was_available: bool = None

  def __iter__(self) -> Iterator[int]:
    return self

  @property
  def was_enough_data_available(self) -> bool:
    return self.__enough_data_was_available

  def __next__(self) -> int:
    if self.__tqdm is None:
      self.__tqdm = tqdm(total=self.__adjusted_until, initial=round(self.__current_total),
                         desc="Until", ncols=200, unit="units")

    try:
      selected_key = next(self.__iterator)
    except StopIteration:
      if self.__current_total == self.__until_value:
        self.__enough_data_was_available = True
      else:
        self.__enough_data_was_available = False
      self.__tqdm.close()
      self.__tqdm = None
      raise StopIteration()

    assert 0 <= selected_key < len(self.__until_values)
    selected_until_value = self.__until_values[selected_key]
    new_total = self.__current_total + selected_until_value
    if new_total <= self.__until_value:
      self.__current_total = new_total
      self.__tqdm.update(round(selected_until_value))
      return selected_key
    else:
      # todo close iterator
      self.__tqdm.close()
      self.__tqdm = None
      self.__enough_data_was_available = True
      raise StopIteration()


def iterate_durations_dict(iterator: Iterator[int], until_values: Dict[int, Union[int, float]], until_value: Union[int, float]) -> Tuple[List[int], bool]:
  iterated_values: List[int] = []
  enough_data_was_available = False
  max_until = sum(until_values)
  adjusted_until = round(min(until_value, max_until))
  current_total = 0.0
  with tqdm(total=len(until_values), initial=0) as progress_bar1:
    with tqdm(total=adjusted_until, initial=round(current_total)) as progress_bar:
      for selected_key in iterator:
        assert selected_key in until_values
        selected_until_value = until_values[selected_key]
        new_total = current_total + selected_until_value
        if new_total <= until_value:
          iterated_values.append(selected_key)
          current_total = new_total
          progress_bar.update(round(selected_until_value))
          progress_bar1.update()
          if current_total == until_value:
            enough_data_was_available = True
        else:
          enough_data_was_available = True
          break
    # Selected: {current_total:.1f}/{until_value:.1f} ({current_total/until_value*100:.2f}%).
    return iterated_values, enough_data_was_available


def iterate_durations(iterator: Iterator[int], until_values: np.ndarray, until_value: Union[int, float]) -> Tuple[List[int], bool]:
  iterated_values: List[int] = []
  enough_data_was_available = False
  max_until = sum(until_values)
  adjusted_until = round(min(until_value, max_until))
  current_total = 0.0
  with tqdm(total=adjusted_until, initial=round(current_total)) as progress_bar:
    for selected_key in iterator:
      assert 0 <= selected_key < len(until_values)
      selected_until_value = until_values[selected_key]
      new_total = current_total + selected_until_value
      if new_total <= until_value:
        iterated_values.append(selected_key)
        current_total = new_total
        progress_bar.update(round(selected_until_value))
        if current_total == until_value:
          enough_data_was_available = True
      else:
        enough_data_was_available = True
        break
  # Selected: {current_total:.1f}/{until_value:.1f} ({current_total/until_value*100:.2f}%).
  return iterated_values, enough_data_was_available
