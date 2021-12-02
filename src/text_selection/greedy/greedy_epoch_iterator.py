from typing import Iterator

import numpy as np
from ordered_set import OrderedSet
from text_selection.greedy.greedy_iterator import GreedyIterator
from text_selection.selection import KeySelector
from tqdm import tqdm


class GreedyEpochIterator(GreedyIterator):
  def __init__(self, data: np.ndarray, data_indices: OrderedSet[int], preselection: np.ndarray, key_selector: KeySelector, epochs: int) -> None:
    assert epochs >= 0
    super().__init__(data, data_indices, preselection, key_selector)
    self.__epochs = epochs
    self.__epochs_tqdm: tqdm = None

  def __iter__(self) -> Iterator[int]:
    return self

  def close(self) -> None:
    super().close()
    if self.__epochs_tqdm is not None:
      self.__epochs_tqdm.close()
      self.__epochs_tqdm = None

  def __next__(self) -> int:
    if self.__epochs_tqdm is None:
      self.__epochs_tqdm = tqdm(total=self.__epochs, desc="Greedy epochs", ncols=200, unit="ep")

    current_epoch = self._current_epoch
    if current_epoch < self.__epochs:
      try:
        result = super().__next__()
        started_next_epoch = self._current_epoch == current_epoch + 1
        if started_next_epoch:
          self.__epochs_tqdm.update()
        return result
      except StopIteration:
        self.__epochs_tqdm.close()
        self.__epochs_tqdm = None
        raise StopIteration()
    else:
      if self._iteration_tqdm is not None:
        self._iteration_tqdm.close()
        self._iteration_tqdm = None
      self.__epochs_tqdm.close()
      self.__epochs_tqdm = None
      raise StopIteration()
