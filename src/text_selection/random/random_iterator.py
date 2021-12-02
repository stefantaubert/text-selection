import random
from typing import Iterator

from ordered_set import OrderedSet
from tqdm import tqdm


class RandomIterator(Iterator[int]):
  def __init__(self, data_indices: OrderedSet[int], seed: int) -> None:
    super().__init__()
    self.__available_data_keys_ordered = data_indices.copy()
    random.seed(seed)
    self._iteration_tqdm: tqdm = None

  def __iter__(self) -> Iterator[int]:
    return self

  def close(self) -> None:
    if self._iteration_tqdm is not None:
      self._iteration_tqdm.close()
      self._iteration_tqdm = None

  def __next__(self) -> int:
    if self._iteration_tqdm is None:
      self._iteration_tqdm = tqdm(total=len(self.__available_data_keys_ordered),
                                  desc="Random iterations", ncols=200, unit="it")

    if len(self.__available_data_keys_ordered) == 0:
      self._iteration_tqdm.close()
      self._iteration_tqdm = None
      raise StopIteration()

    next_index = random.choice(self.__available_data_keys_ordered)
    self.__available_data_keys_ordered.remove(next_index)
    self._iteration_tqdm.update()
    return next_index
