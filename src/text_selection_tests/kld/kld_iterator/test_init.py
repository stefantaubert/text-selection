
import numpy as np
from ordered_set import OrderedSet
from text_selection.kld.kld_iterator import KldIterator, get_uniform_weights
from text_selection.selection import FirstKeySelector


def test_init__empty_indicies():
  data = np.ones(shape=(3, 3), dtype=np.uint32)
  data_indicies = OrderedSet()
  preselection = np.zeros(data.shape, data.dtype)
  iterator = KldIterator(
    data=data,
    data_indicies=data_indicies,
    key_selector=FirstKeySelector(),
    preselection=preselection,
    weights=get_uniform_weights(data.shape[1]),
    maxtasksperchild=None,
    chunksize=1,
    n_jobs=1,
    batches=None,
  )

  result = list(iterator)

  assert result == OrderedSet()
