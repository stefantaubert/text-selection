
from collections import OrderedDict

import numpy as np
from ordered_set import OrderedSet

from text_selection.common.ngram_extractor import NGramExtractor


def test_component():
  data = OrderedDict([
    (4, ("d", "c", "c", "d")),
    (5, ("b", "c", "a", "x")),
    (6, ("a", "b", "a")),
  ])

  x = NGramExtractor(
    data=data,
    chunksize=1,
    maxtasksperchild=None,
    n_jobs=1,
    batches=None,
  )

  x.fit(
    consider_keys={4, 5, 6},
    n_gram=1,
    ignore_symbols={"c"},
  )

  result = x.predict(OrderedSet((4, 6)))

  np.testing.assert_array_equal(result, np.array([
    #a, b, d, x
    [0, 0, 2, 0],  # 4
    [2, 1, 0, 0],  # 5
  ]))
