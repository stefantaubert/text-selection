import pickle
import random
from collections import OrderedDict
from typing import List

from text_selection.random_applied import get_random_seconds
from tqdm import tqdm

seed = 1111
random.seed(seed)

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
            'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def get_random_list(length: int, chars: List[str]) -> List[str]:
  res = [random.choice(chars) for _ in range(length)]
  return res


n_data = 10000
data = OrderedDict({i: get_random_list(random.randint(1, 50), ALPHABET) for i in range(n_data)})

durations = {k: random.randint(1, 10) for k in data.keys()}

potential_seeds = range(3000)
# random.shuffle(potential_seeds)

potential_sets = []
for sample_seed in tqdm(potential_seeds):
  sample_set = get_random_seconds(
    data=data,
    seed=sample_seed,
    durations_s=durations,
    seconds=3600,
  )
  potential_sets.append(sample_set)

with open("/tmp/data.pkl", "wb") as f:
  pickle.dump(potential_sets, f)
